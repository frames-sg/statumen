# statumen Architecture

Canonical crate-local **map** and **invariants** for `statumen`. For the
workspace overview, see [`/architecture.md`](../architecture.md).

`statumen` is a **leaf crate** (no in-workspace deps). It is a pure-Rust whole-slide image
reader: detect format → open dataset → read tiles → optionally surface device-resident
output. The crate forbids `unsafe`.

## 1. Module map

```
src/
├── lib.rs                    re-exports + #![forbid(unsafe_code)]
├── error.rs                  WsiError
├── properties.rs             Properties (key/value metadata)
├── core/                     types, registry, cache, content hash
│   ├── types/                public model, geometry, pixel, request, and output-policy types
│   ├── registry/             FormatProbe / DatasetReader / SlideReader / FormatRegistry / Slide
│   ├── cache.rs              TileCache + CacheKey
│   └── hash.rs               quickhash1 (slide content hashing)
├── decode/                   codec backends (signinum wrappers)
│   ├── jpeg.rs               JPEG via signinum_jpeg
│   ├── jp2k.rs               JPEG2000 (J2K codestream) via signinum_j2k_metal
│   ├── jp2k_backend.rs       narrow-subset validation, YCbCr→RGB
│   └── jp2k_codestream.rs    J2K header / tile-part parsing
├── formats/                  per-vendor backends; each implements core/registry traits
│   ├── dicom/                DICOM WSI (`src/formats/dicom/`: manifest, metadata, image/frame access, decode helpers)
│   ├── mirax.rs              Mirax / 3DHistech (.mrxs)
│   ├── hamamatsu_vms.rs      Hamamatsu VMS
│   ├── olympus_vsi.rs        Olympus VSI/ETS
│   ├── svcache.rs            statumen tile cache container
│   ├── zeiss.rs              Zeiss CZI
│   ├── zeiss_zvi.rs          Zeiss ZVI
│   └── tiff_family/          umbrella for TIFF-shaped vendors
│       ├── container.rs      TiffContainer (full IFD chain)
│       ├── pixel_access/     TiffPixelReader (`src/formats/tiff_family/pixel_access/`) plus JPEG, batching, caches, synthetic/NDPI helpers
│       ├── error.rs
│       └── layout/           per-vendor interpreters (Aperio SVS, NDPI, Leica SCN, Philips, Ventana BIF)
├── output/
│   └── metal.rs              MetalDeviceTile + signinum Metal sessions (feature `metal`)
└── build.rs                  conditional libopenslide link (feature `openslide-bench` only)
```

### Internal dependency layers

```
       core::types
            │
   ┌────────┼─────────┐
   ▼        ▼         ▼
core::cache  core::registry  (defines FormatProbe / DatasetReader / SlideReader)
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
   decode/    formats/    output/   (formats use decode; output is feature-gated)
        └─────────┬──────────┘
                  ▼
                lib.rs (re-exports)
```

A module may only `use` items from modules at or below its level. `formats/*` may use
`decode/*` and `core/*`; `decode/*` may use `core/*`; nothing under `core/` imports from
`formats/`, `decode/`, or `output/`.

## 2. Public surface

### 2.1 Traits — `core::registry`

| Trait | Contract |
|---|---|
| `FormatProbe` | `probe(&Path) -> ProbeResult` returning `{detected, vendor, confidence}`. **Must be cheap** — header sniffing only, no full parse. (`core/registry/traits.rs`) |
| `DatasetReader` | `open(&Path) -> Box<dyn SlideReader>`. Constructs the full `Dataset` (scenes/series/levels/layouts). May be expensive; results should be cacheable across reopens. (`core/registry/traits.rs`) |
| `SlideReader` | `dataset()`, `read_tile_cpu`, `read_tiles` (default impl loops `read_tile_cpu`), `read_region`. Backends override `read_tiles`/`read_region` when batching is cheaper. (`core/registry/traits.rs`) |

### 2.2 Types — `core::types`

- **Dataset hierarchy:** `Dataset` → `Scene` → `Series` → `Level`, with `TileLayout`
  (`Regular { tile_w, tile_h, cols, rows }`, `Irregular { ... }`, `WholeLevel`),
  multi-dim axes (`AxesShape { z, c, t }`, `ChannelInfo`).
- **Tile request:** `TileRequest { scene, series, level, plane, col, row }`.
- **Region request:** `RegionRequest { scene, series, level, plane, origin_px, size_px }`.
- **Tile output:** `CpuTile { width, height, channels, color_space, layout, data: CpuTileData }`
  with `CpuTileData::{U8, U16, F32}`, `ColorSpace::{Grayscale, Rgb, Rgba, YCbCr, Unknown}`,
  `CpuTileLayout::{Interleaved, Planar}`.
- **Output preference:** `TileOutputPreference::{Cpu, PreferDevice, RequireDevice}`.
- **Device tile (feature `metal`):** `MetalDeviceTile { width, height, pitch_bytes, format, storage }`, where storage is currently a Metal buffer plus byte offset.

### 2.3 Registry — `FormatRegistry`

`FormatRegistry::builtin()` registers backends *in this order*
(`core/registry/registry_impl.rs`):

1. `DicomBackend`
2. `MiraxBackend`
3. `HamamatsuVmsBackend`
4. `OlympusVsiBackend`
5. `RawJp2kBackend`
6. `ZeissZviBackend`
7. `ZeissBackend`
8. `TiffFamilyBackend`

`FormatRegistry::builtin()` registers `SvcacheBackend` before the native
backends so explicit `.svcache` opens are handled first.

`FormatRegistry::open(path)` probes all backends, picks the highest `ProbeConfidence`
(`Definite > Likely`), and on a tie returns the **first registered** match.

## 3. Format → decode → output flow

```
path  ──► FormatProbe (each backend, cheap)
            │
            ▼  highest confidence wins
        DatasetReader::open  ──► Dataset (scenes/series/levels/layouts)
            │
            ▼
        SlideReader::read_tiles(&[TileRequest], TileOutputPreference)
            │
            ├── TIFF-family ──► tiff_family::pixel_access::* ──► decode::jpeg / decode::jp2k
            ├── DICOM       ──► decode::jp2k / decode::jpeg / RLE / native (per transfer syntax)
            ├── Mirax       ──► decode::jpeg (zlib-decompressed JPEG tiles)
            ├── Hamamatsu   ──► decode::jpeg (fixed pyramid scales [2, 4, 8])
            ├── Olympus VSI ──► decode::jp2k
            ├── Zeiss CZI   ──► czi-rs + decode::jpeg
            ├── Zeiss ZVI   ──► CFB/OLE + raw/zlib/JPEG plane payloads
            └── .svcache    ──► cached RGB tile payloads
            │
            ▼
        TilePixels {Cpu(CpuTile) | Device(MetalDeviceTile)}
            │
            ▼  optional region composite via TileCache (core/registry/composition.rs)
        Region returned to caller
```

## 4. Invariants

| # | Invariant | Where |
|---|---|---|
| W1 | **Crate forbids `unsafe`.** | `src/lib.rs:1` (`#![forbid(unsafe_code)]`) |
| W2 | **Module dependency direction is one-way:** `core` → `decode`/`formats` → `output` → `lib`. Reverse imports are forbidden. | Convention; verifiable by inspection of `use` statements. |
| W3 | **`FormatProbe::probe` must not perform a full parse.** Probes run for every backend on every open; cost should be bounded by header bytes. | Doc-contract on `FormatProbe`. |
| W4 | **Confidence priority is `Definite > Likely`, ties broken by registration order.** | `core/registry/registry_impl.rs`. |
| W5 | **JPEG decode is bounded.** `MAX_JPEG_DECODE_BYTES = 512 MiB`, `JPEG_MAX_DIMENSION = 65500` to prevent OOM from crafted headers. | `src/decode/jpeg.rs:25–26`. |
| W6 | **`SlideReader::read_tile` is a default thin wrapper over `read_tiles`.** Backends override `read_tiles` when batching is cheaper; semantics for a 1-element batch must equal a single `read_tile`. | `core/registry/traits.rs`. |
| W7 | **`CacheKey` must include all addressing dims:** `(dataset_id, scene, series, level, z, c, t, col, row)`. Dropping a dim is a correctness bug. | `core/registry/composition.rs` / `core/cache.rs`. |
| W8 | **Every source backend module must be registered or intentionally absent.** Dead format modules are public-surface risk because they can advertise unsupported behavior and may be packaged accidentally. | `src/formats/mod.rs`, `core/registry/registry_impl.rs`. |

## 5. Tests, examples, benches

| Area | Path | Purpose |
|---|---|---|
| Tests | `tests/signinum_parity.rs` | CPU vs reference oracle tolerance. |
| Tests | `tests/openslide_parity.rs` | Parity with libopenslide (gated by `openslide-bench` feature). |
| Tests | `tests/real_wsi_behavior.rs` | Real-slide behavioural checks. |
| Tests | `tests/zeiss_czi.rs` | Native Zeiss CZI open and tile-read coverage. |
| Tests | `tests/zeiss_zvi.rs` | Native Zeiss ZVI merged, stacked, and mosaic coverage. |
| Tests | `tests/metal_surface_smoke.rs` | Metal device-tile smoke (feature `metal`). |
| Tests | `tests/phase3_prereq.rs` | Phase-3 prerequisite checks. |
| Examples | `examples/fw01_trace_pattern.rs` | Trace-pattern usage example. |
| Benches | `benches/wsi_pipeline.rs`, `decode_throughput.rs`, `read_paths.rs`, `openslide_parity.rs` | End-to-end throughput, decode-only throughput, read-path comparison, OpenSlide parity benchmark. |

## 6. Recipe — adding a new format

For an agent or engineer adding a new vendor:

1. **Create the backend module.** `src/formats/<vendor>.rs` (or `src/formats/<vendor>/mod.rs` if multi-file). Define `<Vendor>Backend` and any vendor-specific parsers.
2. **Implement `FormatProbe`** on `<Vendor>Backend`. Read only the bytes needed to recognise the format. Return `Definite` only when the bytes are unambiguous; otherwise `Likely`.
3. **Implement `DatasetReader`** on `<Vendor>Backend`. Construct the full `Dataset` — every scene, series, level, and `TileLayout` must be present; lazy fields are an anti-pattern here.
4. **Implement `SlideReader`** for the per-file reader you return from `open`. At minimum override `read_tile_cpu`. Override `read_tiles` and `read_region` only if batching is cheaper than the default loop.
5. **Wire the module** in `src/formats/mod.rs`: `pub(crate) mod <vendor>;`.
6. **Register the backend** in `FormatRegistry::builtin` (`src/core/registry/registry_impl.rs`). Insert at the position that respects the desired tie-break order (W4): more specific backends first.
7. **Add a parity test** under `tests/` that opens a representative file and checks at least one tile against a known oracle.
8. **Update `docs/architecture.md` §1.2 and W8** if this addition changes the surface.

Do **not**:

- Reach into `decode/` from outside `formats/` or `core/registry/`.
- Introduce `unsafe` (W1).
- Skip the registry and expose a vendor-specific public open function.
