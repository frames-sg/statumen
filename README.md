<!-- SPDX-License-Identifier: Apache-2.0 -->

# statumen

[![CI](https://github.com/jcwal1516/statumen/actions/workflows/ci.yml/badge.svg)](https://github.com/jcwal1516/statumen/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/statumen.svg)](https://crates.io/crates/statumen)
[![docs.rs](https://img.shields.io/docsrs/statumen)](https://docs.rs/statumen)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Overview

`statumen` is a Signinum-native whole-slide image (WSI) reader. It owns
container parsing, slide / scene / level / plane geometry, and the
`SlideReader` trait. Codec work is delegated to the `signinum-*` crates.

## Architecture

- Container parsers: TIFF / SVS / NDPI / DICOM / Zeiss CZI / ZVI / Mirax /
  Hamamatsu / Philips TIFF.
- Slide geometry: `Slide`, `Dataset`, `SceneId`, `LevelIdx`, `PlaneIdx`.
- Signinum integration: compressed tile resolution feeds `signinum_jpeg` and
  `signinum_j2k`.
- DICOM is the unified reader for the workspace; `sv-slide` routes `.dcm`
  through the same `statumen` adapter as other WSI formats.
- Parity oracle: vendored `jpeg-decoder` and dynamically loaded compatibility
  library paths are test-only.

## DICOM

The DICOM reader supports VL Whole Slide Microscopy pyramids assembled from a
single file or sibling instances in the same series. Phase 7a coverage includes
JPEG baseline where signinum supports the JPEG bitstream, JPEG 2000, RLE
Lossless for 8-bit RGB/monochrome frames, native uncompressed Explicit VR
Little Endian, Implicit VR Little Endian, Explicit VR Big Endian for 8-bit
frames, row-major multi-frame tile addressing, associated image discovery, and
sparse tiled frame maps.

## Install

```sh
cargo add statumen
```

## Quick Start

The easiest public API is region reading. It opens a slide, reads pixels in
level coordinates, and returns an `image::RgbaImage` that can be saved or passed
to analysis code.

```rust,no_run
use statumen::{
    LevelIdx, PlaneIdx, RegionRequest, SceneId, SeriesId, Slide,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slide = Slide::open("sample.svs")?;
    let region = RegionRequest {
        scene: SceneId(0),
        series: SeriesId(0),
        level: LevelIdx(0),
        plane: PlaneIdx::default(),
        origin_px: (0, 0),
        size_px: (1024, 1024),
    };
    let image = slide.read_region_rgba(&region)?;
    image.save("region.png")?;
    Ok(())
}
```

Use tile-level APIs only when you are writing a viewer, cache, benchmark, or
compressed-tile workflow that needs exact tile coordinates.

## Fast Path For LLM-Assisted Use

If you are a pathologist or researcher asking an LLM to use this repository,
give it this instruction:

> Use `statumen` to open whole-slide image files and read regions or tiles.
> Use `Slide::open` plus `read_region_rgba` for the first working prototype.
> Use `wsi-dicom` only when the task is DICOM export, and use `signinum` only
> when the task is codec-level JPEG or JPEG 2000 work.

For a quick script, ask the LLM to:

1. Add `statumen` as a Rust dependency.
2. Open the slide with `Slide::open("path/to/slide.svs")`.
3. Build a `RegionRequest` for scene 0, series 0, level 0.
4. Call `read_region_rgba`.
5. Save the returned image or pass it to the next analysis step.

## Supported Inputs

Statumen is a reader layer. It detects the container, normalizes slide
geometry, and delegates codec decode to Signinum.

| Input family | Typical extensions | Notes |
| --- | --- | --- |
| TIFF-family WSI | `.svs`, `.tif`, `.tiff`, `.ndpi`, `.scn` | Includes common Aperio, Hamamatsu, Leica, Philips, Ventana, Trestle, and generic tiled TIFF layouts where metadata is available. |
| DICOM VL WSI | `.dcm` or DICOM series directory | Opens single instances or sibling pyramid instances from the same series. |
| MIRAX | `.mrxs` | Reads slide metadata and tiles through the Statumen format adapter. |
| Hamamatsu VMS/VMU | `.vms`, `.vmu` | Reads legacy Hamamatsu multi-file slides. |
| Olympus VSI | `.vsi` | Reads Olympus whole-slide containers. |
| Raw JPEG 2000 / HTJ2K | `.j2k`, `.jp2`, `.jpc` | Useful for codec fixtures and simple single-image workflows. |
| `.svcache` | `.svcache` | Statumen's cache format for prebuilt slide tiles. |

Unsupported or incomplete sources return `WsiError`; they should not silently
produce black or partial pixels.

## OpenSlide Compatibility Shim

The workspace includes `statumen-openslide-shim`, a C ABI library that exports
OpenSlide-compatible symbols and routes reads through Statumen. Use it when an
existing tool already loads `libopenslide` and you want to try Statumen without
rewriting that tool.

Build the shim:

```sh
cargo build -p statumen-openslide-shim --release
```

Then point a test client at the produced dynamic library with a local loader
path, or install it into a private prefix:

```sh
cargo run -p statumen-openslide-shim --bin statumen-openslide-install -- \
  install --shim target/release/libstatumen_openslide_shim.dylib \
  --prefix /tmp/statumen-openslide
```

On Linux the library suffix is `.so`; on macOS it is `.dylib`. Prefer a private
prefix while testing. The installer writes a restore manifest and can restore
backed-up libraries with:

```sh
cargo run -p statumen-openslide-shim --bin statumen-openslide-install -- \
  restore --prefix /tmp/statumen-openslide
```

See [`statumen-openslide-shim/README.md`](statumen-openslide-shim/README.md)
for ABI coverage and loader-path notes.

## Features

| Feature             | Default | Description                                                                                             |
|---------------------|---------|---------------------------------------------------------------------------------------------------------|
| `metal`             | off     | Enables Metal-backed device payload plumbing via `signinum-jpeg-metal` / `signinum-j2k-metal` (macOS).  |
| `cuda`              | off     | Reserved for CUDA-backed payloads.                                                                      |
| `bench`             | off     | Builds the `wsi_bench` binary (no system deps).                                                         |
| `openslide-bench`   | off     | Builds the `openslide_bench` binary (requires `libopenslide` on `PATH` via `pkg-config`).               |
| `parity-openslide`  | off     | Enables the OpenSlide compatibility-oracle parity tests / benches via `libloading`.                     |
| `parity-metal`      | off     | Enables Metal-backed parity comparisons (signinum CPU vs Metal). macOS only.                            |

### Metal example

The `metal` feature opts in to a device-resident output preference. The Metal
backend is macOS-only and is provided by the `signinum-jpeg-metal` and
`signinum-j2k-metal` adapter crates.

```toml
[dependencies]
statumen = { version = "0.2", features = ["metal"] }
```

```rust,ignore
use std::path::Path;
use statumen::{PlaneSelection, Slide, TileOutputPreference, TilePixels, TileRequest};

let slide = Slide::open(Path::new("sample.svs"))?;
let req = TileRequest {
    scene: 0,
    series: 0,
    level: 0,
    plane: PlaneSelection::default(),
    col: 0,
    row: 0,
};

// Prefer a Metal device-resident texture; falls back to CPU when unavailable.
let tile = slide.read_tile(&req, TileOutputPreference::metal())?;
match tile {
    TilePixels::Device(device_tile) => { /* sample on GPU */ }
    TilePixels::Cpu(cpu) => { /* fell back to CPU */ }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Supported Platforms

| Target                    | CPU pipeline | Metal feature |
|---------------------------|:------------:|:-------------:|
| `x86_64-unknown-linux-gnu`| ✅            | ❌             |
| `x86_64-apple-darwin`     | ✅            | ✅             |
| `aarch64-apple-darwin`    | ✅            | ✅             |
| `x86_64-pc-windows-msvc`  | ✅            | ❌             |

## Minimum Supported Rust Version (MSRV)

This crate targets the latest stable Rust toolchain. The current MSRV is
declared in `Cargo.toml` as `rust-version = "1.94"`. Bumping the MSRV is
treated as a minor-version change and is recorded in `CHANGELOG.md`.

## Performance

CPU JPEG tile batches route through Signinum's scoped batch decoder by default
when the jobs share the same TIFF/DICOM color transform and exact decoded tile
dimensions. Mixed or irregular JPEG jobs fall back to the conservative
per-tile path with the same output behavior.

Optional Iris comparison is wired through `scripts/iris_bench.py` and the
existing `bench_driver` workloads. Iris consumes pre-encoded `.iris` slides, so
set `WSI_BENCH_INCLUDE_IRIS=1` plus either `WSI_IRIS_SLIDE_PATH=/path/file.iris`
for a single slide or `WSI_IRIS_SLIDE_DIR=/path/to/iris-slides` for a directory
containing `<source-stem>.iris` files. Set `WSI_BENCH_GATE_IRIS=1` only when
the run should fail if statumen is slower than Iris.

Phase reports and bench harness sources live under `benches/` and `scripts/`
in the project repository (excluded from the published tarball).

## Codec Library

All production JPEG and JPEG 2000 decode is delegated to the sibling
`signinum` repository. Cite signinum's JOSS paper for codec methods,
ROI / restart-marker APIs, batch decode, and decode-performance claims.

<!-- TBD: replace with JOSS-issued DOI after acceptance -->

For reader behavior, container parsing, and SlideViewer integration, cite this
workspace separately until a reader-specific artifact exists.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and our
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The change log lives in
[`CHANGELOG.md`](CHANGELOG.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and the sibling signinum repo for codec
implementation details and its own license metadata.
