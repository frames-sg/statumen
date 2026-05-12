<!-- SPDX-License-Identifier: Apache-2.0 -->

# statumen

[![CI](https://github.com/frames-sg/statumen/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/statumen/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/statumen.svg)](https://crates.io/crates/statumen)
[![docs.rs](https://img.shields.io/docsrs/statumen)](https://docs.rs/statumen)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Overview

`statumen` is a Rust whole-slide image (WSI) reader. It owns container
probing, metadata normalization, slide / scene / series / level geometry,
tile addressing, region composition, associated images, and `.svcache`
read-through policy. JPEG and JPEG 2000 codec work is delegated to the
`signinum-*` crates.

The main crate forbids `unsafe` code. The workspace also includes an optional
OpenSlide-compatible C ABI shim for tools that already load `libopenslide`.

## Install

```sh
cargo add statumen
```

## Quick Start

The simplest public API reads a region in level coordinates and returns an
`image::RgbaImage`.

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

Use tile-level APIs when you are writing a viewer, cache, benchmark, or
compressed-tile workflow that needs exact tile coordinates.

```rust,no_run
use statumen::{PlaneSelection, Slide, TileOutputPreference, TilePixels, TileRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slide = Slide::open("sample.svs")?;
    let req = TileRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };

    match slide.read_tile(&req, TileOutputPreference::cpu())? {
        TilePixels::Cpu(tile) => {
            println!("{}x{} tile with {} channels", tile.width, tile.height, tile.channels);
        }
        TilePixels::Device(_) => unreachable!("CPU output was requested"),
    }
    Ok(())
}
```

Associated images such as labels, macros, and thumbnails are exposed through
the dataset metadata and `read_associated`:

```rust,no_run
use statumen::Slide;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slide = Slide::open("sample.svs")?;
    if slide.dataset().associated_images.contains_key("thumbnail") {
        let thumbnail = slide.read_associated("thumbnail")?;
        println!("thumbnail: {}x{}", thumbnail.width, thumbnail.height);
    }
    Ok(())
}
```

## Fast Path For LLM-Assisted Use

If you are asking an LLM to use this repository, give it this instruction:

> Use `statumen` to open whole-slide image files and read regions or tiles.
> Use `Slide::open` plus `read_region_rgba` for the first working prototype.
> Use `read_tile` / `read_tiles` only when exact source tile coordinates or
> compressed-tile behavior matters.

For a quick script, ask the LLM to:

1. Add `statumen` as a Rust dependency.
2. Open the slide with `Slide::open("path/to/slide.svs")`.
3. Build a `RegionRequest` for scene 0, series 0, level 0.
4. Call `read_region_rgba`.
5. Save the returned image or pass it to the next analysis step.

## Supported Inputs

Statumen detects the container, normalizes slide geometry, and delegates codec
decode to Signinum.

| Input family | Typical paths | Notes |
| --- | --- | --- |
| TIFF-family WSI | `.svs`, `.tif`, `.tiff`, `.ndpi`, `.scn`, `.bif` | Includes common Aperio, Hamamatsu NDPI, Leica, Philips, Ventana, Trestle, and generic tiled TIFF layouts where metadata is available. |
| DICOM VL WSI | `.dcm` files or a DICOM series directory | Opens single instances or sibling pyramid instances from the same series. Supports JPEG baseline, JPEG 2000, HTJ2K transfer syntaxes, RLE lossless 8-bit frames, native uncompressed little/big endian 8-bit frames, associated images, and sparse tiled frame maps. |
| Zeiss | `.czi`, `.zvi` | Reads Zeiss CZI and legacy ZVI slide data. |
| MIRAX | `.mrxs` plus sibling data files | Reads slide metadata and tiles through the Statumen format adapter. |
| Hamamatsu VMS/VMU | `.vms`, `.vmu` plus sibling image files | Reads legacy Hamamatsu multi-file slides. |
| Olympus VSI | `.vsi` plus the matching `_<stem>_` ETS companion directory | Reads Olympus whole-slide containers backed by `frame_t.ets` data. |
| Raw JPEG 2000 codestream | `.j2k`, `.j2c` | Single-image raw codestream workflow for fixtures and codec tests. JP2 boxes are not the raw-file entry point. |
| `.svcache` | `.svcache` | Statumen's zstd-compressed cache format for prebuilt display tiles and associated images. |

Unsupported or incomplete sources return `WsiError`; they should not silently
produce black or partial pixels.

## Opening Options

`Slide::open` uses the built-in registry with deterministic cache defaults and
does not silently rewrite a source path to `.svcache`. Use
`Slide::open_with_options` when callers need explicit cache budgets,
read-through `.svcache` lookup, custom format registries, region limits, or
decode execution settings.

```rust,no_run
use statumen::{CacheConfig, Slide, SlideOpenOptions, SvcachePolicy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = SlideOpenOptions::default()
        .with_cache_config(
            CacheConfig::deterministic()
                .with_shared_tile_bytes(256 * 1024 * 1024)
                .with_display_tile_bytes(32 * 1024 * 1024),
        )
        .with_svcache_policy(SvcachePolicy::PreferFresh);

    let slide = Slide::open_with_options("sample.svs", options)?;
    println!("{} scene(s)", slide.dataset().scenes.len());
    Ok(())
}
```

Build `.svcache` files with the library functions or the `svcache` binary:

```sh
cargo run --release --bin svcache -- build sample.svs --out sample.svs.svcache
cargo run --release --bin svcache -- build-window sample.svs --size 2048x2048 --center 10000,10000
```

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

| Feature | Default | Description |
| --- | --- | --- |
| `metal` | off | Enables Metal-backed device payload plumbing through `signinum-jpeg-metal` and `signinum-j2k-metal` on macOS. |
| `cuda` | off | Reserved for CUDA-backed payloads. |
| `bench` | off | Builds benchmark and cache-gate binaries that have no system dependencies. |
| `openslide-bench` | off | Builds OpenSlide comparison binaries; requires `libopenslide` through `pkg-config`. |
| `parity-openslide` | off | Enables OpenSlide compatibility-oracle parity tests and benches through `libloading`. |
| `parity-metal` | off | Enables Metal-backed parity comparisons on macOS. |

### Metal Example

The `metal` feature opts in to device-resident output. Applications that create
Metal sessions directly should also depend on the adapter crates they name.

```toml
[dependencies]
statumen = { version = "0.3.0", features = ["metal"] }
metal = "0.31"
signinum-jpeg-metal = "0.4"
signinum-j2k-metal = "0.4"
```

```rust,ignore
use statumen::{PlaneSelection, Slide, TileOutputPreference, TilePixels, TileRequest};

let device = metal::Device::system_default()
    .ok_or_else(|| std::io::Error::other("no system Metal device"))?;
let sessions = statumen::output::metal::MetalBackendSessions::new(
    signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
    signinum_j2k_metal::MetalBackendSession::new(device),
);

let slide = Slide::open("sample.svs")?;
let req = TileRequest {
    scene: 0,
    series: 0,
    level: 0,
    plane: PlaneSelection::default(),
    col: 0,
    row: 0,
};

let output = TileOutputPreference::prefer_device_auto_with_metal_and_compressed_decode(sessions);
match slide.read_tile(&req, output)? {
    TilePixels::Device(device_tile) => {
        // Upload or sample the resident Metal buffer in the caller's renderer.
        let _ = device_tile;
    }
    TilePixels::Cpu(cpu_tile) => {
        // `PreferDevice` can fall back to CPU. Use
        // `require_device_auto_with_metal_and_compressed_decode` to reject fallback.
        let _ = cpu_tile;
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Supported Platforms

| Target | CPU pipeline | Metal feature |
| --- | :---: | :---: |
| `x86_64-unknown-linux-gnu` | yes | no |
| `x86_64-apple-darwin` | yes | yes |
| `aarch64-apple-darwin` | yes | yes |
| `x86_64-pc-windows-msvc` | yes | no |

## Minimum Supported Rust Version

This crate tracks the latest stable Rust toolchain. The current MSRV is
declared in `Cargo.toml` as `rust-version = "1.94"`. Bumping the MSRV is
treated as a minor-version change and is recorded in `CHANGELOG.md`.

## Performance And Benchmarks

CPU JPEG tile batches route through Signinum's scoped batch decoder by default
when jobs share the same TIFF/DICOM color transform and exact decoded tile
dimensions. Mixed or irregular JPEG jobs fall back to the conservative
per-tile path with the same output behavior.

Benchmark tools live in the repository under `benches/`, `scripts/`, and
`src/bin/` and are excluded from the published crate tarball. Useful entry
points include:

```sh
cargo test
cargo test --features parity-openslide --test openslide_parity
cargo run --release --features bench --bin wsi_bench -- path/to/slide.svs
cargo run --release --features "bench openslide-bench" --bin bench_driver -- path/to/slide.svs thumbnail
```

Optional Iris comparison is wired through `scripts/iris_bench.py` and
`bench_driver`. Iris consumes pre-encoded `.iris` slides, so set
`WSI_BENCH_INCLUDE_IRIS=1` plus either `WSI_IRIS_SLIDE_PATH=/path/file.iris`
or `WSI_IRIS_SLIDE_DIR=/path/to/iris-slides`. Set `WSI_BENCH_GATE_IRIS=1`
only when the run should fail if Statumen is slower than Iris.

## Development

The repository uses an `xtask` wrapper for CI-style checks:

```sh
cargo xtask fmt
cargo xtask clippy
cargo xtask test
cargo xtask doc
```

`cargo xtask ci` runs the full sequence, including package checks. Some parity
tests require local WSI corpora or external libraries; those tests are ignored
unless the documented environment variables are set.

## Codec Library

Production JPEG and JPEG 2000 decode is delegated to the `signinum-*` crate
family. Cite the relevant Signinum artifact for codec methods, ROI /
restart-marker APIs, batch decode, or decode-performance claims. Cite this
workspace separately for reader behavior, container parsing, normalized slide
geometry, and OpenSlide shim behavior.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and our
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The change log lives in
[`CHANGELOG.md`](CHANGELOG.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
