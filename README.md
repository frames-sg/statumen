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

```toml
[dependencies]
statumen = "0.1"
```

## Quick Start

```rust,no_run
use std::path::Path;
use statumen::{PlaneSelection, Slide, TileOutputPreference, TilePixels, TileRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slide = Slide::open(Path::new("sample.svs"))?;
    let req = TileRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };

    let tile = slide.read_tile(&req, TileOutputPreference::cpu())?;
    if let TilePixels::Cpu(cpu) = tile {
        println!("decoded {}x{} tile", cpu.width, cpu.height);
    }
    Ok(())
}
```

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
statumen = { version = "0.1", features = ["metal"] }
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
