<!-- SPDX-License-Identifier: Apache-2.0 -->

# statumen

## Overview

`statumen` is a Signinum-native WSI reader. It owns container parsing,
slide/scene/level/plane geometry, and the `SlideReader` trait. Codec work is
delegated to the `signinum-*` crates.

## Architecture

- Container parsers: TIFF/SVS/NDPI/DICOM/Zeiss/Mirax/Hamamatsu/Philips TIFF.
- Slide geometry: `Slide`, `Dataset`, `SceneId`, `LevelIdx`, `PlaneIdx`.
- Signinum integration: compressed tile resolution feeds `signinum_jpeg`
  and `signinum_j2k`.
- DICOM is the unified reader for the workspace; `sv-slide` routes `.dcm`
  through the same `statumen` adapter as other WSI formats.
- Parity oracle: vendored `jpeg-decoder` and dynamically loaded compatibility
  library paths are test-only.

## DICOM

The DICOM reader supports VL Whole Slide Microscopy pyramids assembled from a
single file or sibling instances in the same series. Phase 7a coverage includes
JPEG baseline where signinum supports the JPEG bitstream, JPEG 2000, RLE
Lossless for 8-bit RGB/monochrome frames, native uncompressed Explicit VR Little
Endian, Implicit VR Little Endian, Explicit VR Big Endian for 8-bit frames,
row-major multi-frame tile addressing, associated image discovery, and sparse
tiled frame maps.

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

## Performance

Benchmark notes and phase reports are maintained in the project repository.

Optional Iris comparison is wired through `scripts/iris_bench.py` and the
existing `bench_driver` workloads. Iris consumes pre-encoded `.iris` slides, so
set `WSI_BENCH_INCLUDE_IRIS=1` plus either `WSI_IRIS_SLIDE_PATH=/path/file.iris`
for a single slide or `WSI_IRIS_SLIDE_DIR=/path/to/iris-slides` for a directory
containing `<source-stem>.iris` files. Set `WSI_BENCH_GATE_IRIS=1` only when the
run should fail if statumen is slower than Iris.

## Codec Library

All production JPEG and JPEG 2000 decode is delegated to the sibling
`signinum` repository. Cite signinum's JOSS paper
for codec methods, ROI/restart-marker APIs, batch decode, and decode-performance
claims.

<!-- TBD: replace with JOSS-issued DOI after acceptance -->

For reader behavior, container parsing, and SlideViewer integration, cite this
workspace separately until a reader-specific artifact exists.

## License

Apache-2.0. See the sibling signinum repo for codec
implementation details and its own license metadata.
