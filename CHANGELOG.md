<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2026-05-06

### Changed

- Raised the Metal JPEG adapter dependency to `signinum-jpeg-metal` 0.2.2 so
  strict device-decode requests preserve resident Metal output for fast 4:4:4
  JPEG tiles.

## [0.1.4] - 2026-05-06

### Added

- Added a required compressed-device tile output preference so downstream
  callers can reject CPU decode fallback when they need resident device pixels.

## [0.1.3] - 2026-05-05

### Fixed

- Malformed `.ndpi` files now preserve TIFF parser errors during probe
  instead of being reported as a generic unsupported format.
- NDPI files whose first IFD offset points beyond the file length now
  report a truncation-oriented structure error.

## [0.1.2] - 2026-05-05

### Added

- Raw JPEG tile passthrough API for callers that want to forward
  encoded JPEG bitstreams without decoding (`bbd938c`).
- NDPI Metal tile batch decode path (`b209c4d`).

### Changed

- JPEG 2000 decode is now routed through the `signinum` facade
  (`2c49887`).
- Loosened `signinum-*` dependency constraints from exact `=X.Y.Z`
  pins to caret ranges so downstream users receive compatible
  patch releases.
- Pinned the temporary `signinum` patch source to the GPU codec API commit
  used by the Metal passthrough work until those APIs are available from
  crates.io releases.
- Updated `lru` to avoid the `RUSTSEC-2026-0002` advisory.

### Documentation

- Added `CHANGELOG.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md`.
- Expanded `README.md` with badges, a Metal feature example, and an
  MSRV / supported-platform matrix.

## [0.1.1]

Initial public release on crates.io.

### Added

- Container parsers for TIFF, SVS, NDPI, DICOM, Zeiss CZI, ZVI,
  Mirax, Hamamatsu VMS, and Philips TIFF.
- `Slide` / `Dataset` / `Scene` / `Level` / `Plane` geometry model
  and the `SlideReader` trait.
- DICOM VL Whole Slide Microscopy pyramid support assembled from a
  single file or sibling instances in the same series, including
  JPEG baseline, JPEG 2000, RLE Lossless 8-bit, and the uncompressed
  Explicit/Implicit VR Little Endian and Explicit VR Big Endian
  transfer syntaxes.
- Optional Metal-backed device payload plumbing (`metal` feature).
- Bench harness binaries (`wsi_bench`, `openslide_bench`,
  `bench_driver`, `release_gate`) gated behind cargo features.

[Unreleased]: https://github.com/jcwal1516/statumen/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/jcwal1516/statumen/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/jcwal1516/statumen/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/jcwal1516/statumen/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/jcwal1516/statumen/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/jcwal1516/statumen/releases/tag/v0.1.1
