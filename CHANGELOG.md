<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/jcwal1516/statumen/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/jcwal1516/statumen/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/jcwal1516/statumen/releases/tag/v0.1.1
