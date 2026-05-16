# statumen Workspace

`statumen` is a Rust whole-slide image reader workspace.

## Members

- `statumen`: the root library crate for format probing, metadata normalization,
  tile reads, region composition, cache policy, and optional device-output
  plumbing.
- `statumen-openslide-shim`: an OpenSlide-compatible C ABI shim backed by the
  root library.
- `xtask`: repository-local check and release commands.

The library remains at the repository root. Do not move it into a `crates/*`
layout or add new workspace crates for maintainability refactors.

For the detailed module map, invariants, backend registration order, and
contribution rules for new formats, see [`docs/architecture.md`](docs/architecture.md).
