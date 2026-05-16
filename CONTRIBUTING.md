<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to statumen

Thanks for taking the time to contribute. This document describes the
expectations for patches, tests, and review on this crate.

## Code of Conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide
by its terms.

## Development setup

```bash
# Install the toolchain pinned by the MSRV declared in `Cargo.toml`.
rustup show

# Run the same gates that CI runs.
cargo xtask validate
```

`cargo xtask validate` runs `fmt`, `clippy`, `bench-check`, `nextest`, and
`doc`. `cargo xtask bench-check` compiles the Rust benchmark targets without
running timings; use `cargo xtask bench` for the synthetic local Criterion
benchmarks. `cargo xtask ci` runs `validate` plus `package`.

## Branching and commits

- The default branch is `main`. Direct commits to `main` are accepted for
  small, focused changes; larger work goes through a PR.
- Commit messages should be imperative and short. Use prefixes when useful
  (`feat:`, `fix:`, `chore:`, `ci:`, `docs:`).
- Keep each commit self-contained: building / formatting / linting / tests
  should pass at every commit, not only at the tip.

## Refactor boundaries

- Keep the main `statumen` library at the repository root.
- Do not add workspace crates or move the library under `crates/*` for
  maintainability-only work.
- Prefer focused module directories inside the existing crate when a file grows
  too large to review comfortably.
- Preserve deliberate public re-exports from `src/lib.rs`; internal module
  splits should not accidentally expand the public API.

## Tests

- Unit tests live next to the code they cover.
- Integration tests live under `tests/`.
- Behavior-focused tests are preferred over implementation-coupled ones.
- Aim for ≥ 80% changed-path coverage. If something is genuinely
  hard to cover, document the gap in the PR description.

## Reporting issues

Please include:

1. statumen version (`cargo pkgid`).
2. Rust toolchain (`rustc --version`).
3. Operating system and architecture.
4. The smallest reproducer you can share, including the WSI container
   format if relevant.

## Security

If you believe you have found a security vulnerability, please **do not**
open a public issue. Email the maintainers privately and we will
coordinate a fix and disclosure.

## License

By contributing, you agree that your contributions will be licensed under
the Apache License, Version 2.0, as declared in `LICENSE`.
