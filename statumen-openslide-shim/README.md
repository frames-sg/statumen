# statumen-openslide-shim

OpenSlide-compatible C ABI shim backed by `statumen`.

Use this crate when an existing application already loads `libopenslide` and
you want that application to read slides through Statumen without changing the
application source code. Native Rust projects should use `statumen::Slide`
directly.

## Build

```sh
cargo build -p statumen-openslide-shim --release
```

The build produces a dynamic library:

| Platform | Library |
| --- | --- |
| macOS | `target/release/libstatumen_openslide_shim.dylib` |
| Linux | `target/release/libstatumen_openslide_shim.so` |

## Test Without Replacing System Libraries

Use a private directory first. Copy or symlink the built shim to the library
names your test client searches for:

| Platform | Loader-compatible names |
| --- | --- |
| macOS | `libopenslide.1.dylib`, `libopenslide.dylib`, `libopenslide.4.dylib` |
| Linux | `libopenslide.so.1`, `libopenslide.so`, `libopenslide.so.4` |

Then point the loader at that directory for the test command:

```sh
DYLD_LIBRARY_PATH=/tmp/statumen-openslide/lib your-openslide-client slide.svs
```

```sh
LD_LIBRARY_PATH=/tmp/statumen-openslide/lib your-openslide-client slide.svs
```

On macOS, System Integrity Protection can ignore `DYLD_LIBRARY_PATH` for
protected binaries. Use an unprotected test binary or a private prefix for
development.

## Install Into A Private Prefix

The installer copies the shim to all loader-compatible names and writes a
restore manifest. Prefer a private prefix while testing:

```sh
cargo run -p statumen-openslide-shim --bin statumen-openslide-install -- \
  install --shim target/release/libstatumen_openslide_shim.dylib \
  --prefix /tmp/statumen-openslide
```

On Linux, pass `target/release/libstatumen_openslide_shim.so` as `--shim`.

Restore backed-up libraries from the manifest:

```sh
cargo run -p statumen-openslide-shim --bin statumen-openslide-install -- \
  restore --prefix /tmp/statumen-openslide
```

## ABI Coverage

The shim currently implements the OpenSlide-style calls most viewer and test
clients need:

- vendor detection
- open/close/error/version
- level count, dimensions, downsample, and best-level lookup
- `read_region` into premultiplied ARGB
- property names and property values
- associated image names, dimensions, reads, and ICC defaults
- ICC profile size/read defaults
- cache create/set/release compatibility hooks

Unsupported files return a null handle for compatibility. Read errors zero the
destination buffer and make the handle terminal, matching OpenSlide-style error
handling.

## LLM Prompt

If you are asking an LLM to use this shim, tell it:

> Build `statumen-openslide-shim`, expose the produced dynamic library under
> the OpenSlide library names in a private prefix, and run the existing
> OpenSlide-based tool with that prefix on its library search path. Do not
> replace a system OpenSlide installation until the private-prefix test works.
