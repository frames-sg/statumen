# ziggurat Architecture

`ziggurat` owns whole-slide container parsing, metadata normalization, tile
addressing, region composition, and cache policy plumbing. Image-codec work is
delegated to `ashlar`; app runtime policy is owned by SlideViewer.

## Layers

- `core`: public WSI types, `Slide`, `SlideReader`, typed read context, and
  cache primitives.
- `formats`: container readers for TIFF-family WSI, DICOM, MIRAX, Zeiss,
  Hamamatsu VMS, and `.svcache`.
- `decode`: JPEG/JPEG 2000/tile decompression glue that converts WSI requests
  into ashlar calls.
- `output`: optional device-output session plumbing, currently Metal.
- `bin`: benchmark and cache-building entry points.

Dependencies flow from format readers inward to `core` and outward only through
`decode`/`output` adapter glue. Format readers must not type-erase cache state
or reach into SlideViewer runtime policy.

## Public Policy Boundaries

- `Slide::open` is deterministic and does not rewrite a source path to
  `.svcache`.
- `Slide::open_with_options` is the explicit entry point for cache budgets,
  `.svcache` policy, registry selection, and region limits.
- `SvcachePolicy` controls read-through cache resolution. SlideViewer maps env
  vars to this policy; the library does not read `SV_SVCACHE`.
- `TileOutputPreference` uses WSI-rs-owned `OutputBackendRequest`. Conversion to
  `ashlar_core::BackendRequest` happens only inside codec glue.
- `SlideReadContext` is the typed path for read caches and request limits.
  `Any`/downcast-based cache tunnels are not allowed.

## Stability Rules

- Keep `.svcache` builds atomic: write a temp file in the target directory and
  persist only after metadata and payload are complete.
- Sparse `.svcache` updates must preserve fresh existing tiles and append
  missing requested tiles.
- Cache budgets that depend on system memory belong in SlideViewer manager code,
  then flow into WSI-rs through `CacheConfig`.
- `tiff_family/pixel_access.rs` remains the highest-risk module. New TIFF
  behavior should move toward focused helper modules instead of expanding that
  file.
