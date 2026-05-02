These fixtures are generated locally with OpenJPEG tools for the exact raw-J2K subset that
`statumen` supports in production:

- raw codestream only (`.j2k`)
- 3 components
- 8-bit unsigned samples
- irreversible 9/7 DWT (`-I`)
- single tile
- default code-block and precinct settings
- direct RGB and raw YCbCr with 4:4:4 / 4:2:2 / 4:2:0 sampling

Files:

- `rgb_nomct.j2k` / `rgb_nomct.ppm`: direct RGB, no MCT
- `rgb_mct.j2k` / `rgb_mct.ppm`: RGB input with MCT enabled
- `ycbcr_444.j2k` / `ycbcr_444.ppm`: raw YCbCr 4:4:4
- `ycbcr_422.j2k` / `ycbcr_422.ppm`: raw YCbCr 4:2:2
- `ycbcr_420.j2k` / `ycbcr_420.ppm`: raw YCbCr 4:2:0

Regenerate with:

```bash
./.venv/bin/python crates/statumen/tests/fixtures/jp2k/generate.py
```

The generator requires `opj_compress` and `opj_decompress` on `PATH`.
