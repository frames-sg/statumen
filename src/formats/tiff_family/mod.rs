pub(crate) mod container;
pub(crate) mod error;
pub(crate) mod layout;
pub(crate) mod pixel_access;

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use lru::LruCache;

use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::error::WsiError;
use tracing::debug;

use self::container::TiffContainer;
use self::layout::aperio::AperioInterpreter;
use self::layout::generic::GenericTiffInterpreter;
use self::layout::leica::LeicaInterpreter;
use self::layout::ndpi::NdpiInterpreter;
use self::layout::philips::PhilipsInterpreter;
use self::layout::trestle::TrestleInterpreter;
use self::layout::ventana::VentanaInterpreter;
use self::layout::TiffLayoutInterpreter;
use self::pixel_access::TiffPixelReader;

// ── TiffFamilyBackend ────────────────────────────────────────────────

/// Backend that handles all TIFF-based WSI formats. Implements both
/// `FormatProbe` (detection) and `DatasetReader` (opening) traits.
///
/// Probing does a full `TiffContainer::open()` which parses the entire
/// IFD chain. The parsed container is cached so `open()` doesn't
/// redundantly re-parse — the amortized cost of probe+open is a single parse.
pub(crate) struct TiffFamilyBackend {
    probe_cache: Mutex<LruCache<PathBuf, Arc<TiffContainer>>>,
    interpreters: Vec<Box<dyn TiffLayoutInterpreter>>,
}

impl TiffFamilyBackend {
    pub fn new() -> Self {
        Self {
            probe_cache: Mutex::new(LruCache::new(NonZeroUsize::new(16).unwrap())),
            interpreters: vec![
                Box::new(NdpiInterpreter),
                Box::new(AperioInterpreter),
                Box::new(LeicaInterpreter),
                Box::new(PhilipsInterpreter),
                Box::new(TrestleInterpreter),
                Box::new(VentanaInterpreter),
                Box::new(GenericTiffInterpreter), // must be last — catches any tiled TIFF
            ],
        }
    }

    /// Canonical key for the probe cache. Uses fs::canonicalize when
    /// possible, falls back to the raw path.
    fn cache_key(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    /// Find the first interpreter that detects the given container.
    fn find_interpreter(&self, container: &TiffContainer) -> Option<&dyn TiffLayoutInterpreter> {
        self.interpreters
            .iter()
            .find(|i| i.detect(container))
            .map(|i| i.as_ref())
    }
}

impl FormatProbe for TiffFamilyBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        // Parse the full container (IFD chain).
        // If the file isn't a valid TIFF, return detected=false — not an error.
        // This lets other backends in the registry try their probes.
        let container = match TiffContainer::open(path) {
            Ok(c) => c,
            Err(err) => {
                if has_extension(path, "ndpi") {
                    return Err(err.into_wsi_error(path));
                }
                return Ok(ProbeResult {
                    detected: false,
                    vendor: String::new(),
                    confidence: ProbeConfidence::Likely,
                });
            }
        };

        // Try each interpreter's detect() against the parsed container
        if let Some(interp) = self.find_interpreter(&container) {
            let vendor = interp.vendor_name().to_string();
            // Cache the container for open() to consume
            let key = Self::cache_key(path);
            let mut cache = self.probe_cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.put(key, Arc::new(container));

            Ok(ProbeResult {
                detected: true,
                vendor,
                confidence: ProbeConfidence::Definite,
            })
        } else {
            // No interpreter matched — container dropped, not cached
            Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            })
        }
    }
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(expected))
}

impl DatasetReader for TiffFamilyBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let started = Instant::now();
        let key = Self::cache_key(path);

        // Try to consume the cached container from probe()
        let cached_container = {
            let mut cache = self.probe_cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.pop(&key)
        };
        let cache_hit = cached_container.is_some();

        // If cache miss (e.g., open() called without prior probe()), re-parse
        let container = match cached_container {
            Some(c) => c,
            None => {
                let c = TiffContainer::open(path).map_err(|e| e.into_wsi_error(path))?;
                Arc::new(c)
            }
        };

        // Find matching interpreter
        let interpreter = self.find_interpreter(&container).ok_or_else(|| {
            WsiError::UnsupportedFormat(format!(
                "no TIFF layout interpreter detected for: {}",
                path.display(),
            ))
        })?;

        // Interpret the container → DatasetLayout
        let interpret_started = Instant::now();
        let layout = interpreter
            .interpret(&container)
            .map_err(|e| e.into_wsi_error(path))?;

        debug!(
            path = %path.display(),
            vendor = interpreter.vendor_name(),
            cache_hit,
            interpret_elapsed_ms = interpret_started.elapsed().as_secs_f64() * 1000.0,
            open_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "interpreted TIFF dataset layout"
        );

        // Build pixel reader
        let reader = TiffPixelReader::new(container, layout);
        Ok(Box::new(reader))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::ColorSpace;
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn encode_test_jpeg(image: &image::RgbImage) -> Vec<u8> {
        let mut encoded = Vec::new();
        JpegEncoder::new(&mut encoded, 50)
            .encode(
                image.as_raw().as_slice(),
                image.width() as u16,
                image.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        encoded
    }

    /// Build a synthetic NDPI TIFF file with embedded JPEG data.
    /// Each entry is (width, height, source_lens).
    fn build_ndpi_tiff(entries: &[(u32, u32, f32)]) -> NamedTempFile {
        // Step 1: Build minimal JPEG for each entry
        let mut jpeg_blocks: Vec<Vec<u8>> = Vec::new();
        for &(w, h, _) in entries {
            let actual_w = w.min(64);
            let actual_h = h.min(64);
            let rgb = image::RgbImage::new(actual_w, actual_h);
            jpeg_blocks.push(encode_test_jpeg(&rgb));
        }

        let mut buf = Vec::new();

        // TIFF header: little-endian, classic TIFF
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        let first_ifd_offset_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Write JPEG data blocks
        let mut strip_offsets: Vec<u32> = Vec::new();
        let mut strip_byte_counts: Vec<u32> = Vec::new();
        for jpeg in &jpeg_blocks {
            strip_offsets.push(buf.len() as u32);
            strip_byte_counts.push(jpeg.len() as u32);
            buf.extend_from_slice(jpeg);
        }

        // Write IFDs
        let mut ifd_offsets: Vec<u32> = Vec::new();
        let mut next_ifd_patch_positions: Vec<usize> = Vec::new();

        for (i, &(w, h, lens)) in entries.iter().enumerate() {
            let ifd_offset = buf.len() as u32;
            ifd_offsets.push(ifd_offset);

            let mut tags: Vec<(u16, u16, u32, [u8; 4])> = vec![
                (256, 4, 1, w.to_le_bytes()),                    // IMAGE_WIDTH
                (257, 4, 1, h.to_le_bytes()),                    // IMAGE_LENGTH
                (273, 4, 1, strip_offsets[i].to_le_bytes()),     // STRIP_OFFSETS
                (279, 4, 1, strip_byte_counts[i].to_le_bytes()), // STRIP_BYTE_COUNTS
                (65421, 11, 1, lens.to_le_bytes()),              // SOURCELENS (float)
            ];

            // Add NDPI marker tag to first IFD
            if i == 0 {
                tags.push((65420, 4, 1, 1u32.to_le_bytes())); // NDPI marker
            }

            tags.sort_by_key(|t| t.0);

            let entry_count = tags.len() as u16;
            buf.extend_from_slice(&entry_count.to_le_bytes());

            for (tag_id, type_id, count, value) in &tags {
                buf.extend_from_slice(&tag_id.to_le_bytes());
                buf.extend_from_slice(&type_id.to_le_bytes());
                buf.extend_from_slice(&count.to_le_bytes());
                buf.extend_from_slice(value);
            }

            // NDPI uses 8-byte next-IFD pointers
            let next_pos = buf.len();
            buf.extend_from_slice(&0u64.to_le_bytes());
            next_ifd_patch_positions.push(next_pos);
        }

        // Patch first IFD offset
        buf[first_ifd_offset_pos..first_ifd_offset_pos + 4]
            .copy_from_slice(&ifd_offsets[0].to_le_bytes());

        // Chain IFDs
        for i in 0..ifd_offsets.len().saturating_sub(1) {
            let next = ifd_offsets[i + 1] as u64;
            let pos = next_ifd_patch_positions[i];
            buf[pos..pos + 8].copy_from_slice(&next.to_le_bytes());
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn probe_detects_ndpi() {
        let file = build_ndpi_tiff(&[(1024, 768, 40.0)]);
        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path()).unwrap();

        assert!(result.detected);
        assert_eq!(result.vendor, "hamamatsu-ndpi");
        assert_eq!(result.confidence, ProbeConfidence::Definite);
    }

    #[test]
    fn probe_rejects_non_tiff() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"this is not a tiff file at all").unwrap();
        file.flush().unwrap();

        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path());

        // Non-TIFF files should return detected=false, not an error.
        // This lets other backends in the registry try their probes.
        let probe_result = result.unwrap();
        assert!(!probe_result.detected);
    }

    #[test]
    fn probe_reports_malformed_ndpi_instead_of_hiding_parse_error() {
        let mut file = tempfile::Builder::new().suffix(".ndpi").tempfile().unwrap();
        file.write_all(b"II").unwrap();
        file.write_all(&42u16.to_le_bytes()).unwrap();
        file.write_all(&1024u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let backend = TiffFamilyBackend::new();
        let err = backend
            .probe(file.path())
            .expect_err("malformed .ndpi should surface parser error");

        assert!(
            err.to_string().contains("first IFD offset")
                || err.to_string().contains("Error reading TIFF"),
            "got: {err}"
        );
    }

    #[test]
    fn probe_rejects_plain_tiff_without_ndpi() {
        // Build a valid TIFF but without NDPI marker tag
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());

        let ifd_offset = 8u32;
        buf.extend_from_slice(&ifd_offset.to_le_bytes());

        // Simple IFD with just IMAGE_WIDTH and IMAGE_LENGTH
        let entry_count = 2u16;
        buf.extend_from_slice(&entry_count.to_le_bytes());

        // Tag 256 IMAGE_WIDTH
        buf.extend_from_slice(&256u16.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes()); // LONG
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&512u32.to_le_bytes());

        // Tag 257 IMAGE_LENGTH
        buf.extend_from_slice(&257u16.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes()); // LONG
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&384u32.to_le_bytes());

        // Next IFD offset = 0 (end of chain)
        buf.extend_from_slice(&0u32.to_le_bytes());

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();

        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path()).unwrap();

        assert!(!result.detected);
        assert!(result.vendor.is_empty());
    }

    #[test]
    fn open_produces_slide_reader() {
        let file = build_ndpi_tiff(&[(1024, 768, 40.0)]);
        let backend = TiffFamilyBackend::new();

        // First probe, then open (the normal flow)
        let probe_result = backend.probe(file.path()).unwrap();
        assert!(probe_result.detected);

        let source = backend.open(file.path()).unwrap();
        let dataset = source.dataset();

        assert_eq!(dataset.scenes.len(), 1);
        let series = &dataset.scenes[0].series[0];
        assert_eq!(series.levels.len(), 9);
        assert_eq!(series.levels[0].dimensions, (1024, 768));
        assert_eq!(series.levels[1].dimensions, (512, 384));
        assert_eq!(series.levels[2].dimensions, (256, 192));
        assert_eq!(series.levels[8].dimensions, (4, 3));
    }

    #[test]
    fn open_without_prior_probe_works() {
        let file = build_ndpi_tiff(&[(512, 384, 20.0)]);
        let backend = TiffFamilyBackend::new();

        // Skip probe — call open() directly
        let source = backend.open(file.path()).unwrap();
        let dataset = source.dataset();

        assert_eq!(dataset.scenes.len(), 1);
        assert_eq!(dataset.scenes[0].series[0].levels[0].dimensions, (512, 384));
    }

    // ── Registration order tests ────────────────────────────────────

    /// Build a synthetic tiled TIFF with Aperio-style detection tags.
    /// Returns a NamedTempFile with valid JPEG tile data.
    fn build_aperio_tiff(width: u32, height: u32) -> NamedTempFile {
        let tw = 256u32.min(width);
        let th = 256u32.min(height);
        let rgb = image::RgbImage::new(tw, th);
        let jpeg = encode_test_jpeg(&rgb);

        // Write Aperio ImageDescription as out-of-line ASCII
        let desc = b"Aperio Image Library|AppMag = 40|MPP = 0.25\0";

        let mut buf = Vec::new();
        // TIFF header
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Write JPEG tile data
        let tile_offset = buf.len() as u32;
        let tile_byte_count = jpeg.len() as u32;
        buf.extend_from_slice(&jpeg);

        // Write ImageDescription string
        let desc_offset = buf.len() as u32;
        buf.extend_from_slice(desc);

        // Write IFD
        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

        let mut tags_vec: Vec<(u16, u16, u32, [u8; 4])> = vec![
            (256, 4, 1, width.to_le_bytes()),  // IMAGE_WIDTH
            (257, 4, 1, height.to_le_bytes()), // IMAGE_LENGTH
            (259, 3, 1, {
                // COMPRESSION = JPEG (7)
                let mut v = [0u8; 4];
                v[..2].copy_from_slice(&7u16.to_le_bytes());
                v
            }),
            (270, 2, desc.len() as u32, desc_offset.to_le_bytes()), // IMAGE_DESCRIPTION (out-of-line)
            (322, 4, 1, tw.to_le_bytes()),                          // TILE_WIDTH
            (323, 4, 1, th.to_le_bytes()),                          // TILE_LENGTH
            (324, 4, 1, tile_offset.to_le_bytes()),                 // TILE_OFFSETS
            (325, 4, 1, tile_byte_count.to_le_bytes()),             // TILE_BYTE_COUNTS
        ];
        tags_vec.sort_by_key(|t| t.0);

        buf.extend_from_slice(&(tags_vec.len() as u16).to_le_bytes());
        for (tag, typ, count, val) in &tags_vec {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&typ.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(val);
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn probe_detects_aperio() {
        let file = build_aperio_tiff(1024, 768);
        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path()).unwrap();
        assert!(result.detected);
        assert_eq!(result.vendor, "aperio");
    }

    #[test]
    fn specific_vendor_beats_generic() {
        // An Aperio-like file should be detected as "aperio", not "generic-tiff"
        let file = build_aperio_tiff(512, 384);
        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path()).unwrap();
        assert!(result.detected);
        assert_eq!(result.vendor, "aperio");
    }

    #[test]
    fn ndpi_still_detected_first() {
        // Regression test: NDPI files should still be detected, not caught by generic
        let file = build_ndpi_tiff(&[(2048, 1536, 40.0)]);
        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path()).unwrap();
        assert!(result.detected);
        assert_eq!(result.vendor, "hamamatsu-ndpi");
    }

    // ── TiledIfd end-to-end test ─────────────────────────────────

    #[test]
    fn aperio_open_and_read_tile() {
        let file = build_aperio_tiff(64, 64);
        let backend = TiffFamilyBackend::new();

        let source = backend.open(file.path()).unwrap();
        let dataset = source.dataset();
        assert_eq!(dataset.scenes.len(), 1);
        assert_eq!(dataset.scenes[0].series[0].levels[0].dimensions, (64, 64));

        // Read tile (0, 0) — should succeed with JPEG-decoded data
        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        assert_eq!(tile.width, 64);
        assert_eq!(tile.height, 64);
        assert_eq!(tile.channels, 3);
        assert_eq!(tile.color_space, ColorSpace::Rgb);
    }

    /// Build a generic tiled TIFF (no vendor-specific tags).
    fn build_generic_tiled_tiff(width: u32, height: u32) -> NamedTempFile {
        let tw = 256u32.min(width);
        let th = 256u32.min(height);
        let rgb = image::RgbImage::new(tw, th);
        let jpeg = encode_test_jpeg(&rgb);

        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        let tile_offset = buf.len() as u32;
        let tile_byte_count = jpeg.len() as u32;
        buf.extend_from_slice(&jpeg);

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

        let mut tags_vec: Vec<(u16, u16, u32, [u8; 4])> = vec![
            (256, 4, 1, width.to_le_bytes()),
            (257, 4, 1, height.to_le_bytes()),
            (259, 3, 1, {
                let mut v = [0u8; 4];
                v[..2].copy_from_slice(&7u16.to_le_bytes());
                v
            }),
            (322, 4, 1, tw.to_le_bytes()),
            (323, 4, 1, th.to_le_bytes()),
            (324, 4, 1, tile_offset.to_le_bytes()),
            (325, 4, 1, tile_byte_count.to_le_bytes()),
        ];
        tags_vec.sort_by_key(|t| t.0);

        buf.extend_from_slice(&(tags_vec.len() as u16).to_le_bytes());
        for (tag, typ, count, val) in &tags_vec {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&typ.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(val);
        }
        buf.extend_from_slice(&0u32.to_le_bytes());

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn generic_tiff_detected_as_fallback() {
        let file = build_generic_tiled_tiff(256, 256);
        let backend = TiffFamilyBackend::new();
        let result = backend.probe(file.path()).unwrap();
        assert!(result.detected);
        assert_eq!(result.vendor, "generic-tiff");
    }

    #[test]
    fn generic_tiff_open_and_read_tile() {
        let file = build_generic_tiled_tiff(64, 64);
        let backend = TiffFamilyBackend::new();

        let source = backend.open(file.path()).unwrap();
        let dataset = source.dataset();
        assert_eq!(dataset.scenes.len(), 1);
        assert_eq!(dataset.properties.vendor(), Some("generic-tiff"));

        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        assert_eq!(tile.width, 64);
        assert_eq!(tile.height, 64);
        assert_eq!(tile.channels, 3);
    }

    // ── Review finding tests ─────────────────────────────────────

    /// Build a tiled TIFF with uncompressed RGB data (compression=1).
    fn build_uncompressed_tiled_tiff(width: u32, height: u32, big_endian: bool) -> NamedTempFile {
        let spp: u32 = 3;
        let raw_size = width as usize * height as usize * spp as usize;
        // Write test pattern: pixel (x, y) = (x % 256, y % 256, 128)
        let mut raw = vec![0u8; raw_size];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                raw[idx] = (x % 256) as u8;
                raw[idx + 1] = (y % 256) as u8;
                raw[idx + 2] = 128;
            }
        }

        let bom: &[u8] = if big_endian { b"MM" } else { b"II" };

        let mut buf = Vec::new();
        buf.extend_from_slice(bom);
        buf.extend_from_slice(&to_bytes_u16(42, big_endian));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&to_bytes_u32(0, big_endian)); // placeholder

        // Write raw tile data
        let tile_offset = buf.len() as u32;
        let tile_byte_count = raw.len() as u32;
        buf.extend_from_slice(&raw);

        // IFD
        let ifd_offset = buf.len() as u32;
        {
            let p = first_ifd_pos;
            let bytes = to_bytes_u32(ifd_offset, big_endian);
            buf[p..p + 4].copy_from_slice(&bytes);
        }

        // Tags: sorted by ID
        let mut tags_data: Vec<(u16, u16, u32, [u8; 4])> = vec![
            (256, 4, 1, to_bytes_u32_arr(width, big_endian)), // IMAGE_WIDTH
            (257, 4, 1, to_bytes_u32_arr(height, big_endian)), // IMAGE_LENGTH
            (258, 3, 1, to_short_in_long(8, big_endian)),     // BITS_PER_SAMPLE
            (259, 3, 1, to_short_in_long(1, big_endian)),     // COMPRESSION = None
            (262, 3, 1, to_short_in_long(2, big_endian)),     // PHOTOMETRIC = RGB
            (277, 3, 1, to_short_in_long(spp as u16, big_endian)), // SAMPLES_PER_PIXEL
            (322, 4, 1, to_bytes_u32_arr(width, big_endian)), // TILE_WIDTH
            (323, 4, 1, to_bytes_u32_arr(height, big_endian)), // TILE_LENGTH
            (324, 4, 1, to_bytes_u32_arr(tile_offset, big_endian)), // TILE_OFFSETS
            (325, 4, 1, to_bytes_u32_arr(tile_byte_count, big_endian)), // TILE_BYTE_COUNTS
        ];
        tags_data.sort_by_key(|t| t.0);

        buf.extend_from_slice(&to_bytes_u16(tags_data.len() as u16, big_endian));
        for (tag, typ, count, val) in &tags_data {
            buf.extend_from_slice(&to_bytes_u16(*tag, big_endian));
            buf.extend_from_slice(&to_bytes_u16(*typ, big_endian));
            buf.extend_from_slice(&to_bytes_u32(*count, big_endian));
            buf.extend_from_slice(val);
        }
        buf.extend_from_slice(&to_bytes_u32(0, big_endian)); // next IFD = 0

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn to_bytes_u16(v: u16, big_endian: bool) -> [u8; 2] {
        if big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn to_bytes_u32(v: u32, big_endian: bool) -> [u8; 4] {
        if big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn to_bytes_u32_arr(v: u32, big_endian: bool) -> [u8; 4] {
        to_bytes_u32(v, big_endian)
    }

    /// Encode a SHORT value in the 4-byte tag value slot.
    fn to_short_in_long(v: u16, big_endian: bool) -> [u8; 4] {
        let mut arr = [0u8; 4];
        let bytes = to_bytes_u16(v, big_endian);
        arr[..2].copy_from_slice(&bytes);
        arr
    }

    #[test]
    fn uncompressed_tiled_tiff_le_read() {
        let file = build_uncompressed_tiled_tiff(8, 8, false);
        let backend = TiffFamilyBackend::new();
        let source = backend.open(file.path()).unwrap();
        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        assert_eq!(tile.width, 8);
        assert_eq!(tile.height, 8);
        assert_eq!(tile.channels, 3);
        // Verify test pattern: pixel (0,0) = (0, 0, 128)
        let data = tile.data.as_u8().unwrap();
        assert_eq!(data[0], 0); // R
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 128); // B
                                  // pixel (1,0) = (1, 0, 128)
        assert_eq!(data[3], 1);
        assert_eq!(data[4], 0);
        assert_eq!(data[5], 128);
    }

    #[test]
    fn uncompressed_tiled_tiff_be_read() {
        // Big-endian TIFF with uncompressed RGB u8 data.
        // u8 data is endian-neutral, so this tests that the IFD parsing
        // and tag decoding handles big-endian correctly.
        let file = build_uncompressed_tiled_tiff(8, 8, true);
        let backend = TiffFamilyBackend::new();
        let source = backend.open(file.path()).unwrap();
        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        assert_eq!(tile.width, 8);
        assert_eq!(tile.height, 8);
        let data = tile.data.as_u8().unwrap();
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 0);
        assert_eq!(data[2], 128);
    }

    /// Build a tiled TIFF with uncompressed u16 grayscale data to test endianness.
    fn build_u16_grayscale_tiff(width: u32, height: u32, big_endian: bool) -> NamedTempFile {
        let bom: &[u8] = if big_endian { b"MM" } else { b"II" };
        let spp: u32 = 1;
        let pixel_count = (width * height) as usize;

        // Write u16 test pattern: value = x + y * width
        let mut raw = Vec::with_capacity(pixel_count * 2);
        for y in 0..height {
            for x in 0..width {
                let val = (x + y * width) as u16;
                if big_endian {
                    raw.extend_from_slice(&val.to_be_bytes());
                } else {
                    raw.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(bom);
        buf.extend_from_slice(&to_bytes_u16(42, big_endian));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&to_bytes_u32(0, big_endian));

        let tile_offset = buf.len() as u32;
        let tile_byte_count = raw.len() as u32;
        buf.extend_from_slice(&raw);

        let ifd_offset = buf.len() as u32;
        {
            let p = first_ifd_pos;
            buf[p..p + 4].copy_from_slice(&to_bytes_u32(ifd_offset, big_endian));
        }

        let mut tags_data: Vec<(u16, u16, u32, [u8; 4])> = vec![
            (256, 4, 1, to_bytes_u32_arr(width, big_endian)),
            (257, 4, 1, to_bytes_u32_arr(height, big_endian)),
            (258, 3, 1, to_short_in_long(16, big_endian)), // BITS_PER_SAMPLE = 16
            (259, 3, 1, to_short_in_long(1, big_endian)),  // COMPRESSION = None
            (262, 3, 1, to_short_in_long(1, big_endian)),  // PHOTOMETRIC = MinIsBlack
            (277, 3, 1, to_short_in_long(spp as u16, big_endian)),
            (322, 4, 1, to_bytes_u32_arr(width, big_endian)),
            (323, 4, 1, to_bytes_u32_arr(height, big_endian)),
            (324, 4, 1, to_bytes_u32_arr(tile_offset, big_endian)),
            (325, 4, 1, to_bytes_u32_arr(tile_byte_count, big_endian)),
        ];
        tags_data.sort_by_key(|t| t.0);

        buf.extend_from_slice(&to_bytes_u16(tags_data.len() as u16, big_endian));
        for (tag, typ, count, val) in &tags_data {
            buf.extend_from_slice(&to_bytes_u16(*tag, big_endian));
            buf.extend_from_slice(&to_bytes_u16(*typ, big_endian));
            buf.extend_from_slice(&to_bytes_u32(*count, big_endian));
            buf.extend_from_slice(val);
        }
        buf.extend_from_slice(&to_bytes_u32(0, big_endian));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn u16_grayscale_big_endian_decode() {
        let file = build_u16_grayscale_tiff(4, 4, true);
        let backend = TiffFamilyBackend::new();
        let source = backend.open(file.path()).unwrap();
        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        assert_eq!(tile.width, 4);
        assert_eq!(tile.height, 4);
        assert_eq!(tile.channels, 1);
        let data = tile.data.as_u16().unwrap();
        // pixel (0,0) = 0, pixel (1,0) = 1, pixel (0,1) = 4
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 1);
        assert_eq!(data[4], 4);
    }

    #[test]
    fn u16_grayscale_little_endian_decode() {
        let file = build_u16_grayscale_tiff(4, 4, false);
        let backend = TiffFamilyBackend::new();
        let source = backend.open(file.path()).unwrap();
        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        let data = tile.data.as_u16().unwrap();
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 1);
        assert_eq!(data[4], 4);
    }

    /// Build a MinIsWhite grayscale TIFF.
    fn build_min_is_white_tiff(width: u32, height: u32) -> NamedTempFile {
        let spp: u32 = 1;
        let raw_size = (width * height) as usize;
        // White background (0 = white in MinIsWhite), pattern: value = x
        let mut raw = vec![0u8; raw_size];
        for y in 0..height as usize {
            for x in 0..width as usize {
                raw[y * width as usize + x] = (x % 256) as u8;
            }
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        let tile_offset = buf.len() as u32;
        let tile_byte_count = raw.len() as u32;
        buf.extend_from_slice(&raw);

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

        let mut tags_data: Vec<(u16, u16, u32, [u8; 4])> = vec![
            (256, 4, 1, width.to_le_bytes()),
            (257, 4, 1, height.to_le_bytes()),
            (258, 3, 1, {
                let mut v = [0u8; 4];
                v[..2].copy_from_slice(&8u16.to_le_bytes());
                v
            }),
            (259, 3, 1, {
                let mut v = [0u8; 4];
                v[..2].copy_from_slice(&1u16.to_le_bytes());
                v
            }), // None
            (262, 3, 1, {
                let mut v = [0u8; 4];
                v[..2].copy_from_slice(&0u16.to_le_bytes());
                v
            }), // MinIsWhite
            (277, 3, 1, {
                let mut v = [0u8; 4];
                v[..2].copy_from_slice(&(spp as u16).to_le_bytes());
                v
            }),
            (322, 4, 1, width.to_le_bytes()),
            (323, 4, 1, height.to_le_bytes()),
            (324, 4, 1, tile_offset.to_le_bytes()),
            (325, 4, 1, tile_byte_count.to_le_bytes()),
        ];
        tags_data.sort_by_key(|t| t.0);

        buf.extend_from_slice(&(tags_data.len() as u16).to_le_bytes());
        for (tag, typ, count, val) in &tags_data {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&typ.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(val);
        }
        buf.extend_from_slice(&0u32.to_le_bytes());

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn min_is_white_inversion() {
        let file = build_min_is_white_tiff(8, 8);
        let backend = TiffFamilyBackend::new();
        let source = backend.open(file.path()).unwrap();
        let req = crate::core::types::TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = source.read_tile_cpu(&req).unwrap();
        assert_eq!(tile.channels, 1);
        assert_eq!(tile.color_space, ColorSpace::Grayscale);
        let data = tile.data.as_u8().unwrap();
        // In MinIsWhite, raw 0 = white → inverted to 255
        assert_eq!(data[0], 255); // pixel (0,0): raw=0 → 255
        assert_eq!(data[1], 254); // pixel (1,0): raw=1 → 254
        assert_eq!(data[7], 248); // pixel (7,0): raw=7 → 248
    }
}
