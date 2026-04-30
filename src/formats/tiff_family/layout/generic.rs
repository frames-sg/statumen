//! Generic TIFF layout interpreter.
//!
//! Fallback interpreter for any tiled TIFF that is not claimed by a
//! vendor-specific interpreter. Registered last in the interpreter chain
//! so it only fires when all specific vendors decline.

use std::collections::HashMap;

use crate::core::types::*;
use crate::formats::tiff_family::container::{tags, TiffContainer};
use crate::formats::tiff_family::error::{IfdId, TiffParseError};
use crate::properties::Properties;

use super::{
    compute_tiff_dataset_identity, DatasetLayout, TiffLayoutInterpreter, TileSource, TileSourceKey,
};

// ── Helpers ──────────────────────────────────────────────────────────

fn compression_from_tag(val: u32) -> Compression {
    match val {
        1 => Compression::None,
        5 => Compression::Lzw,
        8 | 32946 => Compression::Deflate,
        7 | 6 => Compression::Jpeg,
        50000 => Compression::Zstd,
        33003 | 33005 => Compression::Jp2kYcbcr,
        33004 => Compression::Jp2kRgb,
        _ => Compression::Other(val as u16),
    }
}

// ── Interpreter ──────────────────────────────────────────────────────

pub(crate) struct GenericTiffInterpreter;

impl TiffLayoutInterpreter for GenericTiffInterpreter {
    fn vendor_name(&self) -> &'static str {
        "generic-tiff"
    }

    fn detect(&self, container: &TiffContainer) -> bool {
        // Reject NDPI — handled by NdpiInterpreter.
        if container.is_ndpi() {
            return false;
        }

        // Reject obvious OME-TIFF: ImageDescription on first IFD contains
        // the OME XML namespace marker.
        if let Some(&first_id) = container.top_ifds().first() {
            if let Ok(desc) = container.get_string(first_id, tags::IMAGE_DESCRIPTION) {
                let lower = desc.to_ascii_lowercase();
                if lower.contains("<ome") || lower.contains("ome.xsd") {
                    return false;
                }
            }
        }

        // Accept if at least one top-level IFD has TILE_WIDTH.
        container.top_ifds().iter().any(|&ifd_id| {
            container
                .ifd_by_id(ifd_id)
                .map(|ifd| ifd.tags.contains_key(&tags::TILE_WIDTH))
                .unwrap_or(false)
        })
    }

    fn interpret(&self, container: &TiffContainer) -> Result<DatasetLayout, TiffParseError> {
        let mut tiled_ifds: Vec<TiledIfd> = Vec::new();
        let mut stripped_ifds: Vec<StrippedIfd> = Vec::new();

        // Phase 1: Walk all top-level IFDs and classify.
        for &ifd_id in container.top_ifds() {
            let ifd = container.ifd_by_id(ifd_id)?;

            let width = match container.get_u64(ifd_id, tags::IMAGE_WIDTH) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let height = match container.get_u64(ifd_id, tags::IMAGE_LENGTH) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if width == 0 || height == 0 {
                continue;
            }

            if ifd.tags.contains_key(&tags::TILE_WIDTH) {
                // Tiled IFD — pyramid level.
                let tile_width = container.get_u32(ifd_id, tags::TILE_WIDTH)?;
                let tile_height = container.get_u32(ifd_id, tags::TILE_LENGTH)?;
                let compression_val = container.get_u32(ifd_id, tags::COMPRESSION).unwrap_or(1);

                tiled_ifds.push(TiledIfd {
                    ifd_id,
                    width,
                    height,
                    tile_width,
                    tile_height,
                    compression: compression_from_tag(compression_val),
                });
            } else {
                // Stripped IFD — associated image.
                let compression_val = container.get_u32(ifd_id, tags::COMPRESSION).unwrap_or(1);
                let strip_offsets = container
                    .get_u64_array(ifd_id, tags::STRIP_OFFSETS)
                    .map(|values| values.to_vec())
                    .unwrap_or_default();
                let strip_byte_counts = container
                    .get_u64_array(ifd_id, tags::STRIP_BYTE_COUNTS)
                    .map(|values| values.to_vec())
                    .unwrap_or_default();

                stripped_ifds.push(StrippedIfd {
                    ifd_id,
                    width,
                    height,
                    compression: compression_from_tag(compression_val),
                    strip_offsets,
                    strip_byte_counts,
                });
            }
        }

        if tiled_ifds.is_empty() {
            return Err(TiffParseError::Structure(
                "No tiled IFDs found in generic TIFF".into(),
            ));
        }

        // Phase 2: Sort tiled IFDs by area descending (largest = level 0).
        tiled_ifds.sort_by(|a, b| {
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            area_b.cmp(&area_a)
        });

        let base_w = tiled_ifds[0].width;
        let base_h = tiled_ifds[0].height;

        // Phase 3: JPEG tables from tag 347 on first tiled IFD if present.
        let jpeg_tables: Option<Vec<u8>> = container
            .get_bytes(tiled_ifds[0].ifd_id, tags::JPEG_TABLES)
            .ok()
            .map(|b| b.to_vec());

        // Phase 4: Build levels and tile sources.
        let mut levels = Vec::with_capacity(tiled_ifds.len());
        let mut tile_sources = HashMap::new();

        for (level_idx, tifd) in tiled_ifds.iter().enumerate() {
            if tifd.tile_width == 0 || tifd.tile_height == 0 {
                return Err(TiffParseError::Structure(format!(
                    "Generic TIFF: tile dimensions must be > 0 (got {}x{})",
                    tifd.tile_width, tifd.tile_height
                )));
            }
            let tiles_across = tifd.width.div_ceil(tifd.tile_width as u64);
            let tiles_down = tifd.height.div_ceil(tifd.tile_height as u64);

            let downsample = if level_idx == 0 {
                1.0
            } else {
                let dw = base_w as f64 / tifd.width as f64;
                let dh = base_h as f64 / tifd.height as f64;
                (dw + dh) / 2.0
            };

            levels.push(Level {
                dimensions: (tifd.width, tifd.height),
                downsample,
                tile_layout: TileLayout::Regular {
                    tile_width: tifd.tile_width,
                    tile_height: tifd.tile_height,
                    tiles_across,
                    tiles_down,
                },
            });

            let key = TileSourceKey {
                scene: 0,
                series: 0,
                level: level_idx as u32,
                z: 0,
                c: 0,
                t: 0,
            };
            tile_sources.insert(
                key,
                TileSource::TiledIfd {
                    ifd_id: tifd.ifd_id,
                    jpeg_tables: jpeg_tables.clone(),
                    compression: tifd.compression,
                },
            );
        }

        // Phase 5: Build associated images from stripped IFDs.
        let mut associated_images: HashMap<String, AssociatedImage> = HashMap::new();
        let mut associated_sources: HashMap<String, TileSource> = HashMap::new();

        for (i, sifd) in stripped_ifds.iter().enumerate() {
            let name = format!("image_{}", i);
            associated_images.insert(
                name.clone(),
                AssociatedImage {
                    dimensions: (
                        u32::try_from(sifd.width).unwrap_or(u32::MAX),
                        u32::try_from(sifd.height).unwrap_or(u32::MAX),
                    ),
                    sample_type: SampleType::Uint8,
                    channels: 3,
                },
            );
            associated_sources.insert(
                name,
                TileSource::Stripped {
                    ifd_id: sifd.ifd_id,
                    jpeg_tables: None,
                    compression: sifd.compression,
                    strip_offsets: sifd.strip_offsets.clone(),
                    strip_byte_counts: sifd.strip_byte_counts.clone(),
                },
            );
        }

        // Phase 6: Properties.
        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "generic-tiff");

        if let Some(&first_id) = container.top_ifds().first() {
            if let Ok(desc) = container.get_string(first_id, tags::IMAGE_DESCRIPTION) {
                properties.insert("openslide.comment", desc.to_string());
            }

            // Extract MPP from TIFF XResolution / YResolution tags.
            // ResolutionUnit: 2 = inch (default), 3 = centimeter.
            let res_unit = container
                .get_u32(first_id, tags::RESOLUTION_UNIT)
                .unwrap_or(2); // default: inch
            let unit_to_microns = match res_unit {
                3 => 10_000.0, // 1 cm = 10,000 µm
                _ => 25_400.0, // 1 inch = 25,400 µm
            };
            if let Ok(x_res) = container.get_f64(first_id, tags::X_RESOLUTION) {
                if x_res > 0.0 {
                    let mpp_x = unit_to_microns / x_res;
                    properties.insert("openslide.mpp-x", format!("{mpp_x:.6}"));
                }
            }
            if let Ok(y_res) = container.get_f64(first_id, tags::Y_RESOLUTION) {
                if y_res > 0.0 {
                    let mpp_y = unit_to_microns / y_res;
                    properties.insert("openslide.mpp-y", format!("{mpp_y:.6}"));
                }
            }
        }

        // Phase 7: Dataset identity from TIFF quickhash-compatible content hashing.
        let property_ifd = *container
            .top_ifds()
            .first()
            .ok_or_else(|| TiffParseError::Structure("No IFDs in generic TIFF container".into()))?;
        let identity = compute_tiff_dataset_identity(
            container,
            tiled_ifds.last().unwrap().ifd_id,
            property_ifd,
        )?;
        if let Some(quickhash1) = identity.quickhash1.as_deref() {
            properties.insert("openslide.quickhash-1", quickhash1);
        }
        let dataset_id = identity.dataset_id;

        // Phase 8: Assemble Dataset with single Scene, single Series.
        let dataset = Dataset {
            id: dataset_id,
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels,
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images,
            properties,
            icc_profiles: HashMap::new(),
        };

        Ok(DatasetLayout {
            dataset,
            tile_sources,
            associated_sources,
        })
    }
}

// ── Internal intermediate types ──────────────────────────────────────

struct TiledIfd {
    ifd_id: IfdId,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
    compression: Compression,
}

struct StrippedIfd {
    ifd_id: IfdId,
    width: u64,
    height: u64,
    compression: Compression,
    strip_offsets: Vec<u64>,
    strip_byte_counts: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::tiff_family::container::TiffContainer;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a minimal classic TIFF file in memory with the given IFDs.
    /// Each IFD is a list of (tag, type_id, count, value_bytes).
    /// Supports only inline tags (value fits in 4 bytes) for simplicity.
    #[allow(clippy::type_complexity)]
    fn build_synthetic_tiff(ifds: &[Vec<(u16, u16, u32, [u8; 4])>]) -> NamedTempFile {
        let mut buf = Vec::new();

        // TIFF header: little-endian, classic TIFF
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        let mut ifd_offsets = Vec::new();
        let mut next_ifd_patch_positions = Vec::new();

        for tags in ifds.iter() {
            let ifd_offset = buf.len() as u32;
            ifd_offsets.push(ifd_offset);

            let mut all_tags = tags.clone();
            all_tags.sort_by_key(|t| t.0);

            let entry_count = all_tags.len() as u16;
            buf.extend_from_slice(&entry_count.to_le_bytes());

            for (tag_id, type_id, count, value) in &all_tags {
                buf.extend_from_slice(&tag_id.to_le_bytes());
                buf.extend_from_slice(&type_id.to_le_bytes());
                buf.extend_from_slice(&count.to_le_bytes());
                buf.extend_from_slice(value);
            }

            let next_pos = buf.len();
            buf.extend_from_slice(&0u32.to_le_bytes());
            next_ifd_patch_positions.push(next_pos);
        }

        // Patch first IFD offset.
        let offset_bytes = ifd_offsets[0].to_le_bytes();
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&offset_bytes);

        // Chain IFDs.
        for i in 0..ifd_offsets.len() - 1 {
            let patch_pos = next_ifd_patch_positions[i];
            let next_offset = ifd_offsets[i + 1];
            let bytes = next_offset.to_le_bytes();
            buf[patch_pos..patch_pos + 4].copy_from_slice(&bytes);
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    /// Helper: create a LONG tag value (type_id=4, count=1).
    fn long_tag(tag: u16, value: u32) -> (u16, u16, u32, [u8; 4]) {
        (tag, 4, 1, value.to_le_bytes())
    }

    /// Helper: create a SHORT tag value (type_id=3, count=1), stored in first 2 bytes.
    fn short_tag(tag: u16, value: u16) -> (u16, u16, u32, [u8; 4]) {
        let mut val = [0u8; 4];
        val[0..2].copy_from_slice(&value.to_le_bytes());
        (tag, 3, 1, val)
    }

    fn clone_tempfile(src: &NamedTempFile) -> NamedTempFile {
        let bytes = std::fs::read(src.path()).unwrap();
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        file
    }

    // ── Detection tests ──────────────────────────────────────────────

    #[test]
    fn detect_tiled_tiff() {
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 1024),
            long_tag(tags::IMAGE_LENGTH, 768),
            long_tag(tags::TILE_WIDTH, 256),
            long_tag(tags::TILE_LENGTH, 256),
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        assert!(interp.detect(&container));
    }

    #[test]
    fn reject_non_tiled_tiff() {
        // No TILE_WIDTH tag -> not tiled.
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 1024),
            long_tag(tags::IMAGE_LENGTH, 768),
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        assert!(!interp.detect(&container));
    }

    #[test]
    fn reject_ndpi() {
        // NDPI marker tag present -> NdpiInterpreter should handle it.
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 1024),
            long_tag(tags::IMAGE_LENGTH, 768),
            long_tag(tags::TILE_WIDTH, 256),
            long_tag(tags::TILE_LENGTH, 256),
            long_tag(tags::NDPI_MARKER, 1),
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        assert!(!interp.detect(&container));
    }

    // ── Interpret tests ──────────────────────────────────────────────

    #[test]
    fn interpret_single_level() {
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 1024),
            long_tag(tags::IMAGE_LENGTH, 768),
            long_tag(tags::TILE_WIDTH, 256),
            long_tag(tags::TILE_LENGTH, 256),
            short_tag(tags::COMPRESSION, 7), // JPEG
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let layout = interp.interpret(&container).unwrap();

        assert_eq!(layout.dataset.scenes.len(), 1);
        let series = &layout.dataset.scenes[0].series[0];
        assert_eq!(series.levels.len(), 1);
        assert_eq!(series.levels[0].dimensions, (1024, 768));
        assert!((series.levels[0].downsample - 1.0).abs() < 0.001);

        // tiles_across = ceil(1024/256) = 4, tiles_down = ceil(768/256) = 3
        match &series.levels[0].tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across,
                tiles_down,
            } => {
                assert_eq!(*tile_width, 256);
                assert_eq!(*tile_height, 256);
                assert_eq!(*tiles_across, 4);
                assert_eq!(*tiles_down, 3);
            }
            other => panic!("expected Regular tile layout, got: {:?}", other),
        }

        // Tile source should exist for level 0.
        let key = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        assert!(layout.tile_sources.contains_key(&key));
        match layout.tile_sources.get(&key).unwrap() {
            TileSource::TiledIfd { compression, .. } => {
                assert_eq!(*compression, Compression::Jpeg);
            }
            other => panic!("expected TiledIfd, got: {:?}", other),
        }

        // Vendor property.
        assert_eq!(layout.dataset.properties.vendor(), Some("generic-tiff"),);
        assert!(layout.dataset.properties.quickhash1().is_some());
    }

    #[test]
    fn dataset_identity_is_path_independent_for_same_contents() {
        let file_a = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 1024),
            long_tag(tags::IMAGE_LENGTH, 768),
            long_tag(tags::TILE_WIDTH, 256),
            long_tag(tags::TILE_LENGTH, 256),
            short_tag(tags::COMPRESSION, 7),
        ]]);
        let file_b = clone_tempfile(&file_a);

        let container_a = TiffContainer::open(file_a.path()).unwrap();
        let container_b = TiffContainer::open(file_b.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let layout_a = interp.interpret(&container_a).unwrap();
        let layout_b = interp.interpret(&container_b).unwrap();

        assert_eq!(layout_a.dataset.id, layout_b.dataset.id);
        assert_eq!(
            layout_a.dataset.properties.quickhash1(),
            layout_b.dataset.properties.quickhash1()
        );
    }

    #[test]
    fn interpret_multi_level_sorted() {
        // Two tiled IFDs: smaller first in file, larger second.
        // Interpreter should sort largest as level 0.
        let file = build_synthetic_tiff(&[
            vec![
                long_tag(tags::IMAGE_WIDTH, 512),
                long_tag(tags::IMAGE_LENGTH, 384),
                long_tag(tags::TILE_WIDTH, 256),
                long_tag(tags::TILE_LENGTH, 256),
            ],
            vec![
                long_tag(tags::IMAGE_WIDTH, 2048),
                long_tag(tags::IMAGE_LENGTH, 1536),
                long_tag(tags::TILE_WIDTH, 256),
                long_tag(tags::TILE_LENGTH, 256),
            ],
        ]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let layout = interp.interpret(&container).unwrap();

        let series = &layout.dataset.scenes[0].series[0];
        assert_eq!(series.levels.len(), 2);

        // Level 0 = largest.
        assert_eq!(series.levels[0].dimensions, (2048, 1536));
        assert!((series.levels[0].downsample - 1.0).abs() < 0.001);

        // Level 1 = smaller.
        assert_eq!(series.levels[1].dimensions, (512, 384));
        assert!(series.levels[1].downsample > 1.0);
        // downsample ~ avg(2048/512, 1536/384) / 1 = avg(4.0, 4.0) = 4.0
        assert!((series.levels[1].downsample - 4.0).abs() < 0.01);
    }

    #[test]
    fn interpret_stripped_as_associated() {
        // One tiled IFD + one stripped IFD.
        let file = build_synthetic_tiff(&[
            vec![
                long_tag(tags::IMAGE_WIDTH, 1024),
                long_tag(tags::IMAGE_LENGTH, 768),
                long_tag(tags::TILE_WIDTH, 256),
                long_tag(tags::TILE_LENGTH, 256),
            ],
            vec![
                long_tag(tags::IMAGE_WIDTH, 400),
                long_tag(tags::IMAGE_LENGTH, 300),
                long_tag(tags::STRIP_OFFSETS, 100),
                long_tag(tags::STRIP_BYTE_COUNTS, 500),
            ],
        ]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let layout = interp.interpret(&container).unwrap();

        // Pyramid should have 1 level.
        assert_eq!(layout.dataset.scenes[0].series[0].levels.len(), 1);

        // Associated image should exist.
        assert!(layout.dataset.associated_images.contains_key("image_0"));
        let ai = &layout.dataset.associated_images["image_0"];
        assert_eq!(ai.dimensions, (400, 300));

        // Associated source should exist.
        assert!(layout.associated_sources.contains_key("image_0"));
        match layout.associated_sources.get("image_0").unwrap() {
            TileSource::Stripped {
                strip_offsets,
                strip_byte_counts,
                ..
            } => {
                assert_eq!(strip_offsets.as_slice(), &[100]);
                assert_eq!(strip_byte_counts.as_slice(), &[500]);
            }
            other => panic!("expected Stripped, got: {:?}", other),
        }
    }

    #[test]
    fn interpret_no_tiled_ifds_returns_error() {
        // Only stripped IFDs.
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 400),
            long_tag(tags::IMAGE_LENGTH, 300),
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let result = interp.interpret(&container);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("No tiled IFDs"),
            "expected 'No tiled IFDs', got: {}",
            msg,
        );
    }

    #[test]
    fn interpret_axes_default() {
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 512),
            long_tag(tags::IMAGE_LENGTH, 512),
            long_tag(tags::TILE_WIDTH, 256),
            long_tag(tags::TILE_LENGTH, 256),
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let layout = interp.interpret(&container).unwrap();

        let axes = layout.dataset.scenes[0].series[0].axes;
        assert_eq!(axes, AxesShape { z: 1, c: 1, t: 1 });
    }

    #[test]
    fn interpret_tile_count_ceil() {
        // Width not evenly divisible by tile width.
        // 1000 / 256 = 3.90625 -> tiles_across = 4
        // 500 / 256 = 1.953125 -> tiles_down = 2
        let file = build_synthetic_tiff(&[vec![
            long_tag(tags::IMAGE_WIDTH, 1000),
            long_tag(tags::IMAGE_LENGTH, 500),
            long_tag(tags::TILE_WIDTH, 256),
            long_tag(tags::TILE_LENGTH, 256),
        ]]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interp = GenericTiffInterpreter;
        let layout = interp.interpret(&container).unwrap();

        match &layout.dataset.scenes[0].series[0].levels[0].tile_layout {
            TileLayout::Regular {
                tiles_across,
                tiles_down,
                ..
            } => {
                assert_eq!(*tiles_across, 4);
                assert_eq!(*tiles_down, 2);
            }
            other => panic!("expected Regular, got: {:?}", other),
        }
    }

    #[test]
    fn compression_mapping() {
        assert_eq!(compression_from_tag(1), Compression::None);
        assert_eq!(compression_from_tag(6), Compression::Jpeg);
        assert_eq!(compression_from_tag(7), Compression::Jpeg);
        assert_eq!(compression_from_tag(5), Compression::Lzw);
        assert_eq!(compression_from_tag(8), Compression::Deflate);
        assert_eq!(compression_from_tag(32946), Compression::Deflate);
        assert_eq!(compression_from_tag(50000), Compression::Zstd);
        assert_eq!(compression_from_tag(33003), Compression::Jp2kYcbcr);
        assert_eq!(compression_from_tag(33005), Compression::Jp2kYcbcr);
        assert_eq!(compression_from_tag(33004), Compression::Jp2kRgb);
        assert_eq!(compression_from_tag(999), Compression::Other(999));
    }
}
