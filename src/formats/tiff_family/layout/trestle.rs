use std::collections::HashMap;
use std::path::PathBuf;

use crate::core::types::*;
use crate::decode::jpeg::jpeg_dimensions;
use crate::formats::tiff_family::container::{tags, TiffContainer};
use crate::formats::tiff_family::error::TiffParseError;
use crate::properties::Properties;

use super::{
    compute_tiff_dataset_identity, DatasetLayout, TiffLayoutInterpreter, TileSource, TileSourceKey,
};

const TAG_SOFTWARE: u16 = 305;
const TAG_X_POSITION: u16 = 286;
const TAG_Y_POSITION: u16 = 287;
const TRESTLE_SOFTWARE_PREFIX: &str = "MedScan";

pub(crate) struct TrestleInterpreter;

impl TiffLayoutInterpreter for TrestleInterpreter {
    fn vendor_name(&self) -> &'static str {
        "trestle"
    }

    fn detect(&self, container: &TiffContainer) -> bool {
        let Some(&first_ifd) = container.top_ifds().first() else {
            return false;
        };

        let Ok(software) = container.get_string(first_ifd, TAG_SOFTWARE) else {
            return false;
        };
        if !software.starts_with(TRESTLE_SOFTWARE_PREFIX) {
            return false;
        }

        if container
            .get_string(first_ifd, tags::IMAGE_DESCRIPTION)
            .is_err()
        {
            return false;
        }

        container.top_ifds().iter().all(|&ifd_id| {
            container
                .ifd_by_id(ifd_id)
                .map(|ifd| ifd.tags.contains_key(&tags::TILE_WIDTH))
                .unwrap_or(false)
        })
    }

    fn interpret(&self, container: &TiffContainer) -> Result<DatasetLayout, TiffParseError> {
        let top_ifds = container.top_ifds();
        let (&first_ifd, &lowest_ifd) = top_ifds
            .first()
            .zip(top_ifds.last())
            .ok_or_else(|| TiffParseError::Structure("Trestle slide has no IFDs".into()))?;

        let desc = container
            .get_string(first_ifd, tags::IMAGE_DESCRIPTION)?
            .to_string();
        let parsed_desc = parse_trestle_description(&desc);
        let overlap_pairs = parse_overlap_pairs(parsed_desc.get("OverlapsXY"));

        let mut levels = Vec::with_capacity(top_ifds.len());
        let mut tile_sources = HashMap::with_capacity(top_ifds.len());
        let mut base_dims: Option<(u64, u64)> = None;

        for (level_idx, &ifd_id) in top_ifds.iter().enumerate() {
            let width = container.get_u64(ifd_id, tags::IMAGE_WIDTH)?;
            let height = container.get_u64(ifd_id, tags::IMAGE_LENGTH)?;
            let tile_width = container.get_u32(ifd_id, tags::TILE_WIDTH)?;
            let tile_height = container.get_u32(ifd_id, tags::TILE_LENGTH)?;
            let compression_val = container.get_u32(ifd_id, tags::COMPRESSION).unwrap_or(1);
            let compression = compression_from_tag(compression_val);
            let tiles_across = width.div_ceil(tile_width as u64);
            let tiles_down = height.div_ceil(tile_height as u64);

            let (overlap_x, overlap_y) = overlap_pairs.get(level_idx).copied().unwrap_or((0, 0));
            let public_tile_width = tile_width.saturating_sub(overlap_x);
            let public_tile_height = tile_height.saturating_sub(overlap_y);
            if public_tile_width == 0 || public_tile_height == 0 {
                return Err(TiffParseError::Structure(format!(
                    "Trestle overlap {}x{} consumes tile {}x{} on IFD {}",
                    overlap_x, overlap_y, tile_width, tile_height, ifd_id
                )));
            }

            let public_width = if width >= tile_width as u64 {
                width.saturating_sub((tiles_across.saturating_sub(1)) * overlap_x as u64)
            } else {
                width
            };
            let public_height = if height >= tile_height as u64 {
                height.saturating_sub((tiles_down.saturating_sub(1)) * overlap_y as u64)
            } else {
                height
            };

            let downsample = if let Some((base_w, base_h)) = base_dims {
                let dw = base_w as f64 / public_width as f64;
                let dh = base_h as f64 / public_height as f64;
                (dw + dh) / 2.0
            } else {
                base_dims = Some((public_width, public_height));
                1.0
            };

            let mut tiles = HashMap::with_capacity((tiles_across * tiles_down) as usize);
            for row in 0..tiles_down {
                for col in 0..tiles_across {
                    tiles.insert(
                        (col as i64, row as i64),
                        TileEntry {
                            offset: (0.0, 0.0),
                            dimensions: (tile_width, tile_height),
                            tiff_tile_index: None,
                        },
                    );
                }
            }

            levels.push(Level {
                dimensions: (public_width, public_height),
                downsample,
                tile_layout: TileLayout::Irregular {
                    tile_advance: (public_tile_width as f64, public_tile_height as f64),
                    extra_tiles: (0, 0, 0, 0),
                    tiles,
                },
            });

            tile_sources.insert(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: level_idx as u32,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: container
                        .get_bytes(ifd_id, tags::JPEG_TABLES)
                        .ok()
                        .map(|bytes| bytes.to_vec()),
                    compression,
                },
            );
        }

        let mut associated_images = HashMap::new();
        let mut associated_sources = HashMap::new();
        if let Some(path) = sibling_path(container.path(), ".Full") {
            let data = std::fs::read(&path).map_err(|err| {
                TiffParseError::Structure(format!(
                    "failed to read Trestle macro image {}: {err}",
                    path.display()
                ))
            })?;
            let (width, height) = jpeg_dimensions(&data).map_err(|err| {
                TiffParseError::Structure(format!(
                    "failed to decode Trestle macro image {} dimensions: {err}",
                    path.display()
                ))
            })?;
            associated_images.insert(
                "macro".into(),
                AssociatedImage {
                    dimensions: (width, height),
                    sample_type: SampleType::Uint8,
                    channels: 3,
                },
            );
            associated_sources.insert("macro".into(), TileSource::ExternalJpeg { path });
        }

        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "trestle");
        properties.insert("openslide.comment", desc.clone());
        if let Ok(software) = container.get_string(first_ifd, TAG_SOFTWARE) {
            properties.insert("tiff.Software", software.to_string());
        }
        if let Ok(datetime) = container.get_string(first_ifd, super::TAG_DATETIME) {
            properties.insert("tiff.DateTime", datetime.to_string());
        }
        if let Ok(host) = container.get_string(first_ifd, super::TAG_HOST_COMPUTER) {
            properties.insert("tiff.HostComputer", host.to_string());
        }
        if let Ok(copyright) = container.get_string(first_ifd, super::TAG_COPYRIGHT) {
            properties.insert("tiff.Copyright", copyright.to_string());
        }
        properties.insert("tiff.ImageDescription", desc);

        if let Ok(x_res) = container.get_f64(first_ifd, tags::X_RESOLUTION) {
            properties.insert("tiff.XResolution", x_res.to_string());
            properties.insert("openslide.mpp-x", x_res.to_string());
        }
        if let Ok(y_res) = container.get_f64(first_ifd, tags::Y_RESOLUTION) {
            properties.insert("tiff.YResolution", y_res.to_string());
            properties.insert("openslide.mpp-y", y_res.to_string());
        }
        if let Ok(x_pos) = container.get_f64(first_ifd, TAG_X_POSITION) {
            properties.insert("tiff.XPosition", x_pos.to_string());
        }
        if let Ok(y_pos) = container.get_f64(first_ifd, TAG_Y_POSITION) {
            properties.insert("tiff.YPosition", y_pos.to_string());
        }
        if let Ok(unit) = container.get_u32(first_ifd, tags::RESOLUTION_UNIT) {
            let value = match unit {
                1 => "none",
                2 => "inch",
                3 => "centimeter",
                _ => "unknown",
            };
            properties.insert("tiff.ResolutionUnit", value);
        }

        for (key, value) in parsed_desc {
            properties.insert(format!("trestle.{key}"), value.clone());
            match key.as_str() {
                "Objective Power" => properties.insert("openslide.objective-power", value),
                "Background Color" => properties.insert("openslide.background-color", value),
                _ => {}
            }
        }

        let identity = compute_tiff_dataset_identity(container, lowest_ifd, first_ifd)?;
        if let Some(quickhash) = identity.quickhash1.as_deref() {
            properties.insert("openslide.quickhash-1", quickhash);
        }

        let dataset = Dataset {
            id: identity.dataset_id,
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

fn compression_from_tag(val: u32) -> Compression {
    match val {
        1 => Compression::None,
        5 => Compression::Lzw,
        8 | 32946 => Compression::Deflate,
        6 | 7 => Compression::Jpeg,
        50000 => Compression::Zstd,
        33003 | 33005 => Compression::Jp2kYcbcr,
        33004 => Compression::Jp2kRgb,
        _ => Compression::Other(val as u16),
    }
}

fn parse_trestle_description(desc: &str) -> HashMap<String, String> {
    let mut props = HashMap::new();
    for entry in desc.split(';') {
        let Some((key, value)) = entry.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if !key.is_empty() && !value.is_empty() {
            props.insert(key.to_string(), value.to_string());
        }
    }
    props
}

fn parse_overlap_pairs(value: Option<&String>) -> Vec<(u32, u32)> {
    let Some(value) = value else {
        return Vec::new();
    };

    let values: Vec<u32> = value
        .split_whitespace()
        .filter_map(|part| part.parse::<u32>().ok())
        .collect();
    values
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

fn sibling_path(path: &std::path::Path, extension: &str) -> Option<PathBuf> {
    let stem = path.file_stem()?;
    let mut filename = stem.to_os_string();
    filename.push(extension);
    let sibling = path.with_file_name(filename);
    sibling.is_file().then_some(sibling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_trestle_description_extracts_key_value_pairs() {
        let parsed = parse_trestle_description(
            "Background Color=E6E6E6;White Balance=C0AAA1;Objective Power=10;OverlapsXY= 64 64 32 32",
        );

        assert_eq!(
            parsed.get("Background Color").map(String::as_str),
            Some("E6E6E6")
        );
        assert_eq!(
            parsed.get("White Balance").map(String::as_str),
            Some("C0AAA1")
        );
        assert_eq!(
            parsed.get("Objective Power").map(String::as_str),
            Some("10")
        );
        assert_eq!(
            parsed.get("OverlapsXY").map(String::as_str),
            Some("64 64 32 32")
        );
    }

    #[test]
    fn parse_overlap_pairs_groups_values_by_level() {
        assert_eq!(
            parse_overlap_pairs(Some(&"64 64 32 32 16 16".to_string())),
            vec![(64, 64), (32, 32), (16, 16)]
        );
    }
}
