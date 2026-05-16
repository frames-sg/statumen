//! Ventana BIF layout interpreter.
//!
//! Detects Ventana BIF files by checking for `<iScan` in the XMP tag (700)
//! of any top-level IFD. Builds an irregular tile grid from the embedded
//! XML tile layout metadata (`SlideStitchInfo` / `ImageInfo` / `TileJointInfo`).

use std::collections::HashMap;

use crate::core::types::*;
use crate::decode::xml;
use crate::formats::tiff_family::container::{tags, TiffContainer};
use crate::formats::tiff_family::error::{IfdId, TiffParseError};
use crate::properties::Properties;

use super::{
    compute_tiff_dataset_identity, DatasetLayout, TiffLayoutInterpreter, TileSource, TileSourceKey,
};

// ── VentanaInterpreter ──────────────────────────────────────────────

pub(crate) struct VentanaInterpreter;

impl TiffLayoutInterpreter for VentanaInterpreter {
    fn vendor_name(&self) -> &'static str {
        "ventana"
    }

    fn detect(&self, container: &TiffContainer) -> bool {
        for &ifd_id in container.top_ifds() {
            if has_iscan_xmp(container, ifd_id) {
                return true;
            }
        }
        false
    }

    fn interpret(&self, container: &TiffContainer) -> Result<DatasetLayout, TiffParseError> {
        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "ventana");

        // Phase 1: Find and parse the iScan element from XMP for vendor properties.
        let xmp_str = find_xmp_string(container)?;
        if let Some(ref xmp) = xmp_str {
            parse_iscan_properties(xmp, &mut properties);
        }

        // Phase 2: Classify public associated images and the stitched pyramid.
        let mut pyramid_ifds = Vec::new();
        let mut associated_images: HashMap<String, AssociatedImage> = HashMap::new();
        let mut associated_sources: HashMap<String, TileSource> = HashMap::new();

        for &ifd_id in container.top_ifds() {
            let width = match container.get_u64(ifd_id, tags::IMAGE_WIDTH) {
                Ok(v) if v > 0 => v,
                _ => continue,
            };
            let height = match container.get_u64(ifd_id, tags::IMAGE_LENGTH) {
                Ok(v) if v > 0 => v,
                _ => continue,
            };
            let desc = container
                .get_string(ifd_id, tags::IMAGE_DESCRIPTION)
                .unwrap_or("")
                .to_ascii_lowercase();

            if let Some(name) = classify_associated_image(&desc) {
                let compression =
                    compression_from_tag(container.get_u32(ifd_id, tags::COMPRESSION).unwrap_or(1));
                let source = if let (Ok(tile_width), Ok(tile_height)) = (
                    container.get_u32(ifd_id, tags::TILE_WIDTH),
                    container.get_u32(ifd_id, tags::TILE_LENGTH),
                ) {
                    if tile_width == 0 || tile_height == 0 {
                        continue;
                    }
                    TileSource::TiledIfd {
                        ifd_id,
                        jpeg_tables: container
                            .get_bytes(ifd_id, tags::JPEG_TABLES)
                            .ok()
                            .map(|b| b.to_vec()),
                        compression,
                    }
                } else {
                    TileSource::Stripped {
                        ifd_id,
                        jpeg_tables: container
                            .get_bytes(ifd_id, tags::JPEG_TABLES)
                            .ok()
                            .map(|b| b.to_vec()),
                        compression,
                        strip_offsets: container
                            .get_u64_array(ifd_id, tags::STRIP_OFFSETS)
                            .map(|values| values.to_vec())
                            .unwrap_or_default(),
                        strip_byte_counts: container
                            .get_u64_array(ifd_id, tags::STRIP_BYTE_COUNTS)
                            .map(|values| values.to_vec())
                            .unwrap_or_default(),
                    }
                };

                associated_images.insert(
                    name.clone(),
                    AssociatedImage {
                        dimensions: (
                            u32::try_from(width).unwrap_or(u32::MAX),
                            u32::try_from(height).unwrap_or(u32::MAX),
                        ),
                        sample_type: SampleType::Uint8,
                        channels: 3,
                    },
                );
                associated_sources.insert(name, source);
                continue;
            }

            if !desc.contains("level=") {
                continue;
            }
            let tile_width = match container.get_u32(ifd_id, tags::TILE_WIDTH) {
                Ok(v) if v > 0 => v,
                _ => continue,
            };
            let tile_height = match container.get_u32(ifd_id, tags::TILE_LENGTH) {
                Ok(v) if v > 0 => v,
                _ => continue,
            };
            let compression =
                compression_from_tag(container.get_u32(ifd_id, tags::COMPRESSION).unwrap_or(1));
            let jpeg_tables = container
                .get_bytes(ifd_id, tags::JPEG_TABLES)
                .ok()
                .map(|b| b.to_vec());
            pyramid_ifds.push(VentanaPyramidIfdInfo {
                ifd_id,
                width,
                height,
                tile_width,
                tile_height,
                compression,
                jpeg_tables,
                description: desc,
            });
        }

        if pyramid_ifds.is_empty() {
            return Err(TiffParseError::Structure(
                "Ventana BIF: no tiled pyramid IFDs found".into(),
            ));
        }

        pyramid_ifds.sort_by(|a, b| {
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            area_b.cmp(&area_a)
        });

        // Phase 3: Find level 0 XML (EncodeInfo) for public level-0 bounds.
        let level0_tile_width = pyramid_ifds[0].tile_width as i64;
        let level0_tile_height = pyramid_ifds[0].tile_height as i64;
        let encode_xml = find_encode_info_xml(container)?;
        let bif = parse_level0_xml(&encode_xml, level0_tile_width, level0_tile_height)?;

        if bif.areas.is_empty() {
            return Err(TiffParseError::Structure(
                "Ventana BIF: no scanned areas found in XML".into(),
            ));
        }

        // Phase 4: Build level 0 from the XML-driven irregular tile grid,
        // then keep the lower pyramid levels on the regular overview IFDs.
        let tile_advance_x = bif.tile_advance_x;
        let tile_advance_y = bif.tile_advance_y;
        if !tile_advance_x.is_finite()
            || !tile_advance_y.is_finite()
            || tile_advance_x <= 0.0
            || tile_advance_y <= 0.0
        {
            return Err(TiffParseError::Structure(format!(
                "Ventana: tile advance must be > 0 (got {}x{})",
                tile_advance_x, tile_advance_y
            )));
        }

        let mut level0_tiles: HashMap<(i64, i64), TileEntry> =
            HashMap::with_capacity(bif.tiles.len());
        let mut extra_top = 0u32;
        let mut extra_bottom = 0u32;
        let mut extra_left = 0u32;
        let mut extra_right = 0u32;
        for area in &bif.areas {
            let offset_x = area.x as f64 - area.start_col as f64 * bif.tile_advance_x;
            let offset_y = area.y as f64 - area.start_row as f64 * bif.tile_advance_y;
            let (area_extra_top, area_extra_bottom, area_extra_left, area_extra_right) =
                irregular_extra_tiles(
                    offset_x,
                    offset_y,
                    tile_advance_x,
                    tile_advance_y,
                    level0_tile_width as f64,
                    level0_tile_height as f64,
                );
            extra_top = extra_top.max(area_extra_top);
            extra_bottom = extra_bottom.max(area_extra_bottom);
            extra_left = extra_left.max(area_extra_left);
            extra_right = extra_right.max(area_extra_right);
        }

        let tile_by_coord = bif
            .tiles
            .iter()
            .map(|tile| ((tile.col, tile.row), tile))
            .collect::<HashMap<_, _>>();
        for area in &bif.areas {
            let offset_x = area.x as f64 - area.start_col as f64 * bif.tile_advance_x;
            let offset_y = area.y as f64 - area.start_row as f64 * bif.tile_advance_y;
            for row in area.start_row..area.start_row + area.tiles_down {
                for col in area.start_col..area.start_col + area.tiles_across {
                    let Some(tile) = tile_by_coord.get(&(col, row)) else {
                        continue;
                    };
                    level0_tiles.insert(
                        (col, row),
                        TileEntry {
                            offset: (offset_x, offset_y),
                            dimensions: (tile.width, tile.height),
                            tiff_tile_index: Some(tile.tiff_tile_index),
                        },
                    );
                }
            }
        }
        let level0_dims = ventana_level0_dimensions(&bif, level0_tile_width, level0_tile_height)?;

        let mut levels = Vec::with_capacity(pyramid_ifds.len());
        levels.push(Level {
            dimensions: level0_dims,
            downsample: 1.0,
            tile_layout: TileLayout::Irregular {
                tile_advance: (tile_advance_x, tile_advance_y),
                extra_tiles: (extra_top, extra_bottom, extra_left, extra_right),
                tiles: level0_tiles,
            },
        });

        let mut tile_sources = HashMap::with_capacity(pyramid_ifds.len());
        tile_sources.insert(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::TiledIfd {
                ifd_id: pyramid_ifds[0].ifd_id,
                jpeg_tables: pyramid_ifds[0].jpeg_tables.clone(),
                compression: pyramid_ifds[0].compression,
            },
        );

        for (level_idx, info) in pyramid_ifds.iter().enumerate().skip(1) {
            let dims = ventana_public_level_dimensions(level0_dims, level_idx as u32);
            let tiles_across = info.width.div_ceil(info.tile_width as u64);
            let tiles_down = info.height.div_ceil(info.tile_height as u64);
            levels.push(Level {
                dimensions: dims,
                downsample: (1u64 << level_idx) as f64,
                tile_layout: TileLayout::Regular {
                    tile_width: info.tile_width,
                    tile_height: info.tile_height,
                    tiles_across,
                    tiles_down,
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
                    ifd_id: info.ifd_id,
                    jpeg_tables: info.jpeg_tables.clone(),
                    compression: info.compression,
                },
            );
        }

        if let Some(comment) = pyramid_ifds
            .first()
            .map(|info| info.description.as_str())
            .filter(|value| !value.is_empty())
        {
            properties.insert("openslide.comment", comment);
        }

        // Phase 5: Compute dataset ID.
        let property_ifd = pyramid_ifds
            .first()
            .map(|info| info.ifd_id)
            .ok_or_else(|| {
                TiffParseError::Structure("Ventana BIF: no pyramid IFDs found".into())
            })?;
        let identity = compute_tiff_dataset_identity(
            container,
            pyramid_ifds.last().unwrap().ifd_id,
            property_ifd,
        )?;
        if let Some(quickhash1) = identity.quickhash1.as_deref() {
            properties.insert("openslide.quickhash-1", quickhash1);
        }
        let dataset_id = identity.dataset_id;

        let dataset = Dataset {
            id: dataset_id,
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape { z: 1, c: 1, t: 1 },
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

// ── Detection helper ────────────────────────────────────────────────

/// Check if an IFD has an XMP tag containing `<iScan`.
fn has_iscan_xmp(container: &TiffContainer, ifd_id: IfdId) -> bool {
    // Try get_string first (type ASCII), fall back to get_bytes (type BYTE/Undefined).
    if let Ok(s) = container.get_string(ifd_id, tags::XMP) {
        return s.contains("iScan");
    }
    if let Ok(bytes) = container.get_bytes(ifd_id, tags::XMP) {
        if let Ok(s) = std::str::from_utf8(bytes) {
            return s.contains("iScan");
        }
        // Byte-level search as last resort.
        return bytes.windows(b"iScan".len()).any(|w| w == b"iScan");
    }
    false
}

// ── XMP parsing ─────────────────────────────────────────────────────

/// Find the first XMP tag across all top-level IFDs and return it as a string.
fn find_xmp_string(container: &TiffContainer) -> Result<Option<String>, TiffParseError> {
    for &ifd_id in container.top_ifds() {
        if let Ok(s) = container.get_string(ifd_id, tags::XMP) {
            if let Some(xmp) = extract_iscan_fragment(s) {
                return Ok(Some(xmp));
            }
        }
        if let Ok(bytes) = container.get_bytes(ifd_id, tags::XMP) {
            if let Some(xmp) = extract_iscan_fragment_bytes(bytes) {
                return Ok(Some(xmp));
            }
            if let Ok(s) = std::str::from_utf8(bytes) {
                if let Some(xmp) = extract_iscan_fragment(s) {
                    return Ok(Some(xmp));
                }
            }
        }
    }
    Ok(None)
}

/// Parse iScan attributes into vendor properties.
fn parse_iscan_properties(xmp: &str, properties: &mut Properties) {
    for (key, value) in parse_iscan_attributes(xmp) {
        if !value.is_empty() {
            properties.insert(format!("ventana.{key}"), value);
        }
    }

    if let Some(mag) = properties
        .get("ventana.Magnification")
        .map(|s| s.to_string())
    {
        if let Ok(power) = mag.parse::<f64>() {
            properties.insert("openslide.objective-power", format!("{}", power as u32));
        }
    }
    if let Some(res) = properties.get("ventana.ScanRes").map(|s| s.to_string()) {
        properties.insert("openslide.mpp-x", res.clone());
        properties.insert("openslide.mpp-y", res);
    }
}

fn parse_iscan_attributes(xmp: &str) -> Vec<(String, String)> {
    let start = match xmp.find("<iScan") {
        Some(pos) => pos + "<iScan".len(),
        None => return Vec::new(),
    };
    let end = match xmp[start..].find('>') {
        Some(pos) => start + pos,
        None => return Vec::new(),
    };
    let mut attrs = Vec::new();
    let mut rest = xmp[start..end].trim();

    while !rest.is_empty() {
        rest = rest.trim_start();
        if rest.is_empty() || rest.starts_with('/') {
            break;
        }
        let Some(eq_idx) = rest.find('=') else {
            break;
        };
        let key = rest[..eq_idx].trim();
        if key.is_empty() {
            break;
        }
        let mut value_rest = rest[eq_idx + 1..].trim_start();
        let Some(quote) = value_rest.chars().next() else {
            break;
        };
        if quote != '"' && quote != '\'' {
            break;
        }
        value_rest = &value_rest[quote.len_utf8()..];
        let Some(close_idx) = value_rest.find(quote) else {
            break;
        };
        attrs.push((key.to_string(), value_rest[..close_idx].to_string()));
        rest = &value_rest[close_idx + quote.len_utf8()..];
    }

    attrs
}

// ── Tiled IFD discovery ─────────────────────────────────────────────

struct VentanaPyramidIfdInfo {
    ifd_id: IfdId,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
    compression: Compression,
    jpeg_tables: Option<Vec<u8>>,
    description: String,
}

fn classify_associated_image(desc: &str) -> Option<String> {
    if desc.contains("thumbnail") {
        Some("thumbnail".to_string())
    } else if desc.contains("label image") || desc.contains("label_image") {
        Some("macro".to_string())
    } else {
        None
    }
}

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

// ── EncodeInfo XML discovery ────────────────────────────────────────

/// Search IFD data for `<EncodeInfo>` XML containing the tile layout.
/// Ventana stores this in one of the IFDs (typically as XMP or ImageDescription,
/// or sometimes embedded in the strip/tile data). We search all IFDs for any
/// tag payload that contains an `EncodeInfo` fragment.
fn find_encode_info_xml(container: &TiffContainer) -> Result<String, TiffParseError> {
    for &ifd_id in container.top_ifds() {
        for &tag in &[tags::IMAGE_DESCRIPTION, tags::XMP] {
            if let Ok(bytes) = container.get_bytes(ifd_id, tag) {
                if let Some(xml) = extract_encode_info_bytes(bytes) {
                    return Ok(xml);
                }
                if let Ok(s) = std::str::from_utf8(bytes) {
                    if let Some(xml) = extract_encode_info(s) {
                        return Ok(xml);
                    }
                }
            }
        }
    }
    Err(TiffParseError::Structure(
        "Ventana BIF: no EncodeInfo XML found".into(),
    ))
}

/// Extract `<EncodeInfo>...</EncodeInfo>` from a larger string.
fn extract_encode_info(s: &str) -> Option<String> {
    extract_xml_fragment(s, "<EncodeInfo", "</EncodeInfo>")
}

/// Extract `<EncodeInfo ...>...</EncodeInfo>` from raw tag bytes.
fn extract_encode_info_bytes(bytes: &[u8]) -> Option<String> {
    extract_xml_fragment_bytes(bytes, b"<EncodeInfo", b"</EncodeInfo>")
}

/// Extract `<iScan .../>` or `<iScan ...></iScan>` from a larger string.
fn extract_iscan_fragment(s: &str) -> Option<String> {
    extract_xml_fragment_with_optional_self_closing(s, "<iScan", "</iScan>")
}

/// Extract `<iScan .../>` or `<iScan ...></iScan>` from raw tag bytes.
fn extract_iscan_fragment_bytes(bytes: &[u8]) -> Option<String> {
    extract_xml_fragment_with_optional_self_closing_bytes(bytes, b"<iScan", b"</iScan>")
}

fn extract_xml_fragment(s: &str, start_tag_prefix: &str, end_tag: &str) -> Option<String> {
    let start = s.find(start_tag_prefix)?;
    let end = s[start..].find(end_tag)? + start + end_tag.len();
    Some(s[start..end].to_string())
}

fn extract_xml_fragment_bytes(
    bytes: &[u8],
    start_tag_prefix: &[u8],
    end_tag: &[u8],
) -> Option<String> {
    let start = find_bytes(bytes, start_tag_prefix)?;
    let end = find_bytes(&bytes[start..], end_tag)? + start + end_tag.len();
    Some(String::from_utf8_lossy(&bytes[start..end]).into_owned())
}

fn extract_xml_fragment_with_optional_self_closing(
    s: &str,
    start_tag_prefix: &str,
    end_tag: &str,
) -> Option<String> {
    let start = s.find(start_tag_prefix)?;
    let fragment = &s[start..];
    let end = fragment.find("/>").map(|pos| start + pos + 2).or_else(|| {
        fragment
            .find(end_tag)
            .map(|pos| start + pos + end_tag.len())
    })?;
    Some(s[start..end].to_string())
}

fn extract_xml_fragment_with_optional_self_closing_bytes(
    bytes: &[u8],
    start_tag_prefix: &[u8],
    end_tag: &[u8],
) -> Option<String> {
    let start = find_bytes(bytes, start_tag_prefix)?;
    let fragment = &bytes[start..];
    let end = find_bytes(fragment, b"/>")
        .map(|pos| start + pos + 2)
        .or_else(|| find_bytes(fragment, end_tag).map(|pos| start + pos + end_tag.len()))?;
    Some(String::from_utf8_lossy(&bytes[start..end]).into_owned())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }

    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut idx = 0;

    while idx <= last_start {
        let offset = haystack[idx..=last_start]
            .iter()
            .position(|&byte| byte == first)?;
        let candidate = idx + offset;
        if haystack[candidate..candidate + needle.len()] == *needle {
            return Some(candidate);
        }
        idx = candidate + 1;
    }

    None
}

// ── Level 0 XML parsing ────────────────────────────────────────────

/// Parsed BIF area of interest.
struct BifArea {
    x: i64,
    y: i64,
    start_col: i64,
    start_row: i64,
    tiles_across: i64,
    tiles_down: i64,
}

struct BifTile {
    col: i64,
    row: i64,
    x: f64,
    y: f64,
    width: u32,
    height: u32,
    tiff_tile_index: usize,
}

/// Parsed BIF layout metadata.
struct BifInfo {
    areas: Vec<BifArea>,
    tiles: Vec<BifTile>,
    tile_advance_x: f64,
    tile_advance_y: f64,
}

/// Parse the EncodeInfo XML for BIF tile layout.
fn parse_level0_xml(
    xml_str: &str,
    tile_width: i64,
    tile_height: i64,
) -> Result<BifInfo, TiffParseError> {
    let root = xml::parse_xml(xml_str)
        .map_err(|e| TiffParseError::Structure(format!("Ventana BIF: XML parse error: {}", e)))?;

    let slide_info = root.find("SlideStitchInfo").ok_or_else(|| {
        TiffParseError::Structure("Ventana BIF: no SlideStitchInfo in EncodeInfo XML".into())
    })?;
    let image_infos = slide_info.find_all("ImageInfo");
    let origin_infos = root
        .find("AoiOrigin")
        .map(|node| node.children.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    if !origin_infos.is_empty() && origin_infos.len() != image_infos.len() {
        return Err(TiffParseError::Structure(format!(
            "Ventana BIF: mismatched AOI/ImageInfo counts ({} vs {})",
            origin_infos.len(),
            image_infos.len()
        )));
    }

    let mut areas = Vec::new();
    let mut tiles = Vec::new();
    let mut next_tiff_tile_index: usize = 0;
    let mut total_offset_x: f64 = 0.0;
    let mut total_offset_y: f64 = 0.0;
    let mut total_x_weight: i64 = 0;
    let mut total_y_weight: i64 = 0;
    for (idx, info) in image_infos.into_iter().enumerate() {
        let aoi_scanned: i64 = info
            .attr("AOIScanned")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if aoi_scanned == 0 {
            continue;
        }
        let aoi = origin_infos.get(idx).copied();

        let num_cols: i64 = info
            .attr("NumCols")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let num_rows: i64 = info
            .attr("NumRows")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let pos_x: f64 = info
            .attr("Pos-X")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let pos_y: f64 = info
            .attr("Pos-Y")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let image_width: i64 = info.attr("Width").and_then(|s| s.parse().ok()).unwrap_or(0);
        let image_height: i64 = info
            .attr("Height")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let start_col_x: i64 = aoi
            .and_then(|node| node.attr("OriginX"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let start_row_y: i64 = aoi
            .and_then(|node| node.attr("OriginY"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if start_col_x % tile_width != 0 || start_row_y % tile_height != 0 {
            return Err(TiffParseError::Structure(format!(
                "Ventana BIF: area origin not divisible by tile size: {} % {}, {} % {}",
                start_col_x, tile_width, start_row_y, tile_height
            )));
        }
        let start_col = start_col_x / tile_width;
        let start_row = start_row_y / tile_height;

        // Accumulate joint offsets for tile advance computation.
        for joint_info in info.find_all("TileJointInfo") {
            let overlap_x: f64 = joint_info
                .attr("OverlapX")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let overlap_y: f64 = joint_info
                .attr("OverlapY")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let confidence: i64 = joint_info
                .attr("Confidence")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let direction = joint_info.attr("Direction").unwrap_or("");

            if direction == "UP" {
                total_offset_y += confidence as f64 * (-overlap_y);
                total_y_weight += confidence;
            } else {
                total_offset_x += confidence as f64 * (-overlap_x);
                total_x_weight += confidence;
            }
        }

        areas.push(BifArea {
            x: pos_x as i64,
            y: pos_y as i64,
            start_col,
            start_row,
            tiles_across: num_cols,
            tiles_down: num_rows,
        });

        let exact_positions = parse_area_tile_positions(
            info,
            num_cols,
            num_rows,
            tile_width as f64,
            tile_height as f64,
        );
        let exact_position_map = exact_positions
            .iter()
            .map(|(tile_id, (tile_x, tile_y))| {
                let (local_col, local_row) = ventana_snake_coords(*tile_id, num_cols);
                ((local_col, local_row), (*tile_x, *tile_y))
            })
            .collect::<HashMap<_, _>>();
        let area_tile_count = exact_positions.len();
        for (tile_index, (tile_id, tile_pos)) in exact_positions.into_iter().enumerate() {
            let (tile_x, tile_y) = tile_pos;
            let (local_col, local_row) = ventana_snake_coords(tile_id, num_cols);
            let (width, height) = ventana_exact_tile_dimensions(
                local_col,
                local_row,
                num_cols,
                num_rows,
                &exact_position_map,
                image_width as f64,
                image_height as f64,
                tile_width as f64,
                tile_height as f64,
            );
            tiles.push(BifTile {
                col: start_col + local_col,
                row: start_row + local_row,
                x: pos_x + tile_x,
                y: pos_y + tile_y,
                width,
                height,
                tiff_tile_index: tile_index + next_tiff_tile_index,
            });
        }
        next_tiff_tile_index += area_tile_count;
    }

    let tile_advance_x = if total_x_weight > 0 {
        tile_width as f64 + total_offset_x / total_x_weight as f64
    } else {
        tile_width as f64
    };
    let tile_advance_y = if total_y_weight > 0 {
        tile_height as f64 + total_offset_y / total_y_weight as f64
    } else {
        tile_height as f64
    };

    let mut top = 0i64;
    let heights = areas
        .iter()
        .map(|area| {
            let height =
                ((area.tiles_down - 1) as f64 * tile_advance_y + tile_height as f64).round() as i64;
            top = top.max(area.y + height);
            height
        })
        .collect::<Vec<_>>();
    for (area, height) in areas.iter_mut().zip(heights) {
        area.y = top - area.y - height;
    }

    Ok(BifInfo {
        areas,
        tiles,
        tile_advance_x,
        tile_advance_y,
    })
}

fn irregular_extra_tiles(
    offset_x: f64,
    offset_y: f64,
    tile_advance_x: f64,
    tile_advance_y: f64,
    tile_width: f64,
    tile_height: f64,
) -> (u32, u32, u32, u32) {
    let extra_right = if offset_x < 0.0 {
        (-offset_x / tile_advance_x).ceil() as u32
    } else {
        0
    };
    let offset_xr = offset_x + (tile_width - tile_advance_x);
    let extra_left = if offset_xr > 0.0 {
        (offset_xr / tile_advance_x).ceil() as u32
    } else {
        0
    };

    let extra_bottom = if offset_y < 0.0 {
        (-offset_y / tile_advance_y).ceil() as u32
    } else {
        0
    };
    let offset_yr = offset_y + (tile_height - tile_advance_y);
    let extra_top = if offset_yr > 0.0 {
        (offset_yr / tile_advance_y).ceil() as u32
    } else {
        0
    };

    (extra_top, extra_bottom, extra_left, extra_right)
}

fn parse_area_tile_positions(
    info: &xml::XmlNode,
    num_cols: i64,
    num_rows: i64,
    tile_width: f64,
    tile_height: f64,
) -> Vec<(i64, (f64, f64))> {
    let tile_count = num_cols.max(0) * num_rows.max(0);
    if tile_count == 0 {
        return Vec::new();
    }

    let mut edges: HashMap<i64, Vec<(i64, f64, f64)>> = HashMap::new();
    let mut seed_tile = None;

    for joint_info in info.find_all("TileJointInfo") {
        let Some(tile1) = joint_info.attr("Tile1").and_then(|s| s.parse::<i64>().ok()) else {
            continue;
        };
        let Some(tile2) = joint_info.attr("Tile2").and_then(|s| s.parse::<i64>().ok()) else {
            continue;
        };
        if tile1 <= 0 || tile2 <= 0 || tile1 > tile_count || tile2 > tile_count {
            continue;
        }
        seed_tile.get_or_insert(tile1);

        let overlap_x = joint_info
            .attr("OverlapX")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let overlap_y = joint_info
            .attr("OverlapY")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let Some((dx, dy)) = joint_delta(
            joint_info.attr("Direction").unwrap_or(""),
            tile_width,
            tile_height,
            overlap_x,
            overlap_y,
        ) else {
            continue;
        };

        edges.entry(tile1).or_default().push((tile2, dx, dy));
        edges.entry(tile2).or_default().push((tile1, -dx, -dy));
    }

    let mut positions: HashMap<i64, (f64, f64)> = HashMap::new();
    let mut queue = std::collections::VecDeque::new();
    let root = seed_tile.unwrap_or(1);
    positions.insert(root, (0.0, 0.0));
    queue.push_back(root);

    while let Some(tile_id) = queue.pop_front() {
        let Some((tile_x, tile_y)) = positions.get(&tile_id).copied() else {
            continue;
        };
        for &(neighbor, dx, dy) in edges.get(&tile_id).into_iter().flatten() {
            if positions.contains_key(&neighbor) {
                continue;
            }
            positions.insert(neighbor, (tile_x + dx, tile_y + dy));
            queue.push_back(neighbor);
        }
    }

    for tile_id in 1..=tile_count {
        positions.entry(tile_id).or_insert_with(|| {
            let (col, row) = ventana_snake_coords(tile_id, num_cols);
            (col as f64 * tile_width, row as f64 * tile_height)
        });
    }

    let min_x = positions
        .values()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let min_y = positions
        .values()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);

    let mut result = positions.into_iter().collect::<Vec<_>>();
    result.sort_by_key(|(tile_id, _)| *tile_id);
    for (_, (x, y)) in &mut result {
        *x -= min_x;
        *y -= min_y;
    }
    result
}

fn joint_delta(
    direction: &str,
    tile_width: f64,
    tile_height: f64,
    overlap_x: f64,
    overlap_y: f64,
) -> Option<(f64, f64)> {
    match direction {
        "RIGHT" => Some((tile_width - overlap_x, overlap_y)),
        "LEFT" => Some((-(tile_width - overlap_x), overlap_y)),
        "UP" => Some((overlap_x, tile_height - overlap_y)),
        "DOWN" => Some((overlap_x, -(tile_height - overlap_y))),
        _ => None,
    }
}

fn ventana_snake_coords(tile_id: i64, num_cols: i64) -> (i64, i64) {
    let zero_based = tile_id - 1;
    let row = zero_based.div_euclid(num_cols);
    let col_in_row = zero_based.rem_euclid(num_cols);
    let col = if row % 2 == 0 {
        col_in_row
    } else {
        num_cols - 1 - col_in_row
    };
    (col, row)
}

#[allow(clippy::too_many_arguments)]
fn ventana_exact_tile_dimensions(
    local_col: i64,
    local_row: i64,
    num_cols: i64,
    num_rows: i64,
    positions: &HashMap<(i64, i64), (f64, f64)>,
    area_width: f64,
    area_height: f64,
    fallback_width: f64,
    fallback_height: f64,
) -> (u32, u32) {
    let Some(&(tile_x, tile_y)) = positions.get(&(local_col, local_row)) else {
        return (
            fallback_width.round().max(1.0).min(u32::MAX as f64) as u32,
            fallback_height.round().max(1.0).min(u32::MAX as f64) as u32,
        );
    };

    let width = if local_col + 1 < num_cols {
        if let Some((next_x, _)) = positions.get(&(local_col + 1, local_row)) {
            let delta = next_x - tile_x;
            if delta > 0.5 {
                delta
            } else {
                let edge_width = area_width - tile_x;
                if edge_width > 0.5 {
                    edge_width
                } else if local_col > 0 {
                    positions
                        .get(&(local_col - 1, local_row))
                        .map(|(prev_x, _)| tile_x - prev_x)
                        .filter(|delta| *delta > 0.5)
                        .unwrap_or(fallback_width)
                } else {
                    fallback_width
                }
            }
        } else {
            let edge_width = area_width - tile_x;
            if edge_width > 0.5 {
                edge_width
            } else if local_col > 0 {
                positions
                    .get(&(local_col - 1, local_row))
                    .map(|(prev_x, _)| tile_x - prev_x)
                    .filter(|delta| *delta > 0.5)
                    .unwrap_or(fallback_width)
            } else {
                fallback_width
            }
        }
    } else if local_col > 0 {
        let edge_width = area_width - tile_x;
        if edge_width > 0.5 {
            edge_width
        } else {
            positions
                .get(&(local_col - 1, local_row))
                .map(|(prev_x, _)| tile_x - prev_x)
                .filter(|delta| *delta > 0.5)
                .unwrap_or(fallback_width)
        }
    } else {
        fallback_width
    };

    let height = if local_row + 1 < num_rows {
        if let Some((_, next_y)) = positions.get(&(local_col, local_row + 1)) {
            let delta = next_y - tile_y;
            if delta > 0.5 {
                delta
            } else {
                let edge_height = area_height - tile_y;
                if edge_height > 0.5 {
                    edge_height
                } else if local_row > 0 {
                    positions
                        .get(&(local_col, local_row - 1))
                        .map(|(_, prev_y)| tile_y - prev_y)
                        .filter(|delta| *delta > 0.5)
                        .unwrap_or(fallback_height)
                } else {
                    fallback_height
                }
            }
        } else {
            let edge_height = area_height - tile_y;
            if edge_height > 0.5 {
                edge_height
            } else if local_row > 0 {
                positions
                    .get(&(local_col, local_row - 1))
                    .map(|(_, prev_y)| tile_y - prev_y)
                    .filter(|delta| *delta > 0.5)
                    .unwrap_or(fallback_height)
            } else {
                fallback_height
            }
        }
    } else if local_row > 0 {
        let edge_height = area_height - tile_y;
        if edge_height > 0.5 {
            edge_height
        } else {
            positions
                .get(&(local_col, local_row - 1))
                .map(|(_, prev_y)| tile_y - prev_y)
                .filter(|delta| *delta > 0.5)
                .unwrap_or(fallback_height)
        }
    } else {
        fallback_height
    };

    (
        width.round().max(1.0).min(u32::MAX as f64) as u32,
        height.round().max(1.0).min(u32::MAX as f64) as u32,
    )
}

fn ventana_level0_dimensions(
    bif: &BifInfo,
    tile_width: i64,
    tile_height: i64,
) -> Result<(u64, u64), TiffParseError> {
    // Compatibility level dimensions come from the stitched area model
    // (tile advance plus scanned AOI bounds), not from the exact per-tile extents.
    // Keep exact tile positions for placement, but keep public dimensions aligned
    // with average-overlap geometry whenever the AOI metadata exists.
    if bif.areas.is_empty() && !bif.tiles.is_empty() {
        let min_x = bif
            .tiles
            .iter()
            .map(|tile| tile.x)
            .fold(f64::INFINITY, f64::min);
        let min_y = bif
            .tiles
            .iter()
            .map(|tile| tile.y)
            .fold(f64::INFINITY, f64::min);
        let max_right = bif
            .tiles
            .iter()
            .map(|tile| tile.x + tile.width as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_bottom = bif
            .tiles
            .iter()
            .map(|tile| tile.y + tile.height as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let width = (max_right - min_x).ceil() as u64;
        let height = (max_bottom - min_y).ceil() as u64;
        if width == 0 || height == 0 {
            return Err(TiffParseError::Structure(
                "Ventana BIF: stitched level-0 dimensions resolved to zero".into(),
            ));
        }
        return Ok((width, height));
    }

    let min_x = bif.areas.iter().map(|area| area.x).min().unwrap_or(0) as f64;
    let min_y = bif.areas.iter().map(|area| area.y).min().unwrap_or(0) as f64;
    let mut max_right = 0.0f64;
    let mut max_bottom = 0.0f64;

    for area in &bif.areas {
        if area.tiles_across <= 0 || area.tiles_down <= 0 {
            continue;
        }
        let right = (area.x as f64 - min_x)
            + (area.tiles_across - 1) as f64 * bif.tile_advance_x
            + tile_width as f64;
        let bottom = (area.y as f64 - min_y)
            + (area.tiles_down - 1) as f64 * bif.tile_advance_y
            + tile_height as f64;
        max_right = max_right.max(right);
        max_bottom = max_bottom.max(bottom);
    }

    let width = max_right.ceil() as u64;
    let height = max_bottom.ceil() as u64;
    if width == 0 || height == 0 {
        return Err(TiffParseError::Structure(
            "Ventana BIF: stitched level-0 dimensions resolved to zero".into(),
        ));
    }
    Ok((width, height))
}

fn ventana_public_level_dimensions(level0_dims: (u64, u64), level_idx: u32) -> (u64, u64) {
    let factor = 1u64 << level_idx;
    (
        level0_dims.0.div_ceil(factor),
        level0_dims.1.div_ceil(factor),
    )
}

// ── Overlap validation ──────────────────────────────────────────────

/// Validate that no two tiles in the grid overlap in raster space.
///
/// Uses a sweep-line on X: sort all tile rects by x1, then for each tile
/// check against all subsequent tiles whose x1 < current x2 (early exit
/// when sweep passes the right edge). This is O(n log n) average for
/// non-overlapping grids and catches non-adjacent overlaps that the
/// previous neighbor-only check would miss.
#[cfg(test)]
fn validate_no_adjacent_overlap(
    tiles: &HashMap<(i64, i64), TileEntry>,
    tile_advance_x: f64,
    tile_advance_y: f64,
    _tile_width: u32,
    _tile_height: u32,
) -> Result<(), TiffParseError> {
    let adv_x = tile_advance_x;
    let adv_y = tile_advance_y;

    // Build sorted list of tile rects: (x1, y1, x2, y2, col, row)
    let mut rects: Vec<(f64, f64, f64, f64, i64, i64)> = tiles
        .iter()
        .map(|(&(col, row), entry)| {
            let x1 = col as f64 * adv_x + entry.offset.0;
            let y1 = row as f64 * adv_y + entry.offset.1;
            let x2 = x1 + entry.dimensions.0 as f64;
            let y2 = y1 + entry.dimensions.1 as f64;
            (x1, y1, x2, y2, col, row)
        })
        .collect();

    // Sort by x1 ascending for sweep-line
    rects.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..rects.len() {
        let (x1, y1, x2, y2, col, row) = rects[i];
        for &(nx1, ny1, nx2, ny2, nc, nr) in rects.iter().skip(i + 1) {
            // Sweep-line exit: if next tile's left edge >= our right edge, no
            // further tiles can overlap us in X (they're sorted by x1).
            if nx1 >= x2 {
                break;
            }
            // Check Y overlap
            let intersection_x = (x2.min(nx2) - x1.max(nx1)).max(0.0);
            let intersection_y = (y2.min(ny2) - y1.max(ny1)).max(0.0);
            let overlap_area = intersection_x * intersection_y;
            if overlap_area > 0.0 {
                return Err(TiffParseError::Structure(format!(
                    "Ventana BIF: tiles ({},{}) and ({},{}) overlap by {:.1} pixels",
                    col, row, nc, nr, overlap_area,
                )));
            }
        }
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_level0_xml ────────────────────────────────────────────

    #[test]
    fn parse_level0_xml_basic() {
        let xml = r#"<EncodeInfo>
            <SlideStitchInfo>
                <ImageInfo AOIScanned="1" NumCols="3" NumRows="2" Width="256" Height="256"
                           Pos-X="0" Pos-Y="0">
                    <TileJointInfo Direction="RIGHT" OverlapX="10" OverlapY="0" Confidence="100"
                                   Tile1="1" Tile2="2"/>
                    <TileJointInfo Direction="UP" OverlapX="0" OverlapY="8" Confidence="100"
                                   Tile1="1" Tile2="4"/>
                </ImageInfo>
            </SlideStitchInfo>
        </EncodeInfo>"#;

        let bif = parse_level0_xml(xml, 256, 256).unwrap();
        assert_eq!(bif.areas.len(), 1);
        assert_eq!(bif.areas[0].tiles_across, 3);
        assert_eq!(bif.areas[0].tiles_down, 2);
        // X: 256 + (-10 * 100) / 100 = 246
        assert!((bif.tile_advance_x - 246.0).abs() < 1e-6);
        // Y: 256 + (-8 * 100) / 100 = 248
        assert!((bif.tile_advance_y - 248.0).abs() < 1e-6);
    }

    #[test]
    fn parse_level0_xml_skips_unscanned() {
        let xml = r#"<EncodeInfo>
            <SlideStitchInfo>
                <ImageInfo AOIScanned="0" NumCols="3" NumRows="2"
                           Pos-X="0" Pos-Y="0"/>
                <ImageInfo AOIScanned="1" NumCols="2" NumRows="1"
                           Pos-X="100" Pos-Y="100"/>
            </SlideStitchInfo>
        </EncodeInfo>"#;

        let bif = parse_level0_xml(xml, 256, 256).unwrap();
        assert_eq!(bif.areas.len(), 1);
        assert_eq!(bif.areas[0].tiles_across, 2);
        assert_eq!(bif.areas[0].tiles_down, 1);
    }

    #[test]
    fn parse_level0_xml_no_joints_uses_tile_size() {
        let xml = r#"<EncodeInfo>
            <SlideStitchInfo>
                <ImageInfo AOIScanned="1" NumCols="2" NumRows="2"
                           Pos-X="0" Pos-Y="0"/>
            </SlideStitchInfo>
        </EncodeInfo>"#;

        let bif = parse_level0_xml(xml, 256, 256).unwrap();
        assert!((bif.tile_advance_x - 256.0).abs() < 1e-6);
        assert!((bif.tile_advance_y - 256.0).abs() < 1e-6);
    }

    #[test]
    fn parse_level0_xml_missing_slide_stitch_info_errors() {
        let xml = r#"<EncodeInfo>
            <SomeOtherElement/>
        </EncodeInfo>"#;

        let result = parse_level0_xml(xml, 256, 256);
        assert!(result.is_err());
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(msg.contains("SlideStitchInfo"), "got: {}", msg);
    }

    #[test]
    fn ventana_level0_dimensions_normalize_to_minimum_scanned_origin() {
        let bif = BifInfo {
            areas: vec![
                BifArea {
                    x: 12_000,
                    y: 9_000,
                    start_col: 0,
                    start_row: 0,
                    tiles_across: 2,
                    tiles_down: 2,
                },
                BifArea {
                    x: 12_500,
                    y: 9_250,
                    start_col: 2,
                    start_row: 1,
                    tiles_across: 1,
                    tiles_down: 1,
                },
            ],
            tiles: vec![],
            tile_advance_x: 250.0,
            tile_advance_y: 248.0,
        };

        let dims = ventana_level0_dimensions(&bif, 256, 256).unwrap();
        assert_eq!(dims, (756, 506));
    }

    #[test]
    fn ventana_level0_dimensions_prefers_exact_tile_positions() {
        let bif = BifInfo {
            areas: vec![],
            tiles: vec![
                BifTile {
                    col: 0,
                    row: 0,
                    x: 5.0,
                    y: 7.0,
                    width: 251,
                    height: 249,
                    tiff_tile_index: 0,
                },
                BifTile {
                    col: 1,
                    row: 0,
                    x: 261.0,
                    y: 7.0,
                    width: 240,
                    height: 249,
                    tiff_tile_index: 1,
                },
            ],
            tile_advance_x: 256.0,
            tile_advance_y: 256.0,
        };

        let dims = ventana_level0_dimensions(&bif, 256, 256).unwrap();
        assert_eq!(dims, (496, 249));
    }

    #[test]
    fn ventana_level0_dimensions_prefers_area_model_when_present() {
        let bif = BifInfo {
            areas: vec![BifArea {
                x: 0,
                y: 0,
                start_col: 0,
                start_row: 0,
                tiles_across: 3,
                tiles_down: 2,
            }],
            tiles: vec![
                BifTile {
                    col: 0,
                    row: 0,
                    x: 0.0,
                    y: 0.0,
                    width: 256,
                    height: 256,
                    tiff_tile_index: 0,
                },
                BifTile {
                    col: 1,
                    row: 0,
                    x: 300.0,
                    y: 0.0,
                    width: 256,
                    height: 256,
                    tiff_tile_index: 1,
                },
            ],
            tile_advance_x: 240.0,
            tile_advance_y: 248.0,
        };

        let dims = ventana_level0_dimensions(&bif, 256, 256).unwrap();
        assert_eq!(dims, (736, 504));
    }

    #[test]
    fn joint_delta_supports_left_and_down() {
        assert_eq!(
            joint_delta("LEFT", 256.0, 256.0, 12.0, 8.0),
            Some((-244.0, 8.0))
        );
        assert_eq!(
            joint_delta("DOWN", 256.0, 256.0, 12.0, 8.0),
            Some((12.0, -248.0))
        );
    }

    #[test]
    fn exact_tile_dimensions_use_neighbor_positions() {
        let mut positions = HashMap::new();
        positions.insert((0, 0), (5.0, 7.0));
        positions.insert((1, 0), (261.0, 7.0));
        positions.insert((0, 1), (5.0, 256.0));

        assert_eq!(
            ventana_exact_tile_dimensions(0, 0, 2, 2, &positions, 496.0, 256.0, 256.0, 256.0),
            (256, 249)
        );
        assert_eq!(
            ventana_exact_tile_dimensions(1, 0, 2, 2, &positions, 496.0, 256.0, 256.0, 256.0),
            (235, 249)
        );
    }

    // ── Tile grid building ──────────────────────────────────────────

    #[test]
    fn tile_grid_single_area() {
        let bif = BifInfo {
            areas: vec![BifArea {
                x: 0,
                y: 0,
                start_col: 0,
                start_row: 0,
                tiles_across: 3,
                tiles_down: 2,
            }],
            tiles: vec![],
            tile_advance_x: 250.0,
            tile_advance_y: 248.0,
        };

        let mut tiles: HashMap<(i64, i64), TileEntry> = HashMap::new();
        let mut tiff_idx: usize = 0;
        for area in &bif.areas {
            let offset_x = area.x as f64 - area.start_col as f64 * bif.tile_advance_x;
            let offset_y = area.y as f64 - area.start_row as f64 * bif.tile_advance_y;
            for row in area.start_row..area.start_row + area.tiles_down {
                for col in area.start_col..area.start_col + area.tiles_across {
                    tiles.insert(
                        (col, row),
                        TileEntry {
                            offset: (offset_x, offset_y),
                            dimensions: (256, 256),
                            tiff_tile_index: Some(tiff_idx),
                        },
                    );
                    tiff_idx += 1;
                }
            }
        }

        assert_eq!(tiles.len(), 6);
        assert_eq!(tiles[&(0, 0)].tiff_tile_index, Some(0));
        assert_eq!(tiles[&(1, 0)].tiff_tile_index, Some(1));
        assert_eq!(tiles[&(2, 0)].tiff_tile_index, Some(2));
        assert_eq!(tiles[&(0, 1)].tiff_tile_index, Some(3));
        assert_eq!(tiles[&(1, 1)].tiff_tile_index, Some(4));
        assert_eq!(tiles[&(2, 1)].tiff_tile_index, Some(5));
    }

    #[test]
    fn tile_grid_two_areas_sequential_indices() {
        let bif = BifInfo {
            areas: vec![
                BifArea {
                    x: 0,
                    y: 0,
                    start_col: 0,
                    start_row: 0,
                    tiles_across: 2,
                    tiles_down: 1,
                },
                BifArea {
                    x: 500,
                    y: 0,
                    start_col: 2,
                    start_row: 0,
                    tiles_across: 2,
                    tiles_down: 1,
                },
            ],
            tiles: vec![],
            tile_advance_x: 256.0,
            tile_advance_y: 256.0,
        };

        let mut tiles: HashMap<(i64, i64), TileEntry> = HashMap::new();
        let mut tiff_idx: usize = 0;
        for area in &bif.areas {
            let offset_x = area.x as f64 - area.start_col as f64 * bif.tile_advance_x;
            let offset_y = area.y as f64 - area.start_row as f64 * bif.tile_advance_y;
            for row in area.start_row..area.start_row + area.tiles_down {
                for col in area.start_col..area.start_col + area.tiles_across {
                    tiles.insert(
                        (col, row),
                        TileEntry {
                            offset: (offset_x, offset_y),
                            dimensions: (256, 256),
                            tiff_tile_index: Some(tiff_idx),
                        },
                    );
                    tiff_idx += 1;
                }
            }
        }

        assert_eq!(tiles.len(), 4);
        // First area: (0,0)=0, (1,0)=1
        assert_eq!(tiles[&(0, 0)].tiff_tile_index, Some(0));
        assert_eq!(tiles[&(1, 0)].tiff_tile_index, Some(1));
        // Second area: (2,0)=2, (3,0)=3
        assert_eq!(tiles[&(2, 0)].tiff_tile_index, Some(2));
        assert_eq!(tiles[&(3, 0)].tiff_tile_index, Some(3));

        // Second area has a different offset due to Pos-X=500 vs col*advance=512
        assert_eq!(tiles[&(2, 0)].offset.0, -12.0); // 500 - 2*256 = -12
    }

    #[test]
    fn ventana_snake_coords_reverse_odd_rows() {
        assert_eq!(ventana_snake_coords(1, 4), (0, 0));
        assert_eq!(ventana_snake_coords(4, 4), (3, 0));
        assert_eq!(ventana_snake_coords(5, 4), (3, 1));
        assert_eq!(ventana_snake_coords(8, 4), (0, 1));
    }

    // ── Overlap validation ──────────────────────────────────────────

    #[test]
    fn no_overlap_passes_validation() {
        let mut tiles = HashMap::new();
        tiles.insert(
            (0, 0),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(0),
            },
        );
        tiles.insert(
            (1, 0),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(1),
            },
        );
        tiles.insert(
            (0, 1),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(2),
            },
        );

        // tile_advance = 256 means tiles are exactly adjacent with 0-pixel offsets.
        let result = validate_no_adjacent_overlap(&tiles, 256.0, 256.0, 256, 256);
        assert!(result.is_ok());
    }

    #[test]
    fn overlap_detected_fails_validation() {
        let mut tiles = HashMap::new();
        tiles.insert(
            (0, 0),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(0),
            },
        );
        tiles.insert(
            (1, 0),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(1),
            },
        );

        // tile_advance = 200 means tile at (1,0) starts at pixel 200 but tile at
        // (0,0) extends to pixel 256, causing a 56-pixel overlap.
        let result = validate_no_adjacent_overlap(&tiles, 200.0, 256.0, 256, 256);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("overlap"), "got: {}", msg);
    }

    // ── compression_from_tag ────────────────────────────────────────

    #[test]
    fn compression_tag_mapping() {
        assert_eq!(compression_from_tag(1), Compression::None);
        assert_eq!(compression_from_tag(6), Compression::Jpeg);
        assert_eq!(compression_from_tag(7), Compression::Jpeg);
        assert_eq!(compression_from_tag(33003), Compression::Jp2kYcbcr);
        assert_eq!(compression_from_tag(33004), Compression::Jp2kRgb);
        assert_eq!(compression_from_tag(33005), Compression::Jp2kYcbcr);
        assert_eq!(compression_from_tag(9999), Compression::Other(9999));
    }

    // ── extract_encode_info ─────────────────────────────────────────

    #[test]
    fn extract_encode_info_found() {
        let input = "prefix<EncodeInfo><SlideStitchInfo/></EncodeInfo>suffix";
        let result = extract_encode_info(input);
        assert_eq!(
            result.as_deref(),
            Some("<EncodeInfo><SlideStitchInfo/></EncodeInfo>")
        );
    }

    #[test]
    fn extract_encode_info_with_attributes_found() {
        let input =
            "prefix<?xml version=\"1.0\"?><EncodeInfo Ver='2'><SlideStitchInfo/></EncodeInfo>";
        let result = extract_encode_info(input);
        assert_eq!(
            result.as_deref(),
            Some("<EncodeInfo Ver='2'><SlideStitchInfo/></EncodeInfo>")
        );
    }

    #[test]
    fn extract_encode_info_not_found() {
        let input = "no encode info here";
        assert!(extract_encode_info(input).is_none());
    }

    #[test]
    fn extract_encode_info_from_bytes_with_binary_wrapper() {
        let input = b"\xff\xd9<?xml version=\"1.0\"?>\n<EncodeInfo Ver='2'><SlideStitchInfo/></EncodeInfo>\0tail";
        let result = extract_encode_info_bytes(input);
        assert_eq!(
            result.as_deref(),
            Some("<EncodeInfo Ver='2'><SlideStitchInfo/></EncodeInfo>")
        );
    }

    // ── iScan property parsing ──────────────────────────────────────

    #[test]
    fn extract_iscan_fragment_from_bytes_with_binary_wrapper() {
        let input =
            b"\xff\x00<x:xmpmeta><iScan Magnification=\"40\" ScanRes=\"0.2528\"/></x:xmpmeta>";
        let result = extract_iscan_fragment_bytes(input);
        assert_eq!(
            result.as_deref(),
            Some("<iScan Magnification=\"40\" ScanRes=\"0.2528\"/>")
        );
    }

    #[test]
    fn parse_iscan_properties_basic() {
        let xmp = r#"<iScan Magnification="40" ScanRes="0.2528" SlideID="ABC123"/>"#;
        let mut props = Properties::new();
        parse_iscan_properties(xmp, &mut props);

        assert_eq!(props.get("ventana.Magnification"), Some("40"));
        assert_eq!(props.get("ventana.ScanRes"), Some("0.2528"));
        assert_eq!(props.get("ventana.SlideID"), Some("ABC123"));
        assert_eq!(props.get("openslide.objective-power"), Some("40"));
        assert_eq!(props.get("openslide.mpp-x"), Some("0.2528"));
        assert_eq!(props.get("openslide.mpp-y"), Some("0.2528"));
    }

    #[test]
    fn parse_iscan_properties_no_iscan() {
        let xmp = "<SomeOther attr=\"val\"/>";
        let mut props = Properties::new();
        parse_iscan_properties(xmp, &mut props);
        assert!(props.is_empty());
    }

    #[test]
    fn non_adjacent_overlap_detected() {
        // Tiles at (0,0) and (2,0) with large offsets that make them overlap
        // despite being 2 grid cells apart. The old neighbor-only check would miss this.
        let mut tiles = HashMap::new();
        tiles.insert(
            (0, 0),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(0),
            },
        );
        // (1,0) exists but is normal
        tiles.insert(
            (1, 0),
            TileEntry {
                offset: (0.0, 0.0),
                dimensions: (100, 256), // narrow tile
                tiff_tile_index: Some(1),
            },
        );
        // (2,0) has a large negative offset that pushes it back into (0,0)'s territory
        tiles.insert(
            (2, 0),
            TileEntry {
                offset: (-350.0, 0.0),
                dimensions: (256, 256),
                tiff_tile_index: Some(2),
            },
        );

        // tile_advance = 200
        // (0,0): x=[0, 256)
        // (1,0): x=[200, 300) — no overlap with (0,0) since 200 < 256... actually overlaps!
        // (2,0): x=[400-350, 400-350+256) = [50, 306) — overlaps with (0,0)
        // The sweep-line should catch the (0,0)/(2,0) overlap.
        let result = validate_no_adjacent_overlap(&tiles, 200.0, 256.0, 256, 256);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("overlap"), "got: {}", msg);
    }
}
