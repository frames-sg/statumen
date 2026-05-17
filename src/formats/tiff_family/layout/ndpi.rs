//! NDPI layout interpreter.
//!
//! Classifies IFDs from an NDPI TiffContainer into pyramid levels,
//! associated images, and z-stack planes. Produces a DatasetLayout
//! with TileSource descriptors for each plane.

use std::collections::HashMap;

use crate::core::types::*;
use crate::formats::tiff_family::container::tags;
use crate::formats::tiff_family::container::TiffContainer;
use crate::formats::tiff_family::error::{IfdId, TiffParseError};
use crate::properties::Properties;
use signinum_jpeg::Decoder as SigninumJpegDecoder;

use super::{
    compute_tiff_dataset_identity, DatasetLayout, TiffLayoutInterpreter, TileSource, TileSourceKey,
};

// ── NDPI-specific tag constants ───────────────────────────────────

const NDPI_SOURCELENS: u16 = 65421;
const NDPI_XOFFSET: u16 = 65422;
const NDPI_YOFFSET: u16 = 65423;
const NDPI_FOCAL_PLANE: u16 = 65424;
const NDPI_MCU_STARTS: u16 = 65426;
const NDPI_REFERENCE: u16 = 65427;
const NDPI_PROPERTY_MAP: u16 = 65449;
const JPEG_HEADER_PROBE_BYTES: u64 = 4096;

#[derive(Clone)]
struct JpegGeometryProbe {
    header: Vec<u8>,
    restart_interval: u16,
    mcu_w: u32,
    mcu_h: u32,
}

fn probe_jpeg_geometry_via_signinum(
    container: &TiffContainer,
    ifd_id: IfdId,
) -> Result<JpegGeometryProbe, TiffParseError> {
    let strip_offset = container.get_u64(ifd_id, tags::STRIP_OFFSETS)?;
    let strip_byte_count = container.get_u64(ifd_id, tags::STRIP_BYTE_COUNTS)?;
    let read_len = JPEG_HEADER_PROBE_BYTES.min(strip_byte_count);
    let header = container.pread(strip_offset, read_len)?;
    probe_jpeg_geometry_bytes_via_signinum(header)
}

fn probe_jpeg_geometry_bytes_via_signinum(
    header: Vec<u8>,
) -> Result<JpegGeometryProbe, TiffParseError> {
    match SigninumJpegDecoder::inspect(&header) {
        Ok(info) => Ok(JpegGeometryProbe {
            header: jpeg_header_prefix(&header)?.to_vec(),
            restart_interval: info.restart_interval.unwrap_or(0),
            mcu_w: info.mcu_geometry.width,
            mcu_h: info.mcu_geometry.height,
        }),
        Err(inspect_err) => {
            let probe = probe_jpeg_geometry_bytes_lenient(&header).map_err(|lenient_err| {
                TiffParseError::Structure(format!(
                    "cannot parse JPEG geometry with signinum: {inspect_err}; lenient NDPI probe failed: {lenient_err}"
                ))
            })?;
            Ok(probe)
        }
    }
}

fn probe_jpeg_geometry_bytes_lenient(header: &[u8]) -> Result<JpegGeometryProbe, TiffParseError> {
    if header.len() < 2 || header[0..2] != [0xFF, 0xD8] {
        return Err(TiffParseError::Structure(
            "NDPI JPEG header missing SOI".into(),
        ));
    }

    let mut restart_interval = 0;
    let mut mcu_w = None;
    let mut mcu_h = None;
    let mut i = 2usize;

    while i + 1 < header.len() {
        if header[i] != 0xFF {
            return Err(TiffParseError::Structure(format!(
                "NDPI JPEG marker expected at byte {i}"
            )));
        }
        while i < header.len() && header[i] == 0xFF {
            i += 1;
        }
        if i >= header.len() {
            break;
        }
        let marker = header[i];
        i += 1;

        match marker {
            0xD9 => break,
            0xDA => {
                let prefix = jpeg_header_prefix(header)?;
                let mcu_w = mcu_w.ok_or_else(|| {
                    TiffParseError::Structure("NDPI JPEG header missing SOF marker".into())
                })?;
                let mcu_h = mcu_h.ok_or_else(|| {
                    TiffParseError::Structure("NDPI JPEG header missing SOF marker".into())
                })?;
                return Ok(JpegGeometryProbe {
                    header: prefix.to_vec(),
                    restart_interval,
                    mcu_w,
                    mcu_h,
                });
            }
            0x00 | 0xD0..=0xD7 => continue,
            _ => {}
        }

        if i + 1 >= header.len() {
            return Err(TiffParseError::Structure(format!(
                "NDPI JPEG marker FF{marker:02X} has truncated length"
            )));
        }
        let seg_len = u16::from_be_bytes([header[i], header[i + 1]]) as usize;
        if seg_len < 2 || i + seg_len > header.len() {
            return Err(TiffParseError::Structure(format!(
                "NDPI JPEG marker FF{marker:02X} has invalid length {seg_len}"
            )));
        }
        let payload = &header[i + 2..i + seg_len];
        if is_jpeg_sof_marker(marker) {
            if payload.len() < 6 {
                return Err(TiffParseError::Structure(
                    "NDPI JPEG SOF segment too short".into(),
                ));
            }
            let component_count = payload[5] as usize;
            let components = &payload[6..];
            if components.len() < component_count * 3 {
                return Err(TiffParseError::Structure(
                    "NDPI JPEG SOF component table too short".into(),
                ));
            }
            let mut max_h = 1u8;
            let mut max_v = 1u8;
            for component in components.chunks_exact(3).take(component_count) {
                let sampling = component[1];
                let h = sampling >> 4;
                let v = sampling & 0x0F;
                if h == 0 || v == 0 {
                    return Err(TiffParseError::Structure(format!(
                        "NDPI JPEG invalid sampling {h}x{v}"
                    )));
                }
                max_h = max_h.max(h);
                max_v = max_v.max(v);
            }
            mcu_w = Some(u32::from(max_h) * 8);
            mcu_h = Some(u32::from(max_v) * 8);
        } else if marker == 0xDD {
            if payload.len() < 2 {
                return Err(TiffParseError::Structure(
                    "NDPI JPEG DRI segment too short".into(),
                ));
            }
            restart_interval = u16::from_be_bytes([payload[0], payload[1]]);
        }
        i += seg_len;
    }

    Err(TiffParseError::Structure(
        "NDPI JPEG header missing SOS marker".into(),
    ))
}

fn jpeg_header_prefix(header: &[u8]) -> Result<&[u8], TiffParseError> {
    let mut i = 0usize;
    while i + 1 < header.len() {
        if header[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = header[i + 1];
        if marker == 0xD8 || marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        if i + 3 >= header.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([header[i + 2], header[i + 3]]) as usize;
        if seg_len < 2 || i + 2 + seg_len > header.len() {
            return Err(TiffParseError::Structure(format!(
                "NDPI JPEG marker FF{marker:02X} has invalid length {seg_len}"
            )));
        }
        if marker == 0xDA {
            return Ok(&header[..i + 2 + seg_len]);
        }
        i += 2 + seg_len;
    }
    Err(TiffParseError::Structure(
        "NDPI JPEG header missing SOS marker".into(),
    ))
}

fn is_jpeg_sof_marker(marker: u8) -> bool {
    matches!(
        marker,
        0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF
    )
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
        other => Compression::Other(other as u16),
    }
}

// ── NdpiInterpreter ───────────────────────────────────────────────

pub(crate) struct NdpiInterpreter;

/// Intermediate representation of a classified NDPI IFD.
struct ClassifiedIfd {
    ifd_id: IfdId,
    width: u64,
    height: u64,
    source_lens: f64,
    focal_plane: i64,
    strip_offset: u64,
    strip_byte_count: u64,
}

impl TiffLayoutInterpreter for NdpiInterpreter {
    fn detect(&self, container: &TiffContainer) -> bool {
        container.is_ndpi()
    }

    fn vendor_name(&self) -> &'static str {
        "hamamatsu-ndpi"
    }

    fn interpret(&self, container: &TiffContainer) -> Result<DatasetLayout, TiffParseError> {
        let mut pyramid_ifds: Vec<ClassifiedIfd> = Vec::new();
        let mut associated_images: HashMap<String, AssociatedImage> = HashMap::new();
        let mut associated_sources: HashMap<String, TileSource> = HashMap::new();

        // Phase 1: Classify each top-level IFD
        for &ifd_id in container.top_ifds() {
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

            let source_lens = container.get_f64(ifd_id, NDPI_SOURCELENS).unwrap_or(0.0);

            if source_lens > 0.0 {
                // Pyramid level
                let focal_plane = container
                    .get_f64(ifd_id, NDPI_FOCAL_PLANE)
                    .map(|v| v as i64)
                    .unwrap_or(0);

                let strip_offset = container.get_u64(ifd_id, tags::STRIP_OFFSETS).unwrap_or(0);
                let strip_byte_count = container
                    .get_u64(ifd_id, tags::STRIP_BYTE_COUNTS)
                    .unwrap_or(0);

                if strip_offset == 0 || strip_byte_count == 0 {
                    continue;
                }

                pyramid_ifds.push(ClassifiedIfd {
                    ifd_id,
                    width,
                    height,
                    source_lens,
                    focal_plane,
                    strip_offset,
                    strip_byte_count,
                });
            } else if (source_lens as i64) == -1 {
                let name = "macro";
                let strip_offsets = match container.get_u64_array(ifd_id, tags::STRIP_OFFSETS) {
                    Ok(values) => values.to_vec(),
                    Err(_) => continue,
                };
                let strip_byte_counts =
                    match container.get_u64_array(ifd_id, tags::STRIP_BYTE_COUNTS) {
                        Ok(values) => values.to_vec(),
                        Err(_) => continue,
                    };
                if strip_offsets.is_empty() || strip_offsets.len() != strip_byte_counts.len() {
                    continue;
                }

                let compression =
                    compression_from_tag(container.get_u32(ifd_id, tags::COMPRESSION).unwrap_or(1));
                let jpeg_tables = if compression == Compression::Jpeg {
                    container
                        .get_bytes(ifd_id, tags::JPEG_TABLES)
                        .ok()
                        .map(|bytes| bytes.to_vec())
                } else {
                    None
                };
                let channels = container
                    .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
                    .unwrap_or(3)
                    .clamp(1, u32::from(u16::MAX)) as u16;

                associated_images.insert(
                    name.to_string(),
                    AssociatedImage {
                        dimensions: (
                            u32::try_from(width).unwrap_or(u32::MAX),
                            u32::try_from(height).unwrap_or(u32::MAX),
                        ),
                        sample_type: SampleType::Uint8,
                        channels,
                    },
                );
                associated_sources.insert(
                    name.to_string(),
                    TileSource::Stripped {
                        ifd_id,
                        jpeg_tables,
                        compression,
                        strip_offsets,
                        strip_byte_counts,
                    },
                );
            }
            // Other SOURCELENS values (for example -2.0) remain unclassified in
            // the public dataset model to preserve compatibility associated-image
            // parity.
        }

        if pyramid_ifds.is_empty() {
            return Err(TiffParseError::Structure(
                "No pyramid levels found in NDPI".into(),
            ));
        }

        // Phase 2: Group pyramid IFDs by SOURCELENS, sort by dimensions
        // Within each SOURCELENS group, sub-group by FOCAL_PLANE for z-stack
        let (levels, tile_sources, z_count) = self.build_pyramid(container, &mut pyramid_ifds)?;

        // Phase 3: Parse properties
        let mut properties = self.parse_properties(container)?;

        // Phase 4: Assemble Dataset
        let property_ifd = *container
            .top_ifds()
            .first()
            .ok_or_else(|| TiffParseError::Structure("No IFDs in NDPI container".into()))?;
        let lowest_resolution_ifd = pyramid_ifds
            .last()
            .map(|ifd| ifd.ifd_id)
            .ok_or_else(|| TiffParseError::Structure("No pyramid IFDs in NDPI container".into()))?;
        let identity =
            compute_tiff_dataset_identity(container, lowest_resolution_ifd, property_ifd)?;
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
                    axes: AxesShape {
                        z: z_count,
                        c: 1,
                        t: 1,
                    },
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

impl NdpiInterpreter {
    /// Build pyramid levels from classified IFDs.
    /// Groups physical IFDs by SOURCELENS, then materializes the public NDPI
    /// pyramid as exact power-of-two downsample factors from level 0.
    /// Missing intermediate levels are synthesized from the nearest
    /// higher-resolution public level.
    ///
    /// Returns (levels, tile_sources, z_count).
    #[allow(clippy::type_complexity)]
    fn build_pyramid(
        &self,
        container: &TiffContainer,
        pyramid_ifds: &mut [ClassifiedIfd],
    ) -> Result<(Vec<Level>, HashMap<TileSourceKey, TileSource>, u32), TiffParseError> {
        // Sort by area descending (largest first = level 0)
        pyramid_ifds.sort_by(|a, b| {
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            area_b.cmp(&area_a)
        });

        // Group by SOURCELENS value -- each distinct value is one pyramid level
        // Use a Vec of (source_lens_bits, Vec<&ClassifiedIfd>) to preserve sort order.
        let mut lens_groups: Vec<(u64, Vec<&ClassifiedIfd>)> = Vec::new();
        for ifd in pyramid_ifds.iter() {
            let bits = ifd.source_lens.to_bits();
            if let Some(group) = lens_groups.iter_mut().find(|(b, _)| *b == bits) {
                group.1.push(ifd);
            } else {
                lens_groups.push((bits, vec![ifd]));
            }
        }

        // The first group (sorted by largest area) is level 0.
        let base_dims = (pyramid_ifds[0].width, pyramid_ifds[0].height);

        // Count distinct focal planes across all IFDs
        let mut focal_planes: Vec<i64> = pyramid_ifds.iter().map(|ifd| ifd.focal_plane).collect();
        focal_planes.sort();
        focal_planes.dedup();
        let z_count = focal_planes.len().max(1) as u32;

        let mut physical_groups_by_factor: HashMap<u32, Vec<&ClassifiedIfd>> = HashMap::new();
        for (_lens_bits, group) in lens_groups {
            let representative = group[0];
            let Some(factor) =
                ndpi_power_of_two_factor(base_dims, (representative.width, representative.height))
            else {
                continue;
            };
            physical_groups_by_factor.insert(factor, group);
        }

        let mut expected_factors = Vec::new();
        let mut factor = 1u32;
        while u64::from(factor) <= base_dims.0
            && u64::from(factor) <= base_dims.1
            && base_dims.0.is_multiple_of(u64::from(factor))
            && base_dims.1.is_multiple_of(u64::from(factor))
        {
            expected_factors.push(factor);
            factor = match factor.checked_mul(2) {
                Some(next) => next,
                None => break,
            };
        }

        let mut levels = Vec::new();
        let mut tile_sources = HashMap::new();
        let mut previous_public_level_idx: Option<u32> = None;
        let mut nearest_physical_level: Option<(u32, u32)> = None;

        for expected_factor in expected_factors {
            let level_idx = levels.len() as u32;
            let width = base_dims.0 / u64::from(expected_factor);
            let height = base_dims.1 / u64::from(expected_factor);
            let downsample = expected_factor as f64;

            if let Some(group) = physical_groups_by_factor.remove(&expected_factor) {
                let representative = group[0];

                let representative_probe =
                    probe_jpeg_geometry_via_signinum(container, representative.ifd_id)?;
                let restart_interval = representative_probe.restart_interval;
                let (mcu_w, mcu_h) = (representative_probe.mcu_w, representative_probe.mcu_h);

                let (virtual_tile_width, virtual_tile_height) = if restart_interval > 0 {
                    (
                        mcu_w.checked_mul(restart_interval as u32).ok_or_else(|| {
                            TiffParseError::Structure(format!(
                                "NDPI: virtual tile width overflow (mcu_w={}, restart_interval={})",
                                mcu_w, restart_interval
                            ))
                        })?,
                        mcu_h,
                    )
                } else {
                    (
                        u32::try_from(width).unwrap_or(u32::MAX),
                        u32::try_from(height).unwrap_or(u32::MAX),
                    )
                };

                levels.push(Level {
                    dimensions: (width, height),
                    downsample,
                    tile_layout: TileLayout::WholeLevel {
                        width,
                        height,
                        virtual_tile_width,
                        virtual_tile_height,
                    },
                });

                for ifd in group {
                    let z_index = focal_planes
                        .iter()
                        .position(|&fp| fp == ifd.focal_plane)
                        .unwrap_or(0) as u32;

                    let ifd_probe = if ifd.ifd_id == representative.ifd_id {
                        representative_probe.clone()
                    } else {
                        probe_jpeg_geometry_via_signinum(container, ifd.ifd_id)?
                    };

                    let plane_ri = ifd_probe.restart_interval;
                    let (plane_mcu_w, plane_mcu_h) = (ifd_probe.mcu_w, ifd_probe.mcu_h);

                    let source = if plane_ri > 0 {
                        let plane_vtw =
                            plane_mcu_w.checked_mul(plane_ri as u32).ok_or_else(|| {
                                TiffParseError::Structure(format!(
                                    "NDPI: per-plane virtual tile width overflow (mcu_w={}, ri={})",
                                    plane_mcu_w, plane_ri
                                ))
                            })?;
                        let plane_vth = plane_mcu_h;
                        if plane_vtw == 0 || plane_vth == 0 {
                            return Err(TiffParseError::Structure(format!(
                                "NDPI: virtual tile dimensions must be > 0 (vtw={}, vth={})",
                                plane_vtw, plane_vth
                            )));
                        }
                        let width_u32 = u32::try_from(width).unwrap_or(u32::MAX);
                        let height_u32 = u32::try_from(height).unwrap_or(u32::MAX);
                        let plane_ta = width_u32.saturating_add(plane_vtw - 1) / plane_vtw;
                        let plane_td = height_u32.saturating_add(plane_vth - 1) / plane_vth;
                        TileSource::NdpiJpeg {
                            ifd_id: ifd.ifd_id,
                            jpeg_header: ifd_probe.header,
                            mcu_starts_tag: NDPI_MCU_STARTS,
                            tiles_across: plane_ta,
                            tiles_down: plane_td,
                            restart_interval: plane_ri,
                            strip_offset: ifd.strip_offset,
                            strip_byte_count: ifd.strip_byte_count,
                        }
                    } else {
                        TileSource::NdpiFullDecode {
                            ifd_id: ifd.ifd_id,
                            jpeg_header: ifd_probe.header,
                            strip_offset: ifd.strip_offset,
                            strip_byte_count: ifd.strip_byte_count,
                        }
                    };

                    tile_sources.insert(
                        TileSourceKey {
                            scene: 0,
                            series: 0,
                            level: level_idx,
                            z: z_index,
                            c: 0,
                            t: 0,
                        },
                        source,
                    );
                }
                nearest_physical_level = Some((level_idx, expected_factor));
            } else {
                let direct_physical_base =
                    nearest_physical_level.and_then(|(physical_level, physical_factor)| {
                        if expected_factor % physical_factor != 0 {
                            return None;
                        }
                        let factor = expected_factor / physical_factor;
                        (factor.is_power_of_two() && (2..=8).contains(&factor))
                            .then_some((physical_level, factor))
                    });
                let (base_level, synthetic_factor) = match direct_physical_base {
                    Some(base) => base,
                    None => (
                        previous_public_level_idx.ok_or_else(|| {
                            TiffParseError::Structure(
                                "NDPI: cannot synthesize level without a higher-resolution base"
                                    .into(),
                            )
                        })?,
                        2,
                    ),
                };
                let width_u32 = u32::try_from(width).unwrap_or(u32::MAX);
                let height_u32 = u32::try_from(height).unwrap_or(u32::MAX);
                levels.push(Level {
                    dimensions: (width, height),
                    downsample,
                    tile_layout: TileLayout::WholeLevel {
                        width,
                        height,
                        virtual_tile_width: width_u32,
                        virtual_tile_height: height_u32,
                    },
                });

                for z in 0..z_count {
                    tile_sources.insert(
                        TileSourceKey {
                            scene: 0,
                            series: 0,
                            level: level_idx,
                            z,
                            c: 0,
                            t: 0,
                        },
                        TileSource::SyntheticDownsample {
                            base_level,
                            factor: synthetic_factor,
                        },
                    );
                }
            }

            previous_public_level_idx = Some(level_idx);
        }

        Ok((levels, tile_sources, z_count))
    }

    /// Parse NDPI property map from first IFD and populate Properties.
    fn parse_properties(&self, container: &TiffContainer) -> Result<Properties, TiffParseError> {
        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "hamamatsu");

        let first_ifd = match container.top_ifds().first() {
            Some(&id) => id,
            None => return Ok(properties),
        };

        // Image description -> openslide.comment
        if let Ok(desc) = container.get_string(first_ifd, tags::IMAGE_DESCRIPTION) {
            properties.insert("openslide.comment", desc.to_string());
        }

        // SourceLens -> objective power
        if let Ok(lens) = container.get_f64(first_ifd, NDPI_SOURCELENS) {
            if lens > 0.0 {
                properties.insert("hamamatsu.SourceLens", format!("{}", lens));
                properties.insert("openslide.objective-power", format!("{}", lens));
            }
        }

        // NDPI property map: key=value\r\n pairs
        if let Ok(prop_map) = container.get_string(first_ifd, NDPI_PROPERTY_MAP) {
            for line in prop_map.split("\r\n") {
                if let Some((key, val)) = line.split_once('=') {
                    if !key.is_empty() && !val.is_empty() {
                        properties.insert(format!("hamamatsu.{}", key), val.to_string());
                    }
                }
            }
        }

        // NDPI offsets
        if let Ok(xoff) = container.get_f64(first_ifd, NDPI_XOFFSET) {
            properties.insert(
                "hamamatsu.XOffsetFromSlideCentre",
                format!("{}", xoff as i64),
            );
        }
        if let Ok(yoff) = container.get_f64(first_ifd, NDPI_YOFFSET) {
            properties.insert(
                "hamamatsu.YOffsetFromSlideCentre",
                format!("{}", yoff as i64),
            );
        }
        if let Ok(reference) = container.get_string(first_ifd, NDPI_REFERENCE) {
            properties.insert("hamamatsu.Reference", reference.to_string());
        }

        // MPP from XResolution / YResolution (NDPI stores pixels/cm, unit=3).
        let res_unit = container
            .get_u32(first_ifd, tags::RESOLUTION_UNIT)
            .unwrap_or(3); // NDPI default: centimeter
        let unit_to_microns = match res_unit {
            3 => 10_000.0, // 1 cm = 10,000 µm
            _ => 25_400.0, // 1 inch = 25,400 µm
        };
        if let Ok(x_res) = container.get_f64(first_ifd, tags::X_RESOLUTION) {
            if x_res > 0.0 {
                let mpp_x = unit_to_microns / x_res;
                properties.insert("openslide.mpp-x", format!("{mpp_x:.6}"));
            }
        }
        if let Ok(y_res) = container.get_f64(first_ifd, tags::Y_RESOLUTION) {
            if y_res > 0.0 {
                let mpp_y = unit_to_microns / y_res;
                properties.insert("openslide.mpp-y", format!("{mpp_y:.6}"));
            }
        }

        Ok(properties)
    }
}

fn ndpi_power_of_two_factor(base_dims: (u64, u64), dims: (u64, u64)) -> Option<u32> {
    let (base_w, base_h) = base_dims;
    let (width, height) = dims;
    if width == 0 || height == 0 {
        return None;
    }
    if base_w % width != 0 || base_h % height != 0 {
        return None;
    }
    let factor_w = base_w / width;
    let factor_h = base_h / height;
    if factor_w != factor_h {
        return None;
    }
    let factor = u32::try_from(factor_w).ok()?;
    if factor == 0 || !factor.is_power_of_two() {
        return None;
    }
    Some(factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::tiff_family::container::TiffContainer;
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

    fn synthetic_dri_420_jpeg_header() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xDD, 0x00, 0x04, 0x00, 0x0A, // DRI = 10 MCUs
            0xFF, 0xC0, 0x00, 0x11, // SOF0
            0x08, // precision
            0x00, 0x80, // height = 128
            0x01, 0x00, // width = 256
            0x03, // components
            0x01, 0x22, 0x00, // Y: H=2, V=2
            0x02, 0x11, 0x01, // Cb: H=1, V=1
            0x03, 0x11, 0x01, // Cr: H=1, V=1
            0xFF, 0xDA, 0x00, 0x0C, // SOS
            0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00,
        ]
    }

    #[test]
    fn probe_jpeg_geometry_via_signinum_matches_synthetic_header() {
        let probe = probe_jpeg_geometry_bytes_via_signinum(synthetic_dri_420_jpeg_header())
            .expect("signinum should inspect synthetic DRI JPEG header");
        assert_eq!(probe.restart_interval, 10);
        assert_eq!(probe.mcu_w, 16);
        assert_eq!(probe.mcu_h, 16);
    }

    #[test]
    fn probe_jpeg_geometry_accepts_ndpi_zero_sof_dimensions() {
        let mut header = synthetic_dri_420_jpeg_header();
        let sof = header
            .windows(2)
            .position(|bytes| bytes == [0xFF, 0xC0])
            .expect("synthetic header has SOF0");
        header[sof + 5..sof + 9].copy_from_slice(&[0, 0, 0, 0]);

        let probe = probe_jpeg_geometry_bytes_via_signinum(header)
            .expect("NDPI lenient probe should accept zero SOF dimensions");

        assert_eq!(probe.restart_interval, 10);
        assert_eq!(probe.mcu_w, 16);
        assert_eq!(probe.mcu_h, 16);
        assert!(probe.header.len() < JPEG_HEADER_PROBE_BYTES as usize);
    }

    /// Build a minimal TIFF file in memory with the given IFDs.
    /// Each IFD is a list of (tag, type_id, count, value_bytes).
    /// Supports only inline tags (value fits in 4 bytes) for simplicity.
    /// Returns a NamedTempFile containing the TIFF data.
    #[allow(clippy::type_complexity)]
    fn build_synthetic_tiff(ifds: &[Vec<(u16, u16, u32, [u8; 4])>], ndpi: bool) -> NamedTempFile {
        let mut buf = Vec::new();

        // TIFF header: little-endian, classic TIFF
        buf.extend_from_slice(b"II"); // byte order
        buf.extend_from_slice(&42u16.to_le_bytes()); // magic
                                                     // First IFD offset -- we'll fill this in later
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Build IFDs sequentially
        let mut ifd_offsets = Vec::new();
        let mut next_ifd_patch_positions = Vec::new();

        for (ifd_idx, tags) in ifds.iter().enumerate() {
            let ifd_offset = buf.len() as u32;
            ifd_offsets.push(ifd_offset);

            // Inject NDPI marker tag (65420) into first IFD if requested
            let mut all_tags = tags.clone();
            if ndpi && ifd_idx == 0 {
                all_tags.push((65420, 4, 1, [1, 0, 0, 0])); // LONG, count=1, value=1
            }

            // Sort tags by ID (TIFF requirement)
            all_tags.sort_by_key(|t| t.0);

            let entry_count = all_tags.len() as u16;
            buf.extend_from_slice(&entry_count.to_le_bytes());

            for (tag_id, type_id, count, value) in &all_tags {
                buf.extend_from_slice(&tag_id.to_le_bytes());
                buf.extend_from_slice(&type_id.to_le_bytes());
                buf.extend_from_slice(&count.to_le_bytes());
                buf.extend_from_slice(value);
            }

            // Next IFD offset -- placeholder, will patch
            let next_pos = buf.len();
            if ndpi {
                // NDPI uses 8-byte next-IFD pointers
                buf.extend_from_slice(&0u64.to_le_bytes());
            } else {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
            next_ifd_patch_positions.push((next_pos, ndpi));
        }

        // Patch first IFD offset
        let offset_bytes = ifd_offsets[0].to_le_bytes();
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&offset_bytes);

        // Patch next-IFD pointers to chain them
        for i in 0..ifd_offsets.len() - 1 {
            let (patch_pos, is_ndpi) = next_ifd_patch_positions[i];
            let next_offset = ifd_offsets[i + 1];
            if is_ndpi {
                let bytes = (next_offset as u64).to_le_bytes();
                buf[patch_pos..patch_pos + 8].copy_from_slice(&bytes);
            } else {
                let bytes = next_offset.to_le_bytes();
                buf[patch_pos..patch_pos + 4].copy_from_slice(&bytes);
            }
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

    /// Helper: create a SHORT tag value (type_id=3, count=1).
    fn short_tag(tag: u16, value: u16) -> (u16, u16, u32, [u8; 4]) {
        let mut inline = [0u8; 4];
        inline[..2].copy_from_slice(&value.to_le_bytes());
        (tag, 3, 1, inline)
    }

    /// Helper: create a FLOAT tag value (type_id=11, count=1).
    fn float_tag(tag: u16, value: f32) -> (u16, u16, u32, [u8; 4]) {
        (tag, 11, 1, value.to_le_bytes())
    }

    // ── Task 4: Detection + IFD classification tests ──────────────────

    #[test]
    fn detect_ndpi_container() {
        // Build a TIFF file with NDPI marker tag
        let file = build_synthetic_tiff(
            &[vec![
                long_tag(256, 1024), // IMAGE_WIDTH
                long_tag(257, 768),  // IMAGE_LENGTH
            ]],
            true, // ndpi=true adds tag 65420
        );

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        assert!(interpreter.detect(&container));
    }

    #[test]
    fn reject_non_ndpi_container() {
        // Build a normal TIFF without NDPI marker
        let file = build_synthetic_tiff(&[vec![long_tag(256, 1024), long_tag(257, 768)]], false);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        assert!(!interpreter.detect(&container));
    }

    #[test]
    fn ifd_classification_macro_vs_pyramid() {
        // Build an NDPI with two IFDs:
        // IFD 0: SOURCELENS=40.0 (pyramid)
        // IFD 1: SOURCELENS=-1.0 (macro)
        // Both need valid strip offsets; detect() doesn't require valid JPEG data.
        let file = build_synthetic_tiff(
            &[
                vec![
                    long_tag(256, 2048),    // IMAGE_WIDTH
                    long_tag(257, 1536),    // IMAGE_LENGTH
                    float_tag(65421, 40.0), // SOURCELENS
                    long_tag(273, 0),       // STRIP_OFFSETS (invalid, but detect doesn't care)
                    long_tag(279, 0),       // STRIP_BYTE_COUNTS
                ],
                vec![
                    long_tag(256, 800),         // IMAGE_WIDTH
                    long_tag(257, 600),         // IMAGE_LENGTH
                    float_tag(65421, -1.0_f32), // SOURCELENS = macro
                    long_tag(273, 0),
                    long_tag(279, 0),
                ],
            ],
            true,
        );

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        assert!(interpreter.detect(&container));

        // Verify IFD count
        assert_eq!(container.top_ifds().len(), 2);
    }

    #[test]
    fn interpret_no_pyramid_levels_returns_error() {
        // An NDPI file where all IFDs are macro images (SOURCELENS=-1)
        let file = build_synthetic_tiff(
            &[vec![
                long_tag(256, 800),
                long_tag(257, 600),
                float_tag(65421, -1.0_f32),
                long_tag(273, 100),
                long_tag(279, 500),
            ]],
            true,
        );

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let result = interpreter.interpret(&container);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No pyramid levels"),
            "expected 'No pyramid levels', got: {}",
            err_msg,
        );
    }

    // ── Task 5: Full interpret() tests with embedded JPEG ──────────────

    /// Build a synthetic NDPI TIFF with embedded strip payloads at valid offsets.
    /// Each entry is (width, height, source_lens, focal_plane, compression_tag).
    fn build_ndpi_with_strips(entries: &[(u32, u32, f32, i32, u32)]) -> NamedTempFile {
        let mut strip_blocks: Vec<Vec<u8>> = Vec::new();
        for &(w, h, _, _, compression_tag) in entries {
            let actual_w = w.min(64);
            let actual_h = h.min(64);
            let strip_data = if compression_tag == 1 {
                vec![0u8; actual_w as usize * actual_h as usize * 3]
            } else {
                let rgb = image::RgbImage::new(actual_w, actual_h);
                encode_test_jpeg(&rgb)
            };
            strip_blocks.push(strip_data);
        }

        let mut buf = Vec::new();

        // TIFF header
        buf.extend_from_slice(b"II"); // little-endian
        buf.extend_from_slice(&42u16.to_le_bytes()); // classic TIFF
        let first_ifd_offset_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes()); // first IFD offset placeholder

        // Write JPEG data blocks and remember their offsets
        let mut strip_offsets: Vec<u32> = Vec::new();
        let mut strip_byte_counts: Vec<u32> = Vec::new();
        for strip_data in &strip_blocks {
            strip_offsets.push(buf.len() as u32);
            strip_byte_counts.push(strip_data.len() as u32);
            buf.extend_from_slice(strip_data);
        }

        // Write IFDs
        let mut ifd_offsets: Vec<u32> = Vec::new();
        let mut next_ifd_patch_positions: Vec<usize> = Vec::new();

        for (i, &(w, h, lens, focal, compression_tag)) in entries.iter().enumerate() {
            let ifd_offset = buf.len() as u32;
            ifd_offsets.push(ifd_offset);

            let mut tags: Vec<(u16, u16, u32, [u8; 4])> = vec![
                long_tag(256, w),                                     // IMAGE_WIDTH
                long_tag(257, h),                                     // IMAGE_LENGTH
                short_tag(tags::COMPRESSION, compression_tag as u16), // COMPRESSION
                long_tag(273, strip_offsets[i]),                      // STRIP_OFFSETS
                long_tag(279, strip_byte_counts[i]),                  // STRIP_BYTE_COUNTS
                float_tag(NDPI_SOURCELENS, lens),                     // SOURCELENS
            ];

            // Add FOCAL_PLANE only if non-zero
            if focal != 0 {
                tags.push(float_tag(NDPI_FOCAL_PLANE, focal as f32));
            }

            // Add NDPI marker tag to first IFD
            if i == 0 {
                tags.push(long_tag(65420, 1)); // NDPI marker
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

            // NDPI 8-byte next-IFD pointer
            let next_pos = buf.len();
            buf.extend_from_slice(&0u64.to_le_bytes());
            next_ifd_patch_positions.push(next_pos);
        }

        // Patch first IFD offset
        buf[first_ifd_offset_pos..first_ifd_offset_pos + 4]
            .copy_from_slice(&ifd_offsets[0].to_le_bytes());

        // Chain IFDs
        for i in 0..ifd_offsets.len() - 1 {
            let next = ifd_offsets[i + 1] as u64;
            let pos = next_ifd_patch_positions[i];
            buf[pos..pos + 8].copy_from_slice(&next.to_le_bytes());
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    /// Build a synthetic NDPI TIFF with JPEG-compressed strips.
    /// Each entry is (width, height, source_lens, focal_plane).
    fn build_ndpi_with_jpeg_strips(entries: &[(u32, u32, f32, i32)]) -> NamedTempFile {
        let entries_with_compression: Vec<_> = entries
            .iter()
            .map(|&(w, h, lens, focal)| (w, h, lens, focal, 7u32))
            .collect();
        build_ndpi_with_strips(&entries_with_compression)
    }

    #[test]
    fn interpret_single_level() {
        // Single pyramid level at SOURCELENS=40
        let file = build_ndpi_with_jpeg_strips(&[(1024, 768, 40.0, 0)]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        assert_eq!(layout.dataset.scenes.len(), 1);
        let series = &layout.dataset.scenes[0].series[0];
        assert_eq!(series.levels.len(), 9);
        assert_eq!(series.levels[0].dimensions, (1024, 768));
        assert_eq!(series.levels[1].dimensions, (512, 384));
        assert_eq!(series.levels[2].dimensions, (256, 192));
        assert_eq!(series.levels[8].dimensions, (4, 3));
        assert!((series.levels[0].downsample - 1.0).abs() < 0.001);
        assert_eq!(series.axes.z, 1);

        // Verify tile source exists
        let key = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        assert!(layout.tile_sources.contains_key(&key));
    }

    #[test]
    fn interpret_multi_level_sorted() {
        // Two pyramid levels -- interpreter should sort largest first
        let file = build_ndpi_with_jpeg_strips(&[
            (512, 384, 20.0, 0),   // smaller (level 1)
            (2048, 1536, 40.0, 0), // larger (level 0)
        ]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        let series = &layout.dataset.scenes[0].series[0];
        assert_eq!(series.levels.len(), 10);

        // Level 0 should be the larger image
        assert_eq!(series.levels[0].dimensions, (2048, 1536));
        assert!((series.levels[0].downsample - 1.0).abs() < 0.001);

        // Missing 2x level is synthesized, 4x level stays physical.
        assert_eq!(series.levels[1].dimensions, (1024, 768));
        assert_eq!(series.levels[2].dimensions, (512, 384));
        assert_eq!(series.levels[9].dimensions, (4, 3));
    }

    #[test]
    fn interpret_z_stack() {
        // Two IFDs at same SOURCELENS but different FOCAL_PLANEs
        let file = build_ndpi_with_jpeg_strips(&[
            (1024, 768, 40.0, 0), // z=0
            (1024, 768, 40.0, 1), // z=1
        ]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        let series = &layout.dataset.scenes[0].series[0];
        // Same SOURCELENS -> complete synthetic power-of-two pyramid, z=2
        assert_eq!(series.levels.len(), 9);
        assert_eq!(series.axes.z, 2);

        // Both z planes should have tile sources
        let key_z0 = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        let key_z1 = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 1,
            c: 0,
            t: 0,
        };
        assert!(layout.tile_sources.contains_key(&key_z0));
        assert!(layout.tile_sources.contains_key(&key_z1));
    }

    #[test]
    fn interpret_macro_associated_image() {
        // One pyramid + one macro
        let file = build_ndpi_with_jpeg_strips(&[
            (2048, 1536, 40.0, 0), // pyramid
            (800, 600, -1.0, 0),   // macro
        ]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        // Should have macro in associated images
        assert!(layout.dataset.associated_images.contains_key("macro"));
        let macro_img = &layout.dataset.associated_images["macro"];
        assert_eq!(macro_img.dimensions, (800, 600));

        // Should have macro in associated sources
        assert!(layout.associated_sources.contains_key("macro"));

        // Pyramid should still work
        assert_eq!(layout.dataset.scenes[0].series[0].levels.len(), 10);
    }

    #[test]
    fn negative_two_sourcelens_is_not_exposed_as_public_thumbnail() {
        let file = build_ndpi_with_strips(&[
            (2048, 1536, 40.0, 0, 7), // pyramid
            (196, 572, -2.0, 0, 1),   // thumbnail
        ]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        assert!(!layout.dataset.associated_images.contains_key("thumbnail"));
        assert!(!layout.associated_sources.contains_key("thumbnail"));
        assert!(layout.dataset.associated_images.is_empty());
        assert!(layout.associated_sources.is_empty());
    }

    #[test]
    fn interpret_properties_parsed() {
        let file = build_ndpi_with_jpeg_strips(&[(1024, 768, 40.0, 0)]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        // Vendor should always be set
        assert_eq!(layout.dataset.properties.vendor(), Some("hamamatsu"));
    }

    #[test]
    fn interpret_mcu_geometry_determines_tile_source() {
        // The tiny test JPEG won't have a DRI marker, so it should
        // produce NdpiFullDecode (restart_interval == 0)
        let file = build_ndpi_with_jpeg_strips(&[(1024, 768, 40.0, 0)]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        let key = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        let source = layout.tile_sources.get(&key).unwrap();
        // Our synthetic JPEG has no DRI -> NdpiFullDecode
        match source {
            TileSource::NdpiFullDecode { .. } => {} // expected
            other => panic!("expected NdpiFullDecode, got: {:?}", other),
        }
    }

    #[test]
    fn opens_legacy_wrapped_offset_ndpi_when_corpus_is_available() {
        use crate::core::registry::Slide;

        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let path = workspace_root.join("downloads/openslide-testdata/Hamamatsu/Hamamatsu-1.ndpi");
        if !path.exists() {
            return;
        }

        let slide = Slide::open(&path).expect("open legacy NDPI");
        assert_eq!(
            slide.dataset().scenes[0].series[0].levels[0].dimensions,
            (188160, 101376)
        );
        let tile = slide
            .read_tile(
                &TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: 0,
                    row: 0,
                },
                TileOutputPreference::cpu(),
            )
            .expect("read legacy NDPI tile");
        assert!(matches!(tile, TilePixels::Cpu(_)));
    }

    #[test]
    fn interpret_adds_synthetic_power_of_two_levels_between_sparse_physical_ifds() {
        let file = build_ndpi_with_jpeg_strips(&[(1024, 768, 40.0, 0), (256, 192, 20.0, 0)]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        match layout
            .tile_sources
            .get(&TileSourceKey {
                scene: 0,
                series: 0,
                level: 1,
                z: 0,
                c: 0,
                t: 0,
            })
            .unwrap()
        {
            TileSource::SyntheticDownsample { base_level, factor } => {
                assert_eq!(*base_level, 0);
                assert_eq!(*factor, 2);
            }
            other => panic!("expected SyntheticDownsample, got: {:?}", other),
        }

        match layout
            .tile_sources
            .get(&TileSourceKey {
                scene: 0,
                series: 0,
                level: 2,
                z: 0,
                c: 0,
                t: 0,
            })
            .unwrap()
        {
            TileSource::NdpiFullDecode { .. } => {}
            other => panic!("expected physical NDPI level, got: {:?}", other),
        }

        match layout
            .tile_sources
            .get(&TileSourceKey {
                scene: 0,
                series: 0,
                level: 3,
                z: 0,
                c: 0,
                t: 0,
            })
            .unwrap()
        {
            TileSource::SyntheticDownsample { base_level, factor } => {
                assert_eq!(*base_level, 2);
                assert_eq!(*factor, 2);
            }
            other => panic!("expected SyntheticDownsample, got: {:?}", other),
        }
    }

    #[test]
    fn interpret_points_consecutive_synthetic_levels_at_nearest_physical_base() {
        let file = build_ndpi_with_jpeg_strips(&[(1024, 768, 40.0, 0), (128, 96, 5.0, 0)]);

        let container = TiffContainer::open(file.path()).unwrap();
        let interpreter = NdpiInterpreter;
        let layout = interpreter.interpret(&container).unwrap();

        match layout
            .tile_sources
            .get(&TileSourceKey {
                scene: 0,
                series: 0,
                level: 2,
                z: 0,
                c: 0,
                t: 0,
            })
            .unwrap()
        {
            TileSource::SyntheticDownsample { base_level, factor } => {
                assert_eq!(*base_level, 0);
                assert_eq!(*factor, 4);
            }
            other => panic!("expected SyntheticDownsample, got: {:?}", other),
        }

        match layout
            .tile_sources
            .get(&TileSourceKey {
                scene: 0,
                series: 0,
                level: 3,
                z: 0,
                c: 0,
                t: 0,
            })
            .unwrap()
        {
            TileSource::NdpiFullDecode { .. } => {}
            other => panic!("expected physical NDPI level, got: {:?}", other),
        }
    }

    #[test]
    fn ndpi_power_of_two_factor_requires_exact_power_of_two_dimensions() {
        assert_eq!(
            ndpi_power_of_two_factor((51200, 38144), (12800, 9536)),
            Some(4)
        );
        assert_eq!(
            ndpi_power_of_two_factor((51200, 38144), (25600, 19072)),
            Some(2)
        );
        assert_eq!(
            ndpi_power_of_two_factor((51200, 38144), (200, 149)),
            Some(256)
        );
        assert_eq!(ndpi_power_of_two_factor((51200, 38144), (74, 55)), None);
        assert_eq!(ndpi_power_of_two_factor((51200, 38144), (3200, 2400)), None);
    }
}
