use super::*;

pub(super) fn signinum_decode_options(
    color_transform: SigninumColorTransform,
) -> SigninumDecodeOptions {
    SigninumDecodeOptions::default().with_color_transform(color_transform)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum JpegBitstreamColorHint {
    Rgb,
    RgbComponentIds012,
    YCbCr,
    Unknown,
}

pub(super) fn tiff_jpeg_color_transform(
    photometric: u32,
    samples_per_pixel: u32,
    bitstream_hint: JpegBitstreamColorHint,
) -> SigninumColorTransform {
    if samples_per_pixel == 3 {
        match bitstream_hint {
            JpegBitstreamColorHint::Rgb => return SigninumColorTransform::ForceRgb,
            JpegBitstreamColorHint::RgbComponentIds012 if photometric != 6 => {
                return SigninumColorTransform::ForceRgb;
            }
            JpegBitstreamColorHint::YCbCr => return SigninumColorTransform::ForceYCbCr,
            JpegBitstreamColorHint::RgbComponentIds012 | JpegBitstreamColorHint::Unknown => {}
        }
    }

    match (photometric, samples_per_pixel) {
        (2, 3) => SigninumColorTransform::ForceRgb,
        (6, 3) => SigninumColorTransform::ForceYCbCr,
        _ => SigninumColorTransform::Auto,
    }
}

pub(super) fn jpeg_bitstream_color_hint(
    data: &[u8],
    tables: Option<&[u8]>,
) -> JpegBitstreamColorHint {
    tables
        .map(jpeg_segment_color_hint)
        .filter(|hint| *hint != JpegBitstreamColorHint::Unknown)
        .unwrap_or_else(|| jpeg_segment_color_hint(data))
}

pub(super) fn jpeg_segment_color_hint(data: &[u8]) -> JpegBitstreamColorHint {
    let mut offset = 0usize;
    while offset + 1 < data.len() {
        if data[offset] != 0xFF {
            offset += 1;
            continue;
        }

        let mut marker_offset = offset + 1;
        while marker_offset < data.len() && data[marker_offset] == 0xFF {
            marker_offset += 1;
        }
        if marker_offset >= data.len() {
            return JpegBitstreamColorHint::Unknown;
        }

        let marker = data[marker_offset];
        offset = marker_offset + 1;
        if marker == 0x00 || marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if offset + 2 > data.len() {
            return JpegBitstreamColorHint::Unknown;
        }

        let segment_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        if segment_len < 2 {
            return JpegBitstreamColorHint::Unknown;
        }
        let payload_start = offset + 2;
        let payload_end = offset + segment_len;
        if payload_end > data.len() {
            return JpegBitstreamColorHint::Unknown;
        }
        let payload = &data[payload_start..payload_end];

        match marker {
            0xEE if payload.len() >= 12 && &payload[..5] == b"Adobe" => {
                return match payload[11] {
                    0 => JpegBitstreamColorHint::Rgb,
                    1 | 2 => JpegBitstreamColorHint::YCbCr,
                    _ => JpegBitstreamColorHint::Unknown,
                };
            }
            marker if is_jpeg_sof_marker(marker) => {
                return jpeg_sof_color_hint(payload);
            }
            0xDA => return JpegBitstreamColorHint::Unknown,
            _ => {}
        }

        offset = payload_end;
    }

    JpegBitstreamColorHint::Unknown
}

pub(super) fn is_jpeg_sof_marker(marker: u8) -> bool {
    matches!(
        marker,
        0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF
    )
}

pub(super) fn jpeg_sof_color_hint(payload: &[u8]) -> JpegBitstreamColorHint {
    if payload.len() < 6 {
        return JpegBitstreamColorHint::Unknown;
    }
    let component_count = payload[5] as usize;
    if component_count != 3 || payload.len() < 6 + component_count * 3 {
        return JpegBitstreamColorHint::Unknown;
    }

    let mut ids = [0u8; 3];
    let mut sampling = [(0u8, 0u8); 3];
    for component in 0..component_count {
        let base = 6 + component * 3;
        ids[component] = payload[base];
        sampling[component] = (payload[base + 1] >> 4, payload[base + 1] & 0x0F);
    }

    let first_component_subsampled = sampling[0].0 > sampling[1].0
        || sampling[0].0 > sampling[2].0
        || sampling[0].1 > sampling[1].1
        || sampling[0].1 > sampling[2].1;
    if first_component_subsampled {
        return JpegBitstreamColorHint::YCbCr;
    }

    if ids == [b'R', b'G', b'B'] {
        return JpegBitstreamColorHint::Rgb;
    }
    if ids == [0, 1, 2] {
        return JpegBitstreamColorHint::RgbComponentIds012;
    }

    JpegBitstreamColorHint::Unknown
}

#[derive(Debug, Clone, Copy)]
pub(super) struct JpegFrameInfo {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bits_allocated: u16,
    pub(super) samples_per_pixel: u16,
    pub(super) photometric_interpretation: EncodedTilePhotometricInterpretation,
}

pub(super) fn standalone_jpeg_frame(
    tile_data: &[u8],
    jpeg_tables: Option<&[u8]>,
) -> Result<(Vec<u8>, JpegFrameInfo), WsiError> {
    if !jpeg_has_soi(tile_data) {
        return Err(WsiError::Unsupported {
            reason: "JPEG passthrough requires tile payloads to start with SOI".into(),
        });
    }
    if !tile_data.ends_with(&[0xFF, 0xD9]) {
        return Err(WsiError::Unsupported {
            reason: "JPEG passthrough requires tile payloads to end with EOI".into(),
        });
    }

    let has_dqt = jpeg_has_segment_marker(tile_data, 0xDB)?;
    let has_dht = jpeg_has_segment_marker(tile_data, 0xC4)?;
    let frame = if has_dqt && has_dht {
        tile_data.to_vec()
    } else {
        rebuild_jpeg_frame_with_tables(tile_data, jpeg_tables, !has_dqt, !has_dht)?
    };
    let info = parse_baseline_jpeg_frame_info(&frame)?;
    Ok((frame, info))
}

pub(super) fn jpeg_has_soi(data: &[u8]) -> bool {
    data.starts_with(&[0xFF, 0xD8])
}

pub(super) fn jpeg_has_segment_marker(data: &[u8], needle: u8) -> Result<bool, WsiError> {
    let mut offset = 0usize;
    while let Some(segment) = next_jpeg_segment(data, offset)? {
        if segment.marker == needle {
            return Ok(true);
        }
        if segment.marker == 0xDA || segment.marker == 0xD9 {
            return Ok(false);
        }
        offset = segment.end;
    }
    Ok(false)
}

pub(super) fn rebuild_jpeg_frame_with_tables(
    tile_data: &[u8],
    jpeg_tables: Option<&[u8]>,
    need_dqt: bool,
    need_dht: bool,
) -> Result<Vec<u8>, WsiError> {
    let tables = jpeg_tables.ok_or_else(|| WsiError::Unsupported {
        reason: "JPEG passthrough tile is missing table segments and no JPEGTables are available"
            .into(),
    })?;
    let table_segments = jpeg_table_segments(tables, need_dqt, need_dht)?;
    if table_segments.is_empty() {
        return Err(WsiError::Unsupported {
            reason: "JPEG passthrough could not rebuild required DQT/DHT table segments".into(),
        });
    }
    let mut frame = Vec::with_capacity(tile_data.len() + table_segments.len());
    frame.extend_from_slice(&tile_data[..2]);
    frame.extend_from_slice(&table_segments);
    frame.extend_from_slice(&tile_data[2..]);
    Ok(frame)
}

pub(super) fn jpeg_table_segments(
    data: &[u8],
    need_dqt: bool,
    need_dht: bool,
) -> Result<Vec<u8>, WsiError> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while let Some(segment) = next_jpeg_segment(data, offset)? {
        if segment.marker == 0xDA || segment.marker == 0xD9 {
            break;
        }
        let include = (need_dqt && segment.marker == 0xDB) || (need_dht && segment.marker == 0xC4);
        if include {
            out.extend_from_slice(&data[segment.start..segment.end]);
        }
        offset = segment.end;
    }
    Ok(out)
}

pub(super) fn parse_baseline_jpeg_frame_info(data: &[u8]) -> Result<JpegFrameInfo, WsiError> {
    let mut offset = 0usize;
    while let Some(segment) = next_jpeg_segment(data, offset)? {
        if segment.marker == 0xDA {
            break;
        }
        if segment.marker == 0xC0 {
            return parse_sof0_frame_info(segment.payload);
        }
        if is_jpeg_sof_marker(segment.marker) {
            return Err(WsiError::Unsupported {
                reason: "JPEG passthrough only supports Baseline JPEG SOF0 frames".into(),
            });
        }
        offset = segment.end;
    }
    Err(WsiError::Unsupported {
        reason: "JPEG passthrough could not find a Baseline JPEG SOF0 marker".into(),
    })
}

pub(super) fn parse_sof0_frame_info(payload: &[u8]) -> Result<JpegFrameInfo, WsiError> {
    if payload.len() < 6 {
        return Err(WsiError::Unsupported {
            reason: "JPEG SOF0 segment is truncated".into(),
        });
    }
    let precision = payload[0];
    if precision != 8 {
        return Err(WsiError::Unsupported {
            reason: format!("JPEG passthrough requires 8-bit Baseline JPEG, got {precision}-bit"),
        });
    }
    let height = u16::from_be_bytes([payload[1], payload[2]]) as u32;
    let width = u16::from_be_bytes([payload[3], payload[4]]) as u32;
    let components = payload[5] as usize;
    if width == 0 || height == 0 {
        return Err(WsiError::Unsupported {
            reason: "JPEG passthrough requires nonzero SOF0 dimensions".into(),
        });
    }
    if payload.len() < 6 + components * 3 {
        return Err(WsiError::Unsupported {
            reason: "JPEG SOF0 component table is truncated".into(),
        });
    }
    let photometric_interpretation = match components {
        1 => EncodedTilePhotometricInterpretation::Monochrome2,
        3 => match jpeg_sof_color_hint(payload) {
            JpegBitstreamColorHint::Rgb => EncodedTilePhotometricInterpretation::Rgb,
            _ => EncodedTilePhotometricInterpretation::YbrFull422,
        },
        _ => {
            return Err(WsiError::Unsupported {
                reason: format!("JPEG passthrough supports 1 or 3 components, got {components}"),
            });
        }
    };
    Ok(JpegFrameInfo {
        width,
        height,
        bits_allocated: 8,
        samples_per_pixel: components as u16,
        photometric_interpretation,
    })
}

pub(super) struct JpegSegment<'a> {
    pub(super) marker: u8,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) payload: &'a [u8],
}

pub(super) fn next_jpeg_segment(
    data: &[u8],
    mut offset: usize,
) -> Result<Option<JpegSegment<'_>>, WsiError> {
    while offset + 1 < data.len() {
        if data[offset] != 0xFF {
            offset += 1;
            continue;
        }
        let mut marker_offset = offset + 1;
        while marker_offset < data.len() && data[marker_offset] == 0xFF {
            marker_offset += 1;
        }
        if marker_offset >= data.len() {
            return Ok(None);
        }
        let marker = data[marker_offset];
        if marker == 0x00 {
            offset = marker_offset + 1;
            continue;
        }
        let start = marker_offset - 1;
        let after_marker = marker_offset + 1;
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            return Ok(Some(JpegSegment {
                marker,
                start,
                end: after_marker,
                payload: &[],
            }));
        }
        if after_marker + 2 > data.len() {
            return Err(WsiError::Unsupported {
                reason: "JPEG marker segment is truncated".into(),
            });
        }
        let len = u16::from_be_bytes([data[after_marker], data[after_marker + 1]]) as usize;
        if len < 2 {
            return Err(WsiError::Unsupported {
                reason: "JPEG marker segment has invalid length".into(),
            });
        }
        let end = after_marker
            .checked_add(len)
            .ok_or_else(|| WsiError::Unsupported {
                reason: "JPEG marker segment length overflow".into(),
            })?;
        if end > data.len() {
            return Err(WsiError::Unsupported {
                reason: "JPEG marker segment exceeds payload length".into(),
            });
        }
        return Ok(Some(JpegSegment {
            marker,
            start,
            end,
            payload: &data[after_marker + 2..end],
        }));
    }
    Ok(None)
}

pub(super) fn signinum_downscale_for_factor(factor: u32) -> Option<SigninumDownscale> {
    match factor {
        1 => Some(SigninumDownscale::None),
        2 => Some(SigninumDownscale::Half),
        4 => Some(SigninumDownscale::Quarter),
        8 => Some(SigninumDownscale::Eighth),
        _ => None,
    }
}

pub(super) fn cpu_tile_from_rgb_pixels(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> Result<CpuTile, WsiError> {
    let expected_len = width as usize * height as usize * 3;
    if pixels.len() != expected_len {
        return Err(WsiError::Jpeg(format!(
            "signinum JPEG decode produced {} bytes, expected {} for {}x{} RGB",
            pixels.len(),
            expected_len,
            width,
            height
        )));
    }
    Ok(CpuTile {
        width,
        height,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(pixels),
    })
}

pub(super) fn strip_leading_restart_marker(segment: &[u8]) -> &[u8] {
    if segment.len() >= 2 && segment[0] == 0xFF && (0xD0..=0xD7).contains(&segment[1]) {
        &segment[2..]
    } else {
        segment
    }
}

pub(super) fn strip_trailing_restart_marker(segment: &[u8]) -> &[u8] {
    if segment.len() >= 2 {
        let tail = &segment[segment.len() - 2..];
        if tail[0] == 0xFF && (0xD0..=0xD7).contains(&tail[1]) {
            return &segment[..segment.len() - 2];
        }
    }
    segment
}

pub(super) fn strip_trailing_eoi_marker(segment: &[u8]) -> &[u8] {
    if segment.len() >= 2 && segment[segment.len() - 2..] == [0xFF, 0xD9] {
        &segment[..segment.len() - 2]
    } else {
        segment
    }
}

pub(super) fn disable_jpeg_restart_interval(header: &mut [u8]) {
    let mut i = 0usize;
    while i + 3 < header.len() {
        if header[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = header[i + 1];
        if marker == 0xD8 || marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        let seg_len = u16::from_be_bytes([header[i + 2], header[i + 3]]) as usize;
        if seg_len < 2 || i + 2 + seg_len > header.len() {
            return;
        }
        if marker == 0xDD && seg_len >= 4 {
            header[i + 4] = 0;
            header[i + 5] = 0;
            return;
        }
        if marker == 0xDA {
            return;
        }
        i += 2 + seg_len;
    }
}

pub(super) fn patch_jpeg_sof0_dimensions(
    data: &mut [u8],
    width: u32,
    height: u32,
) -> Result<(), WsiError> {
    if width == 0 || height == 0 || width > u16::MAX as u32 || height > u16::MAX as u32 {
        return Err(WsiError::Unsupported {
            reason: format!(
                "NDPI JPEG passthrough requires u16 SOF dimensions, got {width}x{height}"
            ),
        });
    }

    let mut offset = 0usize;
    while let Some(segment) = next_jpeg_segment(data, offset)? {
        if segment.marker == 0xC0 {
            if segment.payload.len() < 5 {
                return Err(WsiError::Unsupported {
                    reason: "NDPI JPEG passthrough SOF0 segment is truncated".into(),
                });
            }
            let payload_start = segment.start + 4;
            data[payload_start + 1..payload_start + 3]
                .copy_from_slice(&(height as u16).to_be_bytes());
            data[payload_start + 3..payload_start + 5]
                .copy_from_slice(&(width as u16).to_be_bytes());
            return Ok(());
        }
        if is_jpeg_sof_marker(segment.marker) {
            return Err(WsiError::Unsupported {
                reason: "NDPI JPEG passthrough only supports Baseline JPEG SOF0 frames".into(),
            });
        }
        if segment.marker == 0xDA {
            break;
        }
        offset = segment.end;
    }

    Err(WsiError::Unsupported {
        reason: "NDPI JPEG passthrough could not find a Baseline JPEG SOF0 marker".into(),
    })
}

pub(super) fn ndpi_restart_segments_align_to_rows(
    level_width: u64,
    virtual_tile_width: u32,
    restart_interval: u16,
) -> bool {
    if level_width == 0 || virtual_tile_width == 0 || restart_interval == 0 {
        return false;
    }
    let restart_interval = u64::from(restart_interval);
    let virtual_tile_width = u64::from(virtual_tile_width);
    if !virtual_tile_width.is_multiple_of(restart_interval) {
        return false;
    }
    let mcu_width = virtual_tile_width / restart_interval;
    if mcu_width == 0 {
        return false;
    }
    level_width
        .div_ceil(mcu_width)
        .is_multiple_of(restart_interval)
}
