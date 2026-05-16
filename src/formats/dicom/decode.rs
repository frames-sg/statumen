use super::*;

pub(super) fn frame_bytes_to_rgb_tile(
    frame_bytes: &[u8],
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    planar_configuration: u16,
    photometric_interpretation: &str,
) -> Result<CpuTile, WsiError> {
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| WsiError::DisplayConversion("DICOM frame dimensions overflow".into()))?;
    let rgb = match (samples_per_pixel, photometric_interpretation) {
        (3, "RGB") if planar_configuration == 0 => {
            let expected = pixel_count.checked_mul(3).ok_or_else(|| {
                WsiError::DisplayConversion("DICOM RGB frame size overflow".into())
            })?;
            if frame_bytes.len() != expected {
                return Err(WsiError::DisplayConversion(format!(
                    "DICOM RGB frame has {} bytes, expected {expected}",
                    frame_bytes.len()
                )));
            }
            frame_bytes.to_vec()
        }
        (3, "RGB") if planar_configuration == 1 => {
            let expected = pixel_count.checked_mul(3).ok_or_else(|| {
                WsiError::DisplayConversion("DICOM planar RGB frame size overflow".into())
            })?;
            if frame_bytes.len() != expected {
                return Err(WsiError::DisplayConversion(format!(
                    "DICOM planar RGB frame has {} bytes, expected {expected}",
                    frame_bytes.len()
                )));
            }
            let (r_plane, rest) = frame_bytes.split_at(pixel_count);
            let (g_plane, b_plane) = rest.split_at(pixel_count);
            let mut rgb = vec![0; expected];
            for idx in 0..pixel_count {
                let offset = idx * 3;
                rgb[offset] = r_plane[idx];
                rgb[offset + 1] = g_plane[idx];
                rgb[offset + 2] = b_plane[idx];
            }
            rgb
        }
        (1, "MONOCHROME1" | "MONOCHROME2") => {
            if frame_bytes.len() != pixel_count {
                return Err(WsiError::DisplayConversion(format!(
                    "DICOM monochrome frame has {} bytes, expected {pixel_count}",
                    frame_bytes.len()
                )));
            }
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for &gray in frame_bytes {
                // Preserve the legacy sv-slide behavior for consolidation:
                // MONOCHROME1 and MONOCHROME2 are both expanded without inversion.
                rgb.extend_from_slice(&[gray, gray, gray]);
            }
            rgb
        }
        _ => {
            return Err(WsiError::DisplayConversion(format!(
                "unsupported DICOM pixel format: samples_per_pixel={samples_per_pixel}, photometric={photometric_interpretation}, planar_configuration={planar_configuration}"
            )));
        }
    };

    CpuTile::new(
        width,
        height,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(rgb),
    )
}

pub(super) fn decode_rle_lossless_frame(
    frame_bytes: &[u8],
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    photometric_interpretation: &str,
) -> Result<CpuTile, WsiError> {
    if frame_bytes.len() < 64 {
        return Err(WsiError::DisplayConversion(
            "DICOM RLE frame is shorter than its 64-byte header".into(),
        ));
    }
    let segment_count = u32::from_le_bytes(frame_bytes[0..4].try_into().unwrap()) as usize;
    if segment_count == 0 || segment_count > 15 {
        return Err(WsiError::DisplayConversion(format!(
            "DICOM RLE segment count {segment_count} is invalid"
        )));
    }
    let expected_segments = samples_per_pixel as usize;
    if segment_count < expected_segments {
        return Err(WsiError::DisplayConversion(format!(
            "DICOM RLE has {segment_count} segments, expected at least {expected_segments}"
        )));
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| WsiError::DisplayConversion("DICOM RLE dimensions overflow".into()))?;
    let mut planes = Vec::with_capacity(expected_segments);
    for segment in 0..expected_segments {
        let offset_start = 4 + segment * 4;
        let segment_start = u32::from_le_bytes(
            frame_bytes[offset_start..offset_start + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let segment_end = if segment + 1 < segment_count {
            let next_offset_start = 4 + (segment + 1) * 4;
            u32::from_le_bytes(
                frame_bytes[next_offset_start..next_offset_start + 4]
                    .try_into()
                    .unwrap(),
            ) as usize
        } else {
            frame_bytes.len()
        };
        if segment_start < 64 || segment_start > segment_end || segment_end > frame_bytes.len() {
            return Err(WsiError::DisplayConversion(format!(
                "DICOM RLE segment {segment} has invalid byte range {segment_start}..{segment_end}"
            )));
        }
        planes.push(decode_rle_segment(
            &frame_bytes[segment_start..segment_end],
            pixel_count,
        )?);
    }

    let rgb = match (samples_per_pixel, photometric_interpretation) {
        (3, "RGB") => {
            let mut rgb = vec![0; pixel_count * 3];
            for (idx, ((&red, &green), &blue)) in
                planes[0].iter().zip(&planes[1]).zip(&planes[2]).enumerate()
            {
                let offset = idx * 3;
                rgb[offset] = red;
                rgb[offset + 1] = green;
                rgb[offset + 2] = blue;
            }
            rgb
        }
        (1, "MONOCHROME1" | "MONOCHROME2") => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for &gray in &planes[0] {
                rgb.extend_from_slice(&[gray, gray, gray]);
            }
            rgb
        }
        _ => {
            return Err(WsiError::DisplayConversion(format!(
                "unsupported DICOM RLE pixel format: samples_per_pixel={samples_per_pixel}, photometric={photometric_interpretation}"
            )));
        }
    };

    CpuTile::new(
        width,
        height,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(rgb),
    )
}

pub(super) fn decode_rle_segment(segment: &[u8], expected_len: usize) -> Result<Vec<u8>, WsiError> {
    let mut output = Vec::with_capacity(expected_len);
    let mut i = 0;
    while i < segment.len() && output.len() < expected_len {
        let n = segment[i] as i8;
        i += 1;
        match n {
            0..=127 => {
                let count = n as usize + 1;
                let end = i.checked_add(count).ok_or_else(|| {
                    WsiError::DisplayConversion("DICOM RLE literal run overflow".into())
                })?;
                if end > segment.len() {
                    return Err(WsiError::DisplayConversion(
                        "DICOM RLE literal run exceeds segment length".into(),
                    ));
                }
                output.extend_from_slice(&segment[i..end]);
                i = end;
            }
            -127..=-1 => {
                if i >= segment.len() {
                    return Err(WsiError::DisplayConversion(
                        "DICOM RLE repeat run missing value".into(),
                    ));
                }
                let count = 1usize + (-n as usize);
                output.extend(std::iter::repeat_n(segment[i], count));
                i += 1;
            }
            -128 => {}
        }
    }
    if output.len() != expected_len {
        return Err(WsiError::DisplayConversion(format!(
            "DICOM RLE segment decoded to {} bytes, expected {expected_len}",
            output.len()
        )));
    }
    Ok(output)
}

pub(super) fn rgb_image_to_sample_buffer(image: ::image::RgbImage) -> CpuTile {
    CpuTile::new(
        image.width(),
        image.height(),
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(image.into_raw()),
    )
    .expect("RGB image dimensions must match")
}

pub(super) fn crop_sample_buffer_rgb(buffer: CpuTile, width: u32, height: u32) -> CpuTile {
    if buffer.width == width && buffer.height == height {
        return buffer;
    }
    let image = imageops::crop_imm(&buffer.into_rgb_image(), 0, 0, width, height).to_image();
    rgb_image_to_sample_buffer(image)
}

pub(super) fn raw_compression_for_transfer_syntax(
    transfer_syntax_uid: &str,
    photometric_interpretation: &str,
) -> Result<Compression, WsiError> {
    if transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
        return Ok(Compression::Jpeg);
    }
    if JP2K_TRANSFER_SYNTAXES.contains(&transfer_syntax_uid) {
        return Ok(
            if matches!(photometric_interpretation, "YBR_ICT" | "YBR_RCT") {
                Compression::Jp2kYcbcr
            } else {
                Compression::Jp2kRgb
            },
        );
    }
    Err(WsiError::Unsupported {
        reason: format!(
            "raw compressed DICOM tile access requires JPEG Baseline or J2K/HTJ2K transfer syntax, got {transfer_syntax_uid}"
        ),
    })
}

pub(super) fn raw_photometric_interpretation(
    samples_per_pixel: u16,
    photometric_interpretation: &str,
) -> Result<EncodedTilePhotometricInterpretation, WsiError> {
    match (samples_per_pixel, photometric_interpretation) {
        (1, "MONOCHROME1" | "MONOCHROME2") => {
            Ok(EncodedTilePhotometricInterpretation::Monochrome2)
        }
        (3, "RGB") => Ok(EncodedTilePhotometricInterpretation::Rgb),
        (3, "YBR_FULL_422" | "YBR_ICT" | "YBR_RCT") => {
            Ok(EncodedTilePhotometricInterpretation::YbrFull422)
        }
        (_, other) => Err(WsiError::Unsupported {
            reason: format!(
                "raw compressed DICOM tile access does not support photometric interpretation {other}"
            ),
        }),
    }
}

pub(super) fn trim_encapsulated_frame_padding(data: &mut Vec<u8>) {
    if data.len() >= 3
        && data.last() == Some(&0)
        && data[data.len() - 3..data.len() - 1] == [0xFF, 0xD9]
    {
        data.pop();
    }
}

pub(super) trait IntoRgbImage {
    fn into_rgb_image(self) -> ::image::RgbImage;
}

impl IntoRgbImage for CpuTile {
    fn into_rgb_image(self) -> ::image::RgbImage {
        match self {
            CpuTile {
                width,
                height,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::U8(bytes),
            } => ::image::RgbImage::from_raw(width, height, Arc::unwrap_or_clone(bytes))
                .expect("sample buffer must contain valid RGB dimensions"),
            buffer => ::image::DynamicImage::ImageRgba8(
                buffer
                    .to_rgba()
                    .expect("RGB crop fallback must produce RGBA"),
            )
            .into_rgb8(),
        }
    }
}

pub(super) fn black_sample_buffer(width: u32, height: u32) -> CpuTile {
    CpuTile::new(
        width,
        height,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(vec![0; width as usize * height as usize * 3]),
    )
    .expect("black tile dimensions must match")
}
