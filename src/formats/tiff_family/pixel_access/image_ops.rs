use super::*;

/// Validate that tile coordinates are non-negative and fit in u32.
pub(super) fn validate_tile_coords(col: i64, row: i64, level: u32) -> Result<(u32, u32), WsiError> {
    if col < 0 || row < 0 {
        return Err(WsiError::TileRead {
            col,
            row,
            level,
            reason: "negative tile coordinates".into(),
        });
    }
    Ok((col as u32, row as u32))
}

// ── Helpers ──────────────────────────────────────────────────────

pub(super) fn rgba_image_to_sample_buffer(rgba: image::RgbaImage) -> CpuTile {
    let (width, height) = (rgba.width(), rgba.height());
    let rgba_raw = rgba.into_raw();
    let pixel_count = (width as usize) * (height as usize);
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        rgb.push(rgba_raw[i * 4]);
        rgb.push(rgba_raw[i * 4 + 1]);
        rgb.push(rgba_raw[i * 4 + 2]);
    }
    CpuTile {
        width,
        height,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(rgb),
    }
}

pub(super) fn downsample_rgb_2x_box(source: &CpuTile) -> Result<CpuTile, WsiError> {
    if source.layout != CpuTileLayout::Interleaved
        || source.channels != 3
        || source.color_space != ColorSpace::Rgb
    {
        return Err(WsiError::DisplayConversion(
            "synthetic NDPI levels require interleaved RGB input".into(),
        ));
    }

    let src = source.data.as_u8().ok_or_else(|| {
        WsiError::DisplayConversion("synthetic NDPI levels require U8 data".into())
    })?;
    let out_w = source.width.div_ceil(2);
    let out_h = source.height.div_ceil(2);
    let mut out = vec![0u8; out_w as usize * out_h as usize * 3];
    let src_stride = source.width as usize * 3;
    let dst_stride = out_w as usize * 3;
    for out_y in 0..out_h as usize {
        let src_y = out_y * 2;
        for out_x in 0..out_w as usize {
            let src_x = out_x * 2;
            let mut sum = [0u32; 3];
            let mut count = 0u32;
            for dy in 0..2usize {
                let sy = src_y + dy;
                if sy >= source.height as usize {
                    continue;
                }
                let row = sy * src_stride;
                for dx in 0..2usize {
                    let sx = src_x + dx;
                    if sx >= source.width as usize {
                        continue;
                    }
                    let idx = row + sx * 3;
                    sum[0] += u32::from(src[idx]);
                    sum[1] += u32::from(src[idx + 1]);
                    sum[2] += u32::from(src[idx + 2]);
                    count += 1;
                }
            }

            let dst = out_x * 3;
            let row = out_y * dst_stride;
            out[row + dst] = (sum[0] / count) as u8;
            out[row + dst + 1] = (sum[1] / count) as u8;
            out[row + dst + 2] = (sum[2] / count) as u8;
        }
    }

    Ok(CpuTile {
        width: out_w,
        height: out_h,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(out),
    })
}

pub(super) fn downsample_rgb_pow2_box(source: &CpuTile, factor: u32) -> Result<CpuTile, WsiError> {
    if !factor.is_power_of_two() || factor < 2 {
        return Err(WsiError::DisplayConversion(format!(
            "synthetic NDPI levels require power-of-two factor >= 2, got {factor}"
        )));
    }
    let mut current = downsample_rgb_2x_box(source)?;
    let mut current_factor = 2u32;
    while current_factor < factor {
        current = downsample_rgb_2x_box(&current)?;
        current_factor = current_factor.saturating_mul(2);
    }
    Ok(current)
}

pub(super) fn fit_synthetic_rgb_tile_to_dimensions(
    tile: CpuTile,
    width: u32,
    height: u32,
) -> Result<CpuTile, WsiError> {
    if tile.width == width && tile.height == height {
        return Ok(tile);
    }
    if tile.width >= width && tile.height >= height {
        return crop_rgb_interleaved_u8_buffer(&tile, 0, 0, width, height);
    }
    Err(WsiError::DisplayConversion(format!(
        "synthetic NDPI level dimensions mismatch: got {}x{}, expected {}x{}",
        tile.width, tile.height, width, height
    )))
}

pub(super) fn checked_rgb_u8_len(width: u32, height: u32) -> Result<usize, WsiError> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| {
            WsiError::DisplayConversion(format!("RGB region dimensions overflow: {width}x{height}"))
        })?;
    pixels.checked_mul(3).ok_or_else(|| {
        WsiError::DisplayConversion(format!(
            "RGB region byte length overflows usize: {width}x{height}"
        ))
    })
}

pub(super) fn zero_rgb_interleaved_u8_tile(width: u32, height: u32) -> Result<CpuTile, WsiError> {
    Ok(CpuTile {
        width,
        height,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![0u8; checked_rgb_u8_len(width, height)?]),
    })
}

pub(super) fn paste_rgb_interleaved_u8_tile(
    source: &CpuTile,
    width: u32,
    height: u32,
    dst_x: u32,
    dst_y: u32,
) -> Result<CpuTile, WsiError> {
    if source.layout != CpuTileLayout::Interleaved
        || source.channels != 3
        || source.color_space != ColorSpace::Rgb
    {
        return Err(WsiError::DisplayConversion(
            "synthetic NDPI ROI paste requires interleaved RGB input".into(),
        ));
    }
    let src = source.data.as_u8().ok_or_else(|| {
        WsiError::DisplayConversion("synthetic NDPI ROI paste requires U8 data".into())
    })?;
    if dst_x == 0 && dst_y == 0 && source.width == width && source.height == height {
        return Ok(source.clone());
    }
    if u64::from(dst_x) + u64::from(source.width) > u64::from(width)
        || u64::from(dst_y) + u64::from(source.height) > u64::from(height)
    {
        return Err(WsiError::DisplayConversion(format!(
            "synthetic NDPI ROI paste {}x{} at ({dst_x},{dst_y}) exceeds output {width}x{height}",
            source.width, source.height
        )));
    }

    let mut out = vec![0u8; checked_rgb_u8_len(width, height)?];
    let src_stride = source.width as usize * 3;
    let dst_stride = width as usize * 3;
    let dst_x = dst_x as usize;
    let dst_y = dst_y as usize;
    for row in 0..source.height as usize {
        let src_off = row * src_stride;
        let dst_off = (dst_y + row) * dst_stride + dst_x * 3;
        out[dst_off..dst_off + src_stride].copy_from_slice(&src[src_off..src_off + src_stride]);
    }

    Ok(CpuTile {
        width,
        height,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(out),
    })
}

pub(super) fn ensure_interleaved_rgb_u8(tile: CpuTile) -> Result<CpuTile, WsiError> {
    if tile.layout == CpuTileLayout::Interleaved
        && tile.channels == 3
        && tile.color_space == ColorSpace::Rgb
        && tile.data.as_u8().is_some()
    {
        Ok(tile)
    } else {
        Ok(rgba_image_to_sample_buffer(tile.to_rgba()?))
    }
}
