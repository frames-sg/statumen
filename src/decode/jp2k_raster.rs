use crate::core::types::{ColorSpace, CpuTile, CpuTileData, CpuTileLayout};
use crate::decode::jp2k::Jp2kColorSpace;
#[cfg(test)]
use crate::decode::jp2k_backend::DecodedImage;
use crate::decode::jp2k_backend::DecodedInterleavedImage;
use crate::error::WsiError;
#[cfg(test)]
use image::RgbaImage;

#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
pub(crate) fn decoded_image_to_rgba(
    image: DecodedImage,
    colorspace: Jp2kColorSpace,
) -> Result<RgbaImage, WsiError> {
    let w = image.width;
    let h = image.height;
    let [c0, c1, c2] = image.components;

    if c0.width == 0
        || c0.height == 0
        || c1.width == 0
        || c1.height == 0
        || c2.width == 0
        || c2.height == 0
    {
        return Err(WsiError::Jp2k(
            "decoded image has invalid component dimensions".into(),
        ));
    }

    let c0_sub_x = (w / c0.width).max(1);
    let c1_sub_x = (w / c1.width).max(1);
    let c2_sub_x = (w / c2.width).max(1);
    let c0_sub_y = (h / c0.height).max(1);
    let c1_sub_y = (h / c1.height).max(1);
    let c2_sub_y = (h / c2.height).max(1);

    let mut rgba_buf = vec![255u8; w * h * 4];
    match colorspace {
        Jp2kColorSpace::Rgb => {
            for y in 0..h {
                let c0_row = (y / c0_sub_y) * c0.width;
                let c1_row = (y / c1_sub_y) * c1.width;
                let c2_row = (y / c2_sub_y) * c2.width;
                for x in 0..w {
                    let off = (y * w + x) * 4;
                    rgba_buf[off] = clamp_u8(c0.samples[c0_row + x / c0_sub_x]);
                    rgba_buf[off + 1] = clamp_u8(c1.samples[c1_row + x / c1_sub_x]);
                    rgba_buf[off + 2] = clamp_u8(c2.samples[c2_row + x / c2_sub_x]);
                }
            }
        }
        Jp2kColorSpace::YCbCr => {
            for y in 0..h {
                let c0_row = (y / c0_sub_y) * c0.width;
                let c1_row = (y / c1_sub_y) * c1.width;
                let c2_row = (y / c2_sub_y) * c2.width;
                for x in 0..w {
                    let yy = c0.samples[c0_row + x / c0_sub_x];
                    let cb = c1.samples[c1_row + x / c1_sub_x];
                    let cr = c2.samples[c2_row + x / c2_sub_x];
                    let cb_off = cb - 128;
                    let cr_off = cr - 128;
                    let r = yy + ((1402 * cr_off) / 1000);
                    let g = yy - ((344 * cb_off + 714 * cr_off) / 1000);
                    let b = yy + ((1772 * cb_off) / 1000);

                    let off = (y * w + x) * 4;
                    rgba_buf[off] = clamp_u8(r);
                    rgba_buf[off + 1] = clamp_u8(g);
                    rgba_buf[off + 2] = clamp_u8(b);
                }
            }
        }
    }

    RgbaImage::from_raw(w as u32, h as u32, rgba_buf)
        .ok_or_else(|| WsiError::Jp2k("failed to create RgbaImage from decoded data".into()))
}

#[cfg(test)]
pub(crate) fn decoded_image_to_sample_buffer(
    image: DecodedImage,
    colorspace: Jp2kColorSpace,
) -> Result<CpuTile, WsiError> {
    let w = image.width;
    let h = image.height;
    let [c0, c1, c2] = image.components;

    if c0.width == 0
        || c0.height == 0
        || c1.width == 0
        || c1.height == 0
        || c2.width == 0
        || c2.height == 0
    {
        return Err(WsiError::Jp2k(
            "decoded image has invalid component dimensions".into(),
        ));
    }

    let c0_sub_x = (w / c0.width).max(1);
    let c1_sub_x = (w / c1.width).max(1);
    let c2_sub_x = (w / c2.width).max(1);
    let c0_sub_y = (h / c0.height).max(1);
    let c1_sub_y = (h / c1.height).max(1);
    let c2_sub_y = (h / c2.height).max(1);

    let mut rgb = vec![0u8; w * h * 3];
    match colorspace {
        Jp2kColorSpace::Rgb => {
            for y in 0..h {
                let c0_row = (y / c0_sub_y) * c0.width;
                let c1_row = (y / c1_sub_y) * c1.width;
                let c2_row = (y / c2_sub_y) * c2.width;
                for x in 0..w {
                    let off = (y * w + x) * 3;
                    rgb[off] = clamp_u8(c0.samples[c0_row + x / c0_sub_x]);
                    rgb[off + 1] = clamp_u8(c1.samples[c1_row + x / c1_sub_x]);
                    rgb[off + 2] = clamp_u8(c2.samples[c2_row + x / c2_sub_x]);
                }
            }
        }
        Jp2kColorSpace::YCbCr => {
            for y in 0..h {
                let c0_row = (y / c0_sub_y) * c0.width;
                let c1_row = (y / c1_sub_y) * c1.width;
                let c2_row = (y / c2_sub_y) * c2.width;
                for x in 0..w {
                    let yy = c0.samples[c0_row + x / c0_sub_x];
                    let cb = c1.samples[c1_row + x / c1_sub_x];
                    let cr = c2.samples[c2_row + x / c2_sub_x];
                    let cb_off = cb - 128;
                    let cr_off = cr - 128;
                    let r = yy + ((1402 * cr_off) / 1000);
                    let g = yy - ((344 * cb_off + 714 * cr_off) / 1000);
                    let b = yy + ((1772 * cb_off) / 1000);

                    let off = (y * w + x) * 3;
                    rgb[off] = clamp_u8(r);
                    rgb[off + 1] = clamp_u8(g);
                    rgb[off + 2] = clamp_u8(b);
                }
            }
        }
    }

    Ok(CpuTile {
        width: w as u32,
        height: h as u32,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(rgb),
    })
}

pub(crate) fn interleaved_image_to_sample_buffer(
    image: DecodedInterleavedImage,
) -> Result<CpuTile, WsiError> {
    let expected_len = image
        .width
        .checked_mul(image.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| WsiError::Jp2k("decoded JP2K image size overflow".into()))?;
    if image.pixels.len() != expected_len {
        return Err(WsiError::Jp2k(format!(
            "unexpected decoded JP2K buffer length: expected {}, found {}",
            expected_len,
            image.pixels.len()
        )));
    }

    let pixels = match image.colorspace {
        Jp2kColorSpace::Rgb => image.pixels,
        Jp2kColorSpace::YCbCr => {
            let mut rgb = vec![0u8; expected_len];
            for (src, dst) in image.pixels.chunks_exact(3).zip(rgb.chunks_exact_mut(3)) {
                let yy = i32::from(src[0]);
                let cb = i32::from(src[1]);
                let cr = i32::from(src[2]);
                let cb_off = cb - 128;
                let cr_off = cr - 128;
                dst[0] = clamp_u8(yy + ((1402 * cr_off) / 1000));
                dst[1] = clamp_u8(yy - ((344 * cb_off + 714 * cr_off) / 1000));
                dst[2] = clamp_u8(yy + ((1772 * cb_off) / 1000));
            }
            rgb
        }
    };

    Ok(CpuTile {
        width: image.width as u32,
        height: image.height as u32,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(pixels),
    })
}

pub(crate) fn crop_sample_buffer(
    buffer: CpuTile,
    expected_width: u32,
    expected_height: u32,
) -> Result<CpuTile, WsiError> {
    if expected_width == 0 || expected_height == 0 {
        return Err(WsiError::Jp2k(
            "cropped JP2K dimensions must be non-zero".into(),
        ));
    }
    if expected_width > buffer.width || expected_height > buffer.height {
        return Err(WsiError::Jp2k(format!(
            "decoded JP2K buffer too small to crop: decoded {}x{}, requested {}x{}",
            buffer.width, buffer.height, expected_width, expected_height
        )));
    }
    if expected_width == buffer.width && expected_height == buffer.height {
        return Ok(buffer);
    }
    if buffer.layout != CpuTileLayout::Interleaved {
        return Err(WsiError::Jp2k(format!(
            "unsupported JP2K buffer layout for crop: {:?}",
            buffer.layout
        )));
    }

    let channels = buffer.channels as usize;
    let src_width = buffer.width as usize;
    let dst_width = expected_width as usize;
    let dst_height = expected_height as usize;

    let data = match buffer.data {
        CpuTileData::U8(samples) => {
            let mut cropped = Vec::with_capacity(dst_width * dst_height * channels);
            let src_row_stride = src_width * channels;
            let dst_row_width = dst_width * channels;
            for row in 0..dst_height {
                let start = row * src_row_stride;
                cropped.extend_from_slice(&samples[start..start + dst_row_width]);
            }
            CpuTileData::u8(cropped)
        }
        other => {
            return Err(WsiError::Jp2k(format!(
                "unsupported JP2K sample type for crop: {:?}",
                other.sample_type()
            )))
        }
    };

    Ok(CpuTile {
        width: expected_width,
        height: expected_height,
        channels: buffer.channels,
        color_space: buffer.color_space,
        layout: buffer.layout,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::jp2k_backend::{DecodedComponent, DecodedImage, DecodedInterleavedImage};

    #[test]
    fn rgb_components_convert_to_rgba() {
        let image = DecodedImage {
            width: 2,
            height: 1,
            components: [
                DecodedComponent {
                    width: 2,
                    height: 1,
                    samples: vec![10, 20],
                },
                DecodedComponent {
                    width: 2,
                    height: 1,
                    samples: vec![30, 40],
                },
                DecodedComponent {
                    width: 2,
                    height: 1,
                    samples: vec![50, 60],
                },
            ],
        };

        let rgba = decoded_image_to_rgba(image, Jp2kColorSpace::Rgb).unwrap();
        assert_eq!(rgba.as_raw(), &[10, 30, 50, 255, 20, 40, 60, 255]);
    }

    #[test]
    fn ycbcr_components_convert_to_rgba() {
        let image = DecodedImage {
            width: 1,
            height: 1,
            components: [
                DecodedComponent {
                    width: 1,
                    height: 1,
                    samples: vec![100],
                },
                DecodedComponent {
                    width: 1,
                    height: 1,
                    samples: vec![128],
                },
                DecodedComponent {
                    width: 1,
                    height: 1,
                    samples: vec![128],
                },
            ],
        };

        let rgba = decoded_image_to_rgba(image, Jp2kColorSpace::YCbCr).unwrap();
        assert_eq!(rgba.as_raw(), &[100, 100, 100, 255]);
    }

    #[test]
    fn rgb_components_convert_to_sample_buffer() {
        let image = DecodedImage {
            width: 2,
            height: 1,
            components: [
                DecodedComponent {
                    width: 2,
                    height: 1,
                    samples: vec![10, 20],
                },
                DecodedComponent {
                    width: 2,
                    height: 1,
                    samples: vec![30, 40],
                },
                DecodedComponent {
                    width: 2,
                    height: 1,
                    samples: vec![50, 60],
                },
            ],
        };

        let buffer = decoded_image_to_sample_buffer(image, Jp2kColorSpace::Rgb).unwrap();
        assert_eq!(buffer.data.as_u8().unwrap(), &[10, 30, 50, 20, 40, 60]);
    }

    #[test]
    fn crop_sample_buffer_trims_to_requested_bounds() {
        let buffer = CpuTile {
            width: 4,
            height: 3,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8((0..36).collect()),
        };

        let cropped = crop_sample_buffer(buffer, 2, 2).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        assert_eq!(
            cropped.data.as_u8().unwrap(),
            &[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
        );
    }

    #[test]
    fn interleaved_rgb_image_wraps_without_repacking() {
        let image = DecodedInterleavedImage {
            width: 2,
            height: 1,
            colorspace: Jp2kColorSpace::Rgb,
            pixels: vec![10, 20, 30, 40, 50, 60],
        };

        let buffer = interleaved_image_to_sample_buffer(image).unwrap();
        assert_eq!(buffer.data.as_u8().unwrap(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn interleaved_ycbcr_image_converts_to_rgb() {
        let image = DecodedInterleavedImage {
            width: 1,
            height: 1,
            colorspace: Jp2kColorSpace::YCbCr,
            pixels: vec![100, 128, 128],
        };

        let buffer = interleaved_image_to_sample_buffer(image).unwrap();
        assert_eq!(buffer.data.as_u8().unwrap(), &[100, 100, 100]);
    }
}
