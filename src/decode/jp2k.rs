use crate::core::types::CpuTile;
#[cfg(feature = "metal")]
use crate::core::types::{DeviceTile, TilePixels};
use crate::decode::jp2k_backend::{effective_output_colorspace, DecodedInterleavedImage};
use crate::decode::jp2k_codestream::{parse_codestream_header, validate_narrow_subset};
#[cfg(debug_assertions)]
use crate::decode::jp2k_packet::parse_tile_part_packets;
use crate::decode::jp2k_raster::{crop_sample_buffer, interleaved_image_to_sample_buffer};
use crate::error::WsiError;
use image::RgbaImage;
use std::borrow::Cow;

#[cfg(feature = "metal")]
use signinum_core::DeviceSurface as SigninumDeviceSurface;
use signinum_core::{BackendRequest as SigninumBackendRequest, PixelFormat as SigninumPixelFormat};
use signinum_j2k::J2kDecoder as SigninumJp2kDecoder;
#[cfg(feature = "metal")]
use signinum_j2k_metal::{J2kDecoder as SigninumMetalJp2kDecoder, MetalTileBatch};
#[cfg(feature = "metal")]
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2kColorSpace {
    Rgb,
    YCbCr,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct Jp2kDecodeJob<'a> {
    pub data: Cow<'a, [u8]>,
    pub expected_width: u32,
    pub expected_height: u32,
    pub rgb_color_space: bool,
    pub backend: SigninumBackendRequest,
}

#[cfg(test)]
#[inline]
pub(crate) fn dimensions_from_bounds(x0: u32, x1: u32, y0: u32, y1: u32) -> Option<(u32, u32)> {
    Some((x1.checked_sub(x0)?, y1.checked_sub(y0)?))
}

/// Decode a raw JPEG2000 codestream (J2K, not JP2 container) into a
/// premultiplied RGBA image with alpha = 255.
#[allow(dead_code)]
pub fn decode_jp2k(
    data: &[u8],
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
) -> Result<RgbaImage, WsiError> {
    sample_buffer_to_rgba(decode_jp2k_to_sample_buffer(
        data,
        expected_width,
        expected_height,
        colorspace,
    )?)
}

#[allow(dead_code)]
pub(crate) fn decode_jp2k_to_sample_buffer(
    data: &[u8],
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
) -> Result<CpuTile, WsiError> {
    decode_jp2k_to_sample_buffer_with_backend(
        data,
        expected_width,
        expected_height,
        colorspace,
        SigninumBackendRequest::Auto,
    )
}

fn decode_jp2k_to_sample_buffer_with_backend(
    data: &[u8],
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
    backend: SigninumBackendRequest,
) -> Result<CpuTile, WsiError> {
    let header = validate_jp2k_decode_request(data, expected_width, expected_height)?;
    let output_colorspace = effective_output_colorspace(&header, colorspace);
    match backend {
        SigninumBackendRequest::Auto | SigninumBackendRequest::Cpu => {
            decode_jp2k_to_sample_buffer_cpu(
                data,
                expected_width,
                expected_height,
                output_colorspace,
            )
        }
        SigninumBackendRequest::Metal | SigninumBackendRequest::Cuda => {
            Err(WsiError::Unsupported {
                reason: "device backend not available for CPU JP2K sample-buffer decode".into(),
            })
        }
    }
}

pub(crate) fn decode_jp2k_tile_batch_to_sample_buffers(
    reqs: &[Jp2kDecodeJob<'_>],
) -> Result<Vec<CpuTile>, WsiError> {
    if reqs.is_empty() {
        return Ok(Vec::new());
    }
    decode_jp2k_tile_batch_with_signinum(reqs)
}

pub(crate) fn decode_batch_jp2k(jobs: &[Jp2kDecodeJob<'_>]) -> Vec<Result<CpuTile, WsiError>> {
    if jobs.len() <= 1 {
        return jobs.iter().map(decode_one_jp2k_job).collect();
    }
    match decode_jp2k_tile_batch_to_sample_buffers(jobs) {
        Ok(tiles) => tiles.into_iter().map(Ok).collect(),
        Err(_) => jobs.iter().map(decode_one_jp2k_job).collect(),
    }
}

#[cfg(feature = "metal")]
pub(crate) fn decode_batch_jp2k_pixels(
    jobs: &[Jp2kDecodeJob<'_>],
    require_device: bool,
    metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
) -> Vec<Result<TilePixels, WsiError>> {
    if jobs.is_empty() {
        return Vec::new();
    }
    match decode_jp2k_tile_batch_to_pixels(jobs, require_device, metal_sessions) {
        Ok(tiles) => tiles.into_iter().map(Ok).collect(),
        Err(_) => jobs
            .iter()
            .map(|job| decode_one_jp2k_pixels(job, require_device, metal_sessions))
            .collect(),
    }
}

fn decode_one_jp2k_job(job: &Jp2kDecodeJob<'_>) -> Result<CpuTile, WsiError> {
    let colorspace = if job.rgb_color_space {
        Jp2kColorSpace::Rgb
    } else {
        Jp2kColorSpace::YCbCr
    };
    decode_jp2k_to_sample_buffer_with_backend(
        job.data.as_ref(),
        job.expected_width,
        job.expected_height,
        colorspace,
        job.backend,
    )
    .map_err(|err| WsiError::Codec {
        codec: "j2k",
        source: Box::new(err),
    })
}

#[cfg(feature = "metal")]
fn decode_one_jp2k_pixels(
    job: &Jp2kDecodeJob<'_>,
    require_device: bool,
    metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
) -> Result<TilePixels, WsiError> {
    let Some(metal_sessions) = metal_sessions else {
        if require_device {
            return Err(WsiError::Unsupported {
                reason: "device backend not available for j2k without Metal session".into(),
            });
        }
        return decode_one_jp2k_job(job).map(TilePixels::Cpu);
    };
    let header =
        validate_jp2k_decode_request(job.data.as_ref(), job.expected_width, job.expected_height)?;
    let colorspace = effective_output_colorspace(
        &header,
        if job.rgb_color_space {
            Jp2kColorSpace::Rgb
        } else {
            Jp2kColorSpace::YCbCr
        },
    );
    let mut decoder = SigninumMetalJp2kDecoder::new(job.data.as_ref())
        .map_err(|err| WsiError::Jp2k(err.to_string()))?;
    let surface = decoder
        .decode_to_device_with_session(SigninumPixelFormat::Rgb8, metal_sessions.j2k())
        .map_err(|err| WsiError::Jp2k(format!("signinum JP2K device decode failed: {err}")))?;
    tile_pixels_from_jp2k_surface(
        surface,
        job.expected_width,
        job.expected_height,
        colorspace,
        require_device,
    )
}

fn validate_jp2k_decode_request(
    data: &[u8],
    expected_width: u32,
    expected_height: u32,
) -> Result<crate::decode::jp2k_codestream::Jp2kCodestreamInfo, WsiError> {
    if data.is_empty() {
        return Err(WsiError::Jp2k("empty JP2K data".into()));
    }

    let header = parse_codestream_header(data)?;
    validate_narrow_subset(&header)?;
    if header.image_width < expected_width || header.image_height < expected_height {
        return Err(WsiError::Jp2k(format!(
            "dimension mismatch: expected at least {}x{}, got {}x{}",
            expected_width, expected_height, header.image_width, header.image_height
        )));
    }
    if header.components.len() != 3 {
        return Err(WsiError::Jp2k(format!(
            "expected 3 components, found {}",
            header.components.len()
        )));
    }

    #[cfg(debug_assertions)]
    if let Some(tile_part) = header.tile_parts.first() {
        let _ = parse_tile_part_packets(data, &header, tile_part);
    }

    Ok(header)
}

fn decode_jp2k_to_sample_buffer_cpu(
    data: &[u8],
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
) -> Result<CpuTile, WsiError> {
    let mut decoder =
        SigninumJp2kDecoder::new(data).map_err(|err| WsiError::Jp2k(err.to_string()))?;
    let (width, height) = decoder.info().dimensions;
    let row_bytes = (width as usize)
        .checked_mul(SigninumPixelFormat::Rgb8.bytes_per_pixel())
        .ok_or_else(|| WsiError::Jp2k("signinum JP2K row byte count overflow".into()))?;
    let len = row_bytes
        .checked_mul(height as usize)
        .ok_or_else(|| WsiError::Jp2k("signinum JP2K output size overflow".into()))?;
    let mut rgb = vec![0; len];

    decoder
        .decode_into(&mut rgb, row_bytes, SigninumPixelFormat::Rgb8)
        .map_err(|err| WsiError::Jp2k(format!("signinum JP2K decode failed: {err}")))?;

    sample_buffer_from_rgb8_bytes(
        rgb,
        width,
        height,
        expected_width,
        expected_height,
        colorspace,
    )
}

#[allow(dead_code)]
fn decode_jp2k_tile_batch_with_signinum(
    reqs: &[Jp2kDecodeJob<'_>],
) -> Result<Vec<CpuTile>, WsiError> {
    reqs.iter().map(decode_one_jp2k_job).collect()
}

#[cfg(feature = "metal")]
fn decode_jp2k_tile_batch_to_pixels(
    reqs: &[Jp2kDecodeJob<'_>],
    require_device: bool,
    metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
) -> Result<Vec<TilePixels>, WsiError> {
    let Some(metal_sessions) = metal_sessions else {
        return if require_device {
            Err(WsiError::Unsupported {
                reason: "device backend not available for j2k without Metal session".into(),
            })
        } else {
            Err(WsiError::Unsupported {
                reason: "device backend not requested without Metal session".into(),
            })
        };
    };
    if jp2k_device_batch_enabled() {
        if let Ok(tiles) = decode_jp2k_tile_batch_to_device_pixels(reqs, require_device) {
            return Ok(tiles);
        }
    }
    let headers = reqs
        .iter()
        .map(|req| {
            validate_jp2k_decode_request(req.data.as_ref(), req.expected_width, req.expected_height)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let surfaces = reqs
        .iter()
        .map(|req| {
            let mut decoder = SigninumMetalJp2kDecoder::new(req.data.as_ref())
                .map_err(|err| WsiError::Jp2k(err.to_string()))?;
            decoder
                .decode_to_device_with_session(SigninumPixelFormat::Rgb8, metal_sessions.j2k())
                .map_err(|err| WsiError::Jp2k(format!("signinum JP2K device decode failed: {err}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    surfaces
        .into_iter()
        .zip(reqs.iter().zip(headers.iter()))
        .map(|(surface, (req, header))| {
            tile_pixels_from_jp2k_surface(
                surface,
                req.expected_width,
                req.expected_height,
                effective_output_colorspace(
                    header,
                    if req.rgb_color_space {
                        Jp2kColorSpace::Rgb
                    } else {
                        Jp2kColorSpace::YCbCr
                    },
                ),
                require_device,
            )
        })
        .collect()
}

#[cfg(feature = "metal")]
fn jp2k_device_batch_enabled() -> bool {
    parse_jp2k_device_batch_flag(std::env::var("STATUMEN_JP2K_DEVICE_BATCH").ok().as_deref())
}

#[cfg(feature = "metal")]
fn parse_jp2k_device_batch_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

#[cfg(feature = "metal")]
fn decode_jp2k_tile_batch_to_device_pixels(
    reqs: &[Jp2kDecodeJob<'_>],
    require_device: bool,
) -> Result<Vec<TilePixels>, WsiError> {
    let headers = reqs
        .iter()
        .map(|req| {
            validate_jp2k_decode_request(req.data.as_ref(), req.expected_width, req.expected_height)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut batch = MetalTileBatch::with_capacity(reqs.len());
    for req in reqs {
        batch
            .push_shared_tile(
                Arc::<[u8]>::from(req.data.as_ref()),
                SigninumPixelFormat::Rgb8,
                SigninumBackendRequest::Metal,
            )
            .map_err(|err| {
                WsiError::Jp2k(format!("signinum JP2K device batch submit failed: {err}"))
            })?;
    }
    let surfaces = batch.decode_all().map_err(|err| {
        WsiError::Jp2k(format!("signinum JP2K device batch decode failed: {err}"))
    })?;
    surfaces
        .into_iter()
        .zip(reqs.iter().zip(headers.iter()))
        .map(|(surface, (req, header))| {
            tile_pixels_from_jp2k_surface(
                surface,
                req.expected_width,
                req.expected_height,
                effective_output_colorspace(
                    header,
                    if req.rgb_color_space {
                        Jp2kColorSpace::Rgb
                    } else {
                        Jp2kColorSpace::YCbCr
                    },
                ),
                require_device,
            )
        })
        .collect()
}

#[cfg(feature = "metal")]
fn tile_pixels_from_jp2k_surface(
    surface: signinum_j2k_metal::Surface,
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
    require_device: bool,
) -> Result<TilePixels, WsiError> {
    if surface.backend_kind() == signinum_core::BackendKind::Metal {
        if let Some(tile) = crate::output::metal::MetalDeviceTile::from_j2k(surface) {
            return Ok(TilePixels::Device(DeviceTile::Metal(tile)));
        }
        if require_device {
            return Err(WsiError::Unsupported {
                reason: "device backend not available for j2k".into(),
            });
        }
        return Err(WsiError::Jp2k(
            "signinum JP2K returned Metal backend without public buffer".into(),
        ));
    }
    if require_device {
        return Err(WsiError::Unsupported {
            reason: "device backend not available for j2k".into(),
        });
    }
    sample_buffer_from_signinum_surface(surface, expected_width, expected_height, colorspace)
        .map(TilePixels::Cpu)
}

fn sample_buffer_from_rgb8_bytes(
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
) -> Result<CpuTile, WsiError> {
    crop_sample_buffer(
        interleaved_image_to_sample_buffer(DecodedInterleavedImage {
            width: width as usize,
            height: height as usize,
            colorspace,
            pixels: bytes,
        })?,
        expected_width,
        expected_height,
    )
}

#[cfg(feature = "metal")]
fn sample_buffer_from_signinum_surface(
    surface: signinum_j2k_metal::Surface,
    expected_width: u32,
    expected_height: u32,
    colorspace: Jp2kColorSpace,
) -> Result<CpuTile, WsiError> {
    if surface.pixel_format() != SigninumPixelFormat::Rgb8 {
        return Err(WsiError::Jp2k(format!(
            "signinum JP2K returned unsupported pixel format {:?}",
            surface.pixel_format()
        )));
    }
    let (width, height) = surface.dimensions();
    let expected_len = width as usize * height as usize * 3;
    let bytes = surface.as_bytes();
    if bytes.len() != expected_len {
        return Err(WsiError::Jp2k(format!(
            "signinum JP2K returned {} bytes for {}x{} RGB8 surface",
            bytes.len(),
            width,
            height
        )));
    }

    sample_buffer_from_rgb8_bytes(
        bytes.to_vec(),
        width,
        height,
        expected_width,
        expected_height,
        colorspace,
    )
}

fn sample_buffer_to_rgba(buffer: CpuTile) -> Result<RgbaImage, WsiError> {
    if buffer.channels != 3 || buffer.layout != crate::core::types::CpuTileLayout::Interleaved {
        return Err(WsiError::Jp2k(format!(
            "unsupported JP2K sample buffer layout for RGBA conversion: channels={}, layout={:?}",
            buffer.channels, buffer.layout
        )));
    }
    let rgb = buffer.data.as_u8().ok_or_else(|| {
        WsiError::Jp2k("unsupported JP2K sample data type for RGBA conversion".into())
    })?;
    let pixel_count = (buffer.width as usize)
        .checked_mul(buffer.height as usize)
        .ok_or_else(|| WsiError::Jp2k("JP2K RGBA image size overflow".into()))?;
    if rgb.len() != pixel_count * 3 {
        return Err(WsiError::Jp2k(format!(
            "unexpected JP2K RGB buffer length: expected {}, found {}",
            pixel_count * 3,
            rgb.len()
        )));
    }
    let mut rgba = vec![255u8; pixel_count * 4];
    for (src, dst) in rgb.chunks_exact(3).zip(rgba.chunks_exact_mut(4)) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }
    RgbaImage::from_raw(buffer.width, buffer.height, rgba)
        .ok_or_else(|| WsiError::Jp2k("failed to create RgbaImage from decoded JP2K data".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::jp2k_codestream::parse_codestream_header;
    use image::{DynamicImage, ImageFormat, RgbaImage};
    use std::io::Cursor;

    fn load_fixture_rgb(ppm_bytes: &[u8]) -> image::RgbImage {
        match image::load(Cursor::new(ppm_bytes), ImageFormat::Pnm).unwrap() {
            DynamicImage::ImageRgb8(image) => image,
            other => other.to_rgb8(),
        }
    }

    const MAX_CHANNEL_DELTA: u8 = 50;
    const MAX_AVG_CHANNEL_DELTA_X100: u64 = 1600;

    fn assert_rgba_matches_rgb_fixture(decoded: &RgbaImage, expected_rgb: &image::RgbImage) {
        assert_eq!(decoded.width(), expected_rgb.width());
        assert_eq!(decoded.height(), expected_rgb.height());

        let mut total_delta = 0u64;
        let mut max_delta = 0u8;
        let mut channels = 0u64;

        for (decoded_pixel, expected_pixel) in decoded.pixels().zip(expected_rgb.pixels()) {
            for channel in 0..3 {
                let delta = decoded_pixel.0[channel].abs_diff(expected_pixel.0[channel]);
                total_delta += u64::from(delta);
                max_delta = max_delta.max(delta);
                channels += 1;
            }
            assert_eq!(decoded_pixel.0[3], 255);
        }

        let avg_delta_x100 = (total_delta * 100).checked_div(channels).unwrap_or(0);

        assert!(
            max_delta <= MAX_CHANNEL_DELTA,
            "JP2K decode drift too large: max channel delta {max_delta} > {MAX_CHANNEL_DELTA}",
        );
        assert!(
            avg_delta_x100 <= MAX_AVG_CHANNEL_DELTA_X100,
            "JP2K decode drift too large: average channel delta {:.2} > {:.2}",
            avg_delta_x100 as f64 / 100.0,
            MAX_AVG_CHANNEL_DELTA_X100 as f64 / 100.0,
        );
    }

    fn assert_sample_buffer_matches_rgb_fixture(image: &CpuTile, expected_rgb: &image::RgbImage) {
        assert_eq!(image.width, expected_rgb.width());
        assert_eq!(image.height, expected_rgb.height());
        let actual = image.data.as_u8().unwrap();
        let expected = expected_rgb.as_raw();
        assert_eq!(actual.len(), expected.len());

        let mut total_delta = 0u64;
        let mut max_delta = 0u8;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let delta = actual.abs_diff(*expected);
            total_delta += u64::from(delta);
            max_delta = max_delta.max(delta);
        }

        let avg_delta_x100 = if actual.is_empty() {
            0
        } else {
            (total_delta * 100) / actual.len() as u64
        };

        assert!(
            max_delta <= MAX_CHANNEL_DELTA,
            "JP2K decode drift too large: max channel delta {max_delta} > {MAX_CHANNEL_DELTA}",
        );
        assert!(
            avg_delta_x100 <= MAX_AVG_CHANNEL_DELTA_X100,
            "JP2K decode drift too large: average channel delta {:.2} > {:.2}",
            avg_delta_x100 as f64 / 100.0,
            MAX_AVG_CHANNEL_DELTA_X100 as f64 / 100.0,
        );
    }

    fn assert_fixture_decodes_to_expected(
        codestream: &[u8],
        expected_ppm: &[u8],
        colorspace: Jp2kColorSpace,
    ) {
        let header = parse_codestream_header(codestream).unwrap();
        let expected = load_fixture_rgb(expected_ppm);
        let decoded = decode_jp2k(
            codestream,
            header.image_width,
            header.image_height,
            colorspace,
        )
        .unwrap();
        assert_rgba_matches_rgb_fixture(&decoded, &expected);
    }

    #[test]
    fn decode_jp2k_rejects_empty_data() {
        let result = decode_jp2k(&[], 8, 8, Jp2kColorSpace::Rgb);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty"), "unexpected error: {msg}");
    }

    #[test]
    fn decode_jp2k_rejects_invalid_data() {
        let result = decode_jp2k(&[0xFF; 100], 8, 8, Jp2kColorSpace::Rgb);
        assert!(result.is_err());
    }

    #[test]
    fn decode_jp2k_rejects_truncated_stream() {
        let mut buf = vec![0xFF, 0x4F, 0xFF, 0x51];
        buf.extend_from_slice(&[0x00; 50]);
        let result = decode_jp2k(&buf, 8, 8, Jp2kColorSpace::Rgb);
        assert!(result.is_err());
    }

    #[test]
    fn colorspace_enum_values() {
        assert_ne!(Jp2kColorSpace::Rgb, Jp2kColorSpace::YCbCr);
        assert_eq!(Jp2kColorSpace::Rgb, Jp2kColorSpace::Rgb);
    }

    #[test]
    fn dimensions_from_bounds_respects_origin_offsets() {
        assert_eq!(dimensions_from_bounds(10, 18, 20, 32), Some((8, 12)));
        assert_eq!(dimensions_from_bounds(5, 4, 0, 1), None);
    }

    #[test]
    fn fixture_rgb_nomct_decodes_to_reference_rgb() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k");
        let expected = include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.ppm");
        assert_fixture_decodes_to_expected(codestream, expected, Jp2kColorSpace::Rgb);
    }

    #[test]
    fn fixture_rgb_nomct_sample_buffer_matches_rgba_decode_exactly() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k");
        let header = parse_codestream_header(codestream).unwrap();

        for (expected_width, expected_height) in [
            (header.image_width, header.image_height),
            (header.image_width, header.image_height - 1),
        ] {
            let rgba = decode_jp2k(
                codestream,
                expected_width,
                expected_height,
                Jp2kColorSpace::Rgb,
            )
            .unwrap();
            let sample = decode_jp2k_to_sample_buffer(
                codestream,
                expected_width,
                expected_height,
                Jp2kColorSpace::Rgb,
            )
            .unwrap();

            assert_eq!(sample.width, expected_width);
            assert_eq!(sample.height, expected_height);

            let sample_rgb = sample.data.as_u8().unwrap();
            let expected_rgb: Vec<u8> = rgba
                .pixels()
                .flat_map(|pixel| {
                    assert_eq!(pixel.0[3], 255);
                    [pixel.0[0], pixel.0[1], pixel.0[2]]
                })
                .collect();

            assert_eq!(sample_rgb, expected_rgb.as_slice());
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fixture_rgb_device_batch_returns_metal_tiles() {
        if metal::Device::system_default().is_none() {
            eprintln!("skipping JP2K device batch test: no Metal device");
            return;
        }
        let codestream = include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k");
        let header = parse_codestream_header(codestream).unwrap();
        let requests = [
            Jp2kDecodeJob {
                data: Cow::Borrowed(codestream),
                expected_width: header.image_width,
                expected_height: header.image_height,
                rgb_color_space: true,
                backend: SigninumBackendRequest::Auto,
            },
            Jp2kDecodeJob {
                data: Cow::Borrowed(codestream),
                expected_width: header.image_width,
                expected_height: header.image_height,
                rgb_color_space: true,
                backend: SigninumBackendRequest::Auto,
            },
        ];

        let decoded = decode_jp2k_tile_batch_to_device_pixels(&requests, false).unwrap();

        assert_eq!(decoded.len(), 2);
        for tile in decoded {
            let TilePixels::Device(DeviceTile::Metal(tile)) = tile else {
                panic!("expected Metal device tile");
            };
            assert_eq!(
                (tile.width, tile.height),
                (header.image_width, header.image_height)
            );
            assert_eq!(tile.format, SigninumPixelFormat::Rgb8);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn jp2k_device_batch_flag_is_opt_in() {
        assert!(!parse_jp2k_device_batch_flag(None));
        assert!(!parse_jp2k_device_batch_flag(Some("0")));
        assert!(!parse_jp2k_device_batch_flag(Some("false")));
        assert!(parse_jp2k_device_batch_flag(Some("1")));
        assert!(parse_jp2k_device_batch_flag(Some("true")));
        assert!(parse_jp2k_device_batch_flag(Some("ON")));
        assert!(parse_jp2k_device_batch_flag(Some("yes")));
    }

    #[test]
    fn tile_batch_decodes_in_submission_order_with_cpu_fallback_policy() {
        let first_codestream = include_bytes!("../../tests/fixtures/jp2k/ycbcr_420.j2k");
        let first_header = parse_codestream_header(first_codestream).unwrap();
        let first_expected =
            load_fixture_rgb(include_bytes!("../../tests/fixtures/jp2k/ycbcr_420.ppm"));
        let second_codestream = include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k");
        let second_header = parse_codestream_header(second_codestream).unwrap();
        let second_expected =
            load_fixture_rgb(include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.ppm"));

        let requests = [
            Jp2kDecodeJob {
                data: Cow::Borrowed(first_codestream),
                expected_width: first_header.image_width,
                expected_height: first_header.image_height,
                rgb_color_space: false,
                backend: SigninumBackendRequest::Cpu,
            },
            Jp2kDecodeJob {
                data: Cow::Borrowed(second_codestream),
                expected_width: second_header.image_width,
                expected_height: second_header.image_height,
                rgb_color_space: true,
                backend: SigninumBackendRequest::Cpu,
            },
        ];

        let decoded = decode_jp2k_tile_batch_to_sample_buffers(&requests).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_sample_buffer_matches_rgb_fixture(&decoded[0], &first_expected);
        assert_sample_buffer_matches_rgb_fixture(&decoded[1], &second_expected);
    }

    #[test]
    fn rgb_tile_batch_signinum_helper_decodes_in_submission_order() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k");
        let header = parse_codestream_header(codestream).unwrap();
        let expected = load_fixture_rgb(include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.ppm"));

        let requests = [
            Jp2kDecodeJob {
                data: Cow::Borrowed(codestream),
                expected_width: header.image_width,
                expected_height: header.image_height,
                rgb_color_space: true,
                backend: SigninumBackendRequest::Cpu,
            },
            Jp2kDecodeJob {
                data: Cow::Borrowed(codestream),
                expected_width: header.image_width,
                expected_height: header.image_height,
                rgb_color_space: true,
                backend: SigninumBackendRequest::Cpu,
            },
        ];

        let decoded = decode_jp2k_tile_batch_with_signinum(&requests).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_sample_buffer_matches_rgb_fixture(&decoded[0], &expected);
        assert_sample_buffer_matches_rgb_fixture(&decoded[1], &expected);
    }

    #[test]
    fn fixture_rgb_mct_decodes_with_ycbcr_hint() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/rgb_mct.j2k");
        let expected = include_bytes!("../../tests/fixtures/jp2k/rgb_mct.ppm");
        assert_fixture_decodes_to_expected(codestream, expected, Jp2kColorSpace::YCbCr);
    }

    #[test]
    fn fixture_ycbcr_444_decodes_to_reference_rgb() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/ycbcr_444.j2k");
        let expected = include_bytes!("../../tests/fixtures/jp2k/ycbcr_444.ppm");
        assert_fixture_decodes_to_expected(codestream, expected, Jp2kColorSpace::YCbCr);
    }

    #[test]
    fn fixture_ycbcr_422_decodes_to_reference_rgb() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/ycbcr_422.j2k");
        let expected = include_bytes!("../../tests/fixtures/jp2k/ycbcr_422.ppm");
        assert_fixture_decodes_to_expected(codestream, expected, Jp2kColorSpace::YCbCr);
    }

    #[test]
    fn fixture_ycbcr_420_decodes_to_reference_rgb() {
        let codestream = include_bytes!("../../tests/fixtures/jp2k/ycbcr_420.j2k");
        let expected = include_bytes!("../../tests/fixtures/jp2k/ycbcr_420.ppm");
        assert_fixture_decodes_to_expected(codestream, expected, Jp2kColorSpace::YCbCr);
    }
}
