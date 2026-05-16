use std::borrow::Cow;
#[cfg(all(feature = "metal", test, target_os = "macos"))]
use std::cell::Cell;

use crate::core::types::{ColorSpace, CpuTile};
#[cfg(feature = "metal")]
use crate::core::types::{DeviceTile, TilePixels};
use crate::error::WsiError;
#[cfg(test)]
use image::RgbaImage;
use rayon::prelude::*;
#[cfg(feature = "metal")]
use signinum_core::{
    BackendKind as SigninumBackendKind, BackendRequest as SigninumBackendRequest,
    DeviceSurface as SigninumDeviceSurface,
};
#[cfg(feature = "metal")]
use signinum_jpeg::JpegView as SigninumJpegView;
use signinum_jpeg::{
    decode_tiles_into_with_options, decode_tiles_scaled_into_with_options,
    ColorTransform as SigninumColorTransform, DecodeOptions as SigninumDecodeOptions,
    Decoder as SigninumJpegDecoder, Downscale as SigninumDownscale, JpegError as SigninumJpegError,
    PixelFormat as SigninumPixelFormat, SofKind as SigninumSofKind,
    TileBatchOptions as SigninumTileBatchOptions, TileDecodeJob as SigninumTileDecodeJob,
    TileScaledDecodeJob as SigninumTileScaledDecodeJob,
};
#[cfg(feature = "metal")]
use signinum_jpeg_metal::SurfaceResidency as SigninumJpegSurfaceResidency;

/// Maximum total bytes allowed for a single JPEG decode allocation.
/// Set to 512 MB to cover large NDPI full-decode levels while preventing
/// OOM from crafted JPEG headers with extreme dimensions.
const MAX_JPEG_DECODE_BYTES: u64 = 512 * 1024 * 1024;
const JPEG_MAX_DIMENSION: u16 = 65500;
#[cfg(all(feature = "metal", test, target_os = "macos"))]
thread_local! {
    static JPEG_DEVICE_BATCH_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(feature = "metal", test, target_os = "macos"))]
fn reset_jpeg_device_batch_attempts_for_test() {
    JPEG_DEVICE_BATCH_ATTEMPTS.with(|attempts| attempts.set(0));
}

#[cfg(all(feature = "metal", test, target_os = "macos"))]
fn jpeg_device_batch_attempts_for_test() -> usize {
    JPEG_DEVICE_BATCH_ATTEMPTS.with(Cell::get)
}

pub struct DecodedJpegRgb {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct JpegTileGeometry {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
}

#[derive(Debug)]
pub(crate) struct JpegDecodeJob<'a> {
    pub data: Cow<'a, [u8]>,
    pub tables: Option<Cow<'a, [u8]>>,
    pub expected_width: u32,
    pub expected_height: u32,
    pub color_transform: SigninumColorTransform,
    pub force_dimensions: bool,
    pub requested_size: Option<(u32, u32)>,
}

struct ScaledJpegDecode<'a> {
    data: &'a [u8],
    tables: Option<&'a [u8]>,
    expected_width: u32,
    expected_height: u32,
    requested_width: u32,
    requested_height: u32,
    force_dimensions: bool,
    color_transform: SigninumColorTransform,
}

struct PreparedBatchJpeg<'a> {
    input: Cow<'a, [u8]>,
    output_width: u32,
    output_height: u32,
    output_len: usize,
    stride: usize,
    scale: SigninumDownscale,
}

/// Decode JPEG data to premultiplied RGBA (alpha=255 for all decoded pixels).
///
/// If `tables` is provided, it is prepended to `data` before decoding.
/// Tables end with FFD9 (EOI), data starts with FFD8 (SOI).
/// Strip EOI from tables, strip SOI from data, concatenate.
#[cfg(test)]
pub fn decode_jpeg(
    data: &[u8],
    tables: Option<&[u8]>,
    expected_width: u32,
    expected_height: u32,
) -> Result<RgbaImage, WsiError> {
    let decoded = decode_jpeg_rgb(data, tables, expected_width, expected_height)?;
    let pixel_count = (decoded.width as usize)
        .checked_mul(decoded.height as usize)
        .ok_or_else(|| WsiError::Jpeg("pixel dimensions overflow".into()))?;
    let rgba_size = pixel_count
        .checked_mul(4)
        .ok_or_else(|| WsiError::Jpeg("RGBA buffer size overflow".into()))?;
    let mut rgba_buf = vec![255u8; rgba_size];
    for (rgb, rgba) in decoded
        .pixels
        .chunks_exact(3)
        .zip(rgba_buf.chunks_exact_mut(4))
    {
        rgba[0] = rgb[0];
        rgba[1] = rgb[1];
        rgba[2] = rgb[2];
    }
    RgbaImage::from_raw(decoded.width, decoded.height, rgba_buf)
        .ok_or_else(|| WsiError::Jpeg("failed to create RgbaImage".into()))
}

#[cfg(test)]
pub fn decode_jpeg_rgb(
    data: &[u8],
    tables: Option<&[u8]>,
    expected_width: u32,
    expected_height: u32,
) -> Result<DecodedJpegRgb, WsiError> {
    decode_jpeg_rgb_with_color_transform(
        data,
        tables,
        expected_width,
        expected_height,
        SigninumColorTransform::Auto,
    )
}

pub(crate) fn decode_jpeg_rgb_with_size_override(
    data: &[u8],
    tables: Option<&[u8]>,
    image_width: u32,
    image_height: u32,
    requested_width: Option<u32>,
    requested_height: Option<u32>,
    color_transform: SigninumColorTransform,
) -> Result<DecodedJpegRgb, WsiError> {
    if image_width == 0
        || image_height == 0
        || image_width > u16::MAX as u32
        || image_height > u16::MAX as u32
    {
        return Err(WsiError::Jpeg(
            "JPEG size override requires nonzero u16 dimensions".into(),
        ));
    }

    match (requested_width, requested_height) {
        (Some(requested_width), Some(requested_height)) => {
            try_decode_jpeg_rgb_scaled(ScaledJpegDecode {
                data,
                tables,
                expected_width: image_width,
                expected_height: image_height,
                requested_width,
                requested_height,
                force_dimensions: true,
                color_transform,
            })?
            .map_or_else(
                || {
                    decode_jpeg_rgb_with_color_transform_and_patch(
                        data,
                        tables,
                        image_width,
                        image_height,
                        true,
                        color_transform,
                    )
                    .and_then(|decoded| {
                        resize_jpeg_rgb_nearest(decoded, requested_width, requested_height)
                    })
                },
                Ok,
            )
        }
        _ => decode_jpeg_rgb_with_color_transform_and_patch(
            data,
            tables,
            image_width,
            image_height,
            true,
            color_transform,
        ),
    }
}

pub(crate) fn decode_batch_jpeg<'a>(jobs: &[JpegDecodeJob<'a>]) -> Vec<Result<CpuTile, WsiError>> {
    if jobs.len() > 1 {
        if let Some(results) = try_decode_batch_jpeg_with_signinum(jobs) {
            return results;
        }
    }
    if jobs.len() <= 1 {
        return jobs.iter().map(decode_one_jpeg_job).collect();
    }
    jobs.par_iter().map(decode_one_jpeg_job).collect()
}

fn try_decode_batch_jpeg_with_signinum<'a>(
    jobs: &[JpegDecodeJob<'a>],
) -> Option<Vec<Result<CpuTile, WsiError>>> {
    let first = jobs.first()?;
    let color_transform = first.color_transform;
    if jobs
        .iter()
        .any(|job| job.color_transform != color_transform)
    {
        return None;
    }

    let mut prepared = Vec::with_capacity(jobs.len());
    let mut needs_scaled_api = false;
    for job in jobs {
        let prepared_job = prepare_signinum_batch_jpeg_job(job)?;
        needs_scaled_api |= prepared_job.scale != SigninumDownscale::None;
        prepared.push(prepared_job);
    }

    let decode_options = SigninumDecodeOptions::default().with_color_transform(color_transform);
    let mut outputs = prepared
        .iter()
        .map(|job| vec![0u8; job.output_len])
        .collect::<Vec<_>>();
    let batch_options = SigninumTileBatchOptions::default();

    if needs_scaled_api {
        let mut batch_jobs = prepared
            .iter()
            .zip(outputs.iter_mut())
            .map(|(job, output)| SigninumTileScaledDecodeJob {
                input: job.input.as_ref(),
                out: output.as_mut_slice(),
                stride: job.stride,
                scale: job.scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into_with_options(
            &mut batch_jobs,
            SigninumPixelFormat::Rgb8,
            decode_options,
            batch_options,
        )
        .ok()?;
    } else {
        let mut batch_jobs = prepared
            .iter()
            .zip(outputs.iter_mut())
            .map(|(job, output)| SigninumTileDecodeJob {
                input: job.input.as_ref(),
                out: output.as_mut_slice(),
                stride: job.stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into_with_options(
            &mut batch_jobs,
            SigninumPixelFormat::Rgb8,
            decode_options,
            batch_options,
        )
        .ok()?;
    }

    Some(
        prepared
            .into_iter()
            .zip(outputs)
            .map(|(job, pixels)| {
                CpuTile::from_u8_interleaved(
                    job.output_width,
                    job.output_height,
                    3,
                    ColorSpace::Rgb,
                    pixels,
                )
            })
            .collect(),
    )
}

fn prepare_signinum_batch_jpeg_job<'j, 'a>(
    job: &'j JpegDecodeJob<'a>,
) -> Option<PreparedBatchJpeg<'j>> {
    if job.expected_width == 0 || job.expected_height == 0 {
        return None;
    }
    if job.force_dimensions
        && (job.expected_width > u16::MAX as u32 || job.expected_height > u16::MAX as u32)
    {
        return None;
    }

    let (scale, output_width, output_height) = match job.requested_size {
        Some((requested_width, requested_height)) => {
            if requested_width == 0 || requested_height == 0 {
                return None;
            }
            let scale = signinum_downscale_for_dimensions(
                job.expected_width,
                job.expected_height,
                requested_width,
                requested_height,
            )?;
            (scale, requested_width, requested_height)
        }
        None => (
            SigninumDownscale::None,
            job.expected_width,
            job.expected_height,
        ),
    };

    let input = prepare_jpeg_input(
        job.data.as_ref(),
        job.tables.as_deref(),
        job.expected_width,
        job.expected_height,
        job.force_dimensions,
    );
    let encoded_dimensions = inspect_signinum_jpeg_output_size(input.as_ref()).ok()?;
    if encoded_dimensions != (job.expected_width, job.expected_height) {
        return None;
    }
    let output_len = checked_jpeg_rgb_len(output_width, output_height).ok()?;
    let stride = (output_width as usize).checked_mul(3)?;

    Some(PreparedBatchJpeg {
        input,
        output_width,
        output_height,
        output_len,
        stride,
        scale,
    })
}

#[cfg(feature = "metal")]
pub(crate) fn decode_batch_jpeg_pixels<'a>(
    jobs: &[JpegDecodeJob<'a>],
    backend: SigninumBackendRequest,
    require_device: bool,
    metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
) -> Vec<Result<TilePixels, WsiError>> {
    #[cfg(target_os = "macos")]
    if let Some(metal_sessions) = metal_sessions {
        if let Some(decoded) =
            decode_jpeg_tile_batch_to_device_pixels(jobs, backend, require_device, metal_sessions)
        {
            return decoded;
        }
    }

    if jobs.len() <= 1 {
        return jobs
            .iter()
            .map(|job| decode_one_jpeg_pixels(job, backend, require_device, metal_sessions))
            .collect();
    }
    jobs.par_iter()
        .map(|job| decode_one_jpeg_pixels(job, backend, require_device, metal_sessions))
        .collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn decode_jpeg_tile_batch_to_device_pixels<'a>(
    jobs: &[JpegDecodeJob<'a>],
    backend: SigninumBackendRequest,
    require_device: bool,
    metal_sessions: &crate::output::metal::MetalBackendSessions,
) -> Option<Vec<Result<TilePixels, WsiError>>> {
    if jobs.len() < 2
        || metal_sessions.private_jpeg_decode()
        || !matches!(
            backend,
            SigninumBackendRequest::Auto | SigninumBackendRequest::Metal
        )
    {
        return None;
    }

    let mut prepared = Vec::with_capacity(jobs.len());
    for job in jobs {
        if job.force_dimensions
            || job.requested_size.is_some()
            || !matches!(job.color_transform, SigninumColorTransform::Auto)
        {
            return None;
        }

        let input = prepare_jpeg_input(
            job.data.as_ref(),
            job.tables.as_deref(),
            job.expected_width,
            job.expected_height,
            job.force_dimensions,
        );
        let Ok(dimensions) = inspect_signinum_jpeg_output_size(input.as_ref()) else {
            return None;
        };
        if dimensions != (job.expected_width, job.expected_height) {
            return None;
        }
        let Ok(view) = SigninumJpegView::parse_with_options(
            input.as_ref(),
            SigninumDecodeOptions::default().with_color_transform(job.color_transform),
        ) else {
            return None;
        };
        if view.info().sof_kind == SigninumSofKind::Progressive8 {
            return None;
        }
        prepared.push(input);
    }

    #[cfg(all(test, target_os = "macos"))]
    JPEG_DEVICE_BATCH_ATTEMPTS.with(|attempts| attempts.set(attempts.get().saturating_add(1)));

    let inputs = prepared
        .iter()
        .map(|input| input.as_ref())
        .collect::<Vec<_>>();
    let surfaces = match signinum_jpeg_metal::decode_rgb8_batch_to_device_with_session(
        &inputs,
        metal_sessions.jpeg(),
    ) {
        Ok(Some(surfaces)) => surfaces,
        Ok(None) => return None,
        Err(err) if require_device => {
            let reason = format!("JPEG Metal batch decode failed: {err}");
            return Some(
                (0..jobs.len())
                    .map(|_| {
                        Err(WsiError::Unsupported {
                            reason: reason.clone(),
                        })
                    })
                    .collect(),
            );
        }
        Err(_) => return None,
    };

    if surfaces.len() != jobs.len() {
        let reason = format!(
            "JPEG Metal batch returned {} surfaces for {} jobs",
            surfaces.len(),
            jobs.len()
        );
        return Some(
            (0..jobs.len())
                .map(|_| {
                    Err(WsiError::Unsupported {
                        reason: reason.clone(),
                    })
                })
                .collect(),
        );
    }

    Some(
        jobs.iter()
            .zip(surfaces)
            .map(|(job, surface)| match surface {
                Ok(surface) => tile_pixels_from_jpeg_surface(surface, job, require_device),
                Err(err) if require_device => Err(WsiError::Unsupported {
                    reason: format!("JPEG Metal batch decode failed: {err}"),
                }),
                Err(_) => decode_one_jpeg_job(job).map(TilePixels::Cpu),
            })
            .collect(),
    )
}

fn decode_one_jpeg_job(job: &JpegDecodeJob<'_>) -> Result<CpuTile, WsiError> {
    let decoded = if job.force_dimensions {
        decode_jpeg_rgb_with_size_override(
            job.data.as_ref(),
            job.tables.as_deref(),
            job.expected_width,
            job.expected_height,
            job.requested_size.map(|(width, _)| width),
            job.requested_size.map(|(_, height)| height),
            job.color_transform,
        )
    } else if let Some((requested_width, requested_height)) = job.requested_size {
        try_decode_jpeg_rgb_scaled(ScaledJpegDecode {
            data: job.data.as_ref(),
            tables: job.tables.as_deref(),
            expected_width: job.expected_width,
            expected_height: job.expected_height,
            requested_width,
            requested_height,
            force_dimensions: false,
            color_transform: job.color_transform,
        })?
        .map_or_else(
            || {
                decode_jpeg_rgb_with_color_transform(
                    job.data.as_ref(),
                    job.tables.as_deref(),
                    job.expected_width,
                    job.expected_height,
                    job.color_transform,
                )
                .and_then(|decoded| {
                    resize_jpeg_rgb_nearest(decoded, requested_width, requested_height)
                })
            },
            Ok,
        )
    } else {
        decode_jpeg_rgb_with_color_transform(
            job.data.as_ref(),
            job.tables.as_deref(),
            job.expected_width,
            job.expected_height,
            job.color_transform,
        )
    }
    .map_err(|err| WsiError::Codec {
        codec: "jpeg",
        source: Box::new(err),
    })?;

    CpuTile::from_u8_interleaved(
        decoded.width,
        decoded.height,
        3,
        ColorSpace::Rgb,
        decoded.pixels,
    )
}

#[cfg(feature = "metal")]
fn decode_one_jpeg_pixels(
    job: &JpegDecodeJob<'_>,
    _backend: SigninumBackendRequest,
    require_device: bool,
    metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
) -> Result<TilePixels, WsiError> {
    let Some(metal_sessions) = metal_sessions else {
        if require_device {
            return Err(WsiError::Unsupported {
                reason: "device backend not available for jpeg without Metal session".into(),
            });
        }
        return decode_one_jpeg_job(job).map(TilePixels::Cpu);
    };
    let input = prepare_jpeg_input(
        job.data.as_ref(),
        job.tables.as_deref(),
        job.expected_width,
        job.expected_height,
        job.force_dimensions,
    );
    validate_signinum_jpeg_output_size(input.as_ref())?;
    let view = SigninumJpegView::parse_with_options(
        input.as_ref(),
        SigninumDecodeOptions::default().with_color_transform(job.color_transform),
    )
    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
    if progressive_jpeg_requires_cpu_device_route(&view, require_device)? {
        return decode_one_jpeg_job(job).map(TilePixels::Cpu);
    }
    let mut decoder = signinum_jpeg_metal::Decoder::from_view(view)
        .map_err(|err| WsiError::Jpeg(err.to_string()))?;
    if metal_sessions.private_jpeg_decode() {
        match decoder.decode_private_rgb8_tile_with_session(metal_sessions.jpeg()) {
            Ok(tile) => {
                return Ok(TilePixels::Device(DeviceTile::Metal(
                    crate::output::metal::MetalDeviceTile::from_private_jpeg(tile),
                )));
            }
            Err(err) if require_device => {
                return Err(WsiError::Unsupported {
                    reason: format!("JPEG private Metal decode failed: {err}"),
                });
            }
            Err(_) => {}
        }
    }
    let surface = decoder
        .decode_to_device_with_session(SigninumPixelFormat::Rgb8, metal_sessions.jpeg())
        .map_err(|err| WsiError::Jpeg(format!("signinum JPEG device decode failed: {err}")))?;

    tile_pixels_from_jpeg_surface(surface, job, require_device)
}

#[cfg(feature = "metal")]
fn tile_pixels_from_jpeg_surface(
    surface: signinum_jpeg_metal::Surface,
    job: &JpegDecodeJob<'_>,
    require_device: bool,
) -> Result<TilePixels, WsiError> {
    if surface.backend_kind() == SigninumBackendKind::Metal {
        if surface.residency() == SigninumJpegSurfaceResidency::CpuStagedMetalUpload {
            if require_device {
                return Err(WsiError::Unsupported {
                    reason:
                        "JPEG device decode produced CPU-staged Metal upload instead of resident Metal decode"
                            .into(),
                });
            }
            return decode_one_jpeg_job(job).map(TilePixels::Cpu);
        }
        if let Some(tile) = crate::output::metal::MetalDeviceTile::from_jpeg(surface) {
            return Ok(TilePixels::Device(DeviceTile::Metal(tile)));
        }
        if require_device {
            return Err(WsiError::Unsupported {
                reason: "device backend not available for jpeg".into(),
            });
        }
        return decode_one_jpeg_job(job).map(TilePixels::Cpu);
    }

    if require_device {
        return Err(WsiError::Unsupported {
            reason: "device backend not available for jpeg".into(),
        });
    }
    cpu_tile_from_jpeg_surface(surface, job.expected_width, job.expected_height)
        .map(TilePixels::Cpu)
}

#[cfg(feature = "metal")]
fn progressive_jpeg_requires_cpu_device_route(
    view: &SigninumJpegView<'_>,
    require_device: bool,
) -> Result<bool, WsiError> {
    if view.info().sof_kind != SigninumSofKind::Progressive8 {
        return Ok(false);
    }
    if require_device {
        return Err(WsiError::Unsupported {
            reason: "Progressive8 JPEG does not have a resident Metal decode path; use CPU decode or a non-required device output preference".into(),
        });
    }
    Ok(true)
}

#[cfg(feature = "metal")]
fn cpu_tile_from_jpeg_surface(
    surface: signinum_jpeg_metal::Surface,
    expected_width: u32,
    expected_height: u32,
) -> Result<CpuTile, WsiError> {
    if surface.pixel_format() != SigninumPixelFormat::Rgb8 {
        return Err(WsiError::Jpeg(format!(
            "signinum JPEG returned unsupported pixel format {:?}",
            surface.pixel_format()
        )));
    }
    let (width, height) = surface.dimensions();
    let decoded = crop_jpeg_rgb_to_expected(
        DecodedJpegRgb {
            width,
            height,
            pixels: surface.as_bytes().to_vec(),
        },
        expected_width,
        expected_height,
    )?;
    CpuTile::from_u8_interleaved(
        decoded.width,
        decoded.height,
        3,
        ColorSpace::Rgb,
        decoded.pixels,
    )
}

fn prepare_jpeg_input<'a>(
    data: &'a [u8],
    tables: Option<&[u8]>,
    expected_width: u32,
    expected_height: u32,
    force_dimensions: bool,
) -> Cow<'a, [u8]> {
    let input = if let Some(tbl) = tables {
        let tbl_end = if tbl.len() >= 2 && tbl[tbl.len() - 2..] == [0xFF, 0xD9] {
            tbl.len() - 2
        } else {
            tbl.len()
        };
        let data_start = if data.len() >= 2 && data[0..2] == [0xFF, 0xD8] {
            2
        } else {
            0
        };
        Cow::Owned([&tbl[..tbl_end], &data[data_start..]].concat())
    } else {
        Cow::Borrowed(data)
    };
    let patched = patch_jpeg_dimensions(
        input.as_ref(),
        expected_width,
        expected_height,
        force_dimensions,
    );
    match ensure_jpeg_eoi(patched.as_ref()) {
        Cow::Borrowed(bytes) if tables.is_none() && bytes.as_ptr() == data.as_ptr() => {
            Cow::Borrowed(data)
        }
        Cow::Borrowed(bytes) => Cow::Owned(bytes.to_vec()),
        Cow::Owned(bytes) => Cow::Owned(bytes),
    }
}

fn find_sof_position(header: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < header.len().saturating_sub(1) {
        if header[i] == 0xFF && is_sof_marker(header[i + 1]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn patch_sof_dimensions(header: &mut [u8], sof_offset: usize, width: u16, height: u16) {
    if sof_offset + 9 > header.len() {
        return;
    }
    let y = u16::from_be_bytes([header[sof_offset + 5], header[sof_offset + 6]]);
    let x = u16::from_be_bytes([header[sof_offset + 7], header[sof_offset + 8]]);

    let new_y = if y > JPEG_MAX_DIMENSION || y == 0 {
        height.min(JPEG_MAX_DIMENSION)
    } else {
        y
    };
    let new_x = if x > JPEG_MAX_DIMENSION || x == 0 {
        width.min(JPEG_MAX_DIMENSION)
    } else {
        x
    };

    header[sof_offset + 5..sof_offset + 7].copy_from_slice(&new_y.to_be_bytes());
    header[sof_offset + 7..sof_offset + 9].copy_from_slice(&new_x.to_be_bytes());
}

fn set_sof_dimensions(header: &mut [u8], sof_offset: usize, width: u16, height: u16) {
    if sof_offset + 9 > header.len() {
        return;
    }
    header[sof_offset + 5..sof_offset + 7].copy_from_slice(&height.to_be_bytes());
    header[sof_offset + 7..sof_offset + 9].copy_from_slice(&width.to_be_bytes());
}

pub(crate) fn decode_jpeg_rgb_with_color_transform(
    data: &[u8],
    tables: Option<&[u8]>,
    expected_width: u32,
    expected_height: u32,
    color_transform: SigninumColorTransform,
) -> Result<DecodedJpegRgb, WsiError> {
    decode_jpeg_rgb_with_color_transform_and_patch(
        data,
        tables,
        expected_width,
        expected_height,
        false,
        color_transform,
    )
}

fn decode_jpeg_rgb_with_color_transform_and_patch(
    data: &[u8],
    tables: Option<&[u8]>,
    expected_width: u32,
    expected_height: u32,
    force_dimensions: bool,
    color_transform: SigninumColorTransform,
) -> Result<DecodedJpegRgb, WsiError> {
    let input = prepare_jpeg_input(
        data,
        tables,
        expected_width,
        expected_height,
        force_dimensions,
    );
    validate_signinum_jpeg_output_size(input.as_ref())?;
    let decoder = SigninumJpegDecoder::new_with_options(
        input.as_ref(),
        SigninumDecodeOptions::default().with_color_transform(color_transform),
    )
    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
    let (pixels, outcome) = decoder
        .decode(SigninumPixelFormat::Rgb8)
        .map_err(|err| WsiError::Jpeg(err.to_string()))?;
    crop_jpeg_rgb_to_expected(
        DecodedJpegRgb {
            width: outcome.decoded.w,
            height: outcome.decoded.h,
            pixels,
        },
        expected_width,
        expected_height,
    )
}

fn signinum_downscale_for_dimensions(
    expected_width: u32,
    expected_height: u32,
    requested_width: u32,
    requested_height: u32,
) -> Option<SigninumDownscale> {
    if expected_width == requested_width && expected_height == requested_height {
        return Some(SigninumDownscale::None);
    }
    for (scale, denom) in [
        (SigninumDownscale::Half, 2),
        (SigninumDownscale::Quarter, 4),
        (SigninumDownscale::Eighth, 8),
    ] {
        if expected_width.is_multiple_of(denom)
            && expected_height.is_multiple_of(denom)
            && expected_width / denom == requested_width
            && expected_height / denom == requested_height
        {
            return Some(scale);
        }
    }
    None
}

fn try_decode_jpeg_rgb_scaled(
    req: ScaledJpegDecode<'_>,
) -> Result<Option<DecodedJpegRgb>, WsiError> {
    let Some(scale) = signinum_downscale_for_dimensions(
        req.expected_width,
        req.expected_height,
        req.requested_width,
        req.requested_height,
    ) else {
        return Ok(None);
    };

    let input = prepare_jpeg_input(
        req.data,
        req.tables,
        req.expected_width,
        req.expected_height,
        req.force_dimensions,
    );
    validate_signinum_jpeg_output_size(input.as_ref())?;
    let decoder = SigninumJpegDecoder::new_with_options(
        input.as_ref(),
        SigninumDecodeOptions::default().with_color_transform(req.color_transform),
    )
    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
    let decode_result = if scale == SigninumDownscale::None {
        decoder.decode(SigninumPixelFormat::Rgb8)
    } else {
        decoder.decode_scaled(SigninumPixelFormat::Rgb8, scale)
    };
    let (pixels, outcome) = match decode_result {
        Ok(decoded) => decoded,
        Err(err) if should_retry_scaled_jpeg_as_full_decode(&err) => return Ok(None),
        Err(err) => return Err(WsiError::Jpeg(err.to_string())),
    };
    let decoded = if scale == SigninumDownscale::None {
        DecodedJpegRgb {
            width: outcome.decoded.w,
            height: outcome.decoded.h,
            pixels,
        }
    } else {
        DecodedJpegRgb {
            width: req.requested_width,
            height: req.requested_height,
            pixels,
        }
    };
    Ok(Some(crop_jpeg_rgb_to_expected(
        decoded,
        req.requested_width,
        req.requested_height,
    )?))
}

fn should_retry_scaled_jpeg_as_full_decode(err: &SigninumJpegError) -> bool {
    matches!(
        err,
        SigninumJpegError::DownscaleUnsupported { .. }
            | SigninumJpegError::NotImplemented {
                sof: SigninumSofKind::Progressive8
            }
    )
}

pub(crate) fn jpeg_dimensions(data: &[u8]) -> Result<(u32, u32), WsiError> {
    let info = SigninumJpegDecoder::inspect(data).map_err(|err| WsiError::Jpeg(err.to_string()))?;
    Ok(info.dimensions)
}

#[cfg(test)]
pub(crate) fn jpeg_tile_geometry(data: &[u8]) -> Result<JpegTileGeometry, WsiError> {
    let header = parse_jpeg_tile_header(data)?;
    let restart_interval = header.restart_interval;
    let mcu_width = u32::from(header.max_h) * 8;
    let mcu_height = u32::from(header.max_v) * 8;
    let mcus_per_row = header.width.div_ceil(mcu_width as u16);
    if restart_interval > mcus_per_row {
        return Err(WsiError::Jpeg(format!(
            "JPEG restart interval {} exceeds MCUs per row {}",
            restart_interval, mcus_per_row
        )));
    }
    if mcus_per_row % restart_interval != 0 {
        return Err(WsiError::Jpeg(
            "JPEG restart interval does not divide MCUs per row".into(),
        ));
    }

    Ok(JpegTileGeometry {
        width: header.width as u32,
        height: header.height as u32,
        tile_width: mcu_width * u32::from(restart_interval),
        tile_height: mcu_height,
    })
}

#[cfg(test)]
struct ParsedJpegTileHeader {
    width: u16,
    height: u16,
    restart_interval: u16,
    max_h: u8,
    max_v: u8,
}

#[cfg(test)]
fn parse_jpeg_tile_header(data: &[u8]) -> Result<ParsedJpegTileHeader, WsiError> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(WsiError::Jpeg("JPEG missing SOI marker".into()));
    }

    let mut i = 2usize;
    let mut width = None;
    let mut height = None;
    let mut restart_interval = None;
    let mut max_h = 1u8;
    let mut max_v = 1u8;

    while i + 1 < data.len() {
        if data[i] != 0xFF {
            return Err(WsiError::Jpeg(format!(
                "expected JPEG marker at byte {i}, found {:02X}",
                data[i]
            )));
        }

        while i < data.len() && data[i] == 0xFF {
            i += 1;
        }
        if i >= data.len() {
            break;
        }
        let marker = data[i];
        i += 1;

        match marker {
            0xD9 | 0xDA => break,
            0x00 | 0xD0..=0xD7 => continue,
            _ => {}
        }

        if i + 1 >= data.len() {
            return Err(WsiError::Jpeg(format!(
                "truncated JPEG marker length for marker FF{:02X}",
                marker
            )));
        }
        let seg_len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        if seg_len < 2 || i + seg_len > data.len() {
            return Err(WsiError::Jpeg(format!(
                "invalid JPEG segment length {} for marker FF{:02X}",
                seg_len, marker
            )));
        }
        let payload = &data[i + 2..i + seg_len];

        if is_sof_marker(marker) {
            if payload.len() < 6 {
                return Err(WsiError::Jpeg("JPEG SOF segment too short".into()));
            }
            height = Some(u16::from_be_bytes([payload[1], payload[2]]));
            width = Some(u16::from_be_bytes([payload[3], payload[4]]));
            let component_count = payload[5] as usize;
            let components = &payload[6..];
            if components.len() < component_count * 3 {
                return Err(WsiError::Jpeg("JPEG SOF component table too short".into()));
            }
            for component in components.chunks_exact(3).take(component_count) {
                let sampling = component[1];
                max_h = max_h.max(sampling >> 4);
                max_v = max_v.max(sampling & 0x0F);
            }
        } else if marker == 0xDD {
            if payload.len() < 2 {
                return Err(WsiError::Jpeg("JPEG DRI segment too short".into()));
            }
            restart_interval = Some(u16::from_be_bytes([payload[0], payload[1]]));
        }

        i += seg_len;
    }

    let width = width.ok_or_else(|| WsiError::Jpeg("JPEG missing SOF marker".into()))?;
    let height = height.ok_or_else(|| WsiError::Jpeg("JPEG missing SOF marker".into()))?;
    let restart_interval = restart_interval.unwrap_or(0);
    if restart_interval == 0 {
        return Err(WsiError::Jpeg("JPEG missing restart markers".into()));
    }

    Ok(ParsedJpegTileHeader {
        width,
        height,
        restart_interval,
        max_h,
        max_v,
    })
}

fn is_sof_marker(marker: u8) -> bool {
    matches!(
        marker,
        0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF
    )
}

fn ensure_jpeg_eoi<'a>(input: &'a [u8]) -> Cow<'a, [u8]> {
    if input.len() >= 2 && input[input.len() - 2..] == [0xFF, 0xD9] {
        return Cow::Borrowed(input);
    }

    let mut repaired = input.to_vec();
    if repaired.len() >= 2 && repaired[repaired.len() - 2] == 0xFF {
        let last = repaired.len() - 1;
        repaired[last] = 0xD9;
    } else {
        repaired.push(0xFF);
        repaired.push(0xD9);
    }
    Cow::Owned(repaired)
}

fn validate_signinum_jpeg_output_size(input: &[u8]) -> Result<(), WsiError> {
    inspect_signinum_jpeg_output_size(input).map(|_| ())
}

fn inspect_signinum_jpeg_output_size(input: &[u8]) -> Result<(u32, u32), WsiError> {
    let info =
        SigninumJpegDecoder::inspect(input).map_err(|err| WsiError::Jpeg(err.to_string()))?;
    let _ = checked_jpeg_rgb_len(info.dimensions.0, info.dimensions.1)?;
    Ok(info.dimensions)
}

fn checked_jpeg_rgb_len(width: u32, height: u32) -> Result<usize, WsiError> {
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| WsiError::Jpeg("JPEG decode size overflow".into()))?;
    if bytes > MAX_JPEG_DECODE_BYTES {
        return Err(WsiError::Jpeg(format!(
            "JPEG decode size {bytes} bytes exceeds {MAX_JPEG_DECODE_BYTES} byte limit"
        )));
    }
    usize::try_from(bytes).map_err(|_| WsiError::Jpeg("JPEG decode size overflow".into()))
}

fn crop_jpeg_rgb_to_expected(
    decoded: DecodedJpegRgb,
    expected_width: u32,
    expected_height: u32,
) -> Result<DecodedJpegRgb, WsiError> {
    if expected_width == 0 || expected_height == 0 {
        return Ok(decoded);
    }
    if decoded.width <= expected_width && decoded.height <= expected_height {
        return Ok(decoded);
    }

    let crop_w = decoded.width.min(expected_width) as usize;
    let crop_h = decoded.height.min(expected_height) as usize;
    let src_stride = decoded.width as usize * 3;
    let dst_stride = crop_w * 3;
    let mut cropped = Vec::with_capacity(crop_w * crop_h * 3);
    for row in 0..crop_h {
        let start = row * src_stride;
        cropped.extend_from_slice(&decoded.pixels[start..start + dst_stride]);
    }
    Ok(DecodedJpegRgb {
        width: crop_w as u32,
        height: crop_h as u32,
        pixels: cropped,
    })
}

fn resize_jpeg_rgb_nearest(
    decoded: DecodedJpegRgb,
    requested_width: u32,
    requested_height: u32,
) -> Result<DecodedJpegRgb, WsiError> {
    let pixel_count = u64::from(requested_width)
        .checked_mul(u64::from(requested_height))
        .ok_or_else(|| WsiError::Jpeg("scaled JPEG dimensions overflow".into()))?;
    let len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| WsiError::Jpeg("scaled JPEG buffer size overflow".into()))?;
    if len > MAX_JPEG_DECODE_BYTES {
        return Err(WsiError::Jpeg(format!(
            "JPEG scaled decode size {len} bytes exceeds {MAX_JPEG_DECODE_BYTES} byte limit"
        )));
    }

    let mut pixels = vec![0u8; len as usize];
    let src_width = decoded.width as usize;
    let src_height = decoded.height as usize;
    let dst_width = requested_width as usize;
    let dst_height = requested_height as usize;
    for y in 0..dst_height {
        let src_y = y * src_height / dst_height;
        for x in 0..dst_width {
            let src_x = x * src_width / dst_width;
            let src = (src_y * src_width + src_x) * 3;
            let dst = (y * dst_width + x) * 3;
            pixels[dst..dst + 3].copy_from_slice(&decoded.pixels[src..src + 3]);
        }
    }

    Ok(DecodedJpegRgb {
        width: requested_width,
        height: requested_height,
        pixels,
    })
}

fn patch_jpeg_dimensions<'a>(
    input: &'a [u8],
    expected_width: u32,
    expected_height: u32,
    force_dimensions: bool,
) -> Cow<'a, [u8]> {
    if expected_width == 0
        || expected_height == 0
        || expected_width > u16::MAX as u32
        || expected_height > u16::MAX as u32
    {
        return Cow::Borrowed(input);
    }

    let Some(sof_offset) = find_sof_position(input) else {
        return Cow::Borrowed(input);
    };

    if sof_offset + 9 > input.len() {
        return Cow::Borrowed(input);
    }

    let encoded_height = u16::from_be_bytes([input[sof_offset + 5], input[sof_offset + 6]]);
    let encoded_width = u16::from_be_bytes([input[sof_offset + 7], input[sof_offset + 8]]);
    let needs_patch = encoded_width == 0
        || encoded_height == 0
        || (force_dimensions
            && (encoded_width != expected_width as u16
                || encoded_height != expected_height as u16));
    if !needs_patch {
        return Cow::Borrowed(input);
    }

    let mut patched = input.to_vec();
    if force_dimensions {
        set_sof_dimensions(
            &mut patched,
            sof_offset,
            expected_width as u16,
            expected_height as u16,
        );
    } else {
        patch_sof_dimensions(
            &mut patched,
            sof_offset,
            expected_width as u16,
            expected_height as u16,
        );
    }
    Cow::Owned(patched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};

    fn encode_test_jpeg(img: &image::RgbImage) -> Vec<u8> {
        let mut encoded = Vec::new();
        JpegEncoder::new(&mut encoded, 90)
            .encode(
                img.as_raw().as_slice(),
                img.width() as u16,
                img.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        encoded
    }

    fn progressive_8x8_jpeg() -> Vec<u8> {
        const HEX: &str = concat!(
            "ffd8ffe000104a46494600010100000100010000ffdb0043000302020302020303030304030304050805050404050a07",
            "0706080c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ffdb0043010304",
            "0405040509050509140d0b0d141414141414141414141414141414141414141414141414141414141414141414141414",
            "1414141414141414141414141414ffc20011080008000803012200021101031101ffc400150001010000000000000000",
            "0000000000000006ffc4001501010100000000000000000000000000000506ffda000c0301000210031000000188136f",
            "7fffc4001410010000000000000000000000000000000000ffda00080101000105027fffc40014110100000000000000",
            "000000000000000000ffda0008010301013f017fffc40014110100000000000000000000000000000000ffda00080102",
            "01013f017fffc40014100100000000000000000000000000000000ffda0008010100063f027fffc40014100100000000",
            "000000000000000000000000ffda0008010100013f217fffda000c03010002000300000010f7ffc40014110100000000",
            "000000000000000000000000ffda0008010301013f107fffc40014110100000000000000000000000000000000ffda00",
            "08010201013f107fffc40014100100000000000000000000000000000000ffda0008010100013f107fffd9",
        );
        assert_eq!(HEX.len() % 2, 0);
        HEX.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = (pair[0] as char).to_digit(16).unwrap();
                let low = (pair[1] as char).to_digit(16).unwrap();
                ((high << 4) | low) as u8
            })
            .collect()
    }

    #[test]
    fn decode_valid_jpeg() {
        let mut rgb = image::RgbImage::new(8, 8);
        for pixel in rgb.pixels_mut() {
            *pixel = image::Rgb([200, 100, 50]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);
        let decoded = decode_jpeg(&jpeg_data, None, 8, 8).unwrap();
        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 8);
        // All alpha channels should be 255
        for pixel in decoded.pixels() {
            assert_eq!(pixel[3], 255);
        }
    }

    #[test]
    fn decode_empty_data_fails() {
        let result = decode_jpeg(&[], None, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn decode_with_jpeg_tables() {
        // Create a valid JPEG
        let mut rgb = image::RgbImage::new(8, 8);
        for pixel in rgb.pixels_mut() {
            *pixel = image::Rgb([100, 150, 200]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);

        // Find SOS marker (0xFF, 0xDA) to split into tables and scan data.
        // Tables = everything up to (but not including) SOS marker, plus EOI.
        // Data = SOI + SOS marker onward.
        let sos_pos = jpeg_data
            .windows(2)
            .position(|w| w == [0xFF, 0xDA])
            .expect("SOS marker not found");

        // tables: from start to just before SOS, with EOI appended
        let mut tables = jpeg_data[..sos_pos].to_vec();
        tables.extend_from_slice(&[0xFF, 0xD9]); // EOI

        // data: SOI + from SOS onward
        let mut data = vec![0xFF, 0xD8]; // SOI
        data.extend_from_slice(&jpeg_data[sos_pos..]);

        let decoded = decode_jpeg(&data, Some(&tables), 8, 8).unwrap();
        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 8);
        for pixel in decoded.pixels() {
            assert_eq!(pixel[3], 255);
        }
    }

    #[test]
    fn decode_jpeg_rgb_returns_interleaved_rgb() {
        let mut rgb = image::RgbImage::new(4, 4);
        for (idx, pixel) in rgb.pixels_mut().enumerate() {
            *pixel = image::Rgb([idx as u8, 200, 50]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);

        let decoded = decode_jpeg_rgb(&jpeg_data, None, 4, 4).unwrap();
        assert_eq!(decoded.width, 4);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.pixels.len(), 4 * 4 * 3);
    }

    #[test]
    fn decode_progressive_jpeg_rgb_returns_interleaved_rgb() {
        let jpeg_data = progressive_8x8_jpeg();

        let decoded = decode_jpeg_rgb(&jpeg_data, None, 8, 8).unwrap();

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.pixels.len(), 8 * 8 * 3);
    }

    #[test]
    fn progressive_scaled_decode_falls_back_to_full_decode_resize() {
        let jpeg_data = progressive_8x8_jpeg();

        let decoded = decode_jpeg_rgb_with_size_override(
            &jpeg_data,
            None,
            8,
            8,
            Some(4),
            Some(4),
            SigninumColorTransform::Auto,
        )
        .unwrap();

        assert_eq!(decoded.width, 4);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.pixels.len(), 4 * 4 * 3);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn progressive_jpeg_device_route_uses_cpu_unless_device_is_required() {
        let jpeg_data = progressive_8x8_jpeg();
        let view = SigninumJpegView::parse_with_options(
            &jpeg_data,
            SigninumDecodeOptions::default().with_color_transform(SigninumColorTransform::Auto),
        )
        .unwrap();

        assert!(progressive_jpeg_requires_cpu_device_route(&view, false).unwrap());
        let err = progressive_jpeg_requires_cpu_device_route(&view, true).unwrap_err();
        assert!(matches!(
            err,
            WsiError::Unsupported { reason }
                if reason.contains("Progressive8") && reason.contains("Metal")
        ));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn private_metal_jpeg_decode_returns_private_device_tile() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let sessions = crate::output::metal::MetalBackendSessions::new(
            signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
            signinum_j2k_metal::MetalBackendSession::new(device),
        )
        .with_private_jpeg_decode();
        let mut rgb = image::RgbImage::new(16, 16);
        for (idx, pixel) in rgb.pixels_mut().enumerate() {
            *pixel = image::Rgb([
                ((idx * 17) & 0xff) as u8,
                ((idx * 31 + 9) & 0xff) as u8,
                ((idx * 7 + 3) & 0xff) as u8,
            ]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);
        let job = JpegDecodeJob {
            data: Cow::Borrowed(jpeg_data.as_slice()),
            tables: None,
            expected_width: 16,
            expected_height: 16,
            color_transform: SigninumColorTransform::Auto,
            force_dimensions: false,
            requested_size: None,
        };

        let pixels =
            decode_one_jpeg_pixels(&job, SigninumBackendRequest::Metal, true, Some(&sessions))
                .expect("private JPEG Metal tile");
        let TilePixels::Device(DeviceTile::Metal(tile)) = pixels else {
            panic!("expected private Metal tile");
        };
        let crate::output::metal::MetalDeviceStorage::Buffer { buffer, .. } = tile.storage;
        assert_eq!(buffer.storage_mode(), metal::MTLStorageMode::Private);
        assert_eq!(tile.width, 16);
        assert_eq!(tile.height, 16);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn decode_batch_jpeg_pixels_uses_session_backed_device_batch() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let sessions = crate::output::metal::MetalBackendSessions::new(
            signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
            signinum_j2k_metal::MetalBackendSession::new(device),
        );
        let mut first = image::RgbImage::new(16, 16);
        for (idx, pixel) in first.pixels_mut().enumerate() {
            *pixel = image::Rgb([idx as u8, 80, 180]);
        }
        let mut second = image::RgbImage::new(16, 16);
        for (idx, pixel) in second.pixels_mut().enumerate() {
            *pixel = image::Rgb([200, idx as u8, 40]);
        }
        let first_jpeg = encode_test_jpeg(&first);
        let second_jpeg = encode_test_jpeg(&second);
        let jobs = [
            JpegDecodeJob {
                data: Cow::Borrowed(first_jpeg.as_slice()),
                tables: None,
                expected_width: 16,
                expected_height: 16,
                color_transform: SigninumColorTransform::Auto,
                force_dimensions: false,
                requested_size: None,
            },
            JpegDecodeJob {
                data: Cow::Borrowed(second_jpeg.as_slice()),
                tables: None,
                expected_width: 16,
                expected_height: 16,
                color_transform: SigninumColorTransform::Auto,
                force_dimensions: false,
                requested_size: None,
            },
        ];

        reset_jpeg_device_batch_attempts_for_test();
        let pixels =
            decode_batch_jpeg_pixels(&jobs, SigninumBackendRequest::Metal, true, Some(&sessions));

        assert_eq!(jpeg_device_batch_attempts_for_test(), 1);
        assert_eq!(pixels.len(), 2);
        for pixels in pixels {
            assert!(matches!(pixels.unwrap(), TilePixels::Device(_)));
        }
    }

    #[test]
    fn decode_jpeg_rgb_scaled_returns_scaled_dimensions() {
        let mut rgb = image::RgbImage::new(16, 16);
        for (idx, pixel) in rgb.pixels_mut().enumerate() {
            *pixel = image::Rgb([idx as u8, 100, 200]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);

        let decoded = try_decode_jpeg_rgb_scaled(ScaledJpegDecode {
            data: &jpeg_data,
            tables: None,
            expected_width: 16,
            expected_height: 16,
            requested_width: 4,
            requested_height: 4,
            force_dimensions: false,
            color_transform: SigninumColorTransform::Auto,
        })
        .unwrap()
        .expect("power-of-two downscale should use signinum IDCT scale");

        assert_eq!(decoded.width, 4);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.pixels.len(), 4 * 4 * 3);
    }

    #[test]
    fn signinum_batch_fast_path_matches_single_tile_for_forced_color_transform() {
        let mut rgb = image::RgbImage::new(16, 16);
        for (idx, pixel) in rgb.pixels_mut().enumerate() {
            *pixel = image::Rgb([idx as u8, 100, 200]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);
        let jobs = (0..4)
            .map(|_| JpegDecodeJob {
                data: Cow::Borrowed(jpeg_data.as_slice()),
                tables: None,
                expected_width: 16,
                expected_height: 16,
                color_transform: SigninumColorTransform::ForceRgb,
                force_dimensions: false,
                requested_size: None,
            })
            .collect::<Vec<_>>();

        let fast = try_decode_batch_jpeg_with_signinum(&jobs)
            .expect("forced color transform should use signinum batch fast path");
        let sequential = jobs.iter().map(decode_one_jpeg_job).collect::<Vec<_>>();

        assert_eq!(fast.len(), sequential.len());
        for (fast, sequential) in fast.into_iter().zip(sequential) {
            let fast = fast.unwrap();
            let sequential = sequential.unwrap();
            assert_eq!(fast.width, sequential.width);
            assert_eq!(fast.height, sequential.height);
            assert_eq!(fast.data.as_u8(), sequential.data.as_u8());
        }
    }

    #[test]
    fn signinum_batch_fast_path_matches_single_tile_for_scaled_decode() {
        let mut rgb = image::RgbImage::new(16, 16);
        for (idx, pixel) in rgb.pixels_mut().enumerate() {
            *pixel = image::Rgb([idx as u8, 100, 200]);
        }
        let jpeg_data = encode_test_jpeg(&rgb);
        let jobs = (0..4)
            .map(|_| JpegDecodeJob {
                data: Cow::Borrowed(jpeg_data.as_slice()),
                tables: None,
                expected_width: 16,
                expected_height: 16,
                color_transform: SigninumColorTransform::ForceRgb,
                force_dimensions: false,
                requested_size: Some((4, 4)),
            })
            .collect::<Vec<_>>();

        let fast = try_decode_batch_jpeg_with_signinum(&jobs)
            .expect("scaled decode should use signinum batch fast path");
        let sequential = jobs.iter().map(decode_one_jpeg_job).collect::<Vec<_>>();

        assert_eq!(fast.len(), sequential.len());
        for (fast, sequential) in fast.into_iter().zip(sequential) {
            let fast = fast.unwrap();
            let sequential = sequential.unwrap();
            assert_eq!(fast.width, 4);
            assert_eq!(fast.height, 4);
            assert_eq!(fast.data.as_u8(), sequential.data.as_u8());
        }
    }

    #[test]
    fn patch_jpeg_dimensions_overwrites_zero_sized_sof() {
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x11, // length
            0x08, // precision
            0x00, 0x00, // height
            0x00, 0x00, // width
            0x03, // components
            0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
        ];

        let patched = patch_jpeg_dimensions(&jpeg, 512, 256, false);
        let patched = patched.as_ref();
        assert_eq!(&patched[7..9], &256u16.to_be_bytes());
        assert_eq!(&patched[9..11], &512u16.to_be_bytes());

        // Original input is unchanged.
        assert_eq!(&jpeg[7..9], &[0, 0]);
        assert_eq!(&jpeg[9..11], &[0, 0]);
    }

    #[test]
    fn patch_jpeg_dimensions_leaves_nonzero_sof_alone() {
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x11, // length
            0x08, // precision
            0x01, 0x00, // height
            0x02, 0x00, // width
            0x03, // components
            0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
        ];

        let patched = patch_jpeg_dimensions(&jpeg, 512, 256, false);
        assert!(matches!(patched, Cow::Borrowed(_)));
    }

    #[test]
    fn patch_jpeg_dimensions_forces_nonzero_sof_when_requested() {
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x11, // length
            0x08, // precision
            0x00, 0x10, // height = 16
            0x04, 0x00, // width = 1024
            0x03, // components
            0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
        ];

        let patched = patch_jpeg_dimensions(&jpeg, 1024, 4, true);
        let patched = patched.as_ref();
        assert_eq!(&patched[7..9], &4u16.to_be_bytes());
        assert_eq!(&patched[9..11], &1024u16.to_be_bytes());
    }

    #[test]
    fn ensure_jpeg_eoi_appends_missing_marker() {
        let jpeg = vec![0xFF, 0xD8, 0x00, 0x01];
        let repaired = ensure_jpeg_eoi(&jpeg);
        assert_eq!(
            repaired.as_ref()[repaired.as_ref().len() - 2..],
            [0xFF, 0xD9]
        );
    }

    #[test]
    fn ensure_jpeg_eoi_keeps_valid_trailer() {
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let repaired = ensure_jpeg_eoi(&jpeg);
        assert!(matches!(repaired, Cow::Borrowed(_)));
    }

    #[test]
    fn jpeg_tile_geometry_parses_dri_after_sof() {
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x11, // len
            0x08, // precision
            0x00, 0x08, // height
            0x00, 0x20, // width
            0x03, // components
            0x01, 0x22, 0x00, // h=2, v=2
            0x02, 0x11, 0x00, 0x03, 0x11, 0x00, 0xFF, 0xDD, // DRI
            0x00, 0x04, // len
            0x00, 0x02, // restart interval
            0xFF, 0xDA, // SOS
            0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00,
        ];

        let geometry = jpeg_tile_geometry(&jpeg).unwrap();
        assert_eq!(geometry.width, 32);
        assert_eq!(geometry.height, 8);
        assert_eq!(geometry.tile_width, 32);
        assert_eq!(geometry.tile_height, 16);
    }

    #[test]
    fn jpeg_tile_geometry_rejects_missing_restart_markers() {
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x11, // len
            0x08, // precision
            0x00, 0x08, // height
            0x00, 0x10, // width
            0x03, // components
            0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00, 0xFF, 0xDA, // SOS
            0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00,
        ];

        let err = jpeg_tile_geometry(&jpeg).unwrap_err();
        assert!(err.to_string().contains("restart markers"));
    }
}
