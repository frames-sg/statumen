use crate::error::WsiError;
use signinum_core::{DeviceSurface, PixelFormat};
use std::sync::{Arc, Mutex};

/// Codec-specific Metal sessions allocated from one renderer-owned device.
#[derive(Debug, Clone)]
pub struct MetalBackendSessions {
    pub(crate) jpeg: Arc<signinum_jpeg_metal::MetalBackendSession>,
    pub(crate) j2k: Arc<signinum_j2k_metal::MetalBackendSession>,
    ycbcr_to_rgb8: Arc<Mutex<Option<Arc<YcbcrToRgb8Converter>>>>,
    private_jpeg_decode: bool,
}

impl MetalBackendSessions {
    pub fn new(
        jpeg: signinum_jpeg_metal::MetalBackendSession,
        j2k: signinum_j2k_metal::MetalBackendSession,
    ) -> Self {
        Self {
            jpeg: Arc::new(jpeg),
            j2k: Arc::new(j2k),
            ycbcr_to_rgb8: Arc::new(Mutex::new(None)),
            private_jpeg_decode: false,
        }
    }

    pub fn with_private_jpeg_decode(mut self) -> Self {
        self.private_jpeg_decode = true;
        self
    }

    pub(crate) fn jpeg(&self) -> &signinum_jpeg_metal::MetalBackendSession {
        &self.jpeg
    }

    pub(crate) fn private_jpeg_decode(&self) -> bool {
        self.private_jpeg_decode
    }

    pub(crate) fn j2k(&self) -> &signinum_j2k_metal::MetalBackendSession {
        &self.j2k
    }

    pub(crate) fn device_identity(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            self.j2k.device().name().to_string()
        }
        #[cfg(not(target_os = "macos"))]
        {
            "metal".to_string()
        }
    }

    pub(crate) fn ycbcr_to_rgb8_converter(&self) -> Result<Arc<YcbcrToRgb8Converter>, WsiError> {
        let mut cached = self
            .ycbcr_to_rgb8
            .lock()
            .map_err(|_| WsiError::Unsupported {
                reason: "Metal YCbCr converter cache lock is poisoned".into(),
            })?;
        if let Some(converter) = cached.as_ref() {
            return Ok(converter.clone());
        }

        let converter = Arc::new(YcbcrToRgb8Converter::new(self.j2k())?);
        *cached = Some(converter.clone());
        Ok(converter)
    }

    pub(crate) fn ycbcr8_tiles_to_rgb8(
        &self,
        tiles: &[MetalDeviceTile],
    ) -> Result<Vec<MetalDeviceTile>, WsiError> {
        self.ycbcr_to_rgb8_converter()?.convert_tiles(tiles)
    }
}

/// Metal-backed device tile returned from `TilePixels::Device`.
#[derive(Debug, Clone)]
pub struct MetalDeviceTile {
    pub width: u32,
    pub height: u32,
    pub pitch_bytes: usize,
    pub format: PixelFormat,
    pub storage: MetalDeviceStorage,
}

/// Concrete Metal storage backing a [`MetalDeviceTile`].
#[derive(Debug, Clone)]
pub enum MetalDeviceStorage {
    Buffer {
        buffer: metal::Buffer,
        byte_offset: usize,
    },
}

impl MetalDeviceTile {
    pub(crate) fn from_jpeg(surface: signinum_jpeg_metal::Surface) -> Option<Self> {
        let (buffer, byte_offset) = surface.metal_buffer()?;
        Some(Self {
            width: surface.dimensions().0,
            height: surface.dimensions().1,
            pitch_bytes: surface.pitch_bytes(),
            format: surface.pixel_format(),
            storage: MetalDeviceStorage::Buffer {
                buffer: buffer.clone(),
                byte_offset,
            },
        })
    }

    pub(crate) fn from_private_jpeg(tile: signinum_jpeg_metal::ResidentPrivateJpegTile) -> Self {
        Self {
            width: tile.dimensions.0,
            height: tile.dimensions.1,
            pitch_bytes: tile.pitch_bytes,
            format: tile.pixel_format,
            storage: MetalDeviceStorage::Buffer {
                buffer: tile.buffer,
                byte_offset: tile.byte_offset,
            },
        }
    }

    pub(crate) fn from_j2k(surface: signinum_j2k_metal::Surface) -> Option<Self> {
        let (buffer, byte_offset) = surface.metal_buffer()?;
        Some(Self {
            width: surface.dimensions().0,
            height: surface.dimensions().1,
            pitch_bytes: surface.pitch_bytes(),
            format: surface.pixel_format(),
            storage: MetalDeviceStorage::Buffer {
                buffer: buffer.clone(),
                byte_offset,
            },
        })
    }

    pub(crate) fn ycbcr8_to_rgb8(
        &self,
        converter: &YcbcrToRgb8Converter,
    ) -> Result<Self, WsiError> {
        converter.convert_tile(self)
    }
}

pub(crate) struct YcbcrToRgb8Converter {
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

impl core::fmt::Debug for YcbcrToRgb8Converter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("YcbcrToRgb8Converter")
            .finish_non_exhaustive()
    }
}

impl YcbcrToRgb8Converter {
    fn new(session: &signinum_j2k_metal::MetalBackendSession) -> Result<Self, WsiError> {
        let options = metal::CompileOptions::new();
        let library = session
            .device()
            .new_library_with_source(YCBCR_TO_RGB8_METAL, &options)
            .map_err(|message| WsiError::Codec {
                codec: "j2k",
                source: Box::new(WsiError::Jp2k(format!(
                    "Metal YCbCr conversion shader failed to compile: {message}"
                ))),
            })?;
        let function = library
            .get_function("statumen_ycbcr8_to_rgb8", None)
            .map_err(|message| WsiError::Codec {
                codec: "j2k",
                source: Box::new(WsiError::Jp2k(format!(
                    "Metal YCbCr conversion function unavailable: {message}"
                ))),
            })?;
        let pipeline = session
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| WsiError::Codec {
                codec: "j2k",
                source: Box::new(WsiError::Jp2k(format!(
                    "Metal YCbCr conversion pipeline unavailable: {message}"
                ))),
            })?;
        let queue = session.device().new_command_queue();
        Ok(Self { pipeline, queue })
    }

    fn convert_tile(&self, tile: &MetalDeviceTile) -> Result<MetalDeviceTile, WsiError> {
        let mut converted = self.convert_tiles(std::slice::from_ref(tile))?;
        converted.pop().ok_or_else(|| WsiError::Unsupported {
            reason: "Metal YCbCr conversion produced no output tile".into(),
        })
    }

    fn convert_tiles(&self, tiles: &[MetalDeviceTile]) -> Result<Vec<MetalDeviceTile>, WsiError> {
        if tiles.is_empty() {
            return Ok(Vec::new());
        }

        let jobs = tiles
            .iter()
            .map(|tile| self.prepare_job(tile))
            .collect::<Result<Vec<_>, _>>()?;
        let command_buffer = self.queue.new_command_buffer();
        for job in &jobs {
            self.encode_job(command_buffer, job);
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(jobs
            .into_iter()
            .map(|job| MetalDeviceTile {
                width: job.width,
                height: job.height,
                pitch_bytes: job.row_bytes,
                format: PixelFormat::Rgb8,
                storage: MetalDeviceStorage::Buffer {
                    buffer: job.dst_buffer,
                    byte_offset: 0,
                },
            })
            .collect())
    }

    fn prepare_job<'a>(&self, tile: &'a MetalDeviceTile) -> Result<YcbcrToRgb8Job<'a>, WsiError> {
        if tile.format != PixelFormat::Rgb8 {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "Metal YCbCr conversion requires Rgb8-compatible source planes, got {:?}",
                    tile.format
                ),
            });
        }
        let MetalDeviceStorage::Buffer {
            buffer,
            byte_offset,
        } = &tile.storage;
        let bytes_per_pixel = PixelFormat::Rgb8.bytes_per_pixel();
        let row_bytes = (tile.width as usize)
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| WsiError::Unsupported {
                reason: "Metal YCbCr conversion row byte count overflow".into(),
            })?;
        if tile.pitch_bytes < row_bytes {
            return Err(WsiError::Unsupported {
                reason: "Metal YCbCr conversion source pitch is shorter than row bytes".into(),
            });
        }
        let dst_len =
            row_bytes
                .checked_mul(tile.height as usize)
                .ok_or_else(|| WsiError::Unsupported {
                    reason: "Metal YCbCr conversion output byte count overflow".into(),
                })?;
        let dst_buffer = self.queue.device().new_buffer(
            u64::try_from(dst_len).map_err(|_| WsiError::Unsupported {
                reason: "Metal YCbCr conversion output byte count exceeds u64".into(),
            })?,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(YcbcrToRgb8Job {
            src_buffer: buffer,
            src_byte_offset: *byte_offset,
            dst_buffer,
            params: YcbcrToRgb8Params {
                width: tile.width,
                height: tile.height,
                src_pitch: u32::try_from(tile.pitch_bytes).map_err(|_| WsiError::Unsupported {
                    reason: "Metal YCbCr conversion source pitch exceeds u32".into(),
                })?,
                dst_pitch: u32::try_from(row_bytes).map_err(|_| WsiError::Unsupported {
                    reason: "Metal YCbCr conversion destination pitch exceeds u32".into(),
                })?,
            },
            width: tile.width,
            height: tile.height,
            row_bytes,
        })
    }

    fn encode_job(&self, command_buffer: &metal::CommandBufferRef, job: &YcbcrToRgb8Job<'_>) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(job.src_buffer), job.src_byte_offset as u64);
        encoder.set_buffer(1, Some(&job.dst_buffer), 0);
        encoder.set_bytes(
            2,
            core::mem::size_of::<YcbcrToRgb8Params>() as u64,
            std::ptr::from_ref(&job.params).cast(),
        );
        let width = self.pipeline.thread_execution_width().max(1);
        let max_threads = self.pipeline.max_total_threads_per_threadgroup().max(width);
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            metal::MTLSize {
                width: u64::from(job.params.width),
                height: u64::from(job.params.height),
                depth: 1,
            },
            metal::MTLSize {
                width,
                height,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
}

struct YcbcrToRgb8Job<'a> {
    src_buffer: &'a metal::Buffer,
    src_byte_offset: usize,
    dst_buffer: metal::Buffer,
    params: YcbcrToRgb8Params,
    width: u32,
    height: u32,
    row_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct YcbcrToRgb8Params {
    width: u32,
    height: u32,
    src_pitch: u32,
    dst_pitch: u32,
}

const YCBCR_TO_RGB8_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct YcbcrToRgb8Params {
    uint width;
    uint height;
    uint src_pitch;
    uint dst_pitch;
};

static inline uchar clamp_u8_int(int value) {
    return uchar(clamp(value, 0, 255));
}

kernel void statumen_ycbcr8_to_rgb8(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant YcbcrToRgb8Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_idx = gid.y * params.src_pitch + gid.x * 3u;
    const uint dst_idx = gid.y * params.dst_pitch + gid.x * 3u;
    const int yy = int(src[src_idx]);
    const int cb = int(src[src_idx + 1u]) - 128;
    const int cr = int(src[src_idx + 2u]) - 128;
    dst[dst_idx] = clamp_u8_int(yy + ((1402 * cr) / 1000));
    dst[dst_idx + 1u] = clamp_u8_int(yy - ((344 * cb + 714 * cr) / 1000));
    dst[dst_idx + 2u] = clamp_u8_int(yy + ((1772 * cb) / 1000));
}
"#;

const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<MetalDeviceTile>;
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycbcr_to_rgb8_converter_is_cached_per_backend_sessions() {
        let Some(device) = metal::Device::system_default() else {
            eprintln!("skipping Metal converter cache test: no Metal device");
            return;
        };
        let sessions = MetalBackendSessions::new(
            signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
            signinum_j2k_metal::MetalBackendSession::new(device),
        );

        let first = sessions
            .ycbcr_to_rgb8_converter()
            .expect("first YCbCr converter");
        let second = sessions
            .ycbcr_to_rgb8_converter()
            .expect("second YCbCr converter");

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn ycbcr_to_rgb8_tiles_converts_batch_with_one_cached_converter() {
        let Some(device) = metal::Device::system_default() else {
            eprintln!("skipping Metal batch conversion test: no Metal device");
            return;
        };
        let sessions = MetalBackendSessions::new(
            signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
            signinum_j2k_metal::MetalBackendSession::new(device.clone()),
        );
        let tiles = [
            ycbcr_test_tile(&device, &[10, 128, 128, 200, 128, 128]),
            ycbcr_test_tile(&device, &[30, 128, 128, 40, 128, 128]),
        ];

        let converted = sessions
            .ycbcr8_tiles_to_rgb8(&tiles)
            .expect("batch YCbCr conversion");

        assert_eq!(converted.len(), 2);
        assert_eq!(
            tile_rgb_bytes_via_j2k(&converted[0], &sessions),
            vec![10, 10, 10, 200, 200, 200]
        );
        assert_eq!(
            tile_rgb_bytes_via_j2k(&converted[1], &sessions),
            vec![30, 30, 30, 40, 40, 40]
        );
        let first = sessions
            .ycbcr_to_rgb8_converter()
            .expect("cached converter after batch");
        let second = sessions
            .ycbcr_to_rgb8_converter()
            .expect("cached converter after batch");
        assert!(Arc::ptr_eq(&first, &second));
    }

    fn ycbcr_test_tile(device: &metal::Device, bytes: &[u8]) -> MetalDeviceTile {
        let buffer = device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        MetalDeviceTile {
            width: 2,
            height: 1,
            pitch_bytes: 6,
            format: PixelFormat::Rgb8,
            storage: MetalDeviceStorage::Buffer {
                buffer,
                byte_offset: 0,
            },
        }
    }

    fn tile_rgb_bytes_via_j2k(tile: &MetalDeviceTile, sessions: &MetalBackendSessions) -> Vec<u8> {
        let MetalDeviceStorage::Buffer {
            buffer,
            byte_offset,
        } = &tile.storage;
        let encoded = signinum_j2k_metal::encode_lossless_from_padded_metal_buffer_with_report(
            signinum_j2k_metal::MetalLosslessEncodeTile {
                buffer,
                byte_offset: *byte_offset,
                width: tile.width,
                height: tile.height,
                pitch_bytes: tile.pitch_bytes,
                output_width: tile.width,
                output_height: tile.height,
                format: tile.format,
            },
            &signinum_j2k::J2kLosslessEncodeOptions {
                backend: signinum_j2k::EncodeBackendPreference::RequireDevice,
                validation: signinum_j2k::J2kEncodeValidation::External,
                ..signinum_j2k::J2kLosslessEncodeOptions::default()
            },
            sessions.j2k(),
        )
        .expect("encode Metal tile");
        let mut actual = vec![0; tile.width as usize * tile.height as usize * 3];
        signinum_j2k::J2kDecoder::new(&encoded.encoded.codestream)
            .expect("decode encoded Metal tile")
            .decode_into(
                &mut actual,
                tile.width as usize * PixelFormat::Rgb8.bytes_per_pixel(),
                PixelFormat::Rgb8,
            )
            .expect("decode RGB output");
        actual
    }
}
