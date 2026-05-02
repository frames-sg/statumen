use signinum_core::{DeviceSurface, PixelFormat};
use std::sync::Arc;

/// Codec-specific Metal sessions allocated from one renderer-owned device.
#[derive(Debug, Clone)]
pub struct MetalBackendSessions {
    pub(crate) jpeg: Arc<signinum_jpeg_metal::MetalBackendSession>,
    pub(crate) j2k: Arc<signinum_j2k_metal::MetalBackendSession>,
}

impl MetalBackendSessions {
    pub fn new(
        jpeg: signinum_jpeg_metal::MetalBackendSession,
        j2k: signinum_j2k_metal::MetalBackendSession,
    ) -> Self {
        Self {
            jpeg: Arc::new(jpeg),
            j2k: Arc::new(j2k),
        }
    }

    pub(crate) fn jpeg(&self) -> &signinum_jpeg_metal::MetalBackendSession {
        &self.jpeg
    }

    pub(crate) fn j2k(&self) -> &signinum_j2k_metal::MetalBackendSession {
        &self.j2k
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
}

const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<MetalDeviceTile>;
};
