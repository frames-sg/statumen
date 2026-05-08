//! Layer 3: Pixel access — TiffPixelReader and FullDecodeCache.
//!
//! TiffPixelReader implements SlideReader by dispatching tile reads
//! to the appropriate handler based on TileSource variant.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use signinum_core::BackendRequest;
use signinum_jpeg::{
    ColorTransform as SigninumColorTransform, DecodeOptions as SigninumDecodeOptions,
    Decoder as SigninumJpegDecoder, Downscale as SigninumDownscale,
    PixelFormat as SigninumPixelFormat,
};
use signinum_tilecodec::{
    DeflateCodec, DeflatePool, LzwCodec, LzwPool, TileDecompress, ZstdCodec, ZstdPool,
};

use crate::core::cache::CacheKey;
use crate::core::registry::{
    composite_region_from_source, crop_rgb_interleaved_u8_buffer, read_display_tile_from_source,
    SlideReader,
};
use crate::core::types::*;
#[cfg(feature = "metal")]
use crate::decode::jp2k::decode_batch_jp2k_pixels;
use crate::decode::jp2k::{decode_batch_jp2k, Jp2kColorSpace, Jp2kDecodeJob};
#[cfg(feature = "metal")]
use crate::decode::jpeg::decode_batch_jpeg_pixels;
use crate::decode::jpeg::{decode_batch_jpeg, decode_jpeg_rgb_with_size_override, JpegDecodeJob};
use crate::error::WsiError;
use crate::formats::tiff_family::container::{tags, TiffContainer};
use crate::formats::tiff_family::error::IfdId;
use crate::formats::tiff_family::layout::{
    DatasetLayout, StitchedLevelComponent, TileSource, TileSourceKey,
};
use lru::LruCache;
use rayon::prelude::*;

fn signinum_decode_options(color_transform: SigninumColorTransform) -> SigninumDecodeOptions {
    SigninumDecodeOptions::default().with_color_transform(color_transform)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JpegBitstreamColorHint {
    Rgb,
    RgbComponentIds012,
    YCbCr,
    Unknown,
}

fn tiff_jpeg_color_transform(
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

fn jpeg_bitstream_color_hint(data: &[u8], tables: Option<&[u8]>) -> JpegBitstreamColorHint {
    tables
        .map(jpeg_segment_color_hint)
        .filter(|hint| *hint != JpegBitstreamColorHint::Unknown)
        .unwrap_or_else(|| jpeg_segment_color_hint(data))
}

fn jpeg_segment_color_hint(data: &[u8]) -> JpegBitstreamColorHint {
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

fn is_jpeg_sof_marker(marker: u8) -> bool {
    matches!(
        marker,
        0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF
    )
}

fn jpeg_sof_color_hint(payload: &[u8]) -> JpegBitstreamColorHint {
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
struct JpegFrameInfo {
    width: u32,
    height: u32,
    bits_allocated: u16,
    samples_per_pixel: u16,
    photometric_interpretation: EncodedTilePhotometricInterpretation,
}

fn standalone_jpeg_frame(
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

fn jpeg_has_soi(data: &[u8]) -> bool {
    data.starts_with(&[0xFF, 0xD8])
}

fn jpeg_has_segment_marker(data: &[u8], needle: u8) -> Result<bool, WsiError> {
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

fn rebuild_jpeg_frame_with_tables(
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

fn jpeg_table_segments(data: &[u8], need_dqt: bool, need_dht: bool) -> Result<Vec<u8>, WsiError> {
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

fn parse_baseline_jpeg_frame_info(data: &[u8]) -> Result<JpegFrameInfo, WsiError> {
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

fn parse_sof0_frame_info(payload: &[u8]) -> Result<JpegFrameInfo, WsiError> {
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

struct JpegSegment<'a> {
    marker: u8,
    start: usize,
    end: usize,
    payload: &'a [u8],
}

fn next_jpeg_segment(data: &[u8], mut offset: usize) -> Result<Option<JpegSegment<'_>>, WsiError> {
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

fn signinum_downscale_for_factor(factor: u32) -> Option<SigninumDownscale> {
    match factor {
        1 => Some(SigninumDownscale::None),
        2 => Some(SigninumDownscale::Half),
        4 => Some(SigninumDownscale::Quarter),
        8 => Some(SigninumDownscale::Eighth),
        _ => None,
    }
}

fn cpu_tile_from_rgb_pixels(width: u32, height: u32, pixels: Vec<u8>) -> Result<CpuTile, WsiError> {
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

fn strip_leading_restart_marker(segment: &[u8]) -> &[u8] {
    if segment.len() >= 2 && segment[0] == 0xFF && (0xD0..=0xD7).contains(&segment[1]) {
        &segment[2..]
    } else {
        segment
    }
}

fn strip_trailing_restart_marker(segment: &[u8]) -> &[u8] {
    if segment.len() >= 2 {
        let tail = &segment[segment.len() - 2..];
        if tail[0] == 0xFF && (0xD0..=0xD7).contains(&tail[1]) {
            return &segment[..segment.len() - 2];
        }
    }
    segment
}

fn strip_trailing_eoi_marker(segment: &[u8]) -> &[u8] {
    if segment.len() >= 2 && segment[segment.len() - 2..] == [0xFF, 0xD9] {
        &segment[..segment.len() - 2]
    } else {
        segment
    }
}

fn disable_jpeg_restart_interval(header: &mut [u8]) {
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

fn patch_jpeg_sof0_dimensions(data: &mut [u8], width: u32, height: u32) -> Result<(), WsiError> {
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

fn ndpi_restart_segments_align_to_rows(
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

/// Validate that tile coordinates are non-negative and fit in u32.
fn validate_tile_coords(col: i64, row: i64, level: u32) -> Result<(u32, u32), WsiError> {
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

// ── FullDecodeCache ───────────────────────────────────────────────

/// Default maximum cache size: 128 MB.
const DEFAULT_FULL_DECODE_CACHE_BYTES: u64 = 128 * 1024 * 1024;
const FULL_DECODE_CACHE_BYTES_ENV: &str = "STATUMEN_FULL_DECODE_CACHE_BYTES";
/// Default maximum cache size for decoded NDPI strips: 8 MB.
///
/// Large NDPI display traces are effectively one-way walks through the strip
/// space; retaining a much larger working set inflated RSS without improving
/// the measured tail. Keep the budget tight and allow local override for
/// targeted tuning.
const DEFAULT_NDPI_STRIP_CACHE_BYTES: u64 = 1024 * 1024;
const NDPI_STRIP_CACHE_BYTES_ENV: &str = "STATUMEN_NDPI_STRIP_CACHE_BYTES";
/// Default maximum cache size for synthetic NDPI tail levels: 16 MB.
const DEFAULT_SYNTHETIC_LEVEL_CACHE_BYTES: u64 = 2 * 1024 * 1024;
const SYNTHETIC_LEVEL_CACHE_BYTES_ENV: &str = "STATUMEN_SYNTHETIC_LEVEL_CACHE_BYTES";
const DEFAULT_JP2K_SHARED_TILE_CACHE_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_STITCHED_COMPONENT_TILE_CACHE_BYTES: u64 = 16 * 1024 * 1024;
const NDPI_DISPLAY_WIDE_STRIP_BATCH: usize = 4;
const NDPI_DISPLAY_NARROW_STRIP_BATCH: usize = 8;
#[cfg(feature = "metal")]
const JPEG_DEVICE_DECODE_ENV: &str = "STATUMEN_JPEG_DEVICE_DECODE";
#[cfg(feature = "metal")]
const JP2K_DEVICE_DECODE_ENV: &str = "STATUMEN_JP2K_DEVICE_DECODE";

type NdpiMcuStartsCache = HashMap<(IfdId, u16), Arc<Vec<u64>>>;
type SyntheticDeepestKey = (usize, usize, u32, u32, u32);
type SyntheticDeepestValue = (u32, u32, u32);
const NDPI_DISPLAY_WIDE_STRIP_WIDTH: u32 = 1024;

struct NdpiJpegTilePayload {
    jpeg: Vec<u8>,
    width: u32,
    height: u32,
}

#[cfg(feature = "metal")]
fn jpeg_device_decode_enabled() -> bool {
    std::env::var(JPEG_DEVICE_DECODE_ENV).is_ok_and(|value| {
        value.eq_ignore_ascii_case("1")
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("yes")
            || value.eq_ignore_ascii_case("on")
    })
}

#[cfg(feature = "metal")]
fn jp2k_device_decode_enabled() -> bool {
    std::env::var(JP2K_DEVICE_DECODE_ENV).is_ok_and(|value| {
        value.eq_ignore_ascii_case("1")
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("yes")
            || value.eq_ignore_ascii_case("on")
    })
}

/// Byte-budgeted LRU cache for fully decoded NDPI levels.
///
/// NDPI levels without restart markers require decoding the entire JPEG
/// image to extract a single tile. This cache stores the decoded image
/// so subsequent tile requests from the same level are satisfied from
/// memory instead of re-decoding.
///
/// Same pattern as TileCache from Plan 1: byte budget drives eviction,
/// oversize entries are rejected (not cached but still returned).
pub(crate) struct FullDecodeCache {
    entries: LruCache<IfdId, Arc<CpuTile>>,
    current_bytes: u64,
    max_bytes: u64,
}

impl FullDecodeCache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    /// Get a cached decoded image by IFD ID.
    pub fn get(&mut self, key: &IfdId) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    /// Insert a decoded image. Evicts LRU entries to make room.
    /// Rejects entries larger than max_bytes (returns without storing).
    pub fn put(&mut self, key: IfdId, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;

        if byte_size > self.max_bytes {
            return; // Oversize — don't cache
        }

        // Remove existing entry if present
        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        // Evict LRU entries until there's room
        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for FullDecodeCache {
    fn default() -> Self {
        Self::new(DEFAULT_FULL_DECODE_CACHE_BYTES)
    }
}

struct NdpiStripCache {
    entries: LruCache<NdpiStripKey, Arc<CpuTile>>,
    current_bytes: u64,
    max_bytes: u64,
}

impl NdpiStripCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn get(&mut self, key: &NdpiStripKey) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    fn put(&mut self, key: NdpiStripKey, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;

        if byte_size > self.max_bytes {
            return;
        }

        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for NdpiStripCache {
    fn default() -> Self {
        Self::new(DEFAULT_NDPI_STRIP_CACHE_BYTES)
    }
}

struct SyntheticLevelCache {
    entries: LruCache<SyntheticLevelKey, Arc<CpuTile>>,
    current_bytes: u64,
    max_bytes: u64,
}

impl SyntheticLevelCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn get(&mut self, key: &SyntheticLevelKey) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    fn put(&mut self, key: SyntheticLevelKey, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;

        if byte_size > self.max_bytes {
            return;
        }

        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for SyntheticLevelCache {
    fn default() -> Self {
        Self::new(DEFAULT_SYNTHETIC_LEVEL_CACHE_BYTES)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct StitchedComponentTileKey {
    ifd_id: IfdId,
    tile_idx: usize,
    width: u32,
    height: u32,
}

struct StitchedComponentTileCache {
    entries: LruCache<StitchedComponentTileKey, Arc<CpuTile>>,
    current_bytes: u64,
    max_bytes: u64,
}

impl StitchedComponentTileCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn get(&mut self, key: &StitchedComponentTileKey) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    fn put(&mut self, key: StitchedComponentTileKey, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;
        if byte_size > self.max_bytes {
            return;
        }

        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for StitchedComponentTileCache {
    fn default() -> Self {
        Self::new(DEFAULT_STITCHED_COMPONENT_TILE_CACHE_BYTES)
    }
}

#[derive(Clone, Debug, Default)]
struct FullDecodeFlight {
    waiters: usize,
    result: Option<Result<Arc<CpuTile>, String>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct NdpiStripKey {
    ifd_id: IfdId,
    col: u32,
    native_row: u32,
}

#[derive(Clone, Debug, Default)]
struct NdpiStripFlight {
    waiters: usize,
    result: Option<Result<Arc<CpuTile>, String>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SyntheticLevelKey {
    scene: usize,
    series: usize,
    base_level: u32,
    target_level: u32,
    z: u32,
    c: u32,
    t: u32,
}

#[derive(Clone, Debug, Default)]
struct SyntheticLevelFlight {
    waiters: usize,
    result: Option<Result<Arc<CpuTile>, String>>,
}

// ── Helpers ──────────────────────────────────────────────────────

fn rgba_image_to_sample_buffer(rgba: image::RgbaImage) -> CpuTile {
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

fn downsample_rgb_2x_box(source: &CpuTile) -> Result<CpuTile, WsiError> {
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

fn downsample_rgb_pow2_box(source: &CpuTile, factor: u32) -> Result<CpuTile, WsiError> {
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

fn fit_synthetic_rgb_tile_to_dimensions(
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

fn checked_rgb_u8_len(width: u32, height: u32) -> Result<usize, WsiError> {
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

fn zero_rgb_interleaved_u8_tile(width: u32, height: u32) -> Result<CpuTile, WsiError> {
    Ok(CpuTile {
        width,
        height,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![0u8; checked_rgb_u8_len(width, height)?]),
    })
}

fn paste_rgb_interleaved_u8_tile(
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

fn ensure_interleaved_rgb_u8(tile: CpuTile) -> Result<CpuTile, WsiError> {
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

#[derive(Debug)]
enum CodecBatchJob<'a> {
    Jpeg(JpegDecodeJob<'a>),
    Jp2k(Jp2kDecodeJob<'a>),
}

#[derive(Clone, Copy, Debug)]
struct TiffJpegDecodeOptions {
    force_dimensions: bool,
    color_transform: SigninumColorTransform,
}

fn decode_one_jpeg(job: JpegDecodeJob<'_>) -> Result<CpuTile, WsiError> {
    decode_batch_jpeg(&[job])
        .into_iter()
        .next()
        .expect("1-element JPEG batch")
}

fn decode_one_jp2k(job: Jp2kDecodeJob<'_>) -> Result<CpuTile, WsiError> {
    decode_batch_jp2k(&[job])
        .into_iter()
        .next()
        .expect("1-element JP2K batch")
}

fn decode_mixed_batch(jobs: Vec<CodecBatchJob<'_>>) -> Vec<Result<CpuTile, WsiError>> {
    let mut jpeg_jobs = Vec::new();
    let mut jpeg_slots = Vec::new();
    let mut jp2k_jobs = Vec::new();
    let mut jp2k_slots = Vec::new();

    for (slot, job) in jobs.into_iter().enumerate() {
        match job {
            CodecBatchJob::Jpeg(job) => {
                jpeg_slots.push(slot);
                jpeg_jobs.push(job);
            }
            CodecBatchJob::Jp2k(job) => {
                jp2k_slots.push(slot);
                jp2k_jobs.push(job);
            }
        }
    }

    let total = jpeg_slots.len() + jp2k_slots.len();
    let mut out: Vec<Option<Result<CpuTile, WsiError>>> = (0..total).map(|_| None).collect();
    for (slot, result) in jpeg_slots.into_iter().zip(decode_batch_jpeg(&jpeg_jobs)) {
        out[slot] = Some(result);
    }
    for (slot, result) in jp2k_slots.into_iter().zip(decode_batch_jp2k(&jp2k_jobs)) {
        out[slot] = Some(result);
    }

    out.into_iter()
        .map(|result| result.expect("every mixed batch slot filled"))
        .collect()
}

// ── TiffPixelReader ───────────────────────────────────────────────

/// Implements SlideReader by dispatching tile reads based on TileSource type.
/// Holds an Arc<TiffContainer> for concurrent pread access and the layout
/// produced by a TiffLayoutInterpreter.
pub(crate) struct TiffPixelReader {
    container: Arc<TiffContainer>,
    layout: DatasetLayout,
    full_decode_cache: Mutex<FullDecodeCache>,
    full_decode_flights: Mutex<HashMap<IfdId, FullDecodeFlight>>,
    full_decode_ready: Condvar,
    ndpi_strip_cache: Mutex<NdpiStripCache>,
    ndpi_mcu_starts_cache: Mutex<NdpiMcuStartsCache>,
    ndpi_strip_flights: Mutex<HashMap<NdpiStripKey, NdpiStripFlight>>,
    ndpi_strip_ready: Condvar,
    synthetic_level_cache: Mutex<SyntheticLevelCache>,
    synthetic_region_cache: Mutex<SyntheticLevelCache>,
    synthetic_level_flights: Mutex<HashMap<SyntheticLevelKey, SyntheticLevelFlight>>,
    synthetic_level_ready: Condvar,
    synthetic_prime_once: OnceLock<()>,
    stitched_component_tile_cache: Mutex<StitchedComponentTileCache>,
}

impl TiffPixelReader {
    fn stripped_associated_decode_pool() -> Result<&'static rayon::ThreadPool, WsiError> {
        static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
        match POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .use_current_thread()
                .stack_size(2 * 1024 * 1024)
                .thread_name(|idx| format!("wsi-strips-{idx}"))
                .build()
                .map_err(|err| err.to_string())
        }) {
            Ok(pool) => Ok(pool),
            Err(reason) => Err(WsiError::DisplayConversion(format!(
                "failed to build stripped associated decode pool: {reason}"
            ))),
        }
    }

    fn full_decode_cache_bytes() -> u64 {
        std::env::var(FULL_DECODE_CACHE_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_FULL_DECODE_CACHE_BYTES)
    }

    fn ndpi_strip_cache_bytes() -> u64 {
        std::env::var(NDPI_STRIP_CACHE_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_NDPI_STRIP_CACHE_BYTES)
    }

    fn synthetic_level_cache_bytes() -> u64 {
        std::env::var(SYNTHETIC_LEVEL_CACHE_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_SYNTHETIC_LEVEL_CACHE_BYTES)
    }

    fn get_cached_ndpi_strip(&self, strip_key: NdpiStripKey) -> Option<Arc<CpuTile>> {
        self.ndpi_strip_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&strip_key)
    }

    fn synthetic_level_key_for_region(req: &RegionRequest, base_level: u32) -> SyntheticLevelKey {
        let plane = req.plane.0;
        SyntheticLevelKey {
            scene: req.scene.0,
            series: req.series.0,
            base_level,
            target_level: req.level.0,
            z: plane.z,
            c: plane.c,
            t: plane.t,
        }
    }

    fn synthetic_level_key_for_tile(req: &TileRequest, base_level: u32) -> SyntheticLevelKey {
        SyntheticLevelKey {
            scene: req.scene,
            series: req.series,
            base_level,
            target_level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
        }
    }

    fn get_cached_synthetic_level(&self, key: &SyntheticLevelKey) -> Option<Arc<CpuTile>> {
        self.synthetic_region_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }

    fn put_synthetic_level_cache(&self, key: SyntheticLevelKey, image: Arc<CpuTile>) {
        self.synthetic_region_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, image);
    }

    fn try_decode_synthetic_level_with_signinum(
        &self,
        req: &TileRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<Option<CpuTile>, WsiError> {
        let Some(scale) = signinum_downscale_for_factor(factor) else {
            return Ok(None);
        };
        let target =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let base_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: base_level,
            plane: req.plane,
            col: 0,
            row: 0,
        };
        let TileSource::NdpiFullDecode {
            ifd_id,
            strip_offset,
            strip_byte_count,
            ..
        } = self.tile_source_for(&base_req)?
        else {
            return Ok(None);
        };

        let jpeg = self
            .container
            .pread(*strip_offset, *strip_byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let options = signinum_decode_options(
            self.tiff_jpeg_decode_options_for_data(*ifd_id, false, &jpeg, None)
                .color_transform,
        );
        let decoder = SigninumJpegDecoder::new_with_options(&jpeg, options)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let source_dims = decoder.info().dimensions;
        let scale_denom = scale.denominator();
        let scaled_width = source_dims.0.div_ceil(scale_denom);
        let scaled_height = source_dims.1.div_ceil(scale_denom);
        let (pixels, _outcome) = decoder
            .decode_scaled(SigninumPixelFormat::Rgb8, scale)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let scaled = cpu_tile_from_rgb_pixels(scaled_width, scaled_height, pixels)?;

        if scaled.width == target.dimensions.0 as u32 && scaled.height == target.dimensions.1 as u32
        {
            Ok(Some(scaled))
        } else {
            Ok(None)
        }
    }

    fn prime_deepest_synthetic_levels_best_effort(&self) {
        let mut deepest: HashMap<SyntheticDeepestKey, SyntheticDeepestValue> = HashMap::new();
        for (key, source) in &self.layout.tile_sources {
            let TileSource::SyntheticDownsample { base_level, factor } = source else {
                continue;
            };
            deepest
                .entry((key.scene, key.series, key.z, key.c, key.t))
                .and_modify(|current| {
                    if key.level > current.0 {
                        *current = (key.level, *base_level, *factor);
                    }
                })
                .or_insert((key.level, *base_level, *factor));
        }

        for ((scene, series, z, c, t), (target_level, base_level, factor)) in deepest {
            let req = TileRequest {
                scene,
                series,
                level: target_level,
                plane: PlaneSelection { z, c, t },
                col: 0,
                row: 0,
            };
            let key = Self::synthetic_level_key_for_tile(&req, base_level);
            if self.get_cached_synthetic_level(&key).is_some() {
                continue;
            }
            if let Ok(Some(image)) =
                self.try_decode_synthetic_level_with_signinum(&req, base_level, factor)
            {
                self.put_synthetic_level_cache(key, Arc::new(image));
            }
        }
    }

    fn clamp_ndpi_strip_crop(
        src_x: u32,
        src_y: u32,
        width: u32,
        height: u32,
        strip_width: u32,
        strip_height: u32,
    ) -> Option<(u32, u32)> {
        if src_x >= strip_width || src_y >= strip_height {
            return None;
        }

        let clamped_width = width.min(strip_width - src_x);
        let clamped_height = height.min(strip_height - src_y);
        if clamped_width == 0 || clamped_height == 0 {
            return None;
        }

        Some((clamped_width, clamped_height))
    }

    pub fn new(container: Arc<TiffContainer>, layout: DatasetLayout) -> Self {
        Self {
            container,
            layout,
            full_decode_cache: Mutex::new(FullDecodeCache::new(Self::full_decode_cache_bytes())),
            full_decode_flights: Mutex::new(HashMap::new()),
            full_decode_ready: Condvar::new(),
            ndpi_strip_cache: Mutex::new(NdpiStripCache::new(Self::ndpi_strip_cache_bytes())),
            ndpi_mcu_starts_cache: Mutex::new(HashMap::new()),
            ndpi_strip_flights: Mutex::new(HashMap::new()),
            ndpi_strip_ready: Condvar::new(),
            synthetic_level_cache: Mutex::new(SyntheticLevelCache::new(
                Self::synthetic_level_cache_bytes(),
            )),
            synthetic_region_cache: Mutex::new(SyntheticLevelCache::new(
                Self::synthetic_level_cache_bytes(),
            )),
            synthetic_level_flights: Mutex::new(HashMap::new()),
            synthetic_level_ready: Condvar::new(),
            synthetic_prime_once: OnceLock::new(),
            stitched_component_tile_cache: Mutex::new(StitchedComponentTileCache::default()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get_or_decode_stitched_component_tile(
        &self,
        ifd_id: IfdId,
        tile_idx: usize,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        width: u32,
        height: u32,
        offsets: &[u64],
        byte_counts: &[u64],
    ) -> Result<Arc<CpuTile>, WsiError> {
        let key = StitchedComponentTileKey {
            ifd_id,
            tile_idx,
            width,
            height,
        };
        if let Some(cached) = self
            .stitched_component_tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Ok(cached);
        }

        let tile = Arc::new(self.decode_tiled_ifd_tile_index(
            ifd_id,
            tile_idx,
            jpeg_tables,
            compression,
            width,
            height,
            offsets,
            byte_counts,
            BackendRequest::Auto,
        )?);
        self.stitched_component_tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, tile.clone());
        Ok(tile)
    }

    fn ndpi_full_decode_error(req: &TileRequest, reason: impl Into<String>) -> WsiError {
        WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: reason.into(),
        }
    }

    fn read_stripped_data(
        &self,
        name: &str,
        strip_offsets: &[u64],
        strip_byte_counts: &[u64],
    ) -> Result<Vec<u8>, WsiError> {
        if strip_offsets.len() != strip_byte_counts.len() {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has mismatched strip metadata ({} offsets vs {} byte counts)",
                name,
                strip_offsets.len(),
                strip_byte_counts.len()
            )));
        }

        let total_bytes = strip_byte_counts.iter().try_fold(0usize, |acc, count| {
            acc.checked_add(usize::try_from(*count).ok()?)
        });
        let total_bytes = total_bytes.ok_or_else(|| {
            WsiError::UnsupportedFormat(format!(
                "associated image '{}' strip byte counts exceed addressable memory",
                name
            ))
        })?;

        let mut data = Vec::with_capacity(total_bytes);
        for (&offset, &byte_count) in strip_offsets.iter().zip(strip_byte_counts.iter()) {
            if byte_count == 0 {
                continue;
            }
            let bytes = self
                .container
                .pread(offset, byte_count)
                .map_err(|e| e.into_wsi_error(self.container.path()))?;
            data.extend_from_slice(&bytes);
        }
        Ok(data)
    }

    fn read_stripped_jpeg_image(
        &self,
        name: &str,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        dimensions: (u32, u32),
        strip_offsets: &[u64],
        strip_byte_counts: &[u64],
    ) -> Result<CpuTile, WsiError> {
        if strip_offsets.len() != strip_byte_counts.len() {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has mismatched strip metadata ({} offsets vs {} byte counts)",
                name,
                strip_offsets.len(),
                strip_byte_counts.len()
            )));
        }

        let (width, height) = dimensions;
        let rows_per_strip = self
            .container
            .get_u32(ifd_id, tags::ROWS_PER_STRIP)
            .unwrap_or(height)
            .max(1);
        let total_bytes = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| {
                WsiError::UnsupportedFormat(format!(
                    "associated image '{}' dimensions overflow RGB buffer size",
                    name
                ))
            })?;
        let mut composed = vec![0u8; total_bytes];
        let dst_stride = width as usize * 3;
        let strip_count = height.div_ceil(rows_per_strip) as usize;
        if strip_offsets.len() < strip_count || strip_byte_counts.len() < strip_count {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' expected at least {} strips for {} rows, found offsets={} byte_counts={}",
                name,
                strip_count,
                height,
                strip_offsets.len(),
                strip_byte_counts.len()
            )));
        }
        let strip_chunk_bytes =
            dst_stride
                .checked_mul(rows_per_strip as usize)
                .ok_or_else(|| {
                    WsiError::UnsupportedFormat(format!(
                        "associated image '{}' rows_per_strip overflow for width {}",
                        name, width
                    ))
                })?;

        Self::stripped_associated_decode_pool()?.install(|| {
            composed
                .par_chunks_mut(strip_chunk_bytes)
                .zip(
                    strip_offsets[..strip_count]
                        .par_iter()
                        .zip(strip_byte_counts[..strip_count].par_iter())
                        .enumerate(),
                )
                .try_for_each(|(dst_chunk, (strip_idx, (&offset, &byte_count)))| {
                    if byte_count == 0 {
                        return Ok(());
                    }

                    let strip_y = rows_per_strip.saturating_mul(strip_idx as u32);
                    let strip_height = rows_per_strip.min(height - strip_y);
                    let expected_len = strip_height as usize * dst_stride;
                    if dst_chunk.len() != expected_len {
                        return Err(WsiError::UnsupportedFormat(format!(
                            "associated image '{}' destination chunk for strip {} has {} bytes, expected {}",
                            name,
                            strip_idx,
                            dst_chunk.len(),
                            expected_len
                        )));
                    }

                    let data = self
                        .container
                        .pread(offset, byte_count)
                        .map_err(|e| e.into_wsi_error(self.container.path()))?;
                    let decode_options = self.tiff_jpeg_decode_options_for_data(
                        ifd_id,
                        true,
                        &data,
                        jpeg_tables,
                    );
                    let decoded = decode_one_jpeg(
                        JpegDecodeJob {
                            data: Cow::Borrowed(&data),
                            tables: jpeg_tables.map(Cow::Borrowed),
                            expected_width: width,
                            expected_height: strip_height,
                            color_transform: decode_options.color_transform,
                            force_dimensions: decode_options.force_dimensions,
                            requested_size: None,
                        }
                    )
                    .map_err(|err| WsiError::TileRead {
                        col: strip_idx as i64,
                        row: i64::from(strip_y),
                        level: 0,
                        reason: format!(
                            "associated image '{}' JPEG strip {} decode failed (offset={}, bytes={}, dims={}x{}): {}",
                            name, strip_idx, offset, byte_count, width, strip_height, err
                        ),
                    })?;
                    let CpuTileData::U8(decoded_rows) = decoded.data else {
                        return Err(WsiError::DisplayConversion(
                            "stripped JPEG decode expected U8 RGB data".into(),
                        ));
                    };
                    if decoded.width != width || decoded.height != strip_height {
                        return Err(WsiError::UnsupportedFormat(format!(
                            "associated image '{}' decoded strip {} as {}x{} but expected {}x{}",
                            name, strip_idx, decoded.width, decoded.height, width, strip_height
                        )));
                    }
                    if decoded_rows.len() != expected_len {
                        return Err(WsiError::UnsupportedFormat(format!(
                            "associated image '{}' decoded strip {} produced {} bytes, expected {}",
                            name,
                            strip_idx,
                            decoded_rows.len(),
                            expected_len
                        )));
                    }

                    dst_chunk.copy_from_slice(&decoded_rows);
                    Ok(())
                })
        })?;

        Ok(CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(composed),
        })
    }

    fn tiff_jpeg_decode_options_for_data(
        &self,
        ifd_id: IfdId,
        force_dimensions: bool,
        data: &[u8],
        tables: Option<&[u8]>,
    ) -> TiffJpegDecodeOptions {
        self.tiff_jpeg_decode_options_with_hint(
            ifd_id,
            force_dimensions,
            jpeg_bitstream_color_hint(data, tables),
        )
    }

    fn tiff_jpeg_decode_options_with_hint(
        &self,
        ifd_id: IfdId,
        force_dimensions: bool,
        bitstream_hint: JpegBitstreamColorHint,
    ) -> TiffJpegDecodeOptions {
        if self.layout.dataset.properties.vendor() == Some("philips") {
            return TiffJpegDecodeOptions {
                force_dimensions,
                color_transform: SigninumColorTransform::Auto,
            };
        }

        let photometric = self
            .container
            .get_u32(ifd_id, tags::PHOTOMETRIC)
            .unwrap_or(2);
        let samples_per_pixel = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(3);
        let color_transform =
            tiff_jpeg_color_transform(photometric, samples_per_pixel, bitstream_hint);
        TiffJpegDecodeOptions {
            force_dimensions,
            color_transform,
        }
    }

    fn ndpi_mcu_starts(
        &self,
        ifd_id: IfdId,
        mcu_starts_tag: u16,
    ) -> Result<Arc<Vec<u64>>, WsiError> {
        if let Some(starts) = self
            .ndpi_mcu_starts_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(ifd_id, mcu_starts_tag))
            .cloned()
        {
            return Ok(starts);
        }

        let starts = Arc::new(
            self.container
                .get_u64_array(ifd_id, mcu_starts_tag)
                .map_err(|e| e.into_wsi_error(self.container.path()))?
                .to_vec(),
        );
        self.ndpi_mcu_starts_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((ifd_id, mcu_starts_tag), starts.clone());
        Ok(starts)
    }

    fn decode_ndpi_full_image(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let data = self
            .container
            .pread(strip_offset, strip_byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;

        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let options = signinum_decode_options(
            self.tiff_jpeg_decode_options_for_data(ifd_id, false, &data, None)
                .color_transform,
        );
        let decoder = SigninumJpegDecoder::new_with_options(&data, options)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let (pixels, outcome) = decoder
            .decode(SigninumPixelFormat::Rgb8)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let decoded = cpu_tile_from_rgb_pixels(outcome.decoded.w, outcome.decoded.h, pixels)?;
        let decoded = if decoded.width > level_w as u32 || decoded.height > level_h as u32 {
            crop_rgb_interleaved_u8_buffer(&decoded, 0, 0, level_w as u32, level_h as u32)?
        } else {
            decoded
        };

        Ok(Arc::new(decoded))
    }

    #[allow(clippy::too_many_arguments)]
    fn read_ndpi_display_tile(
        &self,
        req: &TileViewRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NdpiJpeg display tile expects WholeLevel layout".into(),
                });
            }
        };

        let tile_origin_x = req.col.saturating_mul(i64::from(req.tile_width));
        let tile_origin_y = req.row.saturating_mul(i64::from(req.tile_height));
        let level_w = level_w as i64;
        let level_h = level_h as i64;
        if tile_origin_x < 0
            || tile_origin_y < 0
            || tile_origin_x >= level_w
            || tile_origin_y >= level_h
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "display tile origin out of bounds".into(),
            });
        }

        let content_width = req.tile_width.min((level_w - tile_origin_x) as u32);
        let content_height = req.tile_height.min((level_h - tile_origin_y) as u32);
        let tile_end_x = tile_origin_x as u32 + content_width;
        let tile_end_y = tile_origin_y as u32 + content_height;
        let native_col_start = tile_origin_x as u32 / vtw;
        let native_col_end = tile_end_x.saturating_sub(1) / vtw;
        let tile_origin_y_u32 = tile_origin_y as u32;
        let native_row_start = tile_origin_y_u32 / vth;
        let native_row_end = tile_end_y.saturating_sub(1) / vth;

        struct NeededNdpiStrip {
            strip_key: NdpiStripKey,
            strip_req: TileRequest,
            copy_start_x: u32,
            copy_start_y: u32,
            copy_width: u32,
            copy_height: u32,
            dest_x: u32,
            dest_y: u32,
            strip: Option<Arc<CpuTile>>,
        }

        let mut needed_strips = Vec::new();
        for native_row in native_row_start..=native_row_end {
            let strip_origin_y = native_row * vth;

            for native_col in native_col_start..=native_col_end {
                let strip_origin_x = native_col * vtw;
                let strip_width = vtw.min((level_w as u32).saturating_sub(strip_origin_x));
                let strip_height = vth.min((level_h as u32).saturating_sub(strip_origin_y));
                let copy_start_x = (tile_origin_x as u32).saturating_sub(strip_origin_x);
                let copy_start_y = tile_origin_y_u32.saturating_sub(strip_origin_y);
                let copy_end_x = tile_end_x.min(strip_origin_x + strip_width);
                let copy_end_y = tile_end_y.min(strip_origin_y + strip_height);
                let desired_width = copy_end_x.saturating_sub(strip_origin_x + copy_start_x);
                let desired_height = copy_end_y.saturating_sub(strip_origin_y + copy_start_y);
                let Some((copy_width, copy_height)) = Self::clamp_ndpi_strip_crop(
                    copy_start_x,
                    copy_start_y,
                    desired_width,
                    desired_height,
                    strip_width,
                    strip_height,
                ) else {
                    continue;
                };

                let strip_key = NdpiStripKey {
                    ifd_id,
                    col: native_col,
                    native_row,
                };
                let dest_x = strip_origin_x
                    .saturating_add(copy_start_x)
                    .saturating_sub(tile_origin_x as u32);
                let dest_y = strip_origin_y
                    .saturating_add(copy_start_y)
                    .saturating_sub(tile_origin_y as u32);
                needed_strips.push(NeededNdpiStrip {
                    strip_key,
                    strip_req: TileRequest {
                        scene: req.scene,
                        series: req.series,
                        level: req.level,
                        plane: req.plane,
                        col: i64::from(native_col),
                        row: i64::from(native_row),
                    },
                    copy_start_x,
                    copy_start_y,
                    copy_width,
                    copy_height,
                    dest_x,
                    dest_y,
                    strip: self.get_cached_ndpi_strip(strip_key),
                });
            }
        }

        let missing_indices: Vec<usize> = needed_strips
            .iter()
            .enumerate()
            .filter_map(|(idx, needed)| needed.strip.is_none().then_some(idx))
            .collect();
        if !missing_indices.is_empty() {
            let decode_batch = if vtw > NDPI_DISPLAY_WIDE_STRIP_WIDTH {
                NDPI_DISPLAY_WIDE_STRIP_BATCH
            } else {
                NDPI_DISPLAY_NARROW_STRIP_BATCH
            };
            let decoded_missing: Result<Vec<(usize, Arc<CpuTile>)>, WsiError> =
                if missing_indices.len() == 1 {
                    let idx = missing_indices[0];
                    let needed = &needed_strips[idx];
                    Ok(vec![(
                        idx,
                        self.get_or_decode_ndpi_strip(
                            &needed.strip_req,
                            ifd_id,
                            jpeg_header,
                            mcu_starts_tag,
                            tiles_across,
                            tiles_down,
                            strip_offset,
                            strip_byte_count,
                            needed.strip_key,
                            vtw,
                            vth,
                            level_w as u32,
                            level_h as u32,
                        )?,
                    )])
                } else {
                    let mut decoded = Vec::with_capacity(missing_indices.len());
                    for batch in missing_indices.chunks(decode_batch) {
                        let mut decoded_batch: Vec<(usize, Arc<CpuTile>)> = batch
                            .par_iter()
                            .map(|idx| {
                                let needed = &needed_strips[*idx];
                                let strip = self.get_or_decode_ndpi_strip(
                                    &needed.strip_req,
                                    ifd_id,
                                    jpeg_header,
                                    mcu_starts_tag,
                                    tiles_across,
                                    tiles_down,
                                    strip_offset,
                                    strip_byte_count,
                                    needed.strip_key,
                                    vtw,
                                    vth,
                                    level_w as u32,
                                    level_h as u32,
                                )?;
                                Ok::<(usize, Arc<CpuTile>), WsiError>((*idx, strip))
                            })
                            .collect::<Result<_, _>>()?;
                        decoded.append(&mut decoded_batch);
                    }
                    Ok(decoded)
                };
            for (idx, strip) in decoded_missing? {
                needed_strips[idx].strip = Some(strip);
            }
        }

        let mut tile_data = vec![255u8; (content_width * content_height * 3) as usize];
        let dst_stride = content_width as usize * 3;

        for needed in needed_strips {
            let strip = needed.strip.ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("missing decoded NDPI strip {:?}", needed.strip_key),
            })?;

            let copy_start_x = needed.copy_start_x;
            let copy_start_y = needed.copy_start_y;
            let copy_width = needed.copy_width;
            let copy_height = needed.copy_height;
            let dest_x = needed.dest_x;
            let dest_y = needed.dest_y;

            if strip.layout != CpuTileLayout::Interleaved || strip.channels != 3 {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI display tile expected interleaved RGB strips".into(),
                });
            }
            let CpuTileData::U8(strip_rgb) = &strip.data else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI display tile expected U8 RGB strip data".into(),
                });
            };
            let src_stride = strip.width as usize * 3;
            let copy_row_bytes = copy_width as usize * 3;
            for row in 0..copy_height as usize {
                let src_off =
                    (copy_start_y as usize + row) * src_stride + copy_start_x as usize * 3;
                let dst_off = (dest_y as usize + row) * dst_stride + dest_x as usize * 3;
                tile_data[dst_off..dst_off + copy_row_bytes]
                    .copy_from_slice(&strip_rgb[src_off..src_off + copy_row_bytes]);
            }
        }

        Ok(CpuTile {
            width: content_width,
            height: content_height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(tile_data),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ndpi_jpeg_tile_payload(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
        strip_key: NdpiStripKey,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
        level_width: u32,
        level_height: u32,
    ) -> Result<NdpiJpegTilePayload, WsiError> {
        if strip_key.native_row >= tiles_down {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("NDPI strip row {} out of range", strip_key.native_row),
            });
        }
        if strip_key.col >= tiles_across {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("NDPI strip column {} out of range", strip_key.col),
            });
        }

        let strip_origin_y = strip_key.native_row * virtual_tile_height;
        let strip_height = virtual_tile_height.min(level_height.saturating_sub(strip_origin_y));
        let strip_width =
            virtual_tile_width.min(level_width.saturating_sub(strip_key.col * virtual_tile_width));

        let mcu_starts = self.ndpi_mcu_starts(ifd_id, mcu_starts_tag)?;

        let idx =
            (strip_key.native_row as u64 * tiles_across as u64 + strip_key.col as u64) as usize;
        if idx >= mcu_starts.len() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI MCU-starts index {} out of range (len={})",
                    idx,
                    mcu_starts.len(),
                ),
            });
        }

        if idx + 1 < mcu_starts.len() && mcu_starts[idx + 1] <= mcu_starts[idx] {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI MCU-starts table is not strictly increasing at index {}",
                    idx
                ),
            });
        }

        let segment_start = *mcu_starts.get(idx).ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: format!("NDPI MCU-starts index {idx} out of range"),
        })?;
        let next_segment_start = if idx + 1 < mcu_starts.len() {
            Some(mcu_starts[idx + 1])
        } else {
            None
        };
        let segment_end = next_segment_start.unwrap_or(strip_byte_count);
        if segment_start >= strip_byte_count
            || segment_end > strip_byte_count
            || segment_end <= segment_start
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI MCU segment [{segment_start}, {segment_end}) exceeds strip byte count {strip_byte_count}"
                ),
            });
        }
        if jpeg_header.is_empty() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "NDPI JPEG header is empty".into(),
            });
        }

        let segment_len = segment_end.saturating_sub(segment_start);
        let read_offset =
            strip_offset
                .checked_add(segment_start)
                .ok_or_else(|| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI strip offset overflow".into(),
                })?;
        let segment = self
            .container
            .pread(read_offset, segment_len)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let entropy = strip_trailing_eoi_marker(strip_trailing_restart_marker(
            strip_leading_restart_marker(&segment),
        ));
        let mut tile_jpeg = Vec::with_capacity(jpeg_header.len() + entropy.len() + 2);
        tile_jpeg.extend_from_slice(jpeg_header);
        disable_jpeg_restart_interval(&mut tile_jpeg);
        tile_jpeg.extend_from_slice(entropy);
        tile_jpeg.extend_from_slice(&[0xFF, 0xD9]);

        Ok(NdpiJpegTilePayload {
            jpeg: tile_jpeg,
            width: strip_width,
            height: strip_height,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_ndpi_strip(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
        strip_key: NdpiStripKey,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
        level_width: u32,
        level_height: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let payload = self.ndpi_jpeg_tile_payload(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            strip_key,
            virtual_tile_width,
            virtual_tile_height,
            level_width,
            level_height,
        )?;
        let decoded = decode_jpeg_rgb_with_size_override(
            &payload.jpeg,
            None,
            payload.width,
            payload.height,
            None,
            None,
            self.tiff_jpeg_decode_options_for_data(ifd_id, false, &payload.jpeg, None)
                .color_transform,
        )?;
        let decoded = cpu_tile_from_rgb_pixels(decoded.width, decoded.height, decoded.pixels)?;

        Ok(Arc::new(decoded))
    }

    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    fn ndpi_jpeg_decode_job<'a>(
        &'a self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<JpegDecodeJob<'a>, WsiError> {
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NdpiJpeg device decode expects WholeLevel tile layout".into(),
                });
            }
        };
        let (col, row) = validate_tile_coords(req.col, req.row, req.level)?;
        let payload = self.ndpi_jpeg_tile_payload(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            NdpiStripKey {
                ifd_id,
                col,
                native_row: row,
            },
            vtw,
            vth,
            level_w as u32,
            level_h as u32,
        )?;
        let color_transform = self
            .tiff_jpeg_decode_options_for_data(ifd_id, false, &payload.jpeg, None)
            .color_transform;
        Ok(JpegDecodeJob {
            data: Cow::Owned(payload.jpeg),
            tables: None,
            expected_width: payload.width,
            expected_height: payload.height,
            color_transform,
            force_dimensions: true,
            requested_size: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn get_or_decode_ndpi_strip(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
        strip_key: NdpiStripKey,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
        level_width: u32,
        level_height: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        if let Some(strip) = {
            let mut cache = self
                .ndpi_strip_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.get(&strip_key)
        } {
            return Ok(strip);
        }

        let mut flights = self
            .ndpi_strip_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut registered_waiter = false;
        loop {
            match flights.get_mut(&strip_key) {
                Some(flight) => {
                    if !registered_waiter {
                        flight.waiters += 1;
                        registered_waiter = true;
                    }
                    if let Some(result) = flight.result.clone() {
                        flight.waiters -= 1;
                        if flight.waiters == 0 {
                            flights.remove(&strip_key);
                        }
                        return result.map_err(|reason| Self::ndpi_full_decode_error(req, reason));
                    }
                    flights = self
                        .ndpi_strip_ready
                        .wait(flights)
                        .unwrap_or_else(|e| e.into_inner());
                }
                None if registered_waiter => {
                    return Err(Self::ndpi_full_decode_error(
                        req,
                        format!("NDPI strip decode flight for {:?} disappeared", strip_key),
                    ));
                }
                None => {
                    flights.insert(strip_key, NdpiStripFlight::default());
                    break;
                }
            }
        }
        drop(flights);

        let decode_result = self
            .decode_ndpi_strip(
                req,
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                strip_offset,
                strip_byte_count,
                strip_key,
                virtual_tile_width,
                virtual_tile_height,
                level_width,
                level_height,
            )
            .map_err(|err| err.to_string());

        if let Ok(strip) = decode_result.as_ref() {
            let mut cache = self
                .ndpi_strip_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.put(strip_key, strip.clone());
        }

        let mut flights = self
            .ndpi_strip_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(flight) = flights.get_mut(&strip_key) {
            flight.result = Some(decode_result.clone());
            if flight.waiters == 0 {
                flights.remove(&strip_key);
            }
        }
        drop(flights);
        self.ndpi_strip_ready.notify_all();

        decode_result.map_err(|reason| Self::ndpi_full_decode_error(req, reason))
    }

    fn get_or_decode_ndpi_full_image(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<Arc<CpuTile>, WsiError> {
        if let Some(img) = {
            let mut cache = self
                .full_decode_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.get(&ifd_id)
        } {
            return Ok(img);
        }

        let mut flights = self
            .full_decode_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut registered_waiter = false;
        loop {
            match flights.get_mut(&ifd_id) {
                Some(flight) => {
                    if !registered_waiter {
                        flight.waiters += 1;
                        registered_waiter = true;
                    }
                    if let Some(result) = flight.result.clone() {
                        flight.waiters -= 1;
                        let should_remove = flight.waiters == 0;
                        if should_remove {
                            flights.remove(&ifd_id);
                        }
                        return result.map_err(|reason| Self::ndpi_full_decode_error(req, reason));
                    }
                    flights = self
                        .full_decode_ready
                        .wait(flights)
                        .unwrap_or_else(|e| e.into_inner());
                }
                None if registered_waiter => {
                    return Err(Self::ndpi_full_decode_error(
                        req,
                        format!("NDPI full decode flight for {ifd_id} disappeared"),
                    ));
                }
                None => {
                    flights.insert(ifd_id, FullDecodeFlight::default());
                    break;
                }
            }
        }
        drop(flights);

        let decode_result = self
            .decode_ndpi_full_image(req, ifd_id, strip_offset, strip_byte_count)
            .map_err(|err| err.to_string());
        if let Ok(image) = decode_result.as_ref() {
            let mut cache = self
                .full_decode_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.put(ifd_id, image.clone());
        }

        let mut flights = self
            .full_decode_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(flight) = flights.get_mut(&ifd_id) {
            flight.result = Some(decode_result.clone());
            if flight.waiters == 0 {
                flights.remove(&ifd_id);
            }
        }
        drop(flights);
        self.full_decode_ready.notify_all();

        decode_result.map_err(|reason| Self::ndpi_full_decode_error(req, reason))
    }

    fn decode_synthetic_level(
        &self,
        req: &TileRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        if !factor.is_power_of_two() || factor < 2 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("invalid synthetic NDPI factor {factor}"),
            });
        }

        if let Some(image) =
            self.try_decode_synthetic_level_with_signinum(req, base_level, factor)?
        {
            return Ok(Arc::new(image));
        }

        let base =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[base_level as usize];
        let target =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let base_tile_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: base_level,
            plane: req.plane,
            col: 0,
            row: 0,
        };
        let mut current = if matches!(
            self.tile_source_for(&base_tile_req),
            Ok(TileSource::NdpiFullDecode { .. })
        ) {
            self.read_tile_cpu(&base_tile_req)?
        } else {
            composite_region_from_source(
                self,
                None,
                &RegionRequest::legacy_xywh(
                    req.scene,
                    req.series,
                    base_level,
                    req.plane,
                    0,
                    0,
                    u32::try_from(base.dimensions.0).unwrap_or(u32::MAX),
                    u32::try_from(base.dimensions.1).unwrap_or(u32::MAX),
                ),
            )?
        };

        if current.layout != CpuTileLayout::Interleaved
            || current.channels != 3
            || current.color_space != ColorSpace::Rgb
            || current.data.as_u8().is_none()
        {
            current = rgba_image_to_sample_buffer(current.to_rgba()?);
        }

        current = fit_synthetic_rgb_tile_to_dimensions(
            downsample_rgb_pow2_box(&current, factor)?,
            target.dimensions.0 as u32,
            target.dimensions.1 as u32,
        )?;

        Ok(Arc::new(current))
    }

    fn read_full_synthetic_region_fastpath(
        &self,
        cache: Option<&crate::core::cache::TileCache>,
        req: &RegionRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<CpuTile, WsiError> {
        if !factor.is_power_of_two() || !(2..=8).contains(&factor) {
            return composite_region_from_source(self, cache, req);
        }

        let (x, y) = req.origin_px;
        let (w, h) = req.size_px;
        let plane = req.plane.0;
        let level = &self.layout.dataset.scenes[req.scene.0].series[req.series.0].levels
            [req.level.0 as usize];
        if x != 0
            || y != 0
            || u64::from(w) != level.dimensions.0
            || u64::from(h) != level.dimensions.1
        {
            return self.read_synthetic_subregion_fastpath(
                cache,
                req,
                base_level,
                factor,
                level.dimensions.0,
                level.dimensions.1,
            );
        }

        let key = CacheKey {
            dataset_id: self.layout.dataset.id,
            scene: req.scene.0 as u32,
            series: req.series.0 as u32,
            level: req.level.0,
            z: plane.z,
            c: plane.c,
            t: plane.t,
            tile_col: 0,
            tile_row: 0,
        };
        if let Some(cache) = cache {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.as_ref().clone());
            }
        }

        let synthetic_key = Self::synthetic_level_key_for_region(req, base_level);
        if let Some(cached) = self.get_cached_synthetic_level(&synthetic_key) {
            if let Some(cache) = cache {
                cache.put(key, cached.clone());
            }
            return Ok(cached.as_ref().clone());
        }

        let base_req = TileRequest {
            scene: req.scene.0,
            series: req.series.0,
            level: base_level,
            plane,
            col: 0,
            row: 0,
        };
        let TileSource::NdpiFullDecode {
            ifd_id,
            strip_offset,
            strip_byte_count,
            ..
        } = self.tile_source_for(&base_req)?
        else {
            return composite_region_from_source(self, cache, req);
        };

        let tile_req = TileRequest {
            scene: req.scene.0,
            series: req.series.0,
            level: req.level.0,
            plane: req.plane.0,
            col: 0,
            row: 0,
        };
        let scaled = if let Some(image) =
            self.try_decode_synthetic_level_with_signinum(&tile_req, base_level, factor)?
        {
            image
        } else {
            let full = self.get_or_decode_ndpi_full_image(
                &base_req,
                *ifd_id,
                *strip_offset,
                *strip_byte_count,
            )?;
            downsample_rgb_pow2_box(full.as_ref(), factor)?
        };
        let image = Arc::new(fit_synthetic_rgb_tile_to_dimensions(scaled, w, h)?);
        if image.width != w || image.height != h {
            return composite_region_from_source(self, cache, req);
        }
        self.put_synthetic_level_cache(synthetic_key, image.clone());
        if let Some(cache) = cache {
            cache.put(key, image.clone());
        }
        Ok(image.as_ref().clone())
    }

    fn read_synthetic_subregion_fastpath(
        &self,
        cache: Option<&crate::core::cache::TileCache>,
        req: &RegionRequest,
        base_level: u32,
        factor: u32,
        target_width: u64,
        target_height: u64,
    ) -> Result<CpuTile, WsiError> {
        let (x, y) = req.origin_px;
        let (w, h) = req.size_px;
        if w == 0 || h == 0 {
            return zero_rgb_interleaved_u8_tile(w, h);
        }

        let x0 = i128::from(x);
        let y0 = i128::from(y);
        let x1 = x0 + i128::from(w);
        let y1 = y0 + i128::from(h);
        let target_w = i128::from(target_width);
        let target_h = i128::from(target_height);
        let clipped_x0 = x0.clamp(0, target_w);
        let clipped_y0 = y0.clamp(0, target_h);
        let clipped_x1 = x1.clamp(0, target_w);
        let clipped_y1 = y1.clamp(0, target_h);

        if clipped_x1 <= clipped_x0 || clipped_y1 <= clipped_y0 {
            return zero_rgb_interleaved_u8_tile(w, h);
        }

        let valid_w = u32::try_from(clipped_x1 - clipped_x0).map_err(|_| {
            WsiError::DisplayConversion(format!(
                "synthetic NDPI ROI width exceeds region API bounds: {}",
                clipped_x1 - clipped_x0
            ))
        })?;
        let valid_h = u32::try_from(clipped_y1 - clipped_y0).map_err(|_| {
            WsiError::DisplayConversion(format!(
                "synthetic NDPI ROI height exceeds region API bounds: {}",
                clipped_y1 - clipped_y0
            ))
        })?;
        let dst_x = u32::try_from(clipped_x0 - x0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI destination x overflow".into())
        })?;
        let dst_y = u32::try_from(clipped_y0 - y0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI destination y overflow".into())
        })?;

        let base_tile_req = TileRequest {
            scene: req.scene.0,
            series: req.series.0,
            level: base_level,
            plane: req.plane.0,
            col: 0,
            row: 0,
        };
        if matches!(
            self.tile_source_for(&base_tile_req)?,
            TileSource::NdpiFullDecode { .. }
        ) {
            let tile_req = TileRequest {
                scene: req.scene.0,
                series: req.series.0,
                level: req.level.0,
                plane: req.plane.0,
                col: 0,
                row: 0,
            };
            if let Some(scaled) =
                self.try_decode_synthetic_level_with_signinum(&tile_req, base_level, factor)?
            {
                let crop_x0 = u32::try_from(clipped_x0).map_err(|_| {
                    WsiError::DisplayConversion(
                        "synthetic NDPI ROI source x exceeds crop bounds".into(),
                    )
                })?;
                let crop_y0 = u32::try_from(clipped_y0).map_err(|_| {
                    WsiError::DisplayConversion(
                        "synthetic NDPI ROI source y exceeds crop bounds".into(),
                    )
                })?;
                let cropped =
                    crop_rgb_interleaved_u8_buffer(&scaled, crop_x0, crop_y0, valid_w, valid_h)?;
                return paste_rgb_interleaved_u8_tile(&cropped, w, h, dst_x, dst_y);
            }
        }

        let series = self
            .layout
            .dataset
            .scenes
            .get(req.scene.0)
            .and_then(|scene| scene.series.get(req.series.0))
            .ok_or_else(|| WsiError::SeriesOutOfRange {
                index: req.series.0,
                count: self
                    .layout
                    .dataset
                    .scenes
                    .get(req.scene.0)
                    .map_or(0, |scene| scene.series.len()),
            })?;
        let base =
            series
                .levels
                .get(base_level as usize)
                .ok_or_else(|| WsiError::LevelOutOfRange {
                    level: base_level,
                    count: series.levels.len() as u32,
                })?;
        let clipped_x0 = u128::try_from(clipped_x0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source x is negative".into())
        })?;
        let clipped_y0 = u128::try_from(clipped_y0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source y is negative".into())
        })?;
        let clipped_x1 = u128::try_from(clipped_x1).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source right is negative".into())
        })?;
        let clipped_y1 = u128::try_from(clipped_y1).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source bottom is negative".into())
        })?;
        let factor = u128::from(factor);
        let base_x0 = clipped_x0.checked_mul(factor).ok_or_else(|| {
            WsiError::DisplayConversion("synthetic NDPI base ROI x overflow".into())
        })?;
        let base_y0 = clipped_y0.checked_mul(factor).ok_or_else(|| {
            WsiError::DisplayConversion("synthetic NDPI base ROI y overflow".into())
        })?;
        let base_x1 = clipped_x1
            .checked_mul(factor)
            .ok_or_else(|| {
                WsiError::DisplayConversion("synthetic NDPI base ROI right overflow".into())
            })?
            .min(u128::from(base.dimensions.0));
        let base_y1 = clipped_y1
            .checked_mul(factor)
            .ok_or_else(|| {
                WsiError::DisplayConversion("synthetic NDPI base ROI bottom overflow".into())
            })?
            .min(u128::from(base.dimensions.1));
        if base_x1 <= base_x0 || base_y1 <= base_y0 {
            return zero_rgb_interleaved_u8_tile(w, h);
        }

        let base_req = RegionRequest::legacy_xywh(
            req.scene.0,
            req.series.0,
            base_level,
            req.plane.0,
            i64::try_from(base_x0).map_err(|_| {
                WsiError::DisplayConversion("synthetic NDPI base ROI x exceeds i64".into())
            })?,
            i64::try_from(base_y0).map_err(|_| {
                WsiError::DisplayConversion("synthetic NDPI base ROI y exceeds i64".into())
            })?,
            u32::try_from(base_x1 - base_x0).map_err(|_| {
                WsiError::DisplayConversion(
                    "synthetic NDPI base ROI width exceeds region API bounds".into(),
                )
            })?,
            u32::try_from(base_y1 - base_y0).map_err(|_| {
                WsiError::DisplayConversion(
                    "synthetic NDPI base ROI height exceeds region API bounds".into(),
                )
            })?,
        );
        let base_source = TiffPixelReaderNoSyntheticPrime { inner: self };
        let base_region = ensure_interleaved_rgb_u8(composite_region_from_source(
            &base_source,
            cache,
            &base_req,
        )?)?;
        let downsampled = fit_synthetic_rgb_tile_to_dimensions(
            downsample_rgb_pow2_box(&base_region, factor as u32)?,
            valid_w,
            valid_h,
        )?;
        paste_rgb_interleaved_u8_tile(&downsampled, w, h, dst_x, dst_y)
    }

    fn read_synthetic_display_tile(
        &self,
        req: &TileViewRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<CpuTile, WsiError> {
        let series = self
            .layout
            .dataset
            .scenes
            .get(req.scene)
            .and_then(|scene| scene.series.get(req.series))
            .ok_or_else(|| WsiError::SeriesOutOfRange {
                index: req.series,
                count: self
                    .layout
                    .dataset
                    .scenes
                    .get(req.scene)
                    .map_or(0, |scene| scene.series.len()),
            })?;
        let level =
            series
                .levels
                .get(req.level as usize)
                .ok_or_else(|| WsiError::LevelOutOfRange {
                    level: req.level,
                    count: series.levels.len() as u32,
                })?;

        let origin_x = req.col.saturating_mul(i64::from(req.tile_width));
        let origin_y = req.row.saturating_mul(i64::from(req.tile_height));
        let level_w = i64::try_from(level.dimensions.0).unwrap_or(i64::MAX);
        let level_h = i64::try_from(level.dimensions.1).unwrap_or(i64::MAX);
        if origin_x >= level_w || origin_y >= level_h {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "display tile origin out of bounds".into(),
            });
        }

        let clipped = RegionRequest::legacy_xywh(
            req.scene,
            req.series,
            req.level,
            req.plane,
            origin_x,
            origin_y,
            req.tile_width.min((level_w - origin_x) as u32),
            req.tile_height.min((level_h - origin_y) as u32),
        );
        self.read_full_synthetic_region_fastpath(None, &clipped, base_level, factor)
    }

    fn get_or_decode_synthetic_level(
        &self,
        req: &TileRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let key = Self::synthetic_level_key_for_tile(req, base_level);

        if let Some(image) = {
            let mut cache = self
                .synthetic_level_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.get(&key)
        } {
            return Ok(image);
        }

        let mut flights = self
            .synthetic_level_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut registered_waiter = false;
        loop {
            match flights.get_mut(&key) {
                Some(flight) => {
                    if !registered_waiter {
                        flight.waiters += 1;
                        registered_waiter = true;
                    }
                    if let Some(result) = flight.result.clone() {
                        flight.waiters -= 1;
                        if flight.waiters == 0 {
                            flights.remove(&key);
                        }
                        return result.map_err(|reason| Self::ndpi_full_decode_error(req, reason));
                    }
                    flights = self
                        .synthetic_level_ready
                        .wait(flights)
                        .unwrap_or_else(|e| e.into_inner());
                }
                None if registered_waiter => {
                    return Err(Self::ndpi_full_decode_error(
                        req,
                        format!(
                            "synthetic NDPI level decode flight for {:?} disappeared",
                            key
                        ),
                    ));
                }
                None => {
                    flights.insert(key, SyntheticLevelFlight::default());
                    break;
                }
            }
        }
        drop(flights);

        let decode_result = self
            .decode_synthetic_level(req, base_level, factor)
            .map_err(|err| err.to_string());
        if let Ok(image) = decode_result.as_ref() {
            let mut cache = self
                .synthetic_level_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.put(key, image.clone());
        }

        let mut flights = self
            .synthetic_level_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(flight) = flights.get_mut(&key) {
            flight.result = Some(decode_result.clone());
            if flight.waiters == 0 {
                flights.remove(&key);
            }
        }
        drop(flights);
        self.synthetic_level_ready.notify_all();

        decode_result.map_err(|reason| Self::ndpi_full_decode_error(req, reason))
    }

    /// Look up the TileSource for a given tile request.
    fn tile_source_for(&self, req: &TileRequest) -> Result<&TileSource, WsiError> {
        let key = TileSourceKey {
            scene: req.scene,
            series: req.series,
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
        };
        self.layout
            .tile_sources
            .get(&key)
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "no tile source for scene={}, series={}, level={}, z={}, c={}, t={}",
                    req.scene, req.series, req.level, req.plane.z, req.plane.c, req.plane.t,
                ),
            })
    }

    /// Read a tile from an NdpiJpeg source (MCU extraction fast path).
    #[allow(clippy::too_many_arguments)]
    fn read_ndpi_restart_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        _restart_interval: u16,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let col = req.col;
        let row = req.row;

        // Bounds check
        if col < 0 || col >= tiles_across as i64 || row < 0 || row >= tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level: req.level,
                reason: format!(
                    "tile ({},{}) out of range ({}x{})",
                    col, row, tiles_across, tiles_down,
                ),
            });
        }

        // Compute tile dimensions first (needed for empty-tile fallback and decode)
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: "NdpiJpeg expects WholeLevel tile layout".into(),
                });
            }
        };

        let strip_key = NdpiStripKey {
            ifd_id,
            col: col as u32,
            native_row: row as u32,
        };
        let strip = self.get_or_decode_ndpi_strip(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            strip_key,
            vtw,
            vth,
            level_w as u32,
            level_h as u32,
        )?;

        Ok(strip.as_ref().clone())
    }

    fn tiled_ifd_tile_index_and_dimensions(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
    ) -> Result<(usize, u32, u32), WsiError> {
        let col = req.col;
        let row = req.row;

        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];

        let tile_idx = match &level.tile_layout {
            TileLayout::Regular {
                tiles_across,
                tiles_down,
                ..
            } => {
                if col < 0 || col >= *tiles_across as i64 || row < 0 || row >= *tiles_down as i64 {
                    return Err(WsiError::TileRead {
                        col,
                        row,
                        level: req.level,
                        reason: format!(
                            "tile ({},{}) out of range ({}x{})",
                            col, row, tiles_across, tiles_down,
                        ),
                    });
                }
                (row as u64 * *tiles_across + col as u64) as usize
            }
            TileLayout::Irregular { tiles, .. } => {
                let entry = tiles.get(&(col, row)).ok_or_else(|| WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: format!("no irregular tile at ({},{})", col, row),
                })?;
                if let Some(tile_idx) = entry.tiff_tile_index {
                    tile_idx
                } else {
                    let image_width =
                        self.container
                            .get_u64(ifd_id, tags::IMAGE_WIDTH)
                            .map_err(|err| WsiError::TileRead {
                                col,
                                row,
                                level: req.level,
                                reason: format!("failed to read tiled IFD image width: {err}"),
                            })?;
                    let tile_width =
                        self.container
                            .get_u32(ifd_id, tags::TILE_WIDTH)
                            .map_err(|err| WsiError::TileRead {
                                col,
                                row,
                                level: req.level,
                                reason: format!("failed to read tiled IFD tile width: {err}"),
                            })?;
                    let tiles_across = image_width.div_ceil(tile_width as u64);
                    if col < 0 || row < 0 {
                        return Err(WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: "irregular tile row/col out of range for TIFF tile grid".into(),
                        });
                    }
                    (row as u64 * tiles_across + col as u64) as usize
                }
            }
            TileLayout::WholeLevel { .. } => {
                return Err(WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: "TiledIfd does not use WholeLevel layout".into(),
                });
            }
        };

        let (level_w, level_h) = level.dimensions;
        let (tw, th) = match &level.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                ..
            } => {
                let tw =
                    (*tile_width).min((level_w as u32).saturating_sub(col as u32 * *tile_width));
                let th =
                    (*tile_height).min((level_h as u32).saturating_sub(row as u32 * *tile_height));
                (tw, th)
            }
            TileLayout::Irregular { .. } => {
                let image_width =
                    self.container
                        .get_u64(ifd_id, tags::IMAGE_WIDTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF image width: {err}"),
                        })?;
                let image_height =
                    self.container
                        .get_u64(ifd_id, tags::IMAGE_LENGTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF image height: {err}"),
                        })?;
                let tile_width =
                    self.container
                        .get_u32(ifd_id, tags::TILE_WIDTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF tile width: {err}"),
                        })?;
                let tile_height =
                    self.container
                        .get_u32(ifd_id, tags::TILE_LENGTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF tile height: {err}"),
                        })?;
                let tw = tile_width.min(
                    image_width
                        .saturating_sub(col.max(0) as u64 * tile_width as u64)
                        .try_into()
                        .unwrap_or(u32::MAX),
                );
                let th = tile_height.min(
                    image_height
                        .saturating_sub(row.max(0) as u64 * tile_height as u64)
                        .try_into()
                        .unwrap_or(u32::MAX),
                );
                (tw, th)
            }
            _ => {
                return Err(WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: "unexpected tile layout for tiled IFD read".into(),
                });
            }
        };

        Ok((tile_idx, tw, th))
    }

    /// Read a tile from a TiledIfd source (standard TIFF tiled IFDs).
    fn read_tiled_ifd_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let (tile_idx, tw, th) = self.tiled_ifd_tile_index_and_dimensions(req, ifd_id)?;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        self.decode_tiled_ifd_tile_index(
            ifd_id,
            tile_idx,
            jpeg_tables,
            compression,
            tw,
            th,
            offsets,
            byte_counts,
            backend,
        )
        .map_err(|err| match err {
            WsiError::TileRead { .. } => err,
            other => WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: other.to_string(),
            },
        })
    }

    fn tiled_ifd_offsets_and_byte_counts(
        &self,
        ifd_id: IfdId,
    ) -> Result<(&[u64], &[u64]), WsiError> {
        let offsets = self
            .container
            .get_u64_array(ifd_id, tags::TILE_OFFSETS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let byte_counts = self
            .container
            .get_u64_array(ifd_id, tags::TILE_BYTE_COUNTS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        Ok((offsets, byte_counts))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tiled_ifd_tile_index(
        &self,
        ifd_id: IfdId,
        tile_idx: usize,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        width: u32,
        height: u32,
        offsets: &[u64],
        byte_counts: &[u64],
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
            return Err(WsiError::UnsupportedFormat(format!(
                "tile index {} out of range (offsets={}, byte_counts={})",
                tile_idx,
                offsets.len(),
                byte_counts.len(),
            )));
        }

        let offset = offsets[tile_idx];
        let byte_count = byte_counts[tile_idx];
        if byte_count == 0 {
            let pixel_count = (width * height * 3) as usize;
            return Ok(CpuTile {
                width,
                height,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::u8(vec![0u8; pixel_count]),
            });
        }

        let tile_data = self
            .container
            .pread(offset, byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        match compression {
            Compression::Jpeg => {
                self.decode_tiled_ifd_jpeg_tile_data(ifd_id, jpeg_tables, &tile_data, width, height)
            }
            Compression::Jp2kRgb => decode_one_jp2k(Jp2kDecodeJob {
                data: Cow::Borrowed(&tile_data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: true,
                backend,
            }),
            Compression::Jp2kYcbcr => decode_one_jp2k(Jp2kDecodeJob {
                data: Cow::Borrowed(&tile_data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: false,
                backend,
            }),
            Compression::None => {
                // Uncompressed: interpret raw bytes using TIFF metadata
                self.decode_uncompressed_tile(ifd_id, &tile_data, width, height)
            }
            Compression::Lzw | Compression::Deflate | Compression::Zstd => self
                .decode_compressed_tiff_tile_data(ifd_id, compression, &tile_data, width, height),
            other => Err(WsiError::UnsupportedFormat(format!(
                "unsupported TiledIfd compression: {:?}",
                other,
            ))),
        }
    }

    fn decode_tiled_ifd_jpeg_tile_data(
        &self,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        tile_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        let options = self.tiff_jpeg_decode_options_for_data(ifd_id, false, tile_data, jpeg_tables);
        decode_one_jpeg(JpegDecodeJob {
            data: Cow::Borrowed(tile_data),
            tables: jpeg_tables.map(Cow::Borrowed),
            expected_width: width,
            expected_height: height,
            color_transform: options.color_transform,
            force_dimensions: options.force_dimensions,
            requested_size: None,
        })
    }

    fn read_tiled_ifd_raw_jpeg_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
    ) -> Result<RawCompressedTile, WsiError> {
        let (tile_idx, _, _) = self.tiled_ifd_tile_index_and_dimensions(req, ifd_id)?;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile index {} out of range (offsets={}, byte_counts={})",
                    tile_idx,
                    offsets.len(),
                    byte_counts.len()
                ),
            });
        }
        let byte_count = byte_counts[tile_idx];
        if byte_count == 0 {
            return Err(WsiError::Unsupported {
                reason: "JPEG passthrough does not support empty TIFF tiles".into(),
            });
        }
        let tile_data = self
            .container
            .pread(offsets[tile_idx], byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let (data, info) = standalone_jpeg_frame(&tile_data, jpeg_tables)?;
        Ok(RawCompressedTile {
            compression: Compression::Jpeg,
            width: info.width,
            height: info.height,
            bits_allocated: info.bits_allocated,
            samples_per_pixel: info.samples_per_pixel,
            photometric_interpretation: info.photometric_interpretation,
            data,
        })
    }

    fn read_tiled_ifd_raw_jp2k_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        compression: Compression,
    ) -> Result<RawCompressedTile, WsiError> {
        let (tile_idx, width, height) = self.tiled_ifd_tile_index_and_dimensions(req, ifd_id)?;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile index {} out of range (offsets={}, byte_counts={})",
                    tile_idx,
                    offsets.len(),
                    byte_counts.len()
                ),
            });
        }
        let byte_count = byte_counts[tile_idx];
        if byte_count == 0 {
            return Err(WsiError::Unsupported {
                reason: "J2K passthrough does not support empty TIFF tiles".into(),
            });
        }

        let data = self
            .container
            .pread(offsets[tile_idx], byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let samples_per_pixel = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(3);
        if samples_per_pixel == 0 || samples_per_pixel > u32::from(u16::MAX) {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "J2K passthrough requires samples per pixel to fit in u16, got {samples_per_pixel}"
                ),
            });
        }
        let bits_allocated = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8);
        if bits_allocated == 0 || bits_allocated > u32::from(u16::MAX) {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "J2K passthrough requires bits per sample to fit in u16, got {bits_allocated}"
                ),
            });
        }
        let photometric = self.container.get_u32(ifd_id, tags::PHOTOMETRIC).unwrap_or(
            match (compression, samples_per_pixel) {
                (_, 1) => 1,
                (Compression::Jp2kYcbcr, _) => 6,
                _ => 2,
            },
        );
        let photometric_interpretation = match samples_per_pixel {
            1 => EncodedTilePhotometricInterpretation::Monochrome2,
            3 => match compression {
                Compression::Jp2kRgb => EncodedTilePhotometricInterpretation::Rgb,
                Compression::Jp2kYcbcr => EncodedTilePhotometricInterpretation::YbrFull422,
                _ if photometric == 2 => EncodedTilePhotometricInterpretation::Rgb,
                _ if photometric == 6 => EncodedTilePhotometricInterpretation::YbrFull422,
                _ => {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "J2K passthrough does not support photometric interpretation {photometric}"
                        ),
                    });
                }
            },
            other => {
                return Err(WsiError::Unsupported {
                    reason: format!("J2K passthrough supports 1 or 3 samples, got {other}"),
                });
            }
        };

        Ok(RawCompressedTile {
            compression,
            width,
            height,
            bits_allocated: bits_allocated as u16,
            samples_per_pixel: samples_per_pixel as u16,
            photometric_interpretation,
            data,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn read_ndpi_raw_jpeg_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        restart_interval: u16,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<RawCompressedTile, WsiError> {
        let (col, row) = validate_tile_coords(req.col, req.row, req.level)?;
        if col >= tiles_across || row >= tiles_down {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI raw JPEG tile ({},{}) out of range ({}x{})",
                    req.col, req.row, tiles_across, tiles_down
                ),
            });
        }

        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (virtual_tile_width, virtual_tile_height) = match level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } if virtual_tile_width > 0 && virtual_tile_height > 0 => {
                (virtual_tile_width, virtual_tile_height)
            }
            TileLayout::WholeLevel { .. } => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI raw JPEG passthrough requires nonzero WholeLevel virtual tile dimensions"
                        .into(),
                });
            }
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI raw JPEG passthrough expects WholeLevel tile layout".into(),
                });
            }
        };
        if !ndpi_restart_segments_align_to_rows(level_w, virtual_tile_width, restart_interval) {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "NDPI raw JPEG passthrough requires restart segments to align to image rows (level width {level_w}, virtual tile width {virtual_tile_width}, restart interval {restart_interval})"
                ),
            });
        }

        let payload = self.ndpi_jpeg_tile_payload(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            NdpiStripKey {
                ifd_id,
                col,
                native_row: row,
            },
            virtual_tile_width,
            virtual_tile_height,
            u32::try_from(level_w).map_err(|_| WsiError::Unsupported {
                reason: "NDPI raw JPEG passthrough requires level width to fit in u32".into(),
            })?,
            u32::try_from(level_h).map_err(|_| WsiError::Unsupported {
                reason: "NDPI raw JPEG passthrough requires level height to fit in u32".into(),
            })?,
        )?;
        let mut data = payload.jpeg;
        patch_jpeg_sof0_dimensions(&mut data, virtual_tile_width, virtual_tile_height)?;
        let info = parse_baseline_jpeg_frame_info(&data)?;
        if info.width != virtual_tile_width || info.height != virtual_tile_height {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "NDPI raw JPEG passthrough SOF dimensions {}x{} do not match virtual tile {}x{}",
                    info.width, info.height, virtual_tile_width, virtual_tile_height
                ),
            });
        }

        Ok(RawCompressedTile {
            compression: Compression::Jpeg,
            width: info.width,
            height: info.height,
            bits_allocated: info.bits_allocated,
            samples_per_pixel: info.samples_per_pixel,
            photometric_interpretation: info.photometric_interpretation,
            data,
        })
    }

    fn empty_rgb_tile(width: u32, height: u32) -> CpuTile {
        let pixel_count = (width * height * 3) as usize;
        CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(vec![0u8; pixel_count]),
        }
    }

    #[allow(dead_code)]
    fn tiled_ifd_batch_compression(
        &self,
        reqs: &[TileRequest],
    ) -> Result<Option<Compression>, WsiError> {
        let mut batch_compression = None;
        for req in reqs {
            let TileSource::TiledIfd { compression, .. } = self.tile_source_for(req)? else {
                return Ok(None);
            };
            if !matches!(
                compression,
                Compression::Jpeg | Compression::Jp2kRgb | Compression::Jp2kYcbcr
            ) {
                return Ok(None);
            }
            match batch_compression {
                Some(existing) if existing != *compression => return Ok(None),
                Some(_) => {}
                None => batch_compression = Some(*compression),
            }
        }
        Ok(batch_compression)
    }

    #[cfg(feature = "metal")]
    fn ndpi_jpeg_batchable(&self, reqs: &[TileRequest]) -> Result<bool, WsiError> {
        if reqs.is_empty() {
            return Ok(false);
        }
        for req in reqs {
            if !matches!(self.tile_source_for(req)?, TileSource::NdpiJpeg { .. }) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn decode_tiled_ifd_mixed_batch(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
    ) -> Result<Option<Vec<CpuTile>>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } = source
            else {
                return Ok(None);
            };
            if !matches!(
                compression,
                Compression::Jpeg | Compression::Jp2kRgb | Compression::Jp2kYcbcr
            ) {
                return Ok(None);
            }

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return Ok(None);
            }
            let data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;

            let job = match compression {
                Compression::Jpeg => {
                    let options = self.tiff_jpeg_decode_options_for_data(
                        *ifd_id,
                        false,
                        &data,
                        jpeg_tables.as_deref(),
                    );
                    CodecBatchJob::Jpeg(JpegDecodeJob {
                        data: Cow::Owned(data),
                        tables: jpeg_tables.as_deref().map(Cow::Borrowed),
                        expected_width: width,
                        expected_height: height,
                        color_transform: options.color_transform,
                        force_dimensions: options.force_dimensions,
                        requested_size: None,
                    })
                }
                Compression::Jp2kRgb | Compression::Jp2kYcbcr => {
                    CodecBatchJob::Jp2k(Jp2kDecodeJob {
                        data: Cow::Owned(data),
                        expected_width: width,
                        expected_height: height,
                        rgb_color_space: matches!(compression, Compression::Jp2kRgb),
                        backend,
                    })
                }
                _ => unreachable!("filtered above"),
            };
            jobs.push(job);
        }

        decode_mixed_batch(jobs)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    #[allow(dead_code)]
    fn decode_tiled_ifd_jpeg_batch(
        &self,
        reqs: &[TileRequest],
        _backend: BackendRequest,
    ) -> Result<Vec<CpuTile>, WsiError> {
        reqs.par_iter()
            .map(|req| {
                let source = self.tile_source_for(req)?;
                let TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables,
                    compression: Compression::Jpeg,
                } = source
                else {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "JPEG tiled batch received a non-JPEG tile source".into(),
                    });
                };

                let (tile_idx, width, height) =
                    self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
                let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
                if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: format!(
                            "tile index {} out of range (offsets={}, byte_counts={})",
                            tile_idx,
                            offsets.len(),
                            byte_counts.len()
                        ),
                    });
                }

                let byte_count = byte_counts[tile_idx];
                if byte_count == 0 {
                    return Ok(Self::empty_rgb_tile(width, height));
                }

                let tile_data = self
                    .container
                    .pread(offsets[tile_idx], byte_count)
                    .map_err(|err| err.into_wsi_error(self.container.path()))?;
                let options = self.tiff_jpeg_decode_options_for_data(
                    *ifd_id,
                    false,
                    &tile_data,
                    jpeg_tables.as_deref(),
                );
                decode_one_jpeg(JpegDecodeJob {
                    data: Cow::Borrowed(&tile_data),
                    tables: jpeg_tables.as_deref().map(Cow::Borrowed),
                    expected_width: width,
                    expected_height: height,
                    color_transform: options.color_transform,
                    force_dimensions: options.force_dimensions,
                    requested_size: None,
                })
                .map_err(|err| match err {
                    WsiError::TileRead { .. } => err,
                    other => WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: other.to_string(),
                    },
                })
            })
            .collect()
    }

    #[cfg(feature = "metal")]
    fn decode_tiled_ifd_jpeg_pixels(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
        require_device: bool,
        metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let jobs = self.collect_tiled_ifd_jpeg_jobs(reqs)?;
        decode_batch_jpeg_pixels(&jobs, backend, require_device, metal_sessions)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect()
    }

    #[cfg(feature = "metal")]
    fn decode_ndpi_jpeg_pixels(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
        require_device: bool,
        metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                strip_offset,
                strip_byte_count,
                ..
            } = source
            else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI JPEG device batch received a non-NDPI tile source".into(),
                });
            };
            jobs.push(self.ndpi_jpeg_decode_job(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *strip_offset,
                *strip_byte_count,
            )?);
        }
        decode_batch_jpeg_pixels(&jobs, backend, require_device, metal_sessions)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect()
    }

    #[cfg(feature = "metal")]
    fn collect_tiled_ifd_jpeg_jobs<'a>(
        &'a self,
        reqs: &[TileRequest],
    ) -> Result<Vec<JpegDecodeJob<'a>>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression: Compression::Jpeg,
            } = source
            else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "JPEG tiled device batch received a non-JPEG tile source".into(),
                });
            };

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return Err(WsiError::Unsupported {
                    reason: "device backend not available for empty jpeg tile".into(),
                });
            }

            let tile_data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;
            let options = self.tiff_jpeg_decode_options_for_data(
                *ifd_id,
                false,
                &tile_data,
                jpeg_tables.as_deref(),
            );
            jobs.push(JpegDecodeJob {
                data: Cow::Owned(tile_data),
                tables: jpeg_tables.as_deref().map(Cow::Borrowed),
                expected_width: width,
                expected_height: height,
                color_transform: options.color_transform,
                force_dimensions: options.force_dimensions,
                requested_size: None,
            });
        }
        Ok(jobs)
    }

    #[cfg(feature = "metal")]
    fn decode_tiled_ifd_jp2k_pixels(
        &self,
        reqs: &[TileRequest],
        compression: Compression,
        backend: BackendRequest,
        require_device: bool,
        metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                compression: actual_compression,
                ..
            } = source
            else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "JP2K tiled device batch received a non-tiled tile source".into(),
                });
            };
            if *actual_compression != compression {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "JP2K tiled device batch received mixed compression".into(),
                });
            }

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return Err(WsiError::Unsupported {
                    reason: "device backend not available for empty jp2k tile".into(),
                });
            }
            let data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;
            jobs.push(Jp2kDecodeJob {
                data: Cow::Owned(data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: matches!(compression, Compression::Jp2kRgb),
                backend,
            });
        }

        decode_batch_jp2k_pixels(&jobs, require_device, metal_sessions)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect()
    }

    fn read_stitched_level_tile(
        &self,
        req: &TileRequest,
        components: &[StitchedLevelComponent],
        direct_tiles: &HashMap<(i64, i64), usize>,
    ) -> Result<CpuTile, WsiError> {
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } = &level.tile_layout
        else {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "stitched level expects regular public tile layout".into(),
            });
        };

        if req.col < 0
            || req.row < 0
            || req.col >= *tiles_across as i64
            || req.row >= *tiles_down as i64
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile ({},{}) out of range ({}x{})",
                    req.col, req.row, tiles_across, tiles_down
                ),
            });
        }

        let public_x = req.col * i64::from(*tile_width);
        let public_y = req.row * i64::from(*tile_height);
        let out_width = (*tile_width)
            .min((level.dimensions.0 as u32).saturating_sub(req.col as u32 * *tile_width));
        let out_height = (*tile_height)
            .min((level.dimensions.1 as u32).saturating_sub(req.row as u32 * *tile_height));

        if let Some(tile) = self.try_read_stitched_level_direct_tile(
            req.col,
            req.row,
            public_x,
            public_y,
            out_width,
            out_height,
            components,
            direct_tiles,
        )? {
            return Ok(tile);
        }

        let mut out = vec![0u8; out_width as usize * out_height as usize * 3];
        let out_stride = out_width as usize * 3;

        for component in components {
            let comp_left = component.origin_x;
            let comp_top = component.origin_y;
            let comp_right = comp_left + component.width as i64;
            let comp_bottom = comp_top + component.height as i64;
            let tile_right = public_x + i64::from(out_width);
            let tile_bottom = public_y + i64::from(out_height);

            let inter_left = public_x.max(comp_left);
            let inter_top = public_y.max(comp_top);
            let inter_right = tile_right.min(comp_right);
            let inter_bottom = tile_bottom.min(comp_bottom);
            if inter_left >= inter_right || inter_top >= inter_bottom {
                continue;
            }

            let local_x = (inter_left - comp_left) as u32;
            let local_y = (inter_top - comp_top) as u32;
            let inter_width = (inter_right - inter_left) as u32;
            let inter_height = (inter_bottom - inter_top) as u32;
            let region = self.read_tiled_ifd_component_region(
                component,
                local_x,
                local_y,
                inter_width,
                inter_height,
            )?;

            let region_data = region.data.as_u8().ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "stitched component produced non-U8 data".into(),
            })?;
            let dst_x = (inter_left - public_x) as usize;
            let dst_y = (inter_top - public_y) as usize;
            let src_stride = inter_width as usize * 3;
            for row in 0..inter_height as usize {
                let src_off = row * src_stride;
                let dst_off = (dst_y + row) * out_stride + dst_x * 3;
                out[dst_off..dst_off + src_stride]
                    .copy_from_slice(&region_data[src_off..src_off + src_stride]);
            }
        }

        Ok(CpuTile {
            width: out_width,
            height: out_height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(out),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn try_read_stitched_level_direct_tile(
        &self,
        public_col: i64,
        public_row: i64,
        public_x: i64,
        public_y: i64,
        out_width: u32,
        out_height: u32,
        components: &[StitchedLevelComponent],
        direct_tiles: &HashMap<(i64, i64), usize>,
    ) -> Result<Option<CpuTile>, WsiError> {
        let tile_right = public_x + i64::from(out_width);
        let tile_bottom = public_y + i64::from(out_height);
        let component = if let Some(&index) = direct_tiles.get(&(public_col, public_row)) {
            components.get(index)
        } else {
            let mut covering_component: Option<&StitchedLevelComponent> = None;

            for component in components {
                let comp_left = component.origin_x;
                let comp_top = component.origin_y;
                let comp_right = comp_left + component.width as i64;
                let comp_bottom = comp_top + component.height as i64;

                let inter_left = public_x.max(comp_left);
                let inter_top = public_y.max(comp_top);
                let inter_right = tile_right.min(comp_right);
                let inter_bottom = tile_bottom.min(comp_bottom);
                if inter_left >= inter_right || inter_top >= inter_bottom {
                    continue;
                }
                if inter_left != public_x
                    || inter_top != public_y
                    || inter_right != tile_right
                    || inter_bottom != tile_bottom
                {
                    return Ok(None);
                }
                if covering_component.is_some() {
                    return Ok(None);
                }
                covering_component = Some(component);
            }
            covering_component
        };

        let Some(component) = component else {
            return Ok(None);
        };

        let local_x = (public_x - component.origin_x) as u32;
        let local_y = (public_y - component.origin_y) as u32;
        let tile_col = local_x / component.tile_width;
        let tile_row = local_y / component.tile_height;
        if u64::from(tile_col) >= component.tiles_across
            || u64::from(tile_row) >= component.tiles_down
        {
            return Ok(None);
        }

        let tile_left = tile_col.saturating_mul(component.tile_width);
        let tile_top = tile_row.saturating_mul(component.tile_height);
        let decoded_width = component.tile_width.min(
            (component.width as u32).saturating_sub(tile_col.saturating_mul(component.tile_width)),
        );
        let decoded_height = component.tile_height.min(
            (component.height as u32)
                .saturating_sub(tile_row.saturating_mul(component.tile_height)),
        );
        if local_x < tile_left
            || local_y < tile_top
            || local_x.saturating_add(out_width) > tile_left.saturating_add(decoded_width)
            || local_y.saturating_add(out_height) > tile_top.saturating_add(decoded_height)
        {
            return Ok(None);
        }

        let tile_idx =
            (u64::from(tile_row) * component.tiles_across + u64::from(tile_col)) as usize;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(component.ifd_id)?;
        let tile = self.get_or_decode_stitched_component_tile(
            component.ifd_id,
            tile_idx,
            component.jpeg_tables.as_deref(),
            component.compression,
            decoded_width,
            decoded_height,
            offsets,
            byte_counts,
        )?;
        let crop_x = local_x - tile_left;
        let crop_y = local_y - tile_top;
        if crop_x == 0 && crop_y == 0 && decoded_width == out_width && decoded_height == out_height
        {
            Ok(Some(tile.as_ref().clone()))
        } else {
            Ok(Some(crop_rgb_interleaved_u8_buffer(
                tile.as_ref(),
                crop_x,
                crop_y,
                out_width,
                out_height,
            )?))
        }
    }

    fn read_tiled_ifd_component_region(
        &self,
        component: &StitchedLevelComponent,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        #[derive(Clone, Copy)]
        struct ComponentTileJob {
            decoded_width: u32,
            decoded_height: u32,
            tile_origin_x: u32,
            tile_origin_y: u32,
            inter_left: u32,
            inter_top: u32,
            inter_right: u32,
            inter_bottom: u32,
        }

        let offsets = self
            .container
            .get_u64_array(component.ifd_id, tags::TILE_OFFSETS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let byte_counts = self
            .container
            .get_u64_array(component.ifd_id, tags::TILE_BYTE_COUNTS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;

        let mut out = vec![0u8; width as usize * height as usize * 3];
        let out_stride = width as usize * 3;
        let tile_width = component.tile_width;
        let tile_height = component.tile_height;
        let start_col = x / tile_width;
        let end_col = (x + width - 1) / tile_width;
        let start_row = y / tile_height;
        let end_row = (y + height - 1) / tile_height;
        let mut tile_jobs = Vec::with_capacity(
            ((end_col - start_col + 1) as usize).saturating_mul((end_row - start_row + 1) as usize),
        );

        for row in start_row..=end_row {
            for col in start_col..=end_col {
                if u64::from(col) >= component.tiles_across
                    || u64::from(row) >= component.tiles_down
                {
                    continue;
                }

                let decoded_width = tile_width
                    .min((component.width as u32).saturating_sub(col.saturating_mul(tile_width)));
                let decoded_height = tile_height
                    .min((component.height as u32).saturating_sub(row.saturating_mul(tile_height)));
                let tile_origin_x = col * tile_width;
                let tile_origin_y = row * tile_height;
                let inter_left = x.max(tile_origin_x);
                let inter_top = y.max(tile_origin_y);
                let inter_right = (x + width).min(tile_origin_x + decoded_width);
                let inter_bottom = (y + height).min(tile_origin_y + decoded_height);
                if inter_left >= inter_right || inter_top >= inter_bottom {
                    continue;
                }

                tile_jobs.push(ComponentTileJob {
                    decoded_width,
                    decoded_height,
                    tile_origin_x,
                    tile_origin_y,
                    inter_left,
                    inter_top,
                    inter_right,
                    inter_bottom,
                });
            }
        }

        let decoded_tiles: Vec<_> = if tile_jobs.len() <= 1 {
            tile_jobs
                .into_iter()
                .map(|job| {
                    let tile_idx = (u64::from(job.tile_origin_y / tile_height)
                        * component.tiles_across
                        + u64::from(job.tile_origin_x / tile_width))
                        as usize;
                    let tile = self.get_or_decode_stitched_component_tile(
                        component.ifd_id,
                        tile_idx,
                        component.jpeg_tables.as_deref(),
                        component.compression,
                        job.decoded_width,
                        job.decoded_height,
                        offsets,
                        byte_counts,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        } else {
            tile_jobs
                .into_par_iter()
                .map(|job| {
                    let tile_idx = (u64::from(job.tile_origin_y / tile_height)
                        * component.tiles_across
                        + u64::from(job.tile_origin_x / tile_width))
                        as usize;
                    let tile = self.get_or_decode_stitched_component_tile(
                        component.ifd_id,
                        tile_idx,
                        component.jpeg_tables.as_deref(),
                        component.compression,
                        job.decoded_width,
                        job.decoded_height,
                        offsets,
                        byte_counts,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        };

        for (job, tile) in decoded_tiles {
            let tile_data = tile.data.as_u8().ok_or_else(|| {
                WsiError::DisplayConversion(
                    "stitched Leica level requires interleaved U8 RGB tiles".into(),
                )
            })?;

            let src_x = (job.inter_left - job.tile_origin_x) as usize;
            let src_y = (job.inter_top - job.tile_origin_y) as usize;
            let dst_x = (job.inter_left - x) as usize;
            let dst_y = (job.inter_top - y) as usize;
            let copy_width = (job.inter_right - job.inter_left) as usize;
            let copy_height = (job.inter_bottom - job.inter_top) as usize;
            let src_stride = job.decoded_width as usize * 3;

            for copy_row in 0..copy_height {
                let src_off = (src_y + copy_row) * src_stride + src_x * 3;
                let dst_off = (dst_y + copy_row) * out_stride + dst_x * 3;
                let len = copy_width * 3;
                out[dst_off..dst_off + len].copy_from_slice(&tile_data[src_off..src_off + len]);
            }
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

    fn read_tiled_associated_image(
        &self,
        name: &str,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        dimensions: (u32, u32),
    ) -> Result<CpuTile, WsiError> {
        #[derive(Clone, Copy)]
        struct AssociatedTileJob {
            tile_w: u32,
            tile_h: u32,
            dest_x: u32,
            dest_y: u32,
            tile_idx: usize,
        }

        let tile_width = self
            .container
            .get_u32(ifd_id, tags::TILE_WIDTH)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let tile_height = self
            .container
            .get_u32(ifd_id, tags::TILE_LENGTH)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        if tile_width == 0 || tile_height == 0 {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has invalid tile size {}x{}",
                name, tile_width, tile_height,
            )));
        }

        let tiles_across = dimensions.0.div_ceil(tile_width);
        let tiles_down = dimensions.1.div_ceil(tile_height);
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        let required_tiles = tiles_across as usize * tiles_down as usize;
        if offsets.len() < required_tiles || byte_counts.len() < required_tiles {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' expected {} tiles, found offsets={} byte_counts={}",
                name,
                required_tiles,
                offsets.len(),
                byte_counts.len(),
            )));
        }

        let mut composed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * 3];
        let composed_stride = dimensions.0 as usize * 3;
        let mut tile_jobs = Vec::with_capacity(tiles_across as usize * tiles_down as usize);
        for row in 0..tiles_down {
            for col in 0..tiles_across {
                tile_jobs.push(AssociatedTileJob {
                    tile_w: tile_width.min(dimensions.0.saturating_sub(col * tile_width)),
                    tile_h: tile_height.min(dimensions.1.saturating_sub(row * tile_height)),
                    dest_x: col * tile_width,
                    dest_y: row * tile_height,
                    tile_idx: (row * tiles_across + col) as usize,
                });
            }
        }

        let decoded_tiles: Vec<_> = if tile_jobs.len() <= 1 {
            tile_jobs
                .into_iter()
                .map(|job| {
                    let tile = self.decode_tiled_ifd_tile_index(
                        ifd_id,
                        job.tile_idx,
                        jpeg_tables,
                        compression,
                        job.tile_w,
                        job.tile_h,
                        offsets,
                        byte_counts,
                        BackendRequest::Auto,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        } else {
            tile_jobs
                .into_par_iter()
                .map(|job| {
                    let tile = self.decode_tiled_ifd_tile_index(
                        ifd_id,
                        job.tile_idx,
                        jpeg_tables,
                        compression,
                        job.tile_w,
                        job.tile_h,
                        offsets,
                        byte_counts,
                        BackendRequest::Auto,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        };

        for (job, tile) in decoded_tiles {
            match (&tile.data, tile.layout, tile.channels, &tile.color_space) {
                (CpuTileData::U8(tile_rgb), CpuTileLayout::Interleaved, 3, ColorSpace::Rgb) => {
                    let tile_src_stride = job.tile_w as usize * 3;
                    for y in 0..job.tile_h as usize {
                        let src_row = y * tile_src_stride;
                        let dst_row =
                            (job.dest_y as usize + y) * composed_stride + job.dest_x as usize * 3;
                        composed[dst_row..dst_row + tile_src_stride]
                            .copy_from_slice(&tile_rgb[src_row..src_row + tile_src_stride]);
                    }
                }
                _ => {
                    let tile_rgba = tile.to_rgba()?;
                    let tile_rgba_raw = tile_rgba.as_raw();
                    let tile_src_stride = job.tile_w as usize * 4;
                    for y in 0..job.tile_h as usize {
                        let src_row = y * tile_src_stride;
                        let dst_row =
                            (job.dest_y as usize + y) * composed_stride + job.dest_x as usize * 3;
                        let src_pixels = &tile_rgba_raw[src_row..src_row + tile_src_stride];
                        let dst_pixels = &mut composed[dst_row..dst_row + job.tile_w as usize * 3];
                        for (src_px, dst_px) in src_pixels
                            .chunks_exact(4)
                            .zip(dst_pixels.chunks_exact_mut(3))
                        {
                            dst_px.copy_from_slice(&src_px[..3]);
                        }
                    }
                }
            }
        }

        Ok(CpuTile {
            width: dimensions.0,
            height: dimensions.1,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(composed),
        })
    }

    /// Decode an uncompressed TIFF tile using IFD metadata.
    fn decode_uncompressed_tile(
        &self,
        ifd_id: IfdId,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        use crate::formats::tiff_family::container::Endian;

        // Resolve TIFF metadata from container
        let spp = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(1);
        let bps_val = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8);
        // Tag 339 SAMPLE_FORMAT: 1=unsigned int (default), 2=signed int, 3=float
        let sample_format = self.container.get_u32(ifd_id, 339).unwrap_or(1);
        // Tag 262 PHOTOMETRIC: 0=MinIsWhite, 1=MinIsBlack, 2=RGB, 3=Palette, 6=YCbCr.
        // When the tag is absent, prefer grayscale for single-sample images and
        // RGB otherwise. Real NDPI associated thumbnails omit PHOTOMETRIC while
        // still storing 8-bit grayscale strips.
        let photometric = self
            .container
            .get_u32(ifd_id, tags::PHOTOMETRIC)
            .unwrap_or(if spp == 1 { 1 } else { 2 });
        // Tag 284 PLANAR_CONFIGURATION: 1=chunky (default), 2=planar
        let planar = self.container.get_u32(ifd_id, 284).unwrap_or(1);

        let endian = self.container.endian();

        if planar == 2 {
            return Err(WsiError::UnsupportedFormat(
                "planar TIFF tiles not supported".into(),
            ));
        }

        let effective_photometric = if spp == 1 && photometric == 2 {
            1
        } else {
            photometric
        };

        // Determine sample type and color space. Some NDPI associated images
        // report RGB photometric with a single 8-bit sample plane; treat those
        // contradictory tags as grayscale because the byte layout is 1 channel.
        let (sample_type, color_space) = match (bps_val, sample_format, spp, effective_photometric)
        {
            (8, 1, 3, 2) => (SampleType::Uint8, ColorSpace::Rgb), // RGB u8
            (8, 1, 1, 0) => (SampleType::Uint8, ColorSpace::Grayscale), // MinIsWhite (inverted below)
            (8, 1, 1, 1) => (SampleType::Uint8, ColorSpace::Grayscale), // MinIsBlack
            (8, 1, 3, 6) => (SampleType::Uint8, ColorSpace::YCbCr),     // YCbCr u8
            (16, 1, 1, 0) | (16, 1, 1, 1) => (SampleType::Uint16, ColorSpace::Grayscale),
            (16, 1, 3, 2) => (SampleType::Uint16, ColorSpace::Rgb), // RGB u16
            (32, 3, 1, _) => (SampleType::Float32, ColorSpace::Grayscale), // Float32 grayscale
            _ => {
                return Err(WsiError::UnsupportedFormat(format!(
                    "unsupported uncompressed format: bps={}, format={}, spp={}, photometric={}",
                    bps_val, sample_format, spp, photometric,
                )));
            }
        };

        let expected_bytes =
            width as usize * height as usize * spp as usize * sample_type.byte_size();
        if data.len() < expected_bytes {
            return Err(WsiError::TileRead {
                col: 0,
                row: 0,
                level: 0,
                reason: format!(
                    "uncompressed tile data too short: {} < {}",
                    data.len(),
                    expected_bytes,
                ),
            });
        }

        let sample_data = match sample_type {
            SampleType::Uint8 => {
                let mut bytes = data[..expected_bytes].to_vec();
                // MinIsWhite: invert grayscale values
                if effective_photometric == 0 {
                    for b in &mut bytes {
                        *b = 255 - *b;
                    }
                }
                CpuTileData::u8(bytes)
            }
            SampleType::Uint16 => {
                let mut samples: Vec<u16> = data[..expected_bytes]
                    .chunks_exact(2)
                    .map(|c| match endian {
                        Endian::Little => u16::from_le_bytes([c[0], c[1]]),
                        Endian::Big => u16::from_be_bytes([c[0], c[1]]),
                    })
                    .collect();
                // MinIsWhite: invert
                if effective_photometric == 0 {
                    for s in &mut samples {
                        *s = u16::MAX - *s;
                    }
                }
                CpuTileData::u16(samples)
            }
            SampleType::Float32 => {
                let samples: Vec<f32> = data[..expected_bytes]
                    .chunks_exact(4)
                    .map(|c| match endian {
                        Endian::Little => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                        Endian::Big => f32::from_be_bytes([c[0], c[1], c[2], c[3]]),
                    })
                    .collect();
                CpuTileData::f32(samples)
            }
        };

        // After MinIsWhite inversion, report as standard Grayscale
        // (the inversion already happened in the sample data)
        if effective_photometric == 0 && color_space == ColorSpace::Grayscale {
            // Already inverted above — color_space stays Grayscale
        }

        Ok(CpuTile {
            width,
            height,
            channels: spp as u16,
            color_space,
            layout: CpuTileLayout::Interleaved,
            data: sample_data,
        })
    }

    fn expected_uncompressed_tile_bytes(
        &self,
        ifd_id: IfdId,
        width: u32,
        height: u32,
    ) -> Result<usize, WsiError> {
        let spp = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(1);
        let bps = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8);
        if bps == 0 || !bps.is_multiple_of(8) {
            return Err(WsiError::UnsupportedFormat(format!(
                "unsupported compressed TIFF bits per sample: {bps}"
            )));
        }
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|value| value.checked_mul(spp as usize))
            .and_then(|value| value.checked_mul((bps / 8) as usize))
            .ok_or_else(|| WsiError::UnsupportedFormat("compressed TIFF tile size overflow".into()))
    }

    fn decompress_tiff_payload(
        &self,
        ifd_id: IfdId,
        compression: Compression,
        input: &[u8],
        expected_bytes: usize,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, WsiError> {
        let mut out = vec![0_u8; expected_bytes];
        let written = match compression {
            Compression::Lzw => {
                let mut pool = LzwPool::new();
                LzwCodec::decompress_into(&mut pool, input, &mut out)
            }
            Compression::Deflate => {
                let mut pool = DeflatePool::new();
                DeflateCodec::decompress_into(&mut pool, input, &mut out)
            }
            Compression::Zstd => {
                let mut pool = ZstdPool::new();
                ZstdCodec::decompress_into(&mut pool, input, &mut out)
            }
            other => {
                return Err(WsiError::UnsupportedFormat(format!(
                    "compression {:?} is not a tilecodec payload",
                    other
                )));
            }
        }
        .map_err(|err| WsiError::Codec {
            codec: match compression {
                Compression::Lzw => "tiff-lzw",
                Compression::Deflate => "tiff-deflate",
                Compression::Zstd => "tiff-zstd",
                _ => "tiff-tilecodec",
            },
            source: Box::new(err),
        })?;
        out.truncate(written);
        self.apply_tiff_predictor(ifd_id, width, height, &mut out)?;
        Ok(out)
    }

    fn apply_tiff_predictor(
        &self,
        ifd_id: IfdId,
        width: u32,
        height: u32,
        data: &mut [u8],
    ) -> Result<(), WsiError> {
        use crate::formats::tiff_family::container::Endian;

        let predictor = self.container.get_u32(ifd_id, tags::PREDICTOR).unwrap_or(1);
        if predictor == 1 {
            return Ok(());
        }
        if predictor != 2 {
            return Err(WsiError::UnsupportedFormat(format!(
                "unsupported TIFF predictor: {predictor}"
            )));
        }

        let spp = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(1) as usize;
        let bps = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8) as usize;
        let width = width as usize;
        let height = height as usize;
        if width == 0 || height == 0 || spp == 0 {
            return Ok(());
        }

        match bps {
            8 => {
                let row_stride = width.checked_mul(spp).ok_or_else(|| {
                    WsiError::UnsupportedFormat("TIFF predictor row stride overflow".into())
                })?;
                if data.len() < row_stride.saturating_mul(height) {
                    return Err(WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "TIFF predictor payload is shorter than expected".into(),
                    });
                }
                for row in data.chunks_exact_mut(row_stride).take(height) {
                    for idx in spp..row_stride {
                        let prior = row[idx - spp];
                        row[idx] = row[idx].wrapping_add(prior);
                    }
                }
                Ok(())
            }
            16 => {
                let row_samples = width.checked_mul(spp).ok_or_else(|| {
                    WsiError::UnsupportedFormat("TIFF predictor row sample overflow".into())
                })?;
                let row_stride = row_samples.checked_mul(2).ok_or_else(|| {
                    WsiError::UnsupportedFormat("TIFF predictor row stride overflow".into())
                })?;
                if data.len() < row_stride.saturating_mul(height) {
                    return Err(WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "TIFF predictor payload is shorter than expected".into(),
                    });
                }
                for row in data.chunks_exact_mut(row_stride).take(height) {
                    for sample_idx in spp..row_samples {
                        let byte_idx = sample_idx * 2;
                        let prior_idx = (sample_idx - spp) * 2;
                        let current = match self.container.endian() {
                            Endian::Little => {
                                u16::from_le_bytes([row[byte_idx], row[byte_idx + 1]])
                            }
                            Endian::Big => u16::from_be_bytes([row[byte_idx], row[byte_idx + 1]]),
                        };
                        let prior = match self.container.endian() {
                            Endian::Little => {
                                u16::from_le_bytes([row[prior_idx], row[prior_idx + 1]])
                            }
                            Endian::Big => u16::from_be_bytes([row[prior_idx], row[prior_idx + 1]]),
                        };
                        let value = current.wrapping_add(prior);
                        let bytes = match self.container.endian() {
                            Endian::Little => value.to_le_bytes(),
                            Endian::Big => value.to_be_bytes(),
                        };
                        row[byte_idx..byte_idx + 2].copy_from_slice(&bytes);
                    }
                }
                Ok(())
            }
            _ => Err(WsiError::UnsupportedFormat(format!(
                "unsupported TIFF predictor bits per sample: {bps}"
            ))),
        }
    }

    fn decode_compressed_tiff_tile_data(
        &self,
        ifd_id: IfdId,
        compression: Compression,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        let expected_bytes = self.expected_uncompressed_tile_bytes(ifd_id, width, height)?;
        let decoded = self.decompress_tiff_payload(
            ifd_id,
            compression,
            input,
            expected_bytes,
            width,
            height,
        )?;
        self.decode_uncompressed_tile(ifd_id, &decoded, width, height)
    }

    /// Read a tile from an NdpiFullDecode source (full JPEG decode fallback).
    ///
    /// Decodes the entire JPEG strip and extracts the requested tile region.
    /// Caches the decoded image in FullDecodeCache for subsequent tile requests
    /// from the same level. Oversize images (larger than cache max) are decoded
    /// per-request without caching.
    fn read_ndpi_full_decode_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        _jpeg_header: &[u8],
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let full_image =
            self.get_or_decode_ndpi_full_image(req, ifd_id, strip_offset, strip_byte_count)?;

        // Extract the requested tile from the full image
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NdpiFullDecode expects WholeLevel tile layout".into(),
                });
            }
        };

        let (col_u32, row_u32) = validate_tile_coords(req.col, req.row, req.level)?;
        let src_x = col_u32 * vtw;
        let src_y = row_u32 * vth;
        let tile_w = vtw.min((level_w as u32).saturating_sub(src_x));
        let tile_h = vth.min((level_h as u32).saturating_sub(src_y));

        if tile_w == 0 || tile_h == 0 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "tile has zero dimensions".into(),
            });
        }

        // Extract sub-region from the full interleaved RGB image
        let full_w = full_image.width as usize;
        let channels = full_image.channels as usize;
        let src_data = full_image.data.as_u8().ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "expected U8 data in full decode cache".into(),
        })?;

        let mut tile_data = Vec::with_capacity(tile_w as usize * tile_h as usize * channels);
        for y in 0..tile_h {
            let row_start = ((src_y + y) as usize * full_w + src_x as usize) * channels;
            let row_end = row_start + tile_w as usize * channels;
            if row_end > src_data.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "decoded image smaller than expected".into(),
                });
            }
            tile_data.extend_from_slice(&src_data[row_start..row_end]);
        }

        Ok(CpuTile {
            width: tile_w,
            height: tile_h,
            channels: full_image.channels,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(tile_data),
        })
    }

    fn read_ndpi_full_display_tile(
        &self,
        req: &TileViewRequest,
        ifd_id: IfdId,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let tile_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: req.level,
            plane: req.plane,
            col: req.col,
            row: req.row,
        };
        let full_image =
            self.get_or_decode_ndpi_full_image(&tile_req, ifd_id, strip_offset, strip_byte_count)?;
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = (level.dimensions.0 as i64, level.dimensions.1 as i64);
        let tile_origin_x = req.col.saturating_mul(i64::from(req.tile_width));
        let tile_origin_y = req.row.saturating_mul(i64::from(req.tile_height));
        if tile_origin_x < 0
            || tile_origin_y < 0
            || tile_origin_x >= level_w
            || tile_origin_y >= level_h
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "display tile origin out of bounds".into(),
            });
        }

        let tile_w = req.tile_width.min((level_w - tile_origin_x) as u32);
        let tile_h = req.tile_height.min((level_h - tile_origin_y) as u32);
        let full_w = full_image.width as usize;
        let channels = full_image.channels as usize;
        let src_data = full_image.data.as_u8().ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "expected U8 data in full decode cache".into(),
        })?;

        let mut tile_data = Vec::with_capacity(tile_w as usize * tile_h as usize * channels);
        for y in 0..tile_h {
            let row_start =
                ((tile_origin_y as u32 + y) as usize * full_w + tile_origin_x as usize) * channels;
            let row_end = row_start + tile_w as usize * channels;
            if row_end > src_data.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "decoded image smaller than expected".into(),
                });
            }
            tile_data.extend_from_slice(&src_data[row_start..row_end]);
        }

        Ok(CpuTile {
            width: tile_w,
            height: tile_h,
            channels: full_image.channels,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(tile_data),
        })
    }

    fn read_tiles_cpu_with_backend(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
    ) -> Result<Vec<CpuTile>, WsiError> {
        if reqs.is_empty() {
            return Ok(Vec::new());
        }

        let first_source = self.tile_source_for(&reqs[0])?;
        if matches!(
            first_source,
            TileSource::TiledIfd {
                compression: Compression::Jpeg | Compression::Jp2kRgb | Compression::Jp2kYcbcr,
                ..
            }
        ) {
            if self.tiled_ifd_batch_compression(reqs)? == Some(Compression::Jpeg) {
                return self.decode_tiled_ifd_jpeg_batch(reqs, backend);
            }
            if let Some(tiles) = self.decode_tiled_ifd_mixed_batch(reqs, backend)? {
                return Ok(tiles);
            }
        }

        let mut decode_reqs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                compression,
                ..
            } = source
            else {
                return reqs
                    .iter()
                    .map(|req| self.read_tile_cpu_with_backend_request(req, backend))
                    .collect();
            };
            let colorspace = match compression {
                Compression::Jp2kRgb => Jp2kColorSpace::Rgb,
                Compression::Jp2kYcbcr => Jp2kColorSpace::YCbCr,
                _ => {
                    return reqs
                        .iter()
                        .map(|req| self.read_tile_cpu_with_backend_request(req, backend))
                        .collect();
                }
            };

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return reqs
                    .iter()
                    .map(|req| self.read_tile_cpu_with_backend_request(req, backend))
                    .collect();
            }
            let data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;
            decode_reqs.push(Jp2kDecodeJob {
                data: Cow::Owned(data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: matches!(colorspace, Jp2kColorSpace::Rgb),
                backend,
            });
        }

        decode_batch_jp2k(&decode_reqs)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                let first = &reqs[0];
                WsiError::TileRead {
                    col: first.col,
                    row: first.row,
                    level: first.level,
                    reason: err.to_string(),
                }
            })
    }

    fn read_tile_cpu_with_backend_request(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        match self.tile_source_for(req)? {
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => self.read_tiled_ifd_tile(
                req,
                *ifd_id,
                jpeg_tables.as_deref(),
                *compression,
                backend,
            ),
            _ => self.read_tile_cpu(req),
        }
    }
}

struct TiffPixelReaderNoSyntheticPrime<'a> {
    inner: &'a TiffPixelReader,
}

impl SlideReader for TiffPixelReaderNoSyntheticPrime<'_> {
    fn dataset(&self) -> &Dataset {
        &self.inner.layout.dataset
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        <TiffPixelReader as SlideReader>::read_tiles(self.inner, reqs, output)
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.inner.read_tile_cpu(req)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.inner.read_associated(name)
    }
}

impl SlideReader for TiffPixelReader {
    fn dataset(&self) -> &Dataset {
        let _ = self
            .synthetic_prime_once
            .get_or_init(|| self.prime_deepest_synthetic_levels_best_effort());
        &self.layout.dataset
    }

    fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        match self.tile_source_for(req) {
            Ok(TileSource::NdpiJpeg { .. } | TileSource::NdpiFullDecode { .. }) => {
                TileCodecKind::Jpeg
            }
            Ok(TileSource::TiledIfd { compression, .. }) => {
                TileCodecKind::from_compression(*compression)
            }
            Ok(TileSource::SyntheticDownsample { base_level, .. }) => {
                let mut base_req = req.clone();
                base_req.level = *base_level;
                self.tile_codec_kind(&base_req)
            }
            Ok(_) | Err(_) => TileCodecKind::Other,
        }
    }

    fn level_source_kind(
        &self,
        scene: usize,
        series: usize,
        level: u32,
    ) -> Result<LevelSourceKind, WsiError> {
        let scene_ref = self
            .layout
            .dataset
            .scenes
            .get(scene)
            .ok_or(WsiError::SceneOutOfRange {
                index: scene,
                count: self.layout.dataset.scenes.len(),
            })?;
        let series_ref = scene_ref
            .series
            .get(series)
            .ok_or(WsiError::SeriesOutOfRange {
                index: series,
                count: scene_ref.series.len(),
            })?;
        if level as usize >= series_ref.levels.len() {
            return Err(WsiError::LevelOutOfRange {
                level,
                count: series_ref.levels.len() as u32,
            });
        }

        let synthetic = self.layout.tile_sources.iter().any(|(key, source)| {
            key.scene == scene
                && key.series == series
                && key.level == level
                && matches!(source, TileSource::SyntheticDownsample { .. })
        });
        if synthetic {
            Ok(LevelSourceKind::SyntheticDownsample)
        } else {
            Ok(LevelSourceKind::Physical)
        }
    }

    fn read_raw_compressed_tile(&self, req: &TileRequest) -> Result<RawCompressedTile, WsiError> {
        match self.tile_source_for(req)? {
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression: Compression::Jpeg,
            } => self.read_tiled_ifd_raw_jpeg_tile(req, *ifd_id, jpeg_tables.as_deref()),
            TileSource::TiledIfd {
                ifd_id,
                compression: compression @ (Compression::Jp2kRgb | Compression::Jp2kYcbcr),
                ..
            } => self.read_tiled_ifd_raw_jp2k_tile(req, *ifd_id, *compression),
            TileSource::TiledIfd { compression, .. } => Err(WsiError::Unsupported {
                reason: format!(
                    "compressed passthrough requires TIFF JPEG or J2K compression, got {:?}",
                    compression
                ),
            }),
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                restart_interval,
                strip_offset,
                strip_byte_count,
                ..
            } => self.read_ndpi_raw_jpeg_tile(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *restart_interval,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::NdpiFullDecode { .. } => Err(WsiError::Unsupported {
                reason: "NDPI JPEG passthrough is not available for whole-level full-decode JPEG sources".into(),
            }),
            TileSource::SyntheticDownsample { .. } => Err(WsiError::Unsupported {
                reason: "JPEG passthrough is not available for synthetic downsample levels".into(),
            }),
            TileSource::StitchedLevel { .. } => Err(WsiError::Unsupported {
                reason: "JPEG passthrough is not available for stitched levels".into(),
            }),
            TileSource::Stripped { .. } | TileSource::ExternalJpeg { .. } => Err(WsiError::Unsupported {
                reason: "JPEG passthrough is only available for tiled image levels".into(),
            }),
        }
    }

    fn use_display_tile_cache(&self, req: &TileViewRequest) -> bool {
        let tile_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: req.level,
            plane: req.plane,
            col: req.col,
            row: req.row,
        };
        match self.tile_source_for(&tile_req) {
            Ok(
                TileSource::NdpiJpeg { .. }
                | TileSource::NdpiFullDecode { .. }
                | TileSource::SyntheticDownsample { .. },
            ) => false,
            Ok(TileSource::TiledIfd { .. }) => true,
            Ok(_) => true,
            Err(_) => true,
        }
    }

    fn read_region_fastpath(
        &self,
        ctx: &mut crate::core::registry::SlideReadContext<'_>,
        req: &RegionRequest,
    ) -> Option<Result<CpuTile, WsiError>> {
        let cache = ctx.tile_cache();
        let series = self
            .layout
            .dataset
            .scenes
            .get(req.scene.0)
            .and_then(|scene| scene.series.get(req.series.0))?;
        let level = series.levels.get(req.level.0 as usize)?;
        if !matches!(level.tile_layout, TileLayout::WholeLevel { .. }) {
            return None;
        }
        let plane = req.plane.0;

        let source = self.layout.tile_sources.get(&TileSourceKey {
            scene: req.scene.0,
            series: req.series.0,
            level: req.level.0,
            z: plane.z,
            c: plane.c,
            t: plane.t,
        })?;
        match source {
            TileSource::SyntheticDownsample { base_level, factor } => {
                Some(self.read_full_synthetic_region_fastpath(cache, req, *base_level, *factor))
            }
            _ => None,
        }
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        let source = self.tile_source_for(req)?;
        match source {
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                restart_interval,
                strip_offset,
                strip_byte_count,
            } => self.read_ndpi_restart_tile(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *restart_interval,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::NdpiFullDecode {
                ifd_id,
                jpeg_header,
                strip_offset,
                strip_byte_count,
            } => self.read_ndpi_full_decode_tile(
                req,
                *ifd_id,
                jpeg_header,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => self.read_tiled_ifd_tile(
                req,
                *ifd_id,
                jpeg_tables.as_deref(),
                *compression,
                BackendRequest::Auto,
            ),
            TileSource::StitchedLevel {
                components,
                direct_tiles,
            } => self.read_stitched_level_tile(req, components, direct_tiles),
            TileSource::SyntheticDownsample { base_level, factor } => {
                if req.col != 0 || req.row != 0 {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "synthetic NDPI whole-level tiles only support tile (0,0)".into(),
                    });
                }
                Ok(self
                    .get_or_decode_synthetic_level(req, *base_level, *factor)?
                    .as_ref()
                    .clone())
            }
            TileSource::Stripped { .. } => Err(WsiError::UnsupportedFormat(
                "Stripped pixel access via read_tile not supported; use read_associated()".into(),
            )),
            TileSource::ExternalJpeg { .. } => Err(WsiError::UnsupportedFormat(
                "External JPEG associated images cannot be read via read_tile()".into(),
            )),
        }
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let backend = output.backend().to_signinum();
        let require_device = output.requires_device();
        #[cfg(feature = "metal")]
        let prefer_device = output.prefers_device();
        #[cfg(feature = "metal")]
        let compressed_device_decode_enabled = output.compressed_device_decode_enabled();
        #[cfg(feature = "metal")]
        let metal_sessions = output.metal_sessions();

        #[cfg(feature = "metal")]
        if prefer_device && !reqs.is_empty() {
            if self.ndpi_jpeg_batchable(reqs)? {
                if compressed_device_decode_enabled || jpeg_device_decode_enabled() {
                    match self.decode_ndpi_jpeg_pixels(
                        reqs,
                        backend,
                        require_device,
                        metal_sessions,
                    ) {
                        Ok(tiles) => return Ok(tiles),
                        Err(err) if require_device => return Err(err),
                        Err(err) => {
                            tracing::debug!(
                                error = %err,
                                fallback_to_cpu = true,
                                fallback_reason = "ndpi_jpeg_device_decode_failed",
                                "NDPI JPEG device tile path failed; retrying through CPU output"
                            );
                        }
                    }
                } else if require_device {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "NDPI JPEG device decode is disabled; set {JPEG_DEVICE_DECODE_ENV}=1 or request compressed device decode to opt in"
                        ),
                    });
                }
            }

            let device_result = match self.tiled_ifd_batch_compression(reqs)? {
                Some(Compression::Jpeg)
                    if compressed_device_decode_enabled || jpeg_device_decode_enabled() =>
                {
                    Some(self.decode_tiled_ifd_jpeg_pixels(
                        reqs,
                        backend,
                        require_device,
                        metal_sessions,
                    ))
                }
                Some(Compression::Jpeg) if require_device => {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "JPEG device decode is disabled; set {JPEG_DEVICE_DECODE_ENV}=1 or request compressed device decode to opt in"
                        ),
                    });
                }
                Some(Compression::Jpeg) => None,
                Some(compression @ (Compression::Jp2kRgb | Compression::Jp2kYcbcr))
                    if compressed_device_decode_enabled || jp2k_device_decode_enabled() =>
                {
                    Some(self.decode_tiled_ifd_jp2k_pixels(
                        reqs,
                        compression,
                        backend,
                        require_device,
                        metal_sessions,
                    ))
                }
                Some(Compression::Jp2kRgb | Compression::Jp2kYcbcr) if require_device => {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "JP2K device decode is disabled; set {JP2K_DEVICE_DECODE_ENV}=1 or request compressed device decode to opt in"
                        ),
                    });
                }
                Some(Compression::Jp2kRgb | Compression::Jp2kYcbcr) => None,
                _ if require_device => {
                    return Err(WsiError::Unsupported {
                        reason: "device backend not available for tiff_family".into(),
                    });
                }
                _ => None,
            };
            if let Some(result) = device_result {
                match result {
                    Ok(tiles) => return Ok(tiles),
                    Err(err) if require_device => return Err(err),
                    Err(err) => {
                        tracing::debug!(
                            error = %err,
                            fallback_to_cpu = true,
                            fallback_reason = "signinum_auto_chose_cpu",
                            "device tile path failed; retrying through CPU output"
                        );
                    }
                }
            }
        }

        #[cfg(not(feature = "metal"))]
        if require_device {
            return Err(WsiError::Unsupported {
                reason: "device backend not available for tiff_family".into(),
            });
        }

        self.read_tiles_cpu_with_backend(reqs, backend)
            .map(|tiles| tiles.into_iter().map(TilePixels::Cpu).collect())
    }

    fn read_tiles_cpu(&self, reqs: &[TileRequest]) -> Result<Vec<CpuTile>, WsiError> {
        self.read_tiles_cpu_with_backend(reqs, BackendRequest::Auto)
    }

    fn read_display_tile(&self, req: &TileViewRequest) -> Result<CpuTile, WsiError> {
        let source = self.tile_source_for(&TileRequest {
            scene: req.scene,
            series: req.series,
            level: req.level,
            plane: req.plane,
            col: req.col,
            row: req.row,
        })?;
        match source {
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                strip_offset,
                strip_byte_count,
                ..
            } => self.read_ndpi_display_tile(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::NdpiFullDecode {
                ifd_id,
                strip_offset,
                strip_byte_count,
                ..
            } => self.read_ndpi_full_display_tile(req, *ifd_id, *strip_offset, *strip_byte_count),
            TileSource::SyntheticDownsample { base_level, factor } => {
                self.read_synthetic_display_tile(req, *base_level, *factor)
            }
            _ => read_display_tile_from_source(self, None, req, TileOutputPreference::cpu()),
        }
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let source = self
            .layout
            .associated_sources
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;

        match source {
            TileSource::Stripped {
                ifd_id,
                jpeg_tables,
                compression,
                strip_offsets,
                strip_byte_counts,
            } => {
                let info = self
                    .layout
                    .dataset
                    .associated_images
                    .get(name)
                    .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;

                match compression {
                    Compression::Jpeg => self.read_stripped_jpeg_image(
                        name,
                        *ifd_id,
                        jpeg_tables.as_deref(),
                        info.dimensions,
                        strip_offsets,
                        strip_byte_counts,
                    ),
                    Compression::None => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        self.decode_uncompressed_tile(
                            *ifd_id,
                            &data,
                            info.dimensions.0,
                            info.dimensions.1,
                        )
                    }
                    Compression::Lzw | Compression::Deflate | Compression::Zstd => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        let expected_bytes = self.expected_uncompressed_tile_bytes(
                            *ifd_id,
                            info.dimensions.0,
                            info.dimensions.1,
                        )?;
                        let decoded = self.decompress_tiff_payload(
                            *ifd_id,
                            *compression,
                            &data,
                            expected_bytes,
                            info.dimensions.0,
                            info.dimensions.1,
                        )?;
                        self.decode_uncompressed_tile(
                            *ifd_id,
                            &decoded,
                            info.dimensions.0,
                            info.dimensions.1,
                        )
                    }
                    Compression::Jp2kRgb => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        decode_one_jp2k(Jp2kDecodeJob {
                            data: Cow::Borrowed(&data),
                            expected_width: info.dimensions.0,
                            expected_height: info.dimensions.1,
                            rgb_color_space: true,
                            backend: BackendRequest::Auto,
                        })
                    }
                    Compression::Jp2kYcbcr => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        decode_one_jp2k(Jp2kDecodeJob {
                            data: Cow::Borrowed(&data),
                            expected_width: info.dimensions.0,
                            expected_height: info.dimensions.1,
                            rgb_color_space: false,
                            backend: BackendRequest::Auto,
                        })
                    }
                    other => Err(WsiError::UnsupportedFormat(format!(
                        "associated image '{}' has unsupported compression {:?}",
                        name, other,
                    ))),
                }
            }
            TileSource::NdpiFullDecode {
                ifd_id,
                strip_offset,
                strip_byte_count,
                ..
            } => {
                let data = self
                    .container
                    .pread(*strip_offset, *strip_byte_count)
                    .map_err(|e| e.into_wsi_error(self.container.path()))?;

                let info = self
                    .layout
                    .dataset
                    .associated_images
                    .get(name)
                    .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;

                let options = signinum_decode_options(
                    self.tiff_jpeg_decode_options_for_data(*ifd_id, false, &data, None)
                        .color_transform,
                );
                let decoder = SigninumJpegDecoder::new_with_options(&data, options)
                    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
                let (pixels, outcome) = decoder
                    .decode(SigninumPixelFormat::Rgb8)
                    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
                let decoded =
                    cpu_tile_from_rgb_pixels(outcome.decoded.w, outcome.decoded.h, pixels)?;
                if decoded.width > info.dimensions.0 || decoded.height > info.dimensions.1 {
                    crop_rgb_interleaved_u8_buffer(
                        &decoded,
                        0,
                        0,
                        info.dimensions.0,
                        info.dimensions.1,
                    )
                } else {
                    Ok(decoded)
                }
            }
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => {
                let info = self
                    .layout
                    .dataset
                    .associated_images
                    .get(name)
                    .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
                self.read_tiled_associated_image(
                    name,
                    *ifd_id,
                    jpeg_tables.as_deref(),
                    *compression,
                    info.dimensions,
                )
            }
            TileSource::ExternalJpeg { path } => {
                let data = std::fs::read(path).map_err(|err| WsiError::InvalidSlide {
                    path: path.clone(),
                    message: format!(
                        "failed to read external JPEG associated image '{}': {err}",
                        path.display()
                    ),
                })?;
                decode_one_jpeg(JpegDecodeJob {
                    data: Cow::Borrowed(&data),
                    tables: None,
                    expected_width: 0,
                    expected_height: 0,
                    color_transform: SigninumColorTransform::Auto,
                    force_dimensions: false,
                    requested_size: None,
                })
            }
            _ => Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has unsupported source type",
                name,
            ))),
        }
    }

    fn recommended_shared_cache_bytes(&self) -> Option<u64> {
        self.layout
            .tile_sources
            .values()
            .any(|source| {
                matches!(
                    source,
                    TileSource::TiledIfd {
                        compression: Compression::Jp2kRgb | Compression::Jp2kYcbcr,
                        ..
                    }
                )
            })
            .then_some(DEFAULT_JP2K_SHARED_TILE_CACHE_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::tiff_family::container::TiffContainer;
    use crate::formats::tiff_family::layout::DatasetLayout;
    use crate::properties::Properties;
    use flate2::write::ZlibEncoder;
    use flate2::Compression as DeflateCompression;
    use image::{DynamicImage, ImageFormat};
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_sample_buffer(size: usize) -> CpuTile {
        CpuTile {
            width: 64,
            height: 64,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(vec![0u8; size]),
        }
    }

    fn jpeg_sof(ids: [u8; 3], sampling: [(u8, u8); 3]) -> Vec<u8> {
        let mut jpeg = vec![
            0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01, 0x00, 0x01, 0x03,
        ];
        for idx in 0..3 {
            jpeg.push(ids[idx]);
            jpeg.push((sampling[idx].0 << 4) | sampling[idx].1);
            jpeg.push(0);
        }
        jpeg
    }

    #[test]
    fn jpeg_rgb_component_ids_zero_one_two_follow_tiff_photometric() {
        let jpeg = jpeg_sof([0, 1, 2], [(1, 1), (1, 1), (1, 1)]);

        assert_eq!(
            jpeg_bitstream_color_hint(&jpeg, None),
            JpegBitstreamColorHint::RgbComponentIds012
        );
        assert_eq!(
            tiff_jpeg_color_transform(2, 3, jpeg_bitstream_color_hint(&jpeg, None)),
            SigninumColorTransform::ForceRgb
        );
        assert_eq!(
            tiff_jpeg_color_transform(6, 3, jpeg_bitstream_color_hint(&jpeg, None)),
            SigninumColorTransform::ForceYCbCr
        );
    }

    #[test]
    fn jpeg_rgb_component_ids_ascii_force_rgb() {
        let jpeg = jpeg_sof([b'R', b'G', b'B'], [(1, 1), (1, 1), (1, 1)]);

        assert_eq!(
            jpeg_bitstream_color_hint(&jpeg, None),
            JpegBitstreamColorHint::Rgb
        );
        assert_eq!(
            tiff_jpeg_color_transform(6, 3, jpeg_bitstream_color_hint(&jpeg, None)),
            SigninumColorTransform::ForceRgb
        );
    }

    #[test]
    fn jpeg_rgb_tiff_with_actual_chroma_subsampling_uses_ycbcr_hint() {
        let jpeg = jpeg_sof([1, 2, 3], [(2, 2), (1, 1), (1, 1)]);

        assert_eq!(
            jpeg_bitstream_color_hint(&jpeg, None),
            JpegBitstreamColorHint::YCbCr
        );
        assert_eq!(
            tiff_jpeg_color_transform(2, 3, jpeg_bitstream_color_hint(&jpeg, None)),
            SigninumColorTransform::ForceYCbCr
        );
    }

    #[test]
    fn jpeg_unknown_bitstream_falls_back_to_tiff_photometric() {
        assert_eq!(
            tiff_jpeg_color_transform(2, 3, JpegBitstreamColorHint::Unknown),
            SigninumColorTransform::ForceRgb
        );
        assert_eq!(
            tiff_jpeg_color_transform(6, 3, JpegBitstreamColorHint::Unknown),
            SigninumColorTransform::ForceYCbCr
        );
    }

    // ── FullDecodeCache tests ─────────────────────────────────────

    #[test]
    fn full_decode_cache_put_and_get() {
        let mut cache = FullDecodeCache::new(1024);
        let buf = Arc::new(make_sample_buffer(100));
        cache.put(IfdId(1000), buf.clone());

        let result = cache.get(&IfdId(1000));
        assert!(result.is_some());
        assert_eq!(result.unwrap().width, 64);
    }

    #[test]
    fn full_decode_cache_eviction() {
        let mut cache = FullDecodeCache::new(250);
        cache.put(IfdId(100), Arc::new(make_sample_buffer(100)));
        cache.put(IfdId(200), Arc::new(make_sample_buffer(100)));
        // 200 bytes used — both fit
        assert!(cache.get(&IfdId(100)).is_some());
        assert!(cache.get(&IfdId(200)).is_some());

        // Third entry pushes over 250 — LRU (IfdId(100)) should be evicted
        // Note: after the two gets above, access order is 100 then 200,
        // so IfdId(100) is older. But LruCache.get() promotes, so after
        // get(100) then get(200), 100 was accessed first, then 200.
        // The LRU is IfdId(100).
        cache.put(IfdId(300), Arc::new(make_sample_buffer(100)));
        assert!(cache.get(&IfdId(100)).is_none()); // evicted
        assert!(cache.get(&IfdId(200)).is_some());
        assert!(cache.get(&IfdId(300)).is_some());
    }

    #[test]
    fn full_decode_cache_oversize_rejected() {
        let mut cache = FullDecodeCache::new(50);
        let buf = Arc::new(make_sample_buffer(100));
        cache.put(IfdId(1000), buf);

        assert!(cache.get(&IfdId(1000)).is_none());
        assert_eq!(cache.current_bytes, 0);
    }

    #[test]
    fn full_decode_cache_miss() {
        let mut cache = FullDecodeCache::new(1024);
        assert!(cache.get(&IfdId(9999)).is_none());
    }

    #[test]
    fn full_decode_cache_replacement_updates_bytes() {
        let mut cache = FullDecodeCache::new(500);
        cache.put(IfdId(100), Arc::new(make_sample_buffer(100)));
        assert_eq!(cache.current_bytes, 100);

        // Replace with larger buffer
        cache.put(IfdId(100), Arc::new(make_sample_buffer(200)));
        assert_eq!(cache.current_bytes, 200);

        // Still retrievable
        assert!(cache.get(&IfdId(100)).is_some());
    }

    #[test]
    fn clamp_ndpi_strip_crop_limits_edge_requests_to_strip_bounds() {
        assert_eq!(
            TiffPixelReader::clamp_ndpi_strip_crop(112, 0, 136, 240, 104, 240),
            None
        );
        assert_eq!(
            TiffPixelReader::clamp_ndpi_strip_crop(0, 0, 136, 240, 104, 240),
            Some((104, 240))
        );
        assert_eq!(
            TiffPixelReader::clamp_ndpi_strip_crop(112, 16, 136, 240, 248, 240),
            Some((136, 224))
        );
    }

    fn build_test_ndpi_reader_for_strip_cache(
        width: u32,
        height: u32,
        tiles_across: u32,
    ) -> (TiffPixelReader, IfdId) {
        let tiles_down = height.div_ceil(16);
        let jpeg = encode_restart_rgb_jpeg(
            &image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0])),
            75,
            8,
        );
        let bitstream_start = find_test_jpeg_bitstream_start(&jpeg).unwrap();
        let jpeg_header = jpeg[..bitstream_start].to_vec();
        let file =
            build_ndpi_full_jpeg_tiff(width, height, &jpeg, (tiles_across * tiles_down) as usize);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(12),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![
                            Level {
                                dimensions: (width as u64, height as u64),
                                downsample: 1.0,
                                tile_layout: TileLayout::WholeLevel {
                                    width: width as u64,
                                    height: height as u64,
                                    virtual_tile_width: 128,
                                    virtual_tile_height: 16,
                                },
                            },
                            Level {
                                dimensions: (width as u64, height as u64),
                                downsample: 2.0,
                                tile_layout: TileLayout::WholeLevel {
                                    width: width as u64,
                                    height: height as u64,
                                    virtual_tile_width: 128,
                                    virtual_tile_height: 16,
                                },
                            },
                        ],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 1,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::NdpiJpeg {
                    ifd_id,
                    jpeg_header,
                    mcu_starts_tag: 65426,
                    tiles_across,
                    tiles_down,
                    restart_interval: 8,
                    strip_offset: 8,
                    strip_byte_count: jpeg.len() as u64,
                },
            )]),
            associated_sources: HashMap::new(),
        };
        (TiffPixelReader::new(container, layout), ifd_id)
    }

    struct TestNdpiJpegLayout {
        ifd_id: IfdId,
        dimensions: (u32, u32),
        virtual_tile: (u32, u32),
        tile_grid: (u32, u32),
        jpeg_header: Vec<u8>,
        strip_byte_count: u64,
    }

    fn build_test_ndpi_layout_from_header(spec: TestNdpiJpegLayout) -> DatasetLayout {
        let (width, height) = spec.dimensions;
        let (virtual_tile_width, virtual_tile_height) = spec.virtual_tile;
        let (tiles_across, tiles_down) = spec.tile_grid;
        DatasetLayout {
            dataset: Dataset {
                id: DatasetId(12),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![Level {
                            dimensions: (width as u64, height as u64),
                            downsample: 1.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: width as u64,
                                height: height as u64,
                                virtual_tile_width,
                                virtual_tile_height,
                            },
                        }],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::NdpiJpeg {
                    ifd_id: spec.ifd_id,
                    jpeg_header: spec.jpeg_header,
                    mcu_starts_tag: 65426,
                    tiles_across,
                    tiles_down,
                    restart_interval: 8,
                    strip_offset: 8,
                    strip_byte_count: spec.strip_byte_count,
                },
            )]),
            associated_sources: HashMap::new(),
        }
    }

    fn make_ndpi_strip(width: u32, height: u32, rgb: [u8; 3]) -> Arc<CpuTile> {
        let mut data = vec![0u8; width as usize * height as usize * 3];
        for pixel in data.chunks_exact_mut(3) {
            pixel.copy_from_slice(&rgb);
        }
        Arc::new(CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(data),
        })
    }

    #[test]
    fn ndpi_display_tile_only_populates_requested_strip_keys() {
        let (reader, ifd_id) = build_test_ndpi_reader_for_strip_cache(680, 72, 5);

        let tile = reader
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
                tile_width: 250,
                tile_height: 32,
            })
            .unwrap();

        assert_eq!(tile.width, 250);
        assert_eq!(tile.height, 32);

        let mut cache = reader.ndpi_strip_cache.lock().unwrap();
        assert!(cache
            .get(&NdpiStripKey {
                ifd_id,
                col: 0,
                native_row: 0
            })
            .is_some());
        assert!(cache
            .get(&NdpiStripKey {
                ifd_id,
                col: 1,
                native_row: 0
            })
            .is_some());
        assert!(cache
            .get(&NdpiStripKey {
                ifd_id,
                col: 0,
                native_row: 1
            })
            .is_some());
        assert!(cache
            .get(&NdpiStripKey {
                ifd_id,
                col: 1,
                native_row: 1
            })
            .is_some());
        assert!(cache
            .get(&NdpiStripKey {
                ifd_id,
                col: 2,
                native_row: 1
            })
            .is_none());
    }

    #[test]
    fn ndpi_display_tile_composites_from_strip_cache_across_rows_and_columns() {
        let (reader, ifd_id) = build_test_ndpi_reader_for_strip_cache(256, 48, 2);
        {
            let mut cache = reader.ndpi_strip_cache.lock().unwrap();
            cache.put(
                NdpiStripKey {
                    ifd_id,
                    col: 0,
                    native_row: 0,
                },
                make_ndpi_strip(128, 16, [10, 0, 0]),
            );
            cache.put(
                NdpiStripKey {
                    ifd_id,
                    col: 1,
                    native_row: 0,
                },
                make_ndpi_strip(128, 16, [20, 0, 0]),
            );
            cache.put(
                NdpiStripKey {
                    ifd_id,
                    col: 0,
                    native_row: 1,
                },
                make_ndpi_strip(128, 16, [30, 0, 0]),
            );
            cache.put(
                NdpiStripKey {
                    ifd_id,
                    col: 1,
                    native_row: 1,
                },
                make_ndpi_strip(128, 16, [40, 0, 0]),
            );
        }

        let tile = reader
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                row: 0,
                col: 0,
                tile_width: 200,
                tile_height: 32,
            })
            .unwrap();

        let CpuTileData::U8(rgb) = tile.data else {
            panic!("expected RGB data");
        };
        assert_eq!(&rgb[0..3], &[10, 0, 0]);
        let right = 128 * 3;
        assert_eq!(&rgb[right..right + 3], &[20, 0, 0]);
        let lower = (16 * tile.width as usize) * 3;
        assert_eq!(&rgb[lower..lower + 3], &[30, 0, 0]);
        let lower_right = ((16 * tile.width as usize) + 128) * 3;
        assert_eq!(&rgb[lower_right..lower_right + 3], &[40, 0, 0]);
    }

    #[test]
    fn ndpi_display_tile_composites_across_multiple_strip_rows_and_columns() {
        let (reader, ifd_id) = build_test_ndpi_reader_for_strip_cache(320, 600, 3);
        {
            let mut cache = reader.ndpi_strip_cache.lock().unwrap();
            for native_row in 16..=31 {
                for col in 0..=1 {
                    cache.put(
                        NdpiStripKey {
                            ifd_id,
                            col,
                            native_row,
                        },
                        make_ndpi_strip(128, 16, [(col * 50) as u8, native_row as u8, 7]),
                    );
                }
            }
        }

        let tile = reader
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 0,
                row: 1,
                tile_width: 256,
                tile_height: 256,
            })
            .unwrap();

        assert_eq!(tile.width, 256);
        assert_eq!(tile.height, 256);
        let rgb = tile.data.as_u8().unwrap();
        let pixel = |x: usize, y: usize| -> [u8; 3] {
            let idx = (y * tile.width as usize + x) * 3;
            [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
        };

        assert_eq!(pixel(50, 4), [0, 16, 7]);
        assert_eq!(pixel(50, 20), [0, 17, 7]);
        assert_eq!(pixel(200, 20), [50, 17, 7]);
    }

    #[test]
    fn ndpi_restart_tile_decodes_target_strip_via_public_read_path() {
        let (file, jpeg_header, strip_byte_count) = build_ndpi_scan_data_tiff_from_blobs(
            128,
            16,
            &[[240, 20, 20], [20, 220, 20], [20, 20, 230], [220, 220, 30]],
            false,
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = build_test_ndpi_layout_from_header(TestNdpiJpegLayout {
            ifd_id,
            dimensions: (128, 16),
            virtual_tile: (64, 8),
            tile_grid: (2, 2),
            jpeg_header,
            strip_byte_count,
        });
        let reader = TiffPixelReader::new(container, layout);

        let tile = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 1,
            })
            .unwrap();

        assert_eq!(tile.width, 64);
        assert_eq!(tile.height, 8);
        let CpuTileData::U8(rgb) = tile.data else {
            panic!("expected RGB data");
        };
        let pixel = [rgb[0], rgb[1], rgb[2]];
        assert!(
            pixel[0] > 170,
            "expected red channel dominance, got {pixel:?}"
        );
        assert!(
            pixel[1] > 170,
            "expected green channel dominance, got {pixel:?}"
        );
        assert!(
            pixel[2] < 120,
            "expected blue channel to stay lower, got {pixel:?}"
        );

        let mut cache = reader.ndpi_strip_cache.lock().unwrap();
        assert!(cache
            .get(&NdpiStripKey {
                ifd_id,
                col: 1,
                native_row: 1,
            })
            .is_some());
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires Metal device decode"]
    fn ndpi_restart_tile_decodes_to_metal_device_tile() {
        let Ok(jpeg_session) = signinum_jpeg_metal::MetalBackendSession::system_default() else {
            return;
        };
        let Ok(j2k_session) = signinum_j2k_metal::MetalBackendSession::system_default() else {
            return;
        };
        std::env::set_var(JPEG_DEVICE_DECODE_ENV, "1");
        let (file, jpeg_header, strip_byte_count) = build_ndpi_scan_data_tiff_from_blobs(
            128,
            16,
            &[[240, 20, 20], [20, 220, 20], [20, 20, 230], [220, 220, 30]],
            false,
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = build_test_ndpi_layout_from_header(TestNdpiJpegLayout {
            ifd_id,
            dimensions: (128, 16),
            virtual_tile: (64, 8),
            tile_grid: (2, 2),
            jpeg_header,
            strip_byte_count,
        });
        let reader = TiffPixelReader::new(container, layout);

        let tiles = reader
            .read_tiles(
                &[TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: 1,
                    row: 1,
                }],
                TileOutputPreference::prefer_device_auto_with_metal(
                    crate::output::metal::MetalBackendSessions::new(jpeg_session, j2k_session),
                ),
            )
            .unwrap();

        assert_eq!(tiles.len(), 1);
        let TilePixels::Device(DeviceTile::Metal(tile)) = &tiles[0] else {
            panic!("expected NDPI tile to decode to Metal");
        };
        assert_eq!((tile.width, tile.height), (64, 8));
        assert_eq!(tile.format, SigninumPixelFormat::Rgb8);
    }

    #[test]
    fn ndpi_restart_tile_does_not_silently_fallback_to_full_decode_on_bad_mcu_table() {
        let jpeg = {
            let mut encoded = Vec::new();
            let image = image::RgbImage::new(8, 8);
            JpegEncoder::new(&mut encoded, 75)
                .encode(
                    image.as_raw().as_slice(),
                    image.width() as u16,
                    image.height() as u16,
                    JpegColorType::Rgb,
                )
                .unwrap();
            encoded
        };
        let file = build_stripped_jpeg_tiff(8, 8, &jpeg);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(3),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![Level {
                            dimensions: (8, 8),
                            downsample: 1.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: 8,
                                height: 8,
                                virtual_tile_width: 8,
                                virtual_tile_height: 8,
                            },
                        }],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::NdpiJpeg {
                    ifd_id,
                    jpeg_header: Vec::new(),
                    mcu_starts_tag: 65426,
                    tiles_across: 1,
                    tiles_down: 1,
                    restart_interval: 1,
                    strip_offset: 8,
                    strip_byte_count: jpeg.len() as u64,
                },
            )]),
            associated_sources: HashMap::new(),
        };
        let reader = TiffPixelReader::new(container, layout);

        let err = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("65426") || err.to_string().contains("MCU"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ndpi_restart_tile_decodes_zero_sof_segment_from_mcu_table() {
        let (file, jpeg_header, strip_byte_count) = build_ndpi_scan_data_tiff_from_blobs(
            128,
            16,
            &[[240, 20, 20], [20, 220, 20], [20, 20, 230], [220, 220, 30]],
            true,
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = build_test_ndpi_layout_from_header(TestNdpiJpegLayout {
            ifd_id,
            dimensions: (128, 16),
            virtual_tile: (64, 8),
            tile_grid: (2, 2),
            jpeg_header,
            strip_byte_count,
        });
        let reader = TiffPixelReader::new(container, layout);

        let tile = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!(tile.width, 64);
        assert_eq!(tile.height, 8);
        let rgb = tile.data.as_u8().expect("expected RGB tile");
        assert!(
            rgb[0] > 180 && rgb[1] < 80 && rgb[2] < 80,
            "unexpected first pixel for zero-SOF NDPI tile: {:?}",
            &rgb[0..3]
        );
    }

    #[test]
    fn synthetic_ndpi_levels_downsample_smallest_physical_level() {
        let mut image = image::RgbImage::new(4, 4);
        let source_pixels = [
            [10, 20, 30],
            [30, 40, 50],
            [50, 60, 70],
            [70, 80, 90],
            [90, 100, 110],
            [110, 120, 130],
            [130, 140, 150],
            [150, 160, 170],
            [20, 30, 40],
            [40, 50, 60],
            [60, 70, 80],
            [80, 90, 100],
            [100, 110, 120],
            [120, 130, 140],
            [140, 150, 160],
            [160, 170, 180],
        ];
        for (pixel, rgb) in image.pixels_mut().zip(source_pixels) {
            *pixel = image::Rgb(rgb);
        }
        let mut jpeg = Vec::new();
        JpegEncoder::new(&mut jpeg, 100)
            .encode(
                image.as_raw().as_slice(),
                image.width() as u16,
                image.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        let file = build_stripped_jpeg_tiff(4, 4, &jpeg);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(99),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![
                            Level {
                                dimensions: (4, 4),
                                downsample: 1.0,
                                tile_layout: TileLayout::WholeLevel {
                                    width: 4,
                                    height: 4,
                                    virtual_tile_width: 4,
                                    virtual_tile_height: 4,
                                },
                            },
                            Level {
                                dimensions: (2, 2),
                                downsample: 2.0,
                                tile_layout: TileLayout::WholeLevel {
                                    width: 2,
                                    height: 2,
                                    virtual_tile_width: 2,
                                    virtual_tile_height: 2,
                                },
                            },
                            Level {
                                dimensions: (1, 1),
                                downsample: 4.0,
                                tile_layout: TileLayout::WholeLevel {
                                    width: 1,
                                    height: 1,
                                    virtual_tile_width: 1,
                                    virtual_tile_height: 1,
                                },
                            },
                        ],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([
                (
                    TileSourceKey {
                        scene: 0,
                        series: 0,
                        level: 0,
                        z: 0,
                        c: 0,
                        t: 0,
                    },
                    TileSource::NdpiFullDecode {
                        ifd_id,
                        jpeg_header: Vec::new(),
                        strip_offset: 8,
                        strip_byte_count: jpeg.len() as u64,
                    },
                ),
                (
                    TileSourceKey {
                        scene: 0,
                        series: 0,
                        level: 1,
                        z: 0,
                        c: 0,
                        t: 0,
                    },
                    TileSource::SyntheticDownsample {
                        base_level: 0,
                        factor: 2,
                    },
                ),
                (
                    TileSourceKey {
                        scene: 0,
                        series: 0,
                        level: 2,
                        z: 0,
                        c: 0,
                        t: 0,
                    },
                    TileSource::SyntheticDownsample {
                        base_level: 0,
                        factor: 4,
                    },
                ),
            ]),
            associated_sources: HashMap::new(),
        };
        let reader = TiffPixelReader::new(container, layout);

        let level1 = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();
        assert_eq!(level1.width, 2);
        assert_eq!(level1.height, 2);
        let level1_rgb = level1.data.as_u8().unwrap();
        assert_rgb_close(&level1_rgb[0..3], &[60, 70, 80], 1);
        assert_rgb_close(&level1_rgb[3..6], &[100, 110, 120], 1);
        assert_rgb_close(&level1_rgb[6..9], &[70, 80, 90], 1);
        assert_rgb_close(&level1_rgb[9..12], &[110, 120, 130], 1);

        let level2 = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 2,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();
        assert_eq!(level2.width, 1);
        assert_eq!(level2.height, 1);
        let level2_rgb = level2.data.as_u8().unwrap();
        assert_rgb_close(&level2_rgb[0..3], &[85, 95, 105], 1);
    }

    fn assert_rgb_close(actual: &[u8], expected: &[u8; 3], tolerance: u8) {
        assert_eq!(actual.len(), 3);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                actual.abs_diff(*expected) <= tolerance,
                "actual RGB channel {actual} differs from expected {expected} by more than {tolerance}"
            );
        }
    }

    fn synthetic_ndpi_base_pixel(x: u32, y: u32) -> [u8; 3] {
        [
            (10 + x.saturating_mul(7) + y.saturating_mul(3)).min(255) as u8,
            (20 + x.saturating_mul(5) + y.saturating_mul(11)).min(255) as u8,
            (30 + x.saturating_mul(13) + y.saturating_mul(2)).min(255) as u8,
        ]
    }

    fn synthetic_ndpi_base_image(width: u32, height: u32) -> image::RgbImage {
        image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb(synthetic_ndpi_base_pixel(x, y))
        })
    }

    fn crop_rgb_with_zero_fill(source: &CpuTile, x: i64, y: i64, w: u32, h: u32) -> CpuTile {
        assert_eq!(source.channels, 3);
        assert_eq!(source.color_space, ColorSpace::Rgb);
        assert_eq!(source.layout, CpuTileLayout::Interleaved);
        let src = source.data.as_u8().unwrap();
        let mut out = vec![0u8; w as usize * h as usize * 3];
        let clipped_x0 = x.max(0).min(i64::from(source.width));
        let clipped_y0 = y.max(0).min(i64::from(source.height));
        let clipped_x1 = x
            .saturating_add(i64::from(w))
            .max(0)
            .min(i64::from(source.width));
        let clipped_y1 = y
            .saturating_add(i64::from(h))
            .max(0)
            .min(i64::from(source.height));
        if clipped_x1 <= clipped_x0 || clipped_y1 <= clipped_y0 {
            return CpuTile {
                width: w,
                height: h,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::u8(out),
            };
        }

        let copy_w = (clipped_x1 - clipped_x0) as usize;
        let copy_h = (clipped_y1 - clipped_y0) as usize;
        let dst_x = (clipped_x0 - x) as usize;
        let dst_y = (clipped_y0 - y) as usize;
        let src_stride = source.width as usize * 3;
        let dst_stride = w as usize * 3;
        for row in 0..copy_h {
            let src_off = (clipped_y0 as usize + row) * src_stride + clipped_x0 as usize * 3;
            let dst_off = (dst_y + row) * dst_stride + dst_x * 3;
            out[dst_off..dst_off + copy_w * 3].copy_from_slice(&src[src_off..src_off + copy_w * 3]);
        }

        CpuTile {
            width: w,
            height: h,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(out),
        }
    }

    fn expected_synthetic_ndpi_region(
        reader: &TiffPixelReader,
        factor: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> CpuTile {
        let tile_req = TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let full = if let Some(image) = reader
            .try_decode_synthetic_level_with_signinum(&tile_req, 0, factor)
            .unwrap()
        {
            image
        } else {
            let mut base = reader
                .read_tile_cpu(&TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: 0,
                    row: 0,
                })
                .unwrap();
            if base.layout != CpuTileLayout::Interleaved
                || base.channels != 3
                || base.color_space != ColorSpace::Rgb
                || base.data.as_u8().is_none()
            {
                base = rgba_image_to_sample_buffer(base.to_rgba().unwrap());
            }
            let target = &reader.layout.dataset.scenes[0].series[0].levels[1];
            fit_synthetic_rgb_tile_to_dimensions(
                downsample_rgb_pow2_box(&base, factor).unwrap(),
                target.dimensions.0 as u32,
                target.dimensions.1 as u32,
            )
            .unwrap()
        };
        crop_rgb_with_zero_fill(&full, x, y, w, h)
    }

    fn assert_tile_eq(actual: &CpuTile, expected: &CpuTile) {
        assert_eq!(
            (actual.width, actual.height),
            (expected.width, expected.height)
        );
        assert_eq!(actual.channels, expected.channels);
        assert_eq!(actual.color_space, expected.color_space);
        assert_eq!(actual.layout, expected.layout);
        assert_eq!(actual.data.as_u8().unwrap(), expected.data.as_u8().unwrap());
    }

    fn read_synthetic_ndpi_region(
        reader: &TiffPixelReader,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> CpuTile {
        let req = RegionRequest::legacy_xywh(0, 0, 1, PlaneSelection::default(), x, y, w, h);
        let mut ctx = crate::core::registry::SlideReadContext::new(
            None,
            TileOutputPreference::cpu(),
            256 * 1024 * 1024,
        );
        reader
            .read_region_fastpath(&mut ctx, &req)
            .expect("synthetic level should have a region fast path")
            .expect("synthetic region fast path should produce pixels")
    }

    fn build_synthetic_ndpi_reader(
        width: u32,
        height: u32,
        synthetic: &[(u64, u64, u32)],
    ) -> TiffPixelReader {
        let image = synthetic_ndpi_base_image(width, height);
        let mut jpeg = Vec::new();
        JpegEncoder::new(&mut jpeg, 95)
            .encode(
                image.as_raw().as_slice(),
                image.width() as u16,
                image.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        let file = build_stripped_jpeg_tiff(width, height, &jpeg);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();

        let mut levels = vec![Level {
            dimensions: (u64::from(width), u64::from(height)),
            downsample: 1.0,
            tile_layout: TileLayout::WholeLevel {
                width: u64::from(width),
                height: u64::from(height),
                virtual_tile_width: width,
                virtual_tile_height: height,
            },
        }];
        let mut tile_sources = HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::NdpiFullDecode {
                ifd_id,
                jpeg_header: Vec::new(),
                strip_offset: 8,
                strip_byte_count: jpeg.len() as u64,
            },
        )]);

        for (idx, (level_width, level_height, factor)) in synthetic.iter().copied().enumerate() {
            let level_idx = (idx + 1) as u32;
            levels.push(Level {
                dimensions: (level_width, level_height),
                downsample: f64::from(factor),
                tile_layout: TileLayout::WholeLevel {
                    width: level_width,
                    height: level_height,
                    virtual_tile_width: level_width as u32,
                    virtual_tile_height: level_height as u32,
                },
            });
            tile_sources.insert(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: level_idx,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::SyntheticDownsample {
                    base_level: 0,
                    factor,
                },
            );
        }

        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(100),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels,
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources,
            associated_sources: HashMap::new(),
        };
        TiffPixelReader::new(container, layout)
    }

    #[test]
    fn synthetic_ndpi_level_source_kind_marks_generated_downsamples() {
        let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);

        assert_eq!(
            reader.level_source_kind(0, 0, 0).unwrap(),
            LevelSourceKind::Physical
        );
        assert_eq!(
            reader.level_source_kind(0, 0, 1).unwrap(),
            LevelSourceKind::SyntheticDownsample
        );
    }

    #[test]
    fn synthetic_ndpi_subregion_fastpath_matches_center_roi_without_materializing_level() {
        let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
        let tile = read_synthetic_ndpi_region(&reader, 1, 1, 2, 2);
        let expected = expected_synthetic_ndpi_region(&reader, 2, 1, 1, 2, 2);

        assert_tile_eq(&tile, &expected);
        assert_eq!(
            reader.synthetic_level_cache.lock().unwrap().current_bytes,
            0,
            "ROI reads must not materialize the whole synthetic level"
        );
        assert_eq!(
            reader.synthetic_region_cache.lock().unwrap().current_bytes,
            0,
            "ROI reads must not populate full synthetic region cache entries"
        );
    }

    #[test]
    fn synthetic_ndpi_display_tile_uses_roi_fastpath_without_materializing_level() {
        let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
        let tile = reader
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 1,
                row: 1,
                tile_width: 2,
                tile_height: 2,
            })
            .unwrap();
        let expected = expected_synthetic_ndpi_region(&reader, 2, 2, 2, 2, 2);

        assert_tile_eq(&tile, &expected);
        assert_eq!(
            reader.synthetic_level_cache.lock().unwrap().current_bytes,
            0,
            "display-tile reads must not materialize the whole synthetic level"
        );
    }

    #[test]
    fn synthetic_ndpi_subregion_fastpath_zero_fills_negative_origin() {
        let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
        let tile = read_synthetic_ndpi_region(&reader, -1, -1, 3, 3);
        let expected = expected_synthetic_ndpi_region(&reader, 2, -1, -1, 3, 3);

        assert_tile_eq(&tile, &expected);
    }

    #[test]
    fn synthetic_ndpi_subregion_fastpath_keeps_odd_ceil_edge_pixels() {
        let reader = build_synthetic_ndpi_reader(5, 5, &[(3, 3, 2)]);
        let tile = read_synthetic_ndpi_region(&reader, 2, 2, 1, 1);
        let expected = expected_synthetic_ndpi_region(&reader, 2, 2, 2, 1, 1);

        assert_tile_eq(&tile, &expected);
    }

    #[test]
    fn synthetic_ndpi_subregion_fastpath_respects_cropped_synthetic_dimensions() {
        let reader = build_synthetic_ndpi_reader(5, 5, &[(2, 2, 2)]);
        let tile = read_synthetic_ndpi_region(&reader, 1, 1, 1, 1);
        let expected = expected_synthetic_ndpi_region(&reader, 2, 1, 1, 1, 1);

        assert_tile_eq(&tile, &expected);
    }

    #[test]
    fn synthetic_ndpi_subregion_fastpath_does_not_prime_deepest_synthetic_level() {
        let reader = build_synthetic_ndpi_reader(8, 8, &[(3, 3, 2), (2, 2, 4)]);
        let tile = read_synthetic_ndpi_region(&reader, 1, 1, 1, 1);
        let expected = expected_synthetic_ndpi_region(&reader, 2, 1, 1, 1, 1);

        assert_tile_eq(&tile, &expected);
        assert_eq!(
            reader.synthetic_level_cache.lock().unwrap().current_bytes,
            0,
            "ROI reads must not materialize the requested synthetic level"
        );
        assert_eq!(
            reader.synthetic_region_cache.lock().unwrap().current_bytes,
            0,
            "ROI reads must not prime unrelated full synthetic levels"
        );
    }

    #[test]
    fn synthetic_ndpi_subregion_fastpath_matches_factor_four_repeated_box_edges() {
        let reader = build_synthetic_ndpi_reader(9, 7, &[(3, 2, 4)]);
        let tile = read_synthetic_ndpi_region(&reader, 1, 1, 2, 1);
        let expected = expected_synthetic_ndpi_region(&reader, 4, 1, 1, 2, 1);

        assert_tile_eq(&tile, &expected);
    }

    #[test]
    fn synthetic_ndpi_tile_path_uses_signinum_downscale_when_dimensions_match() {
        let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
        let direct_req = TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let direct = reader
            .try_decode_synthetic_level_with_signinum(&direct_req, 0, 2)
            .expect("signinum synthetic downscale should decode")
            .expect("matching synthetic dimensions should use signinum downscale");
        assert_eq!((direct.width, direct.height), (4, 4));

        let tile = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!((tile.width, tile.height), (4, 4));
    }

    #[test]
    fn synthetic_ndpi_region_fastpath_falls_back_when_signinum_scaled_dims_do_not_match() {
        let reader = build_synthetic_ndpi_reader(5, 5, &[(2, 2, 2)]);
        let direct_req = TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        assert!(
            reader
                .try_decode_synthetic_level_with_signinum(&direct_req, 0, 2)
                .expect("signinum synthetic downscale should decode")
                .is_none(),
            "odd source dimensions should reject signinum result with mismatched target dimensions"
        );

        let req = RegionRequest::legacy_xywh(0, 0, 1, PlaneSelection::default(), 0, 0, 2, 2);
        let mut ctx = crate::core::registry::SlideReadContext::new(
            None,
            TileOutputPreference::cpu(),
            256 * 1024 * 1024,
        );
        let tile = reader
            .read_region_fastpath(&mut ctx, &req)
            .expect("synthetic fast path should handle whole-level region")
            .expect("odd-dimension signinum downscale mismatch should fall back");

        assert_eq!((tile.width, tile.height), (2, 2));
    }

    fn le_u16(v: u16) -> [u8; 2] {
        v.to_le_bytes()
    }

    fn le_u32(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }

    fn short_in_u32(v: u16) -> [u8; 4] {
        let mut bytes = [0u8; 4];
        bytes[..2].copy_from_slice(&le_u16(v));
        bytes
    }

    fn build_tiled_associated_tiff(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[Vec<u8>],
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let mut tile_offsets = Vec::with_capacity(tiles.len());
        let mut tile_byte_counts = Vec::with_capacity(tiles.len());
        for tile in tiles {
            tile_offsets.push(buf.len() as u32);
            tile_byte_counts.push(tile.len() as u32);
            buf.extend_from_slice(tile);
        }

        let tile_offsets_array_offset = if tile_offsets.len() > 1 {
            let offset = buf.len() as u32;
            for value in &tile_offsets {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let tile_byte_counts_array_offset = if tile_byte_counts.len() > 1 {
            let offset = buf.len() as u32;
            for value in &tile_byte_counts {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (258u16, 3u16, 1u32, short_in_u32(8)),
            (259u16, 3u16, 1u32, short_in_u32(1)),
            (262u16, 3u16, 1u32, short_in_u32(1)),
            (277u16, 3u16, 1u32, short_in_u32(1)),
            (322u16, 4u16, 1u32, le_u32(tile_width)),
            (323u16, 4u16, 1u32, le_u32(tile_height)),
            (
                324u16,
                4u16,
                tile_offsets.len() as u32,
                tile_offsets_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(tile_offsets[0])),
            ),
            (
                325u16,
                4u16,
                tile_byte_counts.len() as u32,
                tile_byte_counts_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
            ),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tiled_encoded_tiff(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[Vec<u8>],
        compression_tag: u16,
        samples_per_pixel: u16,
        photometric: u16,
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let mut tile_offsets = Vec::with_capacity(tiles.len());
        let mut tile_byte_counts = Vec::with_capacity(tiles.len());
        for tile in tiles {
            tile_offsets.push(buf.len() as u32);
            tile_byte_counts.push(tile.len() as u32);
            buf.extend_from_slice(tile);
        }

        let tile_offsets_array_offset = if tile_offsets.len() > 1 {
            let offset = buf.len() as u32;
            for value in &tile_offsets {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let tile_byte_counts_array_offset = if tile_byte_counts.len() > 1 {
            let offset = buf.len() as u32;
            for value in &tile_byte_counts {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (258u16, 3u16, 1u32, short_in_u32(8)),
            (259u16, 3u16, 1u32, short_in_u32(compression_tag)),
            (262u16, 3u16, 1u32, short_in_u32(photometric)),
            (277u16, 3u16, 1u32, short_in_u32(samples_per_pixel)),
            (322u16, 4u16, 1u32, le_u32(tile_width)),
            (323u16, 4u16, 1u32, le_u32(tile_height)),
            (
                324u16,
                4u16,
                tile_offsets.len() as u32,
                tile_offsets_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(tile_offsets[0])),
            ),
            (
                325u16,
                4u16,
                tile_byte_counts.len() as u32,
                tile_byte_counts_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
            ),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn build_stripped_jpeg_tiff(width: u32, height: u32, jpeg_data: &[u8]) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let strip_offset = buf.len() as u32;
        buf.extend_from_slice(jpeg_data);
        let strip_byte_count = jpeg_data.len() as u32;

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (259u16, 3u16, 1u32, short_in_u32(7)),
            (262u16, 3u16, 1u32, short_in_u32(6)),
            (273u16, 4u16, 1u32, le_u32(strip_offset)),
            (277u16, 3u16, 1u32, short_in_u32(3)),
            (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn build_stripped_uncompressed_tiff(
        width: u32,
        height: u32,
        pixels: &[u8],
        samples_per_pixel: u16,
        photometric: Option<u16>,
    ) -> NamedTempFile {
        build_stripped_uncompressed_tiff_with_predictor(
            width,
            height,
            pixels,
            samples_per_pixel,
            photometric,
            None,
        )
    }

    fn build_stripped_uncompressed_tiff_with_predictor(
        width: u32,
        height: u32,
        pixels: &[u8],
        samples_per_pixel: u16,
        photometric: Option<u16>,
        predictor: Option<u16>,
    ) -> NamedTempFile {
        build_stripped_tiff(
            width,
            height,
            pixels,
            samples_per_pixel,
            photometric,
            predictor,
            1,
        )
    }

    fn build_stripped_tiff(
        width: u32,
        height: u32,
        payload: &[u8],
        samples_per_pixel: u16,
        photometric: Option<u16>,
        predictor: Option<u16>,
        compression: u16,
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let strip_offset = buf.len() as u32;
        buf.extend_from_slice(payload);
        let strip_byte_count = payload.len() as u32;

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (258u16, 3u16, 1u32, short_in_u32(8)),
            (259u16, 3u16, 1u32, short_in_u32(compression)),
            (273u16, 4u16, 1u32, le_u32(strip_offset)),
            (277u16, 3u16, 1u32, short_in_u32(samples_per_pixel)),
            (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
        ];
        if let Some(value) = photometric {
            tags.push((262u16, 3u16, 1u32, short_in_u32(value)));
        }
        if let Some(value) = predictor {
            tags.push((317u16, 3u16, 1u32, short_in_u32(value)));
        }
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn build_multi_stripped_jpeg_tiff(
        width: u32,
        height: u32,
        rows_per_strip: u32,
        strips: &[Vec<u8>],
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let mut strip_offsets = Vec::with_capacity(strips.len());
        let mut strip_byte_counts = Vec::with_capacity(strips.len());
        for strip in strips {
            strip_offsets.push(buf.len() as u32);
            buf.extend_from_slice(strip);
            strip_byte_counts.push(strip.len() as u32);
        }

        let strip_offsets_array_offset = buf.len() as u32;
        for value in &strip_offsets {
            buf.extend_from_slice(&le_u32(*value));
        }
        let strip_byte_counts_array_offset = buf.len() as u32;
        for value in &strip_byte_counts {
            buf.extend_from_slice(&le_u32(*value));
        }

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (259u16, 3u16, 1u32, short_in_u32(7)),
            (262u16, 3u16, 1u32, short_in_u32(6)),
            (
                273u16,
                4u16,
                strip_offsets.len() as u32,
                le_u32(strip_offsets_array_offset),
            ),
            (277u16, 3u16, 1u32, short_in_u32(3)),
            (278u16, 4u16, 1u32, le_u32(rows_per_strip)),
            (
                279u16,
                4u16,
                strip_byte_counts.len() as u32,
                le_u32(strip_byte_counts_array_offset),
            ),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn encode_solid_rgb_jpeg(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let image = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
        let mut encoded = Vec::new();
        JpegEncoder::new(&mut encoded, 95)
            .encode(
                image.as_raw().as_slice(),
                image.width() as u16,
                image.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        encoded
    }

    fn encode_restart_rgb_jpeg(
        image: &image::RgbImage,
        quality: u8,
        restart_interval: u16,
    ) -> Vec<u8> {
        let mut encoded = Vec::new();
        let mut encoder = JpegEncoder::new(&mut encoded, quality);
        encoder.set_restart_interval(restart_interval);
        encoder
            .encode(
                image.as_raw().as_slice(),
                image.width() as u16,
                image.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        encoded
    }

    fn find_test_jpeg_bitstream_start(data: &[u8]) -> Option<usize> {
        let mut i = 0;
        while i < data.len().saturating_sub(1) {
            if data[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = data[i + 1];
            if marker == 0xD8 || marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
                i += 2;
                continue;
            }
            if i + 3 >= data.len() {
                break;
            }
            let seg_len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            if marker == 0xDA {
                return Some(i + 2 + seg_len);
            }
            i += 2 + seg_len;
        }
        None
    }

    fn test_jpeg_restart_segment_starts(data: &[u8]) -> Vec<u32> {
        let mut starts = Vec::new();
        if let Some(entropy_start) = find_test_jpeg_bitstream_start(data) {
            starts.push(entropy_start as u32);
        }
        let mut i = starts.first().copied().unwrap_or(0) as usize;
        while i + 1 < data.len() {
            if data[i] == 0xFF && (0xD0..=0xD7).contains(&data[i + 1]) {
                starts.push(i as u32);
                i += 2;
                continue;
            }
            i += 1;
        }
        starts
    }

    fn zero_test_jpeg_sof_dimensions(data: &mut [u8]) {
        let sof = data
            .windows(2)
            .position(|bytes| bytes == [0xFF, 0xC0])
            .expect("test JPEG has SOF0");
        data[sof + 5..sof + 9].copy_from_slice(&[0, 0, 0, 0]);
    }

    fn build_tiled_jpeg_reader(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[Vec<u8>],
    ) -> TiffPixelReader {
        let file = build_tiled_associated_tiff(width, height, tile_width, tile_height, tiles);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let level = Level {
            dimensions: (u64::from(width), u64::from(height)),
            downsample: 1.0,
            tile_layout: TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across: u64::from(width.div_ceil(tile_width)),
                tiles_down: u64::from(height.div_ceil(tile_height)),
            },
        };
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(31),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![level],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::Jpeg,
                },
            )]),
            associated_sources: HashMap::new(),
        };
        TiffPixelReader::new(container, layout)
    }

    fn build_tiled_jpeg_reader_with_tables(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[Vec<u8>],
        jpeg_tables: Vec<u8>,
    ) -> TiffPixelReader {
        let file = build_tiled_jpeg_tiff_with_tables(
            width,
            height,
            tile_width,
            tile_height,
            tiles,
            &jpeg_tables,
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let level = Level {
            dimensions: (u64::from(width), u64::from(height)),
            downsample: 1.0,
            tile_layout: TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across: u64::from(width.div_ceil(tile_width)),
                tiles_down: u64::from(height.div_ceil(tile_height)),
            },
        };
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(32),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![level],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: Some(jpeg_tables),
                    compression: Compression::Jpeg,
                },
            )]),
            associated_sources: HashMap::new(),
        };
        TiffPixelReader::new(container, layout)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tiled_encoded_reader(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[Vec<u8>],
        compression: Compression,
        compression_tag: u16,
        samples_per_pixel: u16,
        photometric: u16,
    ) -> TiffPixelReader {
        let file = build_tiled_encoded_tiff(
            width,
            height,
            tile_width,
            tile_height,
            tiles,
            compression_tag,
            samples_per_pixel,
            photometric,
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let level = Level {
            dimensions: (u64::from(width), u64::from(height)),
            downsample: 1.0,
            tile_layout: TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across: u64::from(width.div_ceil(tile_width)),
                tiles_down: u64::from(height.div_ceil(tile_height)),
            },
        };
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(33),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![level],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: None,
                    compression,
                },
            )]),
            associated_sources: HashMap::new(),
        };
        TiffPixelReader::new(container, layout)
    }

    fn build_tiled_jpeg_tiff_with_tables(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        tiles: &[Vec<u8>],
        jpeg_tables: &[u8],
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let mut tile_offsets = Vec::with_capacity(tiles.len());
        let mut tile_byte_counts = Vec::with_capacity(tiles.len());
        for tile in tiles {
            tile_offsets.push(buf.len() as u32);
            tile_byte_counts.push(tile.len() as u32);
            buf.extend_from_slice(tile);
        }

        let tile_offsets_array_offset = if tile_offsets.len() > 1 {
            let offset = buf.len() as u32;
            for value in &tile_offsets {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let tile_byte_counts_array_offset = if tile_byte_counts.len() > 1 {
            let offset = buf.len() as u32;
            for value in &tile_byte_counts {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let jpeg_tables_offset = buf.len() as u32;
        buf.extend_from_slice(jpeg_tables);

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (258u16, 3u16, 1u32, short_in_u32(8)),
            (259u16, 3u16, 1u32, short_in_u32(7)),
            (262u16, 3u16, 1u32, short_in_u32(6)),
            (277u16, 3u16, 1u32, short_in_u32(3)),
            (322u16, 4u16, 1u32, le_u32(tile_width)),
            (323u16, 4u16, 1u32, le_u32(tile_height)),
            (
                324u16,
                4u16,
                tile_offsets.len() as u32,
                tile_offsets_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(tile_offsets[0])),
            ),
            (
                325u16,
                4u16,
                tile_byte_counts.len() as u32,
                tile_byte_counts_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
            ),
            (
                347u16,
                7u16,
                jpeg_tables.len() as u32,
                le_u32(jpeg_tables_offset),
            ),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn split_test_jpeg_tables(jpeg: &[u8]) -> (Vec<u8>, Vec<u8>) {
        assert!(jpeg.starts_with(&[0xFF, 0xD8]));
        let mut abbreviated = Vec::from(&jpeg[..2]);
        let mut tables = Vec::from(&jpeg[..2]);
        let mut offset = 2usize;
        while offset + 4 <= jpeg.len() {
            assert_eq!(jpeg[offset], 0xFF);
            let marker = jpeg[offset + 1];
            if marker == 0xDA {
                abbreviated.extend_from_slice(&jpeg[offset..]);
                tables.extend_from_slice(&[0xFF, 0xD9]);
                return (abbreviated, tables);
            }
            let len = u16::from_be_bytes([jpeg[offset + 2], jpeg[offset + 3]]) as usize;
            let end = offset + 2 + len;
            assert!(end <= jpeg.len());
            if marker == 0xDB || marker == 0xC4 {
                tables.extend_from_slice(&jpeg[offset..end]);
            } else {
                abbreviated.extend_from_slice(&jpeg[offset..end]);
            }
            offset = end;
        }
        panic!("test JPEG did not contain SOS marker");
    }

    fn build_ndpi_full_jpeg_tiff(
        width: u32,
        height: u32,
        jpeg_data: &[u8],
        blob_count: usize,
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let strip_offset = buf.len() as u32;
        let mut mcu_starts = test_jpeg_restart_segment_starts(jpeg_data);
        if mcu_starts.len() >= blob_count {
            mcu_starts.truncate(blob_count);
        } else {
            mcu_starts = (0..blob_count as u32).collect();
        }
        buf.extend_from_slice(jpeg_data);
        let strip_byte_count = buf.len() as u32 - strip_offset;

        let mcu_starts_array_offset = if mcu_starts.len() > 1 {
            let offset = buf.len() as u32;
            for value in &mcu_starts {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (259u16, 3u16, 1u32, short_in_u32(7)),
            (262u16, 3u16, 1u32, short_in_u32(6)),
            (273u16, 4u16, 1u32, le_u32(strip_offset)),
            (277u16, 3u16, 1u32, short_in_u32(3)),
            (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
            (
                65426u16,
                4u16,
                mcu_starts.len() as u32,
                mcu_starts_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(mcu_starts[0])),
            ),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn build_ndpi_scan_data_tiff_from_blobs(
        width: u32,
        height: u32,
        colors: &[[u8; 3]],
        zero_sof_dimensions: bool,
    ) -> (NamedTempFile, Vec<u8>, u64) {
        let test_tile_width = 64;
        let test_tile_height = 8;
        let tiles_across = width.div_ceil(test_tile_width);
        let mut image = image::RgbImage::new(width, height);
        for (idx, rgb) in colors.iter().enumerate() {
            let tile_col = (idx as u32) % tiles_across;
            let tile_row = (idx as u32) / tiles_across;
            let x0 = tile_col * test_tile_width;
            let y0 = tile_row * test_tile_height;
            for y in y0..(y0 + test_tile_height).min(height) {
                for x in x0..(x0 + test_tile_width).min(width) {
                    image.put_pixel(x, y, image::Rgb(*rgb));
                }
            }
        }
        let mut encoded = encode_restart_rgb_jpeg(&image, 95, 8);
        if zero_sof_dimensions {
            zero_test_jpeg_sof_dimensions(&mut encoded);
        }
        let bitstream_start = find_test_jpeg_bitstream_start(&encoded).unwrap();
        let jpeg_header = encoded[..bitstream_start].to_vec();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&le_u16(42));
        let first_ifd_pos = buf.len();
        buf.extend_from_slice(&le_u32(0));

        let strip_offset = buf.len() as u32;
        let mut mcu_starts = test_jpeg_restart_segment_starts(&encoded);
        mcu_starts.truncate(colors.len());
        assert_eq!(mcu_starts.len(), colors.len());
        buf.extend_from_slice(&encoded);
        let strip_byte_count = buf.len() as u32 - strip_offset;

        let mcu_starts_array_offset = if mcu_starts.len() > 1 {
            let offset = buf.len() as u32;
            for value in &mcu_starts {
                buf.extend_from_slice(&le_u32(*value));
            }
            Some(offset)
        } else {
            None
        };

        let ifd_offset = buf.len() as u32;
        buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

        let mut tags = vec![
            (256u16, 4u16, 1u32, le_u32(width)),
            (257u16, 4u16, 1u32, le_u32(height)),
            (259u16, 3u16, 1u32, short_in_u32(7)),
            (262u16, 3u16, 1u32, short_in_u32(6)),
            (273u16, 4u16, 1u32, le_u32(strip_offset)),
            (277u16, 3u16, 1u32, short_in_u32(3)),
            (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
            (
                65426u16,
                4u16,
                mcu_starts.len() as u32,
                mcu_starts_array_offset
                    .map(le_u32)
                    .unwrap_or_else(|| le_u32(mcu_starts[0])),
            ),
        ];
        tags.sort_by_key(|tag| tag.0);

        buf.extend_from_slice(&le_u16(tags.len() as u16));
        for (tag, typ, count, value) in &tags {
            buf.extend_from_slice(&le_u16(*tag));
            buf.extend_from_slice(&le_u16(*typ));
            buf.extend_from_slice(&le_u32(*count));
            buf.extend_from_slice(value);
        }
        buf.extend_from_slice(&le_u32(0));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        (file, jpeg_header, strip_byte_count as u64)
    }

    // ── TiffPixelReader tests ─────────────────────────────────────

    // Note: Testing TiffPixelReader with NdpiJpeg requires a synthetic NDPI
    // file with valid MCU-starts tags. Since building such files is complex,
    // we test the TiffPixelReader through the full interpret -> read path in
    // Task 9's integration tests. Here we test the FullDecodeCache directly
    // (above) and add integration tests in Task 9.

    #[test]
    fn read_associated_composites_tiled_ifd_images() {
        let tiles = [vec![10u8; 4], vec![20u8; 4], vec![30u8; 4], vec![40u8; 4]];
        let file = build_tiled_associated_tiff(4, 4, 2, 2, &tiles);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(1),
                scenes: vec![],
                associated_images: HashMap::from([(
                    "label".to_string(),
                    AssociatedImage {
                        dimensions: (4, 4),
                        sample_type: SampleType::Uint8,
                        channels: 1,
                    },
                )]),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::from([(
                "label".to_string(),
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::None,
                },
            )]),
        };
        let reader = TiffPixelReader::new(container, layout);

        let image = reader.read_associated("label").unwrap();
        let rgb = image.data.as_u8().unwrap();
        let expected = vec![
            10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20,
            20, 20, 30, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 30, 30, 30, 30, 30, 30, 40, 40,
            40, 40, 40, 40,
        ];
        assert_eq!(rgb, expected.as_slice());
        let pixel = |x: usize, y: usize| -> [u8; 3] {
            let idx = (y * image.width as usize + x) * 3;
            [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
        };

        assert_eq!(pixel(0, 0), [10, 10, 10]);
        assert_eq!(pixel(3, 0), [20, 20, 20]);
        assert_eq!(pixel(0, 3), [30, 30, 30]);
        assert_eq!(pixel(3, 3), [40, 40, 40]);
    }

    #[test]
    fn raw_compressed_tile_returns_standalone_tiled_jpeg_byte_identical() {
        let jpeg = encode_solid_rgb_jpeg(8, 8, [200, 10, 30]);
        let reader = build_tiled_jpeg_reader(8, 8, 8, 8, std::slice::from_ref(&jpeg));

        let raw = reader
            .read_raw_compressed_tile(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!(raw.compression, Compression::Jpeg);
        assert_eq!((raw.width, raw.height), (8, 8));
        assert_eq!(raw.bits_allocated, 8);
        assert_eq!(raw.samples_per_pixel, 3);
        assert_eq!(raw.data, jpeg);
    }

    #[test]
    fn raw_compressed_tile_rebuilds_tiled_jpeg_with_jpeg_tables_without_reencoding_entropy() {
        let jpeg = encode_solid_rgb_jpeg(8, 8, [40, 180, 90]);
        let (abbreviated_tile, jpeg_tables) = split_test_jpeg_tables(&jpeg);
        let reader = build_tiled_jpeg_reader_with_tables(
            8,
            8,
            8,
            8,
            std::slice::from_ref(&abbreviated_tile),
            jpeg_tables,
        );

        let raw = reader
            .read_raw_compressed_tile(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!(raw.compression, Compression::Jpeg);
        assert_eq!((raw.width, raw.height), (8, 8));
        assert!(raw.data.len() > abbreviated_tile.len());
        assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xDB]));
        assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xC4]));
        assert!(raw.data.ends_with(&[0xFF, 0xD9]));
        assert!(raw
            .data
            .windows(abbreviated_tile.len().saturating_sub(2))
            .any(|window| window == &abbreviated_tile[2..]));
    }

    #[test]
    fn raw_compressed_tile_returns_tiled_jp2k_rgb_byte_identical() {
        let codestream = include_bytes!("../../../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
        let expected =
            load_fixture_rgb(include_bytes!("../../../tests/fixtures/jp2k/rgb_nomct.ppm"));
        let reader = build_tiled_encoded_reader(
            expected.width(),
            expected.height(),
            expected.width(),
            expected.height(),
            std::slice::from_ref(&codestream),
            Compression::Jp2kRgb,
            33004,
            3,
            2,
        );

        let raw = reader
            .read_raw_compressed_tile(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!(raw.compression, Compression::Jp2kRgb);
        assert_eq!(
            (raw.width, raw.height),
            (expected.width(), expected.height())
        );
        assert_eq!(raw.bits_allocated, 8);
        assert_eq!(raw.samples_per_pixel, 3);
        assert_eq!(
            raw.photometric_interpretation,
            EncodedTilePhotometricInterpretation::Rgb
        );
        assert_eq!(raw.data, codestream);
    }

    #[test]
    fn raw_compressed_tile_returns_standalone_ndpi_restart_jpeg() {
        let (reader, _) = build_test_ndpi_reader_for_strip_cache(128, 16, 1);

        let raw = reader
            .read_raw_compressed_tile(&TileRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!(raw.compression, Compression::Jpeg);
        assert_eq!((raw.width, raw.height), (128, 16));
        assert_eq!(raw.bits_allocated, 8);
        assert_eq!(raw.samples_per_pixel, 3);
        assert!(raw.data.starts_with(&[0xFF, 0xD8]));
        assert!(raw.data.ends_with(&[0xFF, 0xD9]));
        assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xC0]));
        assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xDA]));

        let decoded = decode_jpeg_rgb_with_size_override(
            &raw.data,
            None,
            raw.width,
            raw.height,
            None,
            None,
            SigninumColorTransform::Auto,
        )
        .expect("decode raw NDPI JPEG tile");
        assert_eq!((decoded.width, decoded.height), (128, 16));
    }

    #[test]
    fn raw_compressed_tile_rejects_ndpi_restart_segments_that_cross_rows() {
        let (reader, _) = build_test_ndpi_reader_for_strip_cache(130, 16, 2);

        let err = reader
            .read_raw_compressed_tile(&TileRequest {
                scene: 0,
                series: 0,
                level: 1,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap_err();

        assert!(
            err.to_string().contains("align to image rows"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn read_associated_thumbnail_assembly_matches_expected_rgb_bytes_with_edge_tiles() {
        let tiles = [
            vec![10u8; 4],
            vec![20u8; 4],
            vec![30u8; 2],
            vec![40u8; 2],
            vec![50u8; 2],
            vec![60u8; 1],
        ];
        let file = build_tiled_associated_tiff(5, 3, 2, 2, &tiles);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(1),
                scenes: vec![],
                associated_images: HashMap::from([(
                    "label".to_string(),
                    AssociatedImage {
                        dimensions: (5, 3),
                        sample_type: SampleType::Uint8,
                        channels: 1,
                    },
                )]),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::from([(
                "label".to_string(),
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::None,
                },
            )]),
        };
        let reader = TiffPixelReader::new(container, layout);

        let image = reader.read_associated("label").unwrap();
        let rgb = image.data.as_u8().unwrap();
        let grayscale_pixels = [10u8, 10, 20, 20, 30, 10, 10, 20, 20, 30, 40, 40, 50, 50, 60];
        let expected: Vec<u8> = grayscale_pixels
            .into_iter()
            .flat_map(|value| [value, value, value])
            .collect();

        assert_eq!(image.width, 5);
        assert_eq!(image.height, 3);
        assert_eq!(rgb, expected.as_slice());
    }

    #[test]
    fn read_associated_composes_multi_strip_jpeg_image() {
        let width = 4;
        let height = 4;
        let rows_per_strip = 2;

        let mut top = image::RgbImage::new(width, rows_per_strip);
        for pixel in top.pixels_mut() {
            *pixel = image::Rgb([220, 40, 10]);
        }
        let mut bottom = image::RgbImage::new(width, rows_per_strip);
        for pixel in bottom.pixels_mut() {
            *pixel = image::Rgb([15, 80, 210]);
        }

        let encode_strip = |img: &image::RgbImage| {
            let mut encoded = Vec::new();
            JpegEncoder::new(&mut encoded, 100)
                .encode(
                    img.as_raw().as_slice(),
                    img.width() as u16,
                    img.height() as u16,
                    JpegColorType::Rgb,
                )
                .unwrap();
            encoded
        };
        let file = build_multi_stripped_jpeg_tiff(
            width,
            height,
            rows_per_strip,
            &[encode_strip(&top), encode_strip(&bottom)],
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let strip_offsets = container
            .get_u64_array(ifd_id, tags::STRIP_OFFSETS)
            .unwrap();
        let strip_byte_counts = container
            .get_u64_array(ifd_id, tags::STRIP_BYTE_COUNTS)
            .unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(17),
                scenes: vec![],
                associated_images: HashMap::from([(
                    "label".to_string(),
                    AssociatedImage {
                        dimensions: (width, height),
                        sample_type: SampleType::Uint8,
                        channels: 3,
                    },
                )]),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::from([(
                "label".to_string(),
                TileSource::Stripped {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::Jpeg,
                    strip_offsets: strip_offsets.to_vec(),
                    strip_byte_counts: strip_byte_counts.to_vec(),
                },
            )]),
        };
        let reader = TiffPixelReader::new(container, layout);

        let image = reader.read_associated("label").unwrap();
        let rgb = image.data.as_u8().unwrap();
        let pixel = |x: usize, y: usize| -> [u8; 3] {
            let idx = (y * image.width as usize + x) * 3;
            [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
        };

        let top_left = pixel(0, 0);
        let top_right = pixel((width - 1) as usize, 0);
        let bottom_left = pixel(0, 3);
        let bottom_right = pixel((width - 1) as usize, 3);
        let strip_delta = |a: [u8; 3], b: [u8; 3]| -> u16 {
            a.into_iter()
                .zip(b)
                .map(|(lhs, rhs)| lhs.abs_diff(rhs) as u16)
                .sum()
        };

        assert!(strip_delta(top_left, top_right) < 20);
        assert!(strip_delta(bottom_left, bottom_right) < 20);
        assert!(strip_delta(top_left, bottom_left) > 80);
    }

    #[test]
    fn read_associated_uncompressed_single_sample_rgb_photometric_treated_as_grayscale() {
        let pixels = [12u8, 34, 56, 78, 90, 123, 150, 210];
        let file = build_stripped_uncompressed_tiff(4, 2, &pixels, 1, Some(2));
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(23),
                scenes: vec![],
                associated_images: HashMap::from([(
                    "thumbnail".to_string(),
                    AssociatedImage {
                        dimensions: (4, 2),
                        sample_type: SampleType::Uint8,
                        channels: 1,
                    },
                )]),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::from([(
                "thumbnail".to_string(),
                TileSource::Stripped {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::None,
                    strip_offsets: vec![container.get_u64(ifd_id, tags::STRIP_OFFSETS).unwrap()],
                    strip_byte_counts: vec![container
                        .get_u64(ifd_id, tags::STRIP_BYTE_COUNTS)
                        .unwrap()],
                },
            )]),
        };
        let reader = TiffPixelReader::new(container, layout);

        let image = reader.read_associated("thumbnail").unwrap();
        assert_eq!(image.width, 4);
        assert_eq!(image.height, 2);
        assert_eq!(image.channels, 1);
        assert_eq!(image.color_space, ColorSpace::Grayscale);
        assert_eq!(image.data.as_u8().unwrap(), pixels.as_slice());
    }

    #[test]
    fn tiff_predictor_reconstructs_8bit_horizontal_deltas() {
        let encoded = [10u8, 5, 5, 1, 2, 3];
        let file =
            build_stripped_uncompressed_tiff_with_predictor(3, 2, &encoded, 1, Some(1), Some(2));
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(24),
                scenes: vec![],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::new(),
        };
        let reader = TiffPixelReader::new(container, layout);
        let mut data = encoded.to_vec();

        reader
            .apply_tiff_predictor(ifd_id, 3, 2, &mut data)
            .unwrap();

        assert_eq!(data, [10, 15, 20, 1, 3, 6]);
    }

    #[test]
    fn read_associated_deflate_predictor_uses_tilecodec_path() {
        let expected = [10u8, 15, 20, 1, 3, 6];
        let predictor_encoded = [10u8, 5, 5, 1, 2, 3];
        let mut encoder = ZlibEncoder::new(Vec::new(), DeflateCompression::fast());
        encoder.write_all(&predictor_encoded).unwrap();
        let compressed = encoder.finish().unwrap();
        let file = build_stripped_tiff(3, 2, &compressed, 1, Some(1), Some(2), 8);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(25),
                scenes: vec![],
                associated_images: HashMap::from([(
                    "thumbnail".to_string(),
                    AssociatedImage {
                        dimensions: (3, 2),
                        sample_type: SampleType::Uint8,
                        channels: 1,
                    },
                )]),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::from([(
                "thumbnail".to_string(),
                TileSource::Stripped {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::Deflate,
                    strip_offsets: vec![container.get_u64(ifd_id, tags::STRIP_OFFSETS).unwrap()],
                    strip_byte_counts: vec![container
                        .get_u64(ifd_id, tags::STRIP_BYTE_COUNTS)
                        .unwrap()],
                },
            )]),
        };
        let reader = TiffPixelReader::new(container, layout);

        let image = reader.read_associated("thumbnail").unwrap();

        assert_eq!(image.data.as_u8().unwrap(), expected.as_slice());
    }

    #[test]
    fn read_tiles_classifies_distinct_jpeg_tiled_ifd_requests_as_batchable() {
        let tiles = [
            encode_solid_rgb_jpeg(8, 8, [200, 10, 10]),
            encode_solid_rgb_jpeg(8, 8, [10, 200, 10]),
            encode_solid_rgb_jpeg(8, 8, [10, 10, 200]),
            encode_solid_rgb_jpeg(8, 8, [220, 220, 20]),
        ];
        let reader = build_tiled_jpeg_reader(16, 16, 8, 8, &tiles);
        let reqs = [
            TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            },
            TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 0,
            },
            TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 1,
            },
            TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 1,
            },
        ];

        assert_eq!(
            reader.tiled_ifd_batch_compression(&reqs).unwrap(),
            Some(Compression::Jpeg)
        );

        let batched = reader.read_tiles_cpu(&reqs).unwrap();
        let sequential = reqs
            .iter()
            .map(|req| reader.read_tile_cpu(req))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(batched.len(), sequential.len());
        for (batched, sequential) in batched.iter().zip(sequential.iter()) {
            assert_eq!((batched.width, batched.height), (8, 8));
            assert_eq!(batched.data.as_u8(), sequential.data.as_u8());
        }
    }

    #[test]
    fn tile_codec_kind_classifies_tiff_jpeg_and_jp2k_sources() {
        let jpeg_tiles = [encode_solid_rgb_jpeg(8, 8, [200, 10, 10])];
        let jpeg_reader = build_tiled_jpeg_reader(8, 8, 8, 8, &jpeg_tiles);
        let req = TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        assert_eq!(jpeg_reader.tile_codec_kind(&req), TileCodecKind::Jpeg);

        let codestream = include_bytes!("../../../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
        let expected =
            load_fixture_rgb(include_bytes!("../../../tests/fixtures/jp2k/rgb_nomct.ppm"));
        let jp2k_reader = build_tiled_encoded_reader(
            expected.width(),
            expected.height(),
            expected.width(),
            expected.height(),
            &[codestream],
            Compression::Jp2kRgb,
            33004,
            3,
            2,
        );
        assert_eq!(jp2k_reader.tile_codec_kind(&req), TileCodecKind::Jp2k);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn prefer_device_empty_tiled_jpeg_falls_back_to_cpu_empty_tile() {
        let tiles = [Vec::new()];
        let reader = build_tiled_jpeg_reader(8, 8, 8, 8, &tiles);
        let req = TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        };

        let tiles = reader
            .read_tiles(&[req], TileOutputPreference::prefer_device_auto())
            .unwrap();

        assert_eq!(tiles.len(), 1);
        let TilePixels::Cpu(tile) = &tiles[0] else {
            panic!("PreferDevice should fall back to CPU for empty tiles");
        };
        assert_eq!((tile.width, tile.height), (8, 8));
        assert_eq!(tile.data.as_u8().unwrap(), &[0u8; 8 * 8 * 3]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn jpeg_device_decode_is_opt_in_by_default() {
        assert!(!jpeg_device_decode_enabled());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn jp2k_device_decode_is_opt_in_by_default() {
        assert!(!jp2k_device_decode_enabled());
    }

    #[test]
    fn jp2k_tiled_sources_request_larger_shared_cache_budget() {
        let tiles = [vec![7u8; 4]];
        let file = build_tiled_associated_tiff(2, 2, 2, 2, &tiles);
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(24),
                scenes: vec![Scene {
                    id: "s0".into(),
                    name: None,
                    series: vec![Series {
                        id: "ser0".into(),
                        axes: AxesShape::default(),
                        levels: vec![Level {
                            dimensions: (2, 2),
                            downsample: 1.0,
                            tile_layout: TileLayout::Regular {
                                tile_width: 2,
                                tile_height: 2,
                                tiles_across: 1,
                                tiles_down: 1,
                            },
                        }],
                        sample_type: SampleType::Uint8,
                        channels: vec![],
                    }],
                }],
                associated_images: HashMap::new(),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::from([(
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: None,
                    compression: Compression::Jp2kRgb,
                },
            )]),
            associated_sources: HashMap::new(),
        };
        let reader = TiffPixelReader::new(container, layout);

        assert_eq!(
            reader.recommended_shared_cache_bytes(),
            Some(DEFAULT_JP2K_SHARED_TILE_CACHE_BYTES)
        );
    }

    fn load_fixture_rgb(ppm_bytes: &[u8]) -> image::RgbImage {
        match image::load(Cursor::new(ppm_bytes), ImageFormat::Pnm).unwrap() {
            DynamicImage::ImageRgb8(image) => image,
            other => other.to_rgb8(),
        }
    }

    fn build_single_tile_jp2k_layout(
        container: Arc<TiffContainer>,
        compression: Compression,
        width: u32,
        height: u32,
    ) -> TiffPixelReader {
        let ifd_id = *container.top_ifds().first().unwrap();
        let layout = DatasetLayout {
            dataset: Dataset {
                id: DatasetId(1),
                scenes: vec![],
                associated_images: HashMap::from([(
                    "label".to_string(),
                    AssociatedImage {
                        dimensions: (width, height),
                        sample_type: SampleType::Uint8,
                        channels: 3,
                    },
                )]),
                properties: Properties::new(),
                icc_profiles: HashMap::new(),
            },
            tile_sources: HashMap::new(),
            associated_sources: HashMap::from([(
                "label".to_string(),
                TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables: None,
                    compression,
                },
            )]),
        };
        TiffPixelReader::new(container, layout)
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
            max_delta <= 50,
            "JP2K tiled decode drift too large: max channel delta {max_delta} > 50",
        );
        assert!(
            avg_delta_x100 <= 1600,
            "JP2K tiled decode drift too large: average channel delta {:.2} > 16.00",
            avg_delta_x100 as f64 / 100.0,
        );
    }

    #[test]
    fn read_associated_decodes_jp2k_rgb_tile_from_tiled_ifd() {
        let codestream = include_bytes!("../../../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
        let expected =
            load_fixture_rgb(include_bytes!("../../../tests/fixtures/jp2k/rgb_nomct.ppm"));
        let file = build_tiled_associated_tiff(
            expected.width(),
            expected.height(),
            expected.width(),
            expected.height(),
            &[codestream],
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let reader = build_single_tile_jp2k_layout(
            container,
            Compression::Jp2kRgb,
            expected.width(),
            expected.height(),
        );

        let image = reader.read_associated("label").unwrap();
        assert_sample_buffer_matches_rgb_fixture(&image, &expected);
    }

    #[test]
    fn read_associated_decodes_jp2k_ycbcr_tile_from_tiled_ifd() {
        let codestream = include_bytes!("../../../tests/fixtures/jp2k/ycbcr_420.j2k").to_vec();
        let expected =
            load_fixture_rgb(include_bytes!("../../../tests/fixtures/jp2k/ycbcr_420.ppm"));
        let file = build_tiled_associated_tiff(
            expected.width(),
            expected.height(),
            expected.width(),
            expected.height(),
            &[codestream],
        );
        let container = Arc::new(TiffContainer::open(file.path()).unwrap());
        let reader = build_single_tile_jp2k_layout(
            container,
            Compression::Jp2kYcbcr,
            expected.width(),
            expected.height(),
        );

        let image = reader.read_associated("label").unwrap();
        assert_sample_buffer_matches_rgb_fixture(&image, &expected);
    }
}
