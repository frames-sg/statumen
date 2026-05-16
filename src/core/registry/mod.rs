use std::path::Path;
use std::sync::Arc;

use crate::core::cache::{CacheConfig, CacheKey, TileCache};
use crate::core::decode_runtime::{AdaptiveDecodeReader, DecodeExecutionOptions, DecodeRuntime};
#[cfg(test)]
use crate::core::decode_runtime::{DecodeRoute, DecodeRouteDecision};
use crate::core::types::*;
use crate::error::WsiError;
use crate::formats::dicom::DicomBackend;
use crate::formats::hamamatsu_vms::HamamatsuVmsBackend;
use crate::formats::mirax::MiraxBackend;
use crate::formats::olympus_vsi::OlympusVsiBackend;
use crate::formats::raw_jp2k::RawJp2kBackend;
use crate::formats::svcache::SvcacheBackend;
use crate::formats::tiff_family::TiffFamilyBackend;
use crate::formats::zeiss::ZeissBackend;
use crate::formats::zeiss_zvi::ZeissZviBackend;

/// Default maximum region size in pixels. Prevents OOM from unreasonably large
/// region requests (256 megapixels = ~768 MB for RGB8).
const DEFAULT_MAX_REGION_PIXELS: u64 = 256 * 1024 * 1024;

mod composition;
mod open_options;
mod registry_impl;
mod slide;
mod traits;

pub(crate) use composition::{
    composite_region_from_source, crop_rgb_interleaved_u8_buffer, read_display_tile_from_source,
};
pub use open_options::SlideOpenOptions;
pub use registry_impl::FormatRegistry;
pub use slide::Slide;
pub use traits::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReadContext, SlideReader,
};

#[cfg(test)]
mod tests;
