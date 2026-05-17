// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

pub(crate) mod core;
pub(crate) mod decode;
pub mod error;
pub(crate) mod formats;
pub mod output;
pub mod properties;

pub use core::cache::CacheConfig;
pub use core::decode_runtime::{DecodeExecutionOptions, DecodeRoute, DecodeRouteDecision};
pub use error::WsiError;
pub use formats::svcache::{
    build_svcache, build_svcache_tile_payloads_replace, build_svcache_tiles,
    build_svcache_tiles_replace, cache_dir_svcache_path, default_svcache_path,
    svcache_candidate_paths, svcache_matches_source, SvcachePolicy, SvcacheTileSelection,
};
pub use properties::Properties;

// Multi-dimensional API
pub use core::registry::{
    DatasetReader, FormatProbe, FormatRegistry, ProbeConfidence, ProbeResult, Slide,
    SlideOpenOptions, SlideReadContext, SlideReader,
};
pub use core::types::{
    AssociatedImage, AxesShape, ChannelInfo, ColorSpace, Compression, CpuTile, CpuTileData,
    CpuTileLayout, Dataset, DatasetId, DeviceTile, DisplayWindow,
    EncodedTilePhotometricInterpretation, Level, LevelIdx, LevelSourceKind, OutputBackendRequest,
    PlaneIdx, PlaneSelection, RawCompressedTile, RegionRequest, SampleType, Scene, SceneId, Series,
    SeriesId, TileCodecKind, TileEntry, TileHit, TileLayout, TileOutputPreference, TilePixels,
    TileRequest, TileViewRequest,
};
