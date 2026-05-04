#![forbid(unsafe_code)]

pub(crate) mod core;
pub(crate) mod decode;
pub mod error;
pub(crate) mod formats;
pub mod output;
pub mod properties;

pub use core::cache::CacheConfig;
pub use error::WsiError;
pub use formats::svcache::{
    build_svcache, build_svcache_tiles, cache_dir_svcache_path, default_svcache_path,
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
    EncodedTilePhotometricInterpretation, Level, LevelIdx, OutputBackendRequest, PlaneIdx,
    PlaneSelection, RawCompressedTile, RegionRequest, SampleType, Scene, SceneId, Series, SeriesId,
    TileEntry, TileHit, TileLayout, TileOutputPreference, TilePixels, TileRequest, TileViewRequest,
};
