use super::*;

// ── Request types ──────────────────────────────────────────────────

/// Selects a z/c/t plane. Default is (0,0,0) for plain 2D reads.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct PlaneSelection {
    pub z: u32,
    pub c: u32,
    pub t: u32,
}

/// A region request — used by Slide (public API), not by backends.
#[derive(Debug, Clone)]
pub struct RegionRequest {
    pub scene: SceneId,
    pub series: SeriesId,
    pub level: LevelIdx,
    pub plane: PlaneIdx,
    /// Signed top-left pixel position in level coordinates.
    pub origin_px: (i64, i64),
    /// Unsigned region size in pixels.
    pub size_px: (u32, u32),
}

/// A single-tile request — the backend primitive.
#[derive(Debug, Clone)]
pub struct TileRequest {
    pub scene: usize,
    pub series: usize,
    pub level: u32,
    pub plane: PlaneSelection,
    pub col: i64,
    pub row: i64,
}

/// A viewer/display-tile request.
///
/// Unlike `TileRequest`, this is expressed in the viewer's regular display grid
/// rather than the storage-native tile layout. Backends may satisfy it from
/// native tiles, cached decode bands, or region composition as appropriate.
#[derive(Debug, Clone)]
pub struct TileViewRequest {
    pub scene: usize,
    pub series: usize,
    pub level: u32,
    pub plane: PlaneSelection,
    pub col: i64,
    pub row: i64,
    pub tile_width: u32,
    pub tile_height: u32,
}
