use super::*;

// ── Probe traits ───────────────────────────────────────────────────

/// Detects whether a file is a given format. Fast, no full parse.
pub trait FormatProbe: Send + Sync {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError>;
}

#[derive(Debug)]
pub struct ProbeResult {
    pub detected: bool,
    pub vendor: String,
    pub confidence: ProbeConfidence,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProbeConfidence {
    Definite,
    Likely,
}

/// Opens a file and returns a SlideReader.
pub trait DatasetReader: Send + Sync {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError>;
}

// ── Read interface ─────────────────────────────────────────────────

pub struct SlideReadContext<'a> {
    tile_cache: Option<&'a TileCache>,
    output: TileOutputPreference,
    max_region_pixels: u64,
}

impl<'a> SlideReadContext<'a> {
    pub fn new(
        tile_cache: Option<&'a TileCache>,
        output: TileOutputPreference,
        max_region_pixels: u64,
    ) -> Self {
        Self {
            tile_cache,
            output,
            max_region_pixels,
        }
    }

    pub(crate) fn tile_cache(&self) -> Option<&'a TileCache> {
        self.tile_cache
    }

    pub fn output(&self) -> &TileOutputPreference {
        &self.output
    }

    pub fn max_region_pixels(&self) -> u64 {
        self.max_region_pixels
    }
}

/// Phase-2 read interface.
///
/// `read_tile` is a default impl over a 1-element slice into `read_tiles`. A
/// backend that overrides `read_tiles` automatically gets the right
/// `read_tile` for free:
///
/// ```
/// use statumen::{
///     ColorSpace, CpuTile, Dataset, SlideReader, TileOutputPreference, TilePixels, TileRequest,
///     WsiError,
/// };
/// # fn _example() {
/// struct Mock;
/// impl SlideReader for Mock {
///     fn dataset(&self) -> &Dataset { unimplemented!() }
///     fn read_tiles(
///         &self,
///         reqs: &[TileRequest],
///         _: TileOutputPreference,
///     ) -> Result<Vec<TilePixels>, WsiError> {
///         Ok(reqs
///             .iter()
///             .map(|_| {
///                 TilePixels::Cpu(
///                     CpuTile::from_u8_interleaved(1, 1, 3, ColorSpace::Rgb, vec![255, 0, 0])
///                         .unwrap(),
///                 )
///             })
///             .collect())
///     }
///     fn read_tile_cpu(&self, _: &TileRequest) -> Result<CpuTile, WsiError> {
///         Ok(CpuTile::from_u8_interleaved(1, 1, 3, ColorSpace::Rgb, vec![255, 0, 0]).unwrap())
///     }
///     fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
///         Err(WsiError::AssociatedImageNotFound(name.into()))
///     }
/// }
/// let m = Mock;
/// let _ = m.read_tile(
///     &TileRequest {
///         scene: 0,
///         series: 0,
///         level: 0,
///         plane: Default::default(),
///         col: 0,
///         row: 0,
///     },
///     TileOutputPreference::cpu(),
/// );
/// # }
/// ```
pub trait SlideReader: Send + Sync {
    fn dataset(&self) -> &Dataset;
    fn tile_codec_kind(&self, _req: &TileRequest) -> TileCodecKind {
        TileCodecKind::Other
    }
    fn level_source_kind(
        &self,
        scene: usize,
        series: usize,
        level: u32,
    ) -> Result<LevelSourceKind, WsiError> {
        let dataset = self.dataset();
        let scene_ref = dataset.scenes.get(scene).ok_or(WsiError::SceneOutOfRange {
            index: scene,
            count: dataset.scenes.len(),
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
        Ok(LevelSourceKind::Physical)
    }
    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        if matches!(output, TileOutputPreference::RequireDevice { .. }) {
            return Err(WsiError::Unsupported {
                reason: "RequireDevice not supported by this reader in Phase 2".into(),
            });
        }
        reqs.iter()
            .map(|req| self.read_tile_cpu(req).map(TilePixels::Cpu))
            .collect()
    }
    fn read_tile(
        &self,
        req: &TileRequest,
        output: TileOutputPreference,
    ) -> Result<TilePixels, WsiError> {
        let mut tiles = self.read_tiles(std::slice::from_ref(req), output)?;
        match tiles.len() {
            1 => Ok(tiles.remove(0)),
            0 => Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "empty tile batch result".into(),
            }),
            count => Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("single tile read returned {count} tiles"),
            }),
        }
    }
    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError>;
    fn read_raw_compressed_tile(&self, req: &TileRequest) -> Result<RawCompressedTile, WsiError> {
        Err(WsiError::Unsupported {
            reason: format!(
                "raw compressed tile access is not available for tile ({}, {}) at level {}",
                req.col, req.row, req.level
            ),
        })
    }
    fn read_tiles_cpu(&self, reqs: &[TileRequest]) -> Result<Vec<CpuTile>, WsiError> {
        self.read_tiles(reqs, TileOutputPreference::cpu())?
            .into_iter()
            .map(|tile| match tile {
                TilePixels::Cpu(cpu) => Ok(cpu),
                TilePixels::Device(_) => Err(WsiError::Unsupported {
                    reason: "CPU tile request returned device payload".into(),
                }),
            })
            .collect()
    }
    fn use_display_tile_cache(&self, _req: &TileViewRequest) -> bool {
        true
    }
    fn read_region_fastpath(
        &self,
        _ctx: &mut SlideReadContext<'_>,
        _req: &RegionRequest,
    ) -> Option<Result<CpuTile, WsiError>> {
        None
    }
    fn read_region(
        &self,
        req: &RegionRequest,
        output: TileOutputPreference,
    ) -> Result<TilePixels, WsiError> {
        if matches!(output, TileOutputPreference::RequireDevice { .. }) {
            return Err(WsiError::Unsupported {
                reason: "region requires CPU composition; RequireDevice not supported in Phase 2"
                    .into(),
            });
        }
        composite_region_from_source(self, None, req).map(TilePixels::Cpu)
    }
    fn read_display_tile(&self, req: &TileViewRequest) -> Result<CpuTile, WsiError> {
        read_display_tile_from_source(self, None, req, TileOutputPreference::cpu())
    }
    fn associated_image(&self, name: &str) -> Result<Option<CpuTile>, WsiError> {
        match self.read_associated(name) {
            Ok(tile) => Ok(Some(tile)),
            Err(WsiError::AssociatedImageNotFound(_)) => Ok(None),
            Err(err) => Err(err),
        }
    }
    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError>;
    fn recommended_shared_cache_bytes(&self) -> Option<u64> {
        None
    }
}
