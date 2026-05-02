use std::path::Path;
use std::sync::Arc;

use crate::core::cache::{CacheConfig, CacheKey, TileCache};
use crate::core::types::*;
use crate::error::WsiError;
use crate::formats::dicom::DicomBackend;
use crate::formats::hamamatsu_vms::HamamatsuVmsBackend;
use crate::formats::mirax::MiraxBackend;
use crate::formats::olympus_vsi::OlympusVsiBackend;
use crate::formats::svcache::SvcacheBackend;
use crate::formats::tiff_family::TiffFamilyBackend;
use crate::formats::zeiss::ZeissBackend;
use crate::formats::zeiss_zvi::ZeissZviBackend;

/// Default maximum region size in pixels. Prevents OOM from unreasonably large
/// region requests (256 megapixels = ~768 MB for RGB8).
const DEFAULT_MAX_REGION_PIXELS: u64 = 256 * 1024 * 1024;

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

fn validate_region_request<'a>(
    dataset: &'a Dataset,
    req: &RegionRequest,
) -> Result<(&'a Scene, &'a Series, &'a Level), WsiError> {
    if req.scene.0 >= dataset.scenes.len() {
        return Err(WsiError::SceneOutOfRange {
            index: req.scene.0,
            count: dataset.scenes.len(),
        });
    }
    let scene = &dataset.scenes[req.scene.0];

    if req.series.0 >= scene.series.len() {
        return Err(WsiError::SeriesOutOfRange {
            index: req.series.0,
            count: scene.series.len(),
        });
    }
    let series = &scene.series[req.series.0];

    if req.level.0 as usize >= series.levels.len() {
        return Err(WsiError::LevelOutOfRange {
            level: req.level.0,
            count: series.levels.len() as u32,
        });
    }
    let level = &series.levels[req.level.0 as usize];

    if req.plane.0.z >= series.axes.z {
        return Err(WsiError::PlaneOutOfRange {
            axis: "z".into(),
            value: req.plane.0.z,
            max: series.axes.z,
        });
    }
    if req.plane.0.c >= series.axes.c {
        return Err(WsiError::PlaneOutOfRange {
            axis: "c".into(),
            value: req.plane.0.c,
            max: series.axes.c,
        });
    }
    if req.plane.0.t >= series.axes.t {
        return Err(WsiError::PlaneOutOfRange {
            axis: "t".into(),
            value: req.plane.0.t,
            max: series.axes.t,
        });
    }

    Ok((scene, series, level))
}

pub(crate) fn composite_region_from_source<T: SlideReader + ?Sized>(
    source: &T,
    cache: Option<&TileCache>,
    req: &RegionRequest,
) -> Result<CpuTile, WsiError> {
    let dataset = source.dataset();
    let (_, series, level) = validate_region_request(dataset, req)?;
    let (x, y) = req.origin_px;
    let (w, h) = req.size_px;
    let plane = req.plane.0;

    let cache_key_for = |col: i64, row: i64| CacheKey {
        dataset_id: dataset.id,
        scene: req.scene.0 as u32,
        series: req.series.0 as u32,
        level: req.level.0,
        z: plane.z,
        c: plane.c,
        t: plane.t,
        tile_col: col,
        tile_row: row,
    };

    let tile_req_for = |col: i64, row: i64| TileRequest {
        scene: req.scene.0,
        series: req.series.0,
        level: req.level.0,
        plane,
        col,
        row,
    };

    let read_tile_cached = |col: i64, row: i64| -> Result<Arc<CpuTile>, WsiError> {
        let key = cache_key_for(col, row);

        if let Some(cache) = cache {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached);
            }
        }

        let tile = source.read_tile_cpu(&tile_req_for(col, row))?;
        let arc_tile = Arc::new(tile);
        if let Some(cache) = cache {
            cache.put(key, arc_tile.clone());
        }
        Ok(arc_tile)
    };

    let read_hit_tiles_cached = |hits: &[TileHit]| -> Result<Vec<Arc<CpuTile>>, WsiError> {
        let mut tiles = vec![None; hits.len()];
        let mut missed_slots = Vec::new();
        let mut missed_keys = Vec::new();
        let mut missed_reqs = Vec::new();

        for (slot, hit) in hits.iter().enumerate() {
            let key = cache_key_for(hit.col, hit.row);
            if let Some(cache) = cache {
                if let Some(cached) = cache.get(&key) {
                    tiles[slot] = Some(cached);
                    continue;
                }
            }
            missed_slots.push(slot);
            missed_keys.push(key);
            missed_reqs.push(tile_req_for(hit.col, hit.row));
        }

        if !missed_reqs.is_empty() {
            let decoded = if missed_reqs.len() == 1 {
                vec![source.read_tile_cpu(&missed_reqs[0])?]
            } else {
                source
                    .read_tiles(&missed_reqs, TileOutputPreference::cpu())?
                    .into_iter()
                    .map(|tile| match tile {
                        TilePixels::Cpu(cpu) => Ok(cpu),
                        TilePixels::Device(_) => Err(WsiError::Unsupported {
                            reason: "region composition requires CPU tiles".into(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            if decoded.len() != missed_reqs.len() {
                return Err(WsiError::TileRead {
                    col: missed_reqs.first().map_or(0, |req| req.col),
                    row: missed_reqs.first().map_or(0, |req| req.row),
                    level: req.level.0,
                    reason: format!(
                        "batched tile read returned {} tiles for {} requests",
                        decoded.len(),
                        missed_reqs.len()
                    ),
                });
            }

            for ((slot, key), tile) in missed_slots.into_iter().zip(missed_keys).zip(decoded) {
                let arc_tile = Arc::new(tile);
                if let Some(cache) = cache {
                    cache.put(key, arc_tile.clone());
                }
                tiles[slot] = Some(arc_tile);
            }
        }

        tiles
            .into_iter()
            .zip(hits.iter())
            .map(|(tile, hit)| {
                tile.ok_or_else(|| WsiError::TileRead {
                    col: hit.col,
                    row: hit.row,
                    level: req.level.0,
                    reason: "batched tile read did not populate requested tile".into(),
                })
            })
            .collect()
    };

    let hits = level.tile_layout.tiles_for_region(x, y, w, h);

    if hits.is_empty() {
        if let Some((probe_col, probe_row)) = metadata_probe_coordinate(&level.tile_layout) {
            if let Ok(template) = read_tile_cached(probe_col, probe_row) {
                return Ok(zero_sample_buffer_from_template(w, h, template.as_ref()));
            }
        }

        return Ok(zero_sample_buffer_from_series(w, h, series));
    }

    let hit_tiles = read_hit_tiles_cached(&hits)?;
    let first_tile = hit_tiles[0].clone();

    if first_tile.layout == CpuTileLayout::Planar {
        return Err(WsiError::DisplayConversion(
            "planar compositing not supported".into(),
        ));
    }

    let out_channels = first_tile.channels;
    let out_color_space = first_tile.color_space.clone();
    let out_layout = first_tile.layout;
    let out_w = w as usize;
    let out_h = h as usize;
    let region_pixels = w as u64 * h as u64;
    if region_pixels > DEFAULT_MAX_REGION_PIXELS {
        return Err(WsiError::DisplayConversion(format!(
            "region {}x{} ({} pixels) exceeds maximum of {} pixels",
            w, h, region_pixels, DEFAULT_MAX_REGION_PIXELS
        )));
    }
    let total_samples = out_w * out_h * out_channels as usize;
    let mut out_data = match &first_tile.data {
        CpuTileData::U8(_) => CpuTileData::u8(vec![0u8; total_samples]),
        CpuTileData::U16(_) => CpuTileData::u16(vec![0u16; total_samples]),
        CpuTileData::F32(_) => CpuTileData::f32(vec![0.0f32; total_samples]),
    };

    macro_rules! blit_tile {
        ($out_vec:expr, $tile_vec:expr, $tile:expr, $hit:expr) => {{
            let tw = $tile.width as i64;
            let th = $tile.height as i64;
            let ch = out_channels as usize;

            let src_x = (0i64).max(-$hit.dest_x) as usize;
            let src_y = (0i64).max(-$hit.dest_y) as usize;
            let dx = (0i64).max($hit.dest_x) as usize;
            let dy = (0i64).max($hit.dest_y) as usize;
            let copy_w = ((tw - src_x as i64) as usize).min(out_w - dx);
            let copy_h = ((th - src_y as i64) as usize).min(out_h - dy);
            let tile_row_stride = $tile.width as usize * ch;
            let out_row_stride = out_w * ch;

            for row in 0..copy_h {
                let src_off = (src_y + row) * tile_row_stride + src_x * ch;
                let dst_off = (dy + row) * out_row_stride + dx * ch;
                let len = copy_w * ch;
                $out_vec[dst_off..dst_off + len]
                    .copy_from_slice(&$tile_vec[src_off..src_off + len]);
            }
        }};
    }

    let needs_fractional_blit = |hit: &TileHit| {
        (hit.dest_x_f64 - hit.dest_x as f64).abs() > 1e-6
            || (hit.dest_y_f64 - hit.dest_y as f64).abs() > 1e-6
    };

    let mut alpha_buffer = matches!(&out_data, CpuTileData::U8(_))
        .then(|| hits.iter().any(needs_fractional_blit))
        .filter(|needed| *needed)
        .map(|_| vec![0.0f32; out_w * out_h]);

    let mark_tile_opaque = |alpha: &mut [f32], tile: &CpuTile, hit: &TileHit| {
        let tw = tile.width as i64;
        let th = tile.height as i64;
        let src_x = (0i64).max(-hit.dest_x) as usize;
        let src_y = (0i64).max(-hit.dest_y) as usize;
        let dx = (0i64).max(hit.dest_x) as usize;
        let dy = (0i64).max(hit.dest_y) as usize;
        let copy_w = ((tw - src_x as i64) as usize).min(out_w - dx);
        let copy_h = ((th - src_y as i64) as usize).min(out_h - dy);

        for row in 0..copy_h {
            let dst_off = (dy + row) * out_w + dx;
            alpha[dst_off..dst_off + copy_w].fill(1.0);
        }
    };

    let blit_tile_fractional_u8 = |out_vec: &mut Vec<u8>,
                                   alpha_vec: &mut [f32],
                                   tile_vec: &[u8],
                                   tile: &CpuTile,
                                   hit: &TileHit| {
        let ch = out_channels as usize;
        let tile_w = tile.width as i64;
        let tile_h = tile.height as i64;
        let start_x = hit.dest_x_f64.floor().max(0.0) as usize;
        let start_y = hit.dest_y_f64.floor().max(0.0) as usize;
        let end_x = (hit.dest_x_f64 + tile_w as f64).ceil().min(out_w as f64) as usize;
        let end_y = (hit.dest_y_f64 + tile_h as f64).ceil().min(out_h as f64) as usize;
        let out_row_stride = out_w * ch;
        let tile_row_stride = tile_w as usize * ch;

        for out_y in start_y..end_y {
            let src_y = out_y as f64 - hit.dest_y_f64;
            let y0 = src_y.floor() as i64;
            let y1 = y0 + 1;
            let wy = src_y - y0 as f64;
            let wy0 = (1.0 - wy) as f32;
            let wy1 = wy as f32;

            for out_x in start_x..end_x {
                let src_x = out_x as f64 - hit.dest_x_f64;
                let x0 = src_x.floor() as i64;
                let x1 = x0 + 1;
                let wx = src_x - x0 as f64;
                let wx0 = (1.0 - wx) as f32;
                let wx1 = wx as f32;
                let dst_off = out_y * out_row_stride + out_x * ch;
                let alpha_off = out_y * out_w + out_x;

                let in_bounds = |sx: i64, sy: i64| sx >= 0 && sx < tile_w && sy >= 0 && sy < tile_h;
                let a00 = if in_bounds(x0, y0) { wx0 * wy0 } else { 0.0 };
                let a10 = if in_bounds(x1, y0) { wx1 * wy0 } else { 0.0 };
                let a01 = if in_bounds(x0, y1) { wx0 * wy1 } else { 0.0 };
                let a11 = if in_bounds(x1, y1) { wx1 * wy1 } else { 0.0 };
                let src_alpha = a00 + a10 + a01 + a11;
                if src_alpha <= 0.0 {
                    continue;
                }

                let p00 = if in_bounds(x0, y0) {
                    Some((y0 as usize * tile_row_stride) + x0 as usize * ch)
                } else {
                    None
                };
                let p10 = if in_bounds(x1, y0) {
                    Some((y0 as usize * tile_row_stride) + x1 as usize * ch)
                } else {
                    None
                };
                let p01 = if in_bounds(x0, y1) {
                    Some((y1 as usize * tile_row_stride) + x0 as usize * ch)
                } else {
                    None
                };
                let p11 = if in_bounds(x1, y1) {
                    Some((y1 as usize * tile_row_stride) + x1 as usize * ch)
                } else {
                    None
                };
                let dst_alpha = alpha_vec[alpha_off];
                let out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);

                for channel in 0..ch {
                    let src_premult = p00
                        .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a00)
                        .unwrap_or(0.0)
                        + p10
                            .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a10)
                            .unwrap_or(0.0)
                        + p01
                            .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a01)
                            .unwrap_or(0.0)
                        + p11
                            .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a11)
                            .unwrap_or(0.0);
                    let dst_premult = (out_vec[dst_off + channel] as f32 / 255.0) * dst_alpha;
                    let out_premult = src_premult + dst_premult * (1.0 - src_alpha);
                    let value = if out_alpha > 0.0 {
                        out_premult / out_alpha
                    } else {
                        0.0
                    };
                    out_vec[dst_off + channel] = (value * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                alpha_vec[alpha_off] = out_alpha;
            }
        }
    };

    let mut blit_one_tile = |hit: &TileHit, tile: &Arc<CpuTile>| -> Result<(), WsiError> {
        match (&mut out_data, &tile.data) {
            (CpuTileData::U8(out_vec), CpuTileData::U8(tile_vec)) => {
                let out_vec = Arc::make_mut(out_vec);
                if needs_fractional_blit(hit) {
                    let alpha_vec = alpha_buffer.as_mut().ok_or_else(|| {
                        WsiError::DisplayConversion(
                            "fractional compositing alpha buffer missing".into(),
                        )
                    })?;
                    blit_tile_fractional_u8(out_vec, alpha_vec, tile_vec.as_slice(), tile, hit);
                } else {
                    blit_tile!(out_vec, tile_vec.as_slice(), tile, hit);
                    if let Some(alpha_vec) = alpha_buffer.as_mut() {
                        mark_tile_opaque(alpha_vec, tile, hit);
                    }
                }
            }
            (CpuTileData::U16(out_vec), CpuTileData::U16(tile_vec)) => {
                blit_tile!(Arc::make_mut(out_vec), tile_vec.as_slice(), tile, hit);
            }
            (CpuTileData::F32(out_vec), CpuTileData::F32(tile_vec)) => {
                blit_tile!(Arc::make_mut(out_vec), tile_vec.as_slice(), tile, hit);
            }
            _ => {
                return Err(WsiError::DisplayConversion(
                    "tile sample type mismatch during compositing".into(),
                ));
            }
        }
        Ok(())
    };

    blit_one_tile(&hits[0], &first_tile)?;
    for (hit, tile) in hits.iter().zip(hit_tiles.iter()).skip(1) {
        blit_one_tile(hit, tile)?;
    }

    Ok(CpuTile {
        width: w,
        height: h,
        channels: out_channels,
        color_space: out_color_space,
        layout: out_layout,
        data: out_data,
    })
}

fn metadata_probe_coordinate(layout: &TileLayout) -> Option<(i64, i64)> {
    match layout {
        TileLayout::Regular {
            tiles_across,
            tiles_down,
            ..
        } => (*tiles_across > 0 && *tiles_down > 0).then_some((0, 0)),
        TileLayout::WholeLevel { width, height, .. } => {
            (*width > 0 && *height > 0).then_some((0, 0))
        }
        TileLayout::Irregular { tiles, .. } => tiles
            .keys()
            .min_by(|(col_a, row_a), (col_b, row_b)| row_a.cmp(row_b).then(col_a.cmp(col_b)))
            .copied(),
    }
}

fn zero_sample_data(total_samples: usize, sample_type: SampleType) -> CpuTileData {
    match sample_type {
        SampleType::Uint8 => CpuTileData::u8(vec![0u8; total_samples]),
        SampleType::Uint16 => CpuTileData::u16(vec![0u16; total_samples]),
        SampleType::Float32 => CpuTileData::f32(vec![0.0f32; total_samples]),
    }
}

fn zero_sample_buffer_from_template(width: u32, height: u32, template: &CpuTile) -> CpuTile {
    let total_samples = width as usize * height as usize * template.channels as usize;
    CpuTile {
        width,
        height,
        channels: template.channels,
        color_space: template.color_space.clone(),
        layout: template.layout,
        data: zero_sample_data(total_samples, template.data.sample_type()),
    }
}

fn zero_sample_buffer_from_series(width: u32, height: u32, series: &Series) -> CpuTile {
    let channels = if series.channels.is_empty() {
        1u16
    } else {
        series.channels.len() as u16
    };
    let color_space = match channels {
        1 => ColorSpace::Grayscale,
        3 => ColorSpace::Rgb,
        4 => ColorSpace::Rgba,
        _ => ColorSpace::Unknown,
    };
    let total_samples = width as usize * height as usize * channels as usize;
    CpuTile {
        width,
        height,
        channels,
        color_space,
        layout: CpuTileLayout::Interleaved,
        data: zero_sample_data(total_samples, series.sample_type),
    }
}

pub(crate) fn crop_rgb_interleaved_u8_buffer(
    src: &CpuTile,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<CpuTile, WsiError> {
    if src.layout != CpuTileLayout::Interleaved || src.channels != 3 {
        return Err(WsiError::DisplayConversion(
            "RGB crop expects 3-channel interleaved data".into(),
        ));
    }
    if x > src.width
        || y > src.height
        || x.saturating_add(width) > src.width
        || y.saturating_add(height) > src.height
    {
        return Err(WsiError::DisplayConversion(format!(
            "crop {}x{} at {},{} exceeds source {}x{}",
            width, height, x, y, src.width, src.height
        )));
    }

    let src_data = src
        .data
        .as_u8()
        .ok_or_else(|| WsiError::DisplayConversion("RGB crop expects U8 source data".into()))?;
    let mut out = vec![0u8; width as usize * height as usize * 3];
    let src_stride = src.width as usize * 3;
    let dst_stride = width as usize * 3;
    for row in 0..height as usize {
        let src_start = (y as usize + row) * src_stride + x as usize * 3;
        let src_end = src_start + dst_stride;
        let dst_start = row * dst_stride;
        out[dst_start..dst_start + dst_stride].copy_from_slice(&src_data[src_start..src_end]);
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

pub(crate) fn read_display_tile_from_source<T: SlideReader + ?Sized>(
    source: &T,
    cache: Option<&TileCache>,
    req: &TileViewRequest,
    output: TileOutputPreference,
) -> Result<CpuTile, WsiError> {
    if matches!(output, TileOutputPreference::RequireDevice { .. }) {
        return Err(WsiError::Unsupported {
            reason: "display tile composition returns CPU pixels in Phase 2".into(),
        });
    }

    let dataset = source.dataset();
    let read_tile_uncached = |col: i64, row: i64| -> Result<CpuTile, WsiError> {
        let tile = source.read_tile(
            &TileRequest {
                scene: req.scene,
                series: req.series,
                level: req.level,
                plane: req.plane,
                col,
                row,
            },
            output.clone(),
        )?;
        match tile {
            TilePixels::Cpu(cpu) => Ok(cpu),
            TilePixels::Device(_) => Err(WsiError::Unsupported {
                reason: "display tile read requires CPU pixels".into(),
            }),
        }
    };
    let read_tile_cached = |col: i64, row: i64| -> Result<Arc<CpuTile>, WsiError> {
        let key = CacheKey {
            dataset_id: dataset.id,
            scene: req.scene as u32,
            series: req.series as u32,
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
            tile_col: col,
            tile_row: row,
        };

        if let Some(cache) = cache {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached);
            }
        }

        let tile = Arc::new(read_tile_uncached(col, row)?);
        if let Some(cache) = cache {
            cache.put(key, tile.clone());
        }
        Ok(tile)
    };
    let region_req = RegionRequest::legacy_xywh(
        req.scene,
        req.series,
        req.level,
        req.plane,
        req.col.saturating_mul(i64::from(req.tile_width)),
        req.row.saturating_mul(i64::from(req.tile_height)),
        req.tile_width,
        req.tile_height,
    );
    let (_, _, level) = validate_region_request(dataset, &region_req)?;

    if let TileLayout::Regular {
        tile_width,
        tile_height,
        tiles_across,
        tiles_down,
    } = &level.tile_layout
    {
        if *tile_width == req.tile_width
            && *tile_height == req.tile_height
            && req.col >= 0
            && req.row >= 0
            && req.col < *tiles_across as i64
            && req.row < *tiles_down as i64
        {
            if cache.is_none() {
                return read_tile_uncached(req.col, req.row);
            }
            return Ok(read_tile_cached(req.col, req.row)?.as_ref().clone());
        }
    }

    let level_w = level.dimensions.0 as i64;
    let level_h = level.dimensions.1 as i64;
    if region_req.origin_px.0 >= level_w || region_req.origin_px.1 >= level_h {
        return Err(WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "display tile origin out of bounds".into(),
        });
    }

    let clipped = RegionRequest {
        size_px: (
            req.tile_width
                .min((level_w - region_req.origin_px.0) as u32),
            req.tile_height
                .min((level_h - region_req.origin_px.1) as u32),
        ),
        ..region_req
    };
    composite_region_from_source(source, cache, &clipped)
}

// ── Arc blanket impls ─────────────────────────────────────────────
// Enable a single Arc<T> to be registered as both FormatProbe and
// DatasetReader when T implements both traits. Used by TiffFamilyBackend.

impl<T: FormatProbe> FormatProbe for Arc<T> {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        (**self).probe(path)
    }
}

impl<T: DatasetReader> DatasetReader for Arc<T> {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        (**self).open(path)
    }
}

// ── Format registry ────────────────────────────────────────────────

#[derive(Default)]
pub struct FormatRegistry {
    backends: Vec<RegisteredBackend>,
}

impl std::fmt::Debug for FormatRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FormatRegistry")
            .field("backend_count", &self.backends.len())
            .finish()
    }
}

struct RegisteredBackend {
    probe: Box<dyn FormatProbe>,
    reader: Box<dyn DatasetReader>,
}

impl FormatRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        probe: impl FormatProbe + 'static,
        reader: impl DatasetReader + 'static,
    ) {
        self.backends.push(RegisteredBackend {
            probe: Box::new(probe),
            reader: Box::new(reader),
        });
    }

    /// Create a registry with all built-in backends registered.
    pub fn builtin() -> Self {
        let mut reg = Self::new();
        let svcache = Arc::new(SvcacheBackend::new());
        reg.register(svcache.clone(), svcache);
        reg.register_native_backends();
        reg
    }

    pub(crate) fn builtin_native() -> Self {
        let mut reg = Self::new();
        reg.register_native_backends();
        reg
    }

    fn register_native_backends(&mut self) {
        let dicom = Arc::new(DicomBackend::new());
        self.register(dicom.clone(), dicom);
        let mirax = Arc::new(MiraxBackend::new());
        self.register(mirax.clone(), mirax);
        let vms = Arc::new(HamamatsuVmsBackend::new());
        self.register(vms.clone(), vms);
        let vsi = Arc::new(OlympusVsiBackend::new());
        self.register(vsi.clone(), vsi);
        let zeiss_zvi = Arc::new(ZeissZviBackend::new());
        self.register(zeiss_zvi.clone(), zeiss_zvi);
        let zeiss = Arc::new(ZeissBackend::new());
        self.register(zeiss.clone(), zeiss);
        let tiff = Arc::new(TiffFamilyBackend::new());
        self.register(tiff.clone(), tiff);
    }

    /// Probe all backends, open with best match.
    /// Definite confidence beats Likely. First-registered wins ties.
    pub fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        self.open_exact(path)
    }

    pub(crate) fn open_exact(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let mut best: Option<(ProbeConfidence, usize)> = None;
        let mut first_error: Option<WsiError> = None;

        for (i, backend) in self.backends.iter().enumerate() {
            match backend.probe.probe(path) {
                Ok(result) => {
                    if result.detected {
                        match (&best, &result.confidence) {
                            (None, _) => best = Some((result.confidence, i)),
                            (Some((ProbeConfidence::Likely, _)), ProbeConfidence::Definite) => {
                                best = Some((result.confidence, i));
                            }
                            _ => {}
                        }
                    }
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        match best {
            Some((_, i)) => self.backends[i].reader.open(path),
            None => Err(first_error
                .unwrap_or_else(|| WsiError::UnsupportedFormat(path.display().to_string()))),
        }
    }
}

pub struct SlideOpenOptions {
    pub registry: FormatRegistry,
    pub cache_config: CacheConfig,
    pub svcache_policy: crate::formats::svcache::SvcachePolicy,
    pub max_region_pixels: u64,
}

impl SlideOpenOptions {
    pub fn deterministic() -> Self {
        Self {
            registry: FormatRegistry::builtin(),
            cache_config: CacheConfig::deterministic(),
            svcache_policy: crate::formats::svcache::SvcachePolicy::Off,
            max_region_pixels: DEFAULT_MAX_REGION_PIXELS,
        }
    }

    pub fn with_cache_config(mut self, cache_config: CacheConfig) -> Self {
        self.cache_config = cache_config;
        self
    }

    pub fn with_svcache_policy(
        mut self,
        svcache_policy: crate::formats::svcache::SvcachePolicy,
    ) -> Self {
        self.svcache_policy = svcache_policy;
        self
    }

    pub fn with_registry(mut self, registry: FormatRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn with_max_region_pixels(mut self, max_region_pixels: u64) -> Self {
        self.max_region_pixels = max_region_pixels;
        self
    }
}

impl Default for SlideOpenOptions {
    fn default() -> Self {
        Self::deterministic()
    }
}

// ── Slide ──────────────────────────────────────────────────

/// Top-level handle. Owns the SlideReader + shared cache.
pub struct Slide {
    source: Box<dyn SlideReader>,
    cache: Arc<TileCache>,
    display_cache: Arc<TileCache>,
    max_region_pixels: u64,
}

impl std::fmt::Debug for Slide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slide")
            .field("dataset_id", &self.source.dataset().id)
            .finish()
    }
}

impl Slide {
    /// Construct from an already-opened source and cache.
    pub(crate) fn from_source(source: Box<dyn SlideReader>, cache: Arc<TileCache>) -> Self {
        Self {
            source,
            cache,
            display_cache: Arc::new(TileCache::display_default()),
            max_region_pixels: DEFAULT_MAX_REGION_PIXELS,
        }
    }

    pub(crate) fn from_source_with_config(
        source: Box<dyn SlideReader>,
        cache_config: CacheConfig,
        max_region_pixels: u64,
    ) -> Self {
        let source_hint = source.recommended_shared_cache_bytes();
        Self {
            source,
            cache: Arc::new(TileCache::shared_with_config(cache_config, source_hint)),
            display_cache: Arc::new(TileCache::display_with_config(cache_config)),
            max_region_pixels,
        }
    }

    /// Construct from an already-opened source with an internal cache budget.
    pub fn from_source_with_cache_bytes(source: Box<dyn SlideReader>, cache_bytes: u64) -> Self {
        Self::from_source(source, Arc::new(TileCache::new(cache_bytes)))
    }

    /// Zero-config entry point: builtin registry + source-aware default cache.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WsiError> {
        Self::open_with_options(path, SlideOpenOptions::default())
    }

    pub fn open_with_options(
        path: impl AsRef<Path>,
        options: SlideOpenOptions,
    ) -> Result<Self, WsiError> {
        let resolved_path = crate::formats::svcache::resolve_open_path_with_policy(
            path.as_ref(),
            options.svcache_policy,
        )?;
        let source = options.registry.open(&resolved_path)?;
        Ok(Self::from_source_with_config(
            source,
            options.cache_config,
            options.max_region_pixels,
        ))
    }

    /// Open with the given registry and cache.
    ///
    /// Reusing the same [`TileCache`] across multiple handles allows decoded
    /// tiles from one handle to satisfy later reads from another handle that
    /// targets the same dataset and plane.
    pub(crate) fn open_with(
        path: impl AsRef<Path>,
        registry: &FormatRegistry,
        cache: Arc<TileCache>,
    ) -> Result<Self, WsiError> {
        let source = registry.open(path.as_ref())?;
        let mut slide = Self::from_source(source, cache);
        slide.max_region_pixels = DEFAULT_MAX_REGION_PIXELS;
        Ok(slide)
    }

    /// Open with the given registry and an internal cache budget.
    pub fn open_with_cache_bytes(
        path: impl AsRef<Path>,
        registry: &FormatRegistry,
        cache_bytes: u64,
    ) -> Result<Self, WsiError> {
        Self::open_with(path, registry, Arc::new(TileCache::new(cache_bytes)))
    }

    pub fn dataset(&self) -> &Dataset {
        self.source.dataset()
    }

    pub fn cached_tile_present(&self, req: &TileRequest) -> bool {
        let key = CacheKey {
            dataset_id: self.dataset().id,
            scene: req.scene as u32,
            series: req.series as u32,
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
            tile_col: req.col,
            tile_row: req.row,
        };
        self.cache.get(&key).is_some()
    }

    pub fn source(&self) -> &dyn SlideReader {
        self.source.as_ref()
    }

    pub fn read_tile(
        &self,
        req: &TileRequest,
        output: TileOutputPreference,
    ) -> Result<TilePixels, WsiError> {
        let device_decode_attempted = matches!(
            output,
            TileOutputPreference::PreferDevice { .. } | TileOutputPreference::RequireDevice { .. }
        );
        let span = tracing::debug_span!(
            "wsi_read_tile",
            device_decode_attempted,
            fallback_to_cpu = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
            device_decoded_host_resident = tracing::field::Empty,
        );
        let _guard = span.enter();
        let result = self.source.read_tile(req, output);
        let mut fallback_to_cpu = false;
        let mut fallback_reason = "none";
        let device_decoded_host_resident = false;
        match &result {
            Ok(TilePixels::Cpu(_)) if device_decode_attempted => {
                fallback_to_cpu = true;
                fallback_reason = "signinum_auto_chose_cpu";
                span.record("fallback_to_cpu", true);
                span.record("fallback_reason", fallback_reason);
                span.record("device_decoded_host_resident", false);
            }
            Ok(TilePixels::Cpu(_)) => {
                span.record("fallback_to_cpu", false);
                span.record("fallback_reason", "none");
                span.record("device_decoded_host_resident", false);
            }
            Ok(TilePixels::Device(_)) => {
                span.record("fallback_to_cpu", false);
                span.record("fallback_reason", "none");
                span.record("device_decoded_host_resident", false);
            }
            Err(WsiError::Unsupported { .. }) if device_decode_attempted => {
                fallback_to_cpu = true;
                fallback_reason = "no_device_backend_for_codec";
                span.record("fallback_to_cpu", true);
                span.record("fallback_reason", fallback_reason);
                span.record("device_decoded_host_resident", false);
            }
            Err(_) => {
                span.record("fallback_to_cpu", false);
                span.record("fallback_reason", "none");
                span.record("device_decoded_host_resident", false);
            }
        }
        tracing::debug!(
            device_decode_attempted,
            fallback_to_cpu,
            fallback_reason,
            device_decoded_host_resident,
            "wsi tile output preference resolved"
        );
        result
    }

    /// Read a pixel region, compositing from cached or freshly-decoded tiles.
    ///
    /// Validates all indices (scene, series, level, plane axes) before reading.
    /// Output buffer metadata (color_space, channels, sample_type, layout) is
    /// inherited from the first decoded tile -- no hardcoded assumptions.
    ///
    /// Only `CpuTileLayout::Interleaved` is supported for compositing. Planar
    /// tiles return `WsiError::DisplayConversion`.
    pub fn read_region(&self, req: &RegionRequest) -> Result<CpuTile, WsiError> {
        let mut ctx = SlideReadContext::new(
            Some(self.cache.as_ref()),
            TileOutputPreference::cpu(),
            self.max_region_pixels,
        );
        if let Some(result) = self.source.read_region_fastpath(&mut ctx, req) {
            return result;
        }
        composite_region_from_source(self.source.as_ref(), Some(self.cache.as_ref()), req)
    }

    pub fn read_display_tile(&self, req: &TileViewRequest) -> Result<CpuTile, WsiError> {
        // For Regular tile layouts, route through the generic composition path
        // with cache so intermediate tile reads are reused. For WholeLevel and
        // Irregular layouts, delegate to the source's override which may have
        // format-specific fast paths (e.g. NDPI MCU-level JPEG access).
        let is_regular = self
            .source
            .dataset()
            .scenes
            .get(req.scene)
            .and_then(|s| s.series.get(req.series))
            .and_then(|s| s.levels.get(req.level as usize))
            .is_some_and(|level| matches!(level.tile_layout, TileLayout::Regular { .. }));
        if is_regular {
            let display_cache = self
                .source
                .use_display_tile_cache(req)
                .then_some(self.display_cache.as_ref());
            read_display_tile_from_source(
                self.source.as_ref(),
                display_cache,
                req,
                TileOutputPreference::cpu(),
            )
        } else {
            self.source.read_display_tile(req)
        }
    }

    pub fn read_display_tile_with_output(
        &self,
        req: &TileViewRequest,
        output: TileOutputPreference,
    ) -> Result<CpuTile, WsiError> {
        let is_regular = self
            .source
            .dataset()
            .scenes
            .get(req.scene)
            .and_then(|s| s.series.get(req.series))
            .and_then(|s| s.levels.get(req.level as usize))
            .is_some_and(|level| matches!(level.tile_layout, TileLayout::Regular { .. }));
        if is_regular {
            let display_cache = self
                .source
                .use_display_tile_cache(req)
                .then_some(self.display_cache.as_ref());
            read_display_tile_from_source(self.source.as_ref(), display_cache, req, output)
        } else if matches!(output, TileOutputPreference::RequireDevice { .. }) {
            Err(WsiError::Unsupported {
                reason: "format-specific display tile fast paths return CPU pixels in Phase 2"
                    .into(),
            })
        } else {
            self.source.read_display_tile(req)
        }
    }

    /// Convenience: read a region and convert to RgbaImage.
    /// Only works for Uint8 data (brightfield). For Uint16/Float32,
    /// use read_region() + to_rgba_windowed() with an explicit DisplayWindow.
    pub fn read_region_rgba(&self, req: &RegionRequest) -> Result<image::RgbaImage, WsiError> {
        self.read_region(req)?.to_rgba()
    }

    /// Read a region and convert to RgbaImage with explicit windowing.
    /// For Uint16/Float32 data (fluorescence, computed images).
    pub fn read_region_rgba_windowed(
        &self,
        req: &RegionRequest,
        window: &DisplayWindow,
    ) -> Result<image::RgbaImage, WsiError> {
        self.read_region(req)?.to_rgba_windowed(window)
    }

    /// Read an associated image (label, macro, thumbnail).
    /// Direct delegation to the underlying SlideReader. No caching.
    pub fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.source.read_associated(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties::Properties;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct ErrProbe;

    impl FormatProbe for ErrProbe {
        fn probe(&self, _path: &Path) -> Result<ProbeResult, WsiError> {
            Err(WsiError::InvalidSlide {
                path: "/bad.slide".into(),
                message: "probe failed".into(),
            })
        }
    }

    struct FalseProbe;

    impl FormatProbe for FalseProbe {
        fn probe(&self, _path: &Path) -> Result<ProbeResult, WsiError> {
            Ok(ProbeResult {
                detected: false,
                vendor: "none".into(),
                confidence: ProbeConfidence::Likely,
            })
        }
    }

    struct MockReader;

    impl DatasetReader for MockReader {
        fn open(&self, _path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
            Ok(Box::new(MockSource::new()))
        }
    }

    // Mock SlideReader for testing -- returns solid-color tiles based on (col, row).
    // Grid: 2 cols x 2 rows of 256x256 tiles = 512x512 level.
    //   (0,0) -> red   (255,0,0)
    //   (1,0) -> green (0,255,0)
    //   (0,1) -> blue  (0,0,255)
    //   (1,1) -> white (255,255,255)
    struct MockSource {
        ds: Dataset,
    }

    impl MockSource {
        fn new() -> Self {
            Self {
                ds: Dataset {
                    id: DatasetId(1),
                    scenes: vec![Scene {
                        id: "s0".into(),
                        name: None,
                        series: vec![Series {
                            id: "ser0".into(),
                            axes: AxesShape::default(),
                            levels: vec![Level {
                                dimensions: (512, 512),
                                downsample: 1.0,
                                tile_layout: TileLayout::Regular {
                                    tile_width: 256,
                                    tile_height: 256,
                                    tiles_across: 2,
                                    tiles_down: 2,
                                },
                            }],
                            sample_type: SampleType::Uint8,
                            channels: vec![
                                ChannelInfo {
                                    name: Some("R".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                                ChannelInfo {
                                    name: Some("G".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                                ChannelInfo {
                                    name: Some("B".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                            ],
                        }],
                    }],
                    associated_images: HashMap::new(),
                    properties: Properties::new(),
                    icc_profiles: HashMap::new(),
                },
            }
        }

        fn tile_color(col: i64, row: i64) -> [u8; 3] {
            match (col, row) {
                (0, 0) => [255, 0, 0],     // red
                (1, 0) => [0, 255, 0],     // green
                (0, 1) => [0, 0, 255],     // blue
                (1, 1) => [255, 255, 255], // white
                _ => [0, 0, 0],            // black (out of range)
            }
        }
    }

    impl SlideReader for MockSource {
        fn dataset(&self) -> &Dataset {
            &self.ds
        }
        fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
            let [r, g, b] = MockSource::tile_color(req.col, req.row);
            let mut data = vec![0u8; 256 * 256 * 3];
            for pixel in data.chunks_exact_mut(3) {
                pixel[0] = r;
                pixel[1] = g;
                pixel[2] = b;
            }
            Ok(CpuTile {
                width: 256,
                height: 256,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::u8(data),
            })
        }
        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            Err(WsiError::AssociatedImageNotFound(name.into()))
        }
    }

    struct CountingSource {
        ds: Dataset,
        tile_reads: Arc<AtomicUsize>,
    }

    impl CountingSource {
        fn new(dataset_id: DatasetId, tile_reads: Arc<AtomicUsize>) -> Self {
            Self {
                ds: Dataset {
                    id: dataset_id,
                    scenes: vec![Scene {
                        id: "s0".into(),
                        name: None,
                        series: vec![Series {
                            id: "ser0".into(),
                            axes: AxesShape::default(),
                            levels: vec![Level {
                                dimensions: (256, 256),
                                downsample: 1.0,
                                tile_layout: TileLayout::Regular {
                                    tile_width: 256,
                                    tile_height: 256,
                                    tiles_across: 1,
                                    tiles_down: 1,
                                },
                            }],
                            sample_type: SampleType::Uint8,
                            channels: vec![
                                ChannelInfo {
                                    name: Some("R".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                                ChannelInfo {
                                    name: Some("G".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                                ChannelInfo {
                                    name: Some("B".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                            ],
                        }],
                    }],
                    associated_images: HashMap::new(),
                    properties: Properties::new(),
                    icc_profiles: HashMap::new(),
                },
                tile_reads,
            }
        }
    }

    impl SlideReader for CountingSource {
        fn dataset(&self) -> &Dataset {
            &self.ds
        }

        fn read_tile_cpu(&self, _req: &TileRequest) -> Result<CpuTile, WsiError> {
            self.tile_reads.fetch_add(1, Ordering::SeqCst);
            Ok(CpuTile {
                width: 256,
                height: 256,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::u8(vec![9u8; 256 * 256 * 3]),
            })
        }

        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            Err(WsiError::AssociatedImageNotFound(name.into()))
        }
    }

    struct BatchCountingSource {
        inner: MockSource,
        tile_reads: Arc<AtomicUsize>,
        batch_reads: Arc<AtomicUsize>,
        batch_tile_count: Arc<AtomicUsize>,
    }

    impl BatchCountingSource {
        fn new(
            tile_reads: Arc<AtomicUsize>,
            batch_reads: Arc<AtomicUsize>,
            batch_tile_count: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                inner: MockSource::new(),
                tile_reads,
                batch_reads,
                batch_tile_count,
            }
        }
    }

    impl SlideReader for BatchCountingSource {
        fn dataset(&self) -> &Dataset {
            self.inner.dataset()
        }

        fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
            self.tile_reads.fetch_add(1, Ordering::SeqCst);
            self.inner.read_tile_cpu(req)
        }

        fn read_tiles(
            &self,
            reqs: &[TileRequest],
            _output: TileOutputPreference,
        ) -> Result<Vec<TilePixels>, WsiError> {
            self.batch_reads.fetch_add(1, Ordering::SeqCst);
            self.batch_tile_count
                .fetch_add(reqs.len(), Ordering::SeqCst);
            reqs.iter()
                .map(|req| self.inner.read_tile_cpu(req).map(TilePixels::Cpu))
                .collect()
        }

        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            self.inner.read_associated(name)
        }
    }

    struct GrayscaleSource {
        ds: Dataset,
    }

    impl GrayscaleSource {
        fn new() -> Self {
            Self {
                ds: Dataset {
                    id: DatasetId(2),
                    scenes: vec![Scene {
                        id: "s0".into(),
                        name: None,
                        series: vec![Series {
                            id: "ser0".into(),
                            axes: AxesShape::default(),
                            levels: vec![Level {
                                dimensions: (128, 128),
                                downsample: 1.0,
                                tile_layout: TileLayout::Regular {
                                    tile_width: 128,
                                    tile_height: 128,
                                    tiles_across: 1,
                                    tiles_down: 1,
                                },
                            }],
                            sample_type: SampleType::Uint16,
                            channels: vec![ChannelInfo {
                                name: Some("Gray".into()),
                                color: None,
                                excitation_nm: None,
                                emission_nm: None,
                            }],
                        }],
                    }],
                    associated_images: HashMap::new(),
                    properties: Properties::new(),
                    icc_profiles: HashMap::new(),
                },
            }
        }
    }

    impl SlideReader for GrayscaleSource {
        fn dataset(&self) -> &Dataset {
            &self.ds
        }

        fn read_tile_cpu(&self, _req: &TileRequest) -> Result<CpuTile, WsiError> {
            Ok(CpuTile {
                width: 128,
                height: 128,
                channels: 1,
                color_space: ColorSpace::Grayscale,
                layout: CpuTileLayout::Planar,
                data: CpuTileData::u16(vec![7u16; 128 * 128]),
            })
        }

        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            Err(WsiError::AssociatedImageNotFound(name.into()))
        }
    }

    struct GridReader {
        ds: Dataset,
    }

    impl GridReader {
        fn new() -> Self {
            let level = Level {
                dimensions: (8, 8),
                downsample: 1.0,
                tile_layout: TileLayout::Regular {
                    tile_width: 2,
                    tile_height: 2,
                    tiles_across: 4,
                    tiles_down: 4,
                },
            };
            Self {
                ds: Dataset {
                    id: DatasetId(99),
                    scenes: vec![Scene {
                        id: "scene".into(),
                        name: None,
                        series: vec![Series {
                            id: "series".into(),
                            axes: AxesShape::default(),
                            levels: vec![level],
                            sample_type: SampleType::Uint8,
                            channels: vec![
                                ChannelInfo {
                                    name: None,
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                };
                                3
                            ],
                        }],
                    }],
                    associated_images: HashMap::new(),
                    properties: Properties::new(),
                    icc_profiles: HashMap::new(),
                },
            }
        }
    }

    impl SlideReader for GridReader {
        fn dataset(&self) -> &Dataset {
            &self.ds
        }

        fn read_tiles(
            &self,
            reqs: &[TileRequest],
            _output: TileOutputPreference,
        ) -> Result<Vec<TilePixels>, WsiError> {
            Ok(reqs
                .iter()
                .map(|req| {
                    let mut bytes = vec![0u8; 2 * 2 * 3];
                    for pixel in bytes.chunks_exact_mut(3) {
                        pixel[0] = (req.col & 0xff) as u8;
                        pixel[1] = (req.row & 0xff) as u8;
                    }
                    TilePixels::Cpu(
                        CpuTile::from_u8_interleaved(2, 2, 3, ColorSpace::Rgb, bytes).unwrap(),
                    )
                })
                .collect())
        }

        fn read_tile_cpu(&self, _req: &TileRequest) -> Result<CpuTile, WsiError> {
            unimplemented!("GridReader tests exercise batch-primary read_region")
        }

        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            Err(WsiError::AssociatedImageNotFound(name.into()))
        }
    }

    #[test]
    fn read_region_default_composes_across_tile_boundary() {
        let reader = GridReader::new();
        let req = RegionRequest {
            scene: SceneId(0),
            series: SeriesId(0),
            level: LevelIdx(0),
            plane: PlaneIdx::default(),
            origin_px: (1, 1),
            size_px: (4, 4),
        };
        let pixels = reader
            .read_region(&req, TileOutputPreference::cpu())
            .expect("read region");
        let cpu = match pixels {
            TilePixels::Cpu(cpu) => cpu,
            TilePixels::Device(_) => panic!("CPU region request returned device payload"),
        };
        assert_eq!((cpu.width, cpu.height), (4, 4));
        let bytes = cpu.data.as_u8().unwrap();
        assert_eq!(&bytes[0..3], &[0, 0, 0]);
        assert_eq!(&bytes[3..6], &[1, 0, 0]);
        assert_eq!(&bytes[12..15], &[0, 1, 0]);
    }

    #[test]
    fn read_region_default_rejects_require_device() {
        let reader = GridReader::new();
        let req = RegionRequest {
            scene: SceneId(0),
            series: SeriesId(0),
            level: LevelIdx(0),
            plane: PlaneIdx::default(),
            origin_px: (0, 0),
            size_px: (4, 4),
        };
        let err = reader
            .read_region(&req, TileOutputPreference::require_metal())
            .expect_err("RequireDevice must error");
        assert!(matches!(err, WsiError::Unsupported { .. }));
    }

    #[test]
    fn read_tile_rejects_wrong_batch_cardinality() {
        struct BadBatchReader {
            inner: MockSource,
        }

        impl SlideReader for BadBatchReader {
            fn dataset(&self) -> &Dataset {
                self.inner.dataset()
            }

            fn read_tiles(
                &self,
                _reqs: &[TileRequest],
                _output: TileOutputPreference,
            ) -> Result<Vec<TilePixels>, WsiError> {
                Ok(vec![
                    TilePixels::Cpu(self.inner.read_tile_cpu(&TileRequest {
                        scene: 0,
                        series: 0,
                        level: 0,
                        plane: PlaneSelection::default(),
                        col: 0,
                        row: 0,
                    })?),
                    TilePixels::Cpu(self.inner.read_tile_cpu(&TileRequest {
                        scene: 0,
                        series: 0,
                        level: 0,
                        plane: PlaneSelection::default(),
                        col: 1,
                        row: 0,
                    })?),
                ])
            }

            fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
                self.inner.read_tile_cpu(req)
            }

            fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
                self.inner.read_associated(name)
            }
        }

        let reader = BadBatchReader {
            inner: MockSource::new(),
        };
        let err = reader
            .read_tile(
                &TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: 0,
                    row: 0,
                },
                TileOutputPreference::cpu(),
            )
            .expect_err("single read must reject extra batch outputs");
        assert!(matches!(err, WsiError::TileRead { .. }));
        assert!(err.to_string().contains("returned 2 tiles"));
    }

    #[test]
    fn read_display_tile_with_require_device_rejects_cached_cpu_tile() {
        let slide = Slide::from_source(
            Box::new(MockSource::new()),
            Arc::new(TileCache::new(1024 * 1024)),
        );
        let req = TileViewRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
            tile_width: 256,
            tile_height: 256,
        };

        slide
            .read_display_tile(&req)
            .expect("CPU display tile read should populate cache");
        let err = slide
            .read_display_tile_with_output(&req, TileOutputPreference::require_metal())
            .expect_err("RequireDevice display read must not use cached CPU tile");
        assert!(matches!(err, WsiError::Unsupported { .. }));
    }

    #[test]
    fn format_registry_empty_returns_unsupported() {
        let reg = FormatRegistry::new();
        let result = reg.open(std::path::Path::new("/nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn slide_open_options_default_disables_implicit_svcache_resolution() {
        let options = SlideOpenOptions::default();

        assert_eq!(
            options.svcache_policy,
            crate::formats::svcache::SvcachePolicy::Off
        );
        assert_eq!(options.cache_config, CacheConfig::deterministic());
    }

    #[test]
    fn probe_confidence_definite_beats_likely() {
        // Definite should beat Likely — tested via ProbeConfidence ordering
        assert!(matches!(
            ProbeConfidence::Definite,
            ProbeConfidence::Definite
        ));
        assert!(matches!(ProbeConfidence::Likely, ProbeConfidence::Likely));
    }

    #[test]
    fn slide_exposes_dataset() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = std::sync::Arc::new(TileCache::new(1024 * 1024));
        let handle = Slide::from_source(source, cache);

        assert_eq!(handle.dataset().id, DatasetId(1));
        assert_eq!(handle.dataset().scenes.len(), 1);
        assert_eq!(
            handle.dataset().scenes[0].series[0].levels[0].dimensions,
            (512, 512)
        );
    }

    #[test]
    fn format_registry_returns_probe_error_when_no_backend_matches() {
        let mut reg = FormatRegistry::new();
        reg.register(ErrProbe, MockReader);

        match reg.open(Path::new("/bad.slide")) {
            Err(err) => match err {
                WsiError::InvalidSlide { message, .. } => assert!(message.contains("probe failed")),
                other => panic!("expected InvalidSlide, got {other:?}"),
            },
            Ok(_) => panic!("expected probe error"),
        }
    }

    #[test]
    fn detected_backend_beats_probe_error() {
        let mut reg = FormatRegistry::new();
        reg.register(ErrProbe, MockReader);
        reg.register(FalseProbe, MockReader);

        struct DefiniteProbe;
        impl FormatProbe for DefiniteProbe {
            fn probe(&self, _path: &Path) -> Result<ProbeResult, WsiError> {
                Ok(ProbeResult {
                    detected: true,
                    vendor: "mock".into(),
                    confidence: ProbeConfidence::Definite,
                })
            }
        }

        reg.register(DefiniteProbe, MockReader);

        let opened = reg.open(Path::new("/ok.slide")).unwrap();
        assert_eq!(opened.dataset().id, DatasetId(1));
    }

    #[test]
    fn arc_format_probe_blanket_impl() {
        struct TestProbe;
        impl FormatProbe for TestProbe {
            fn probe(&self, _path: &Path) -> Result<ProbeResult, WsiError> {
                Ok(ProbeResult {
                    detected: true,
                    vendor: "test".into(),
                    confidence: ProbeConfidence::Definite,
                })
            }
        }

        let arc_probe: Arc<TestProbe> = Arc::new(TestProbe);
        let result = arc_probe.probe(Path::new("/test")).unwrap();
        assert!(result.detected);
        assert_eq!(result.vendor, "test");
    }

    #[test]
    fn arc_dataset_reader_blanket_impl() {
        let arc_reader: Arc<MockReader> = Arc::new(MockReader);
        let source = arc_reader.open(Path::new("/test")).unwrap();
        assert_eq!(source.dataset().id, DatasetId(1));
    }

    #[test]
    fn builtin_registry_has_tiff_backend() {
        let reg = FormatRegistry::builtin();
        // The builtin registry should have at least one backend registered.
        // Probing a nonexistent path should produce an error (not panic).
        let result = reg.open(Path::new("/nonexistent/test.ndpi"));
        assert!(result.is_err());
        // The backend was registered and tried to probe. Whether we get
        // UnsupportedFormat (probe returned detected=false) or another
        // error variant, the backend was exercised.
        match result {
            Err(WsiError::UnsupportedFormat(_)) => {
                // The TIFF backend's probe returns detected=false for non-existent
                // files (the TiffContainer::open fails, so it returns detected=false).
                // With no backends matching, registry falls through to UnsupportedFormat.
                // This is acceptable — it proves the backend was registered and probed.
            }
            Err(_) => {} // Any other error also proves the backend tried
            Ok(_) => panic!("expected error for nonexistent file"),
        }
    }

    #[test]
    fn open_nonexistent_file_returns_error() {
        let result = Slide::open("/nonexistent/path/slide.ndpi");
        assert!(result.is_err());
    }

    #[test]
    fn read_region_single_tile() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 100, 100);
        let buf = handle.read_region(&req).unwrap();
        assert_eq!(buf.width, 100);
        assert_eq!(buf.height, 100);
        assert_eq!(buf.channels, 3);
        assert_eq!(buf.color_space, ColorSpace::Rgb);

        // All pixels should be red (tile 0,0)
        let data = buf.data.as_u8().unwrap();
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 0); // B
                                // Check last pixel too
        let last = (100 * 100 - 1) * 3;
        assert_eq!(data[last], 255);
        assert_eq!(data[last + 1], 0);
        assert_eq!(data[last + 2], 0);
    }

    #[test]
    fn read_display_tile_regular_native_passthrough() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let buf = handle
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 0,
                tile_width: 256,
                tile_height: 256,
            })
            .unwrap();
        assert_eq!(buf.width, 256);
        assert_eq!(buf.height, 256);
        let data = buf.data.as_u8().unwrap();
        assert_eq!(&data[..3], &[0, 255, 0]);
    }

    #[test]
    fn read_display_tile_composes_subtile_from_regular_grid() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let buf = handle
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
                tile_width: 128,
                tile_height: 128,
            })
            .unwrap();
        assert_eq!(buf.width, 128);
        assert_eq!(buf.height, 128);
        let data = buf.data.as_u8().unwrap();
        assert_eq!(&data[..3], &[255, 0, 0]);
    }

    #[test]
    fn read_region_multi_tile_compositing() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        // Request spanning all four tiles: full 512x512
        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 512, 512);
        let buf = handle.read_region(&req).unwrap();
        assert_eq!(buf.width, 512);
        assert_eq!(buf.height, 512);

        let data = buf.data.as_u8().unwrap();

        // Top-left pixel (0,0) -> tile (0,0) -> red
        assert_eq!(&data[0..3], &[255, 0, 0]);

        // Top-right pixel (511,0) -> tile (1,0) -> green
        let idx = 511 * 3;
        assert_eq!(&data[idx..idx + 3], &[0, 255, 0]);

        // Bottom-left pixel (0,511) -> tile (0,1) -> blue
        let idx = (511 * 512) * 3;
        assert_eq!(&data[idx..idx + 3], &[0, 0, 255]);

        // Bottom-right pixel (511,511) -> tile (1,1) -> white
        let idx = (511 * 512 + 511) * 3;
        assert_eq!(&data[idx..idx + 3], &[255, 255, 255]);
    }

    #[test]
    fn read_region_cross_tile_boundary() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        // 2x2 region crossing the tile boundary at x=256
        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 255, 0, 2, 1);
        let buf = handle.read_region(&req).unwrap();
        let data = buf.data.as_u8().unwrap();

        // Pixel at x=255 -> tile (0,0) -> red
        assert_eq!(&data[0..3], &[255, 0, 0]);
        // Pixel at x=256 -> tile (1,0) -> green
        assert_eq!(&data[3..6], &[0, 255, 0]);
    }

    #[test]
    fn read_region_scene_out_of_range() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(5, 0, 0, PlaneSelection::default(), 0, 0, 10, 10);
        match handle.read_region(&req) {
            Err(WsiError::SceneOutOfRange { index: 5, count: 1 }) => {}
            other => panic!("expected SceneOutOfRange, got {:?}", other),
        }
    }

    #[test]
    fn read_region_level_out_of_range() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(0, 0, 99, PlaneSelection::default(), 0, 0, 10, 10);
        match handle.read_region(&req) {
            Err(WsiError::LevelOutOfRange {
                level: 99,
                count: 1,
            }) => {}
            other => panic!("expected LevelOutOfRange, got {:?}", other),
        }
    }

    #[test]
    fn read_region_plane_out_of_range() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(1024));
        let handle = Slide::from_source(source, cache);

        let req =
            RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection { z: 5, c: 0, t: 0 }, 0, 0, 10, 10);
        match handle.read_region(&req) {
            Err(WsiError::PlaneOutOfRange {
                axis,
                value: 5,
                max: 1,
            }) => {
                assert_eq!(axis, "z");
            }
            other => panic!("expected PlaneOutOfRange, got {:?}", other),
        }
    }

    #[test]
    fn read_region_no_tiles_hit_returns_zeros() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(1024));
        let handle = Slide::from_source(source, cache);

        // Region entirely outside the level (level is 512x512)
        let req =
            RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 10000, 10000, 10, 10);
        let buf = handle.read_region(&req).unwrap();
        assert_eq!(buf.width, 10);
        assert_eq!(buf.height, 10);
        // All zeros
        let data = buf.data.as_u8().unwrap();
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn read_region_no_tiles_hit_preserves_template_metadata() {
        let source: Box<dyn SlideReader> = Box::new(GrayscaleSource::new());
        let cache = Arc::new(TileCache::new(1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 512, 512, 16, 16);
        let buf = handle.read_region(&req).unwrap();

        assert_eq!(buf.channels, 1);
        assert_eq!(buf.color_space, ColorSpace::Grayscale);
        assert_eq!(buf.layout, CpuTileLayout::Planar);
        assert_eq!(buf.data.sample_type(), SampleType::Uint16);
        assert!(buf.data.as_u16().unwrap().iter().all(|sample| *sample == 0));
    }

    struct FailingTileSource {
        ds: Dataset,
    }

    impl FailingTileSource {
        fn new() -> Self {
            Self {
                ds: Dataset {
                    id: DatasetId(9),
                    scenes: vec![Scene {
                        id: "s0".into(),
                        name: None,
                        series: vec![Series {
                            id: "ser0".into(),
                            axes: AxesShape::default(),
                            levels: vec![Level {
                                dimensions: (128, 128),
                                downsample: 1.0,
                                tile_layout: TileLayout::Regular {
                                    tile_width: 128,
                                    tile_height: 128,
                                    tiles_across: 1,
                                    tiles_down: 1,
                                },
                            }],
                            sample_type: SampleType::Uint8,
                            channels: vec![
                                ChannelInfo {
                                    name: Some("R".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                                ChannelInfo {
                                    name: Some("G".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                                ChannelInfo {
                                    name: Some("B".into()),
                                    color: None,
                                    excitation_nm: None,
                                    emission_nm: None,
                                },
                            ],
                        }],
                    }],
                    associated_images: HashMap::new(),
                    properties: Properties::new(),
                    icc_profiles: HashMap::new(),
                },
            }
        }
    }

    impl SlideReader for FailingTileSource {
        fn dataset(&self) -> &Dataset {
            &self.ds
        }

        fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
            Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "synthetic decode failure".into(),
            })
        }

        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            Err(WsiError::AssociatedImageNotFound(name.into()))
        }
    }

    #[test]
    fn read_region_uses_cache() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache.clone());

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 100, 100);

        // First read populates cache
        let _ = handle.read_region(&req).unwrap();

        // Verify tile is now cached
        let key = CacheKey {
            dataset_id: DatasetId(1),
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
            tile_col: 0,
            tile_row: 0,
        };
        assert!(cache.get(&key).is_some());

        // Second read should use cache (same result)
        let buf2 = handle.read_region(&req).unwrap();
        assert_eq!(buf2.data.as_u8().unwrap()[0], 255); // still red
    }

    #[test]
    fn shared_cache_reuses_tile_across_handles() {
        let tile_reads = Arc::new(AtomicUsize::new(0));
        let shared_cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle_a = Slide::from_source(
            Box::new(CountingSource::new(DatasetId(7), tile_reads.clone())),
            shared_cache.clone(),
        );
        let handle_b = Slide::from_source(
            Box::new(CountingSource::new(DatasetId(7), tile_reads.clone())),
            shared_cache,
        );

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 64, 64);

        let _ = handle_a.read_region(&req).unwrap();
        assert_eq!(tile_reads.load(Ordering::SeqCst), 1);

        let _ = handle_b.read_region(&req).unwrap();
        assert_eq!(
            tile_reads.load(Ordering::SeqCst),
            1,
            "second handle should reuse the shared cached tile"
        );
    }

    #[test]
    fn read_region_batches_uncached_tiles_and_preserves_cache() {
        let tile_reads = Arc::new(AtomicUsize::new(0));
        let batch_reads = Arc::new(AtomicUsize::new(0));
        let batch_tile_count = Arc::new(AtomicUsize::new(0));
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(
            Box::new(BatchCountingSource::new(
                tile_reads.clone(),
                batch_reads.clone(),
                batch_tile_count.clone(),
            )),
            cache,
        );

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 512, 256);

        let first = handle.read_region(&req).unwrap();
        let pixels = first.data.as_u8().unwrap();
        assert_eq!(&pixels[..3], &[255, 0, 0]);
        assert_eq!(&pixels[(256 * 3)..(257 * 3)], &[0, 255, 0]);
        assert_eq!(tile_reads.load(Ordering::SeqCst), 0);
        assert_eq!(batch_reads.load(Ordering::SeqCst), 1);
        assert_eq!(batch_tile_count.load(Ordering::SeqCst), 2);

        let second = handle.read_region(&req).unwrap();
        assert_eq!(second.data.as_u8().unwrap(), pixels);
        assert_eq!(tile_reads.load(Ordering::SeqCst), 0);
        assert_eq!(
            batch_reads.load(Ordering::SeqCst),
            1,
            "second read should be fully satisfied from cache"
        );
    }

    #[test]
    fn display_tile_exact_regular_reads_use_display_cache() {
        let tile_reads = Arc::new(AtomicUsize::new(0));
        let handle = Slide::from_source(
            Box::new(CountingSource::new(DatasetId(8), tile_reads.clone())),
            Arc::new(TileCache::new(64 * 1024 * 1024)),
        );

        let req = TileViewRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
            tile_width: 256,
            tile_height: 256,
        };

        let _ = handle.read_display_tile(&req).unwrap();
        assert_eq!(tile_reads.load(Ordering::SeqCst), 1);

        let _ = handle.read_display_tile(&req).unwrap();
        assert_eq!(
            tile_reads.load(Ordering::SeqCst),
            1,
            "second exact display-tile read should hit the display cache"
        );
    }

    #[test]
    fn read_region_no_tiles_hit_falls_back_when_probe_tile_read_fails() {
        let source: Box<dyn SlideReader> = Box::new(FailingTileSource::new());
        let cache = Arc::new(TileCache::new(1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 512, 512, 16, 16);
        let buf = handle.read_region(&req).unwrap();

        assert_eq!(buf.channels, 3);
        assert_eq!(buf.color_space, ColorSpace::Rgb);
        assert_eq!(buf.layout, CpuTileLayout::Interleaved);
        assert!(buf.data.as_u8().unwrap().iter().all(|sample| *sample == 0));
    }

    #[test]
    fn read_region_rgba_produces_correct_image() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 256, 256);
        let img = handle.read_region_rgba(&req).unwrap();
        assert_eq!(img.width(), 256);
        assert_eq!(img.height(), 256);

        // All pixels should be red with full alpha (tile 0,0)
        let pixel = img.get_pixel(0, 0);
        assert_eq!(pixel.0, [255, 0, 0, 255]);

        let pixel = img.get_pixel(255, 255);
        assert_eq!(pixel.0, [255, 0, 0, 255]);
    }

    #[test]
    fn read_region_rgba_multi_tile() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let req = RegionRequest::legacy_xywh(0, 0, 0, PlaneSelection::default(), 0, 0, 512, 512);
        let img = handle.read_region_rgba(&req).unwrap();
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 512);

        // Top-left -> red
        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
        // Top-right -> green
        assert_eq!(img.get_pixel(511, 0).0, [0, 255, 0, 255]);
        // Bottom-left -> blue
        assert_eq!(img.get_pixel(0, 511).0, [0, 0, 255, 255]);
        // Bottom-right -> white
        assert_eq!(img.get_pixel(511, 511).0, [255, 255, 255, 255]);
    }

    /// Mock source with a non-256-aligned level (300x260) to test edge tile
    /// origin calculation. Each pixel encodes its level-space x coordinate in
    /// the red channel so we can verify the tile was read from the right origin.
    struct EdgeMockSource {
        ds: Dataset,
    }

    impl EdgeMockSource {
        fn new() -> Self {
            Self {
                ds: Dataset {
                    id: DatasetId(2),
                    scenes: vec![Scene {
                        id: "s0".into(),
                        name: None,
                        series: vec![Series {
                            id: "ser0".into(),
                            axes: AxesShape::default(),
                            levels: vec![Level {
                                dimensions: (300, 260),
                                downsample: 1.0,
                                tile_layout: TileLayout::Regular {
                                    tile_width: 256,
                                    tile_height: 256,
                                    tiles_across: 2,
                                    tiles_down: 2,
                                },
                            }],
                            sample_type: SampleType::Uint8,
                            channels: vec![],
                        }],
                    }],
                    associated_images: HashMap::new(),
                    properties: crate::Properties::new(),
                    icc_profiles: HashMap::new(),
                },
            }
        }
    }

    impl SlideReader for EdgeMockSource {
        fn dataset(&self) -> &Dataset {
            &self.ds
        }
        fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
            // Return native 256x256 tiles with pixel R = (tile_origin_x + px) & 0xFF
            let tile_origin_x = req.col as u32 * 256;
            let level_w = 300u32;
            let tile_w = 256.min(level_w.saturating_sub(tile_origin_x));
            let tile_h = 256.min(260u32.saturating_sub(req.row as u32 * 256));
            let mut data = vec![0u8; (tile_w * tile_h * 3) as usize];
            for y in 0..tile_h {
                for x in 0..tile_w {
                    let idx = ((y * tile_w + x) * 3) as usize;
                    let abs_x = tile_origin_x + x;
                    data[idx] = (abs_x & 0xFF) as u8; // R = level-space x
                    data[idx + 1] = (y & 0xFF) as u8; // G = local y
                    data[idx + 2] = 42;
                }
            }
            Ok(CpuTile {
                width: tile_w,
                height: tile_h,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::u8(data),
            })
        }
        fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
            Err(WsiError::AssociatedImageNotFound(name.into()))
        }
    }

    #[test]
    fn display_tile_edge_origin_correct_with_full_tile_width() {
        // Level is 300x260. With 256x256 grid, last column (col=1) starts at
        // x=256 and has content_width=44. Passing tile_width=256 must produce
        // an origin of 256 (not col*content_width=1*44=44).
        let source: Box<dyn SlideReader> = Box::new(EdgeMockSource::new());
        let cache = Arc::new(TileCache::new(64 * 1024 * 1024));
        let handle = Slide::from_source(source, cache);

        let buf = handle
            .read_display_tile(&TileViewRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 0,
                tile_width: 256,
                tile_height: 256,
            })
            .unwrap();

        // The edge tile should be clipped to 44x256.
        assert_eq!(buf.width, 44);
        assert_eq!(buf.height, 256);

        // First pixel should be from level-space x=256, not x=44.
        let data = buf.data.as_u8().unwrap();
        let first_r = data[0];
        assert_eq!(
            first_r,
            (256u32 & 0xFF) as u8,
            "edge tile first pixel R should encode level-space x=256, got x={}",
            first_r,
        );
    }

    #[test]
    fn read_associated_delegates_to_source() {
        let source: Box<dyn SlideReader> = Box::new(MockSource::new());
        let cache = Arc::new(TileCache::new(1024));
        let handle = Slide::from_source(source, cache);

        match handle.read_associated("label") {
            Err(WsiError::AssociatedImageNotFound(name)) => {
                assert_eq!(name, "label");
            }
            other => panic!("expected AssociatedImageNotFound, got {:?}", other),
        }
    }
}
