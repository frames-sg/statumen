use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use czi_rs::{
    AttachmentBlob, CompressionMode as CziCompressionMode, CziFile, Dimension as CziDimension,
    IntRect, PixelType as CziPixelType,
};
use image::imageops::{self, FilterType};
use lru::LruCache;
use signinum_core::BackendRequest;
use std::collections::HashMap as StdHashMap;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    crop_rgb_interleaved_u8_buffer, DatasetReader, FormatProbe, ProbeConfidence, ProbeResult,
    SlideReader,
};
use crate::core::types::*;
use crate::decode::jpeg::{decode_batch_jpeg, JpegDecodeJob};
use crate::error::WsiError;
use crate::properties::Properties;

const FILE_MAGIC: &[u8; 16] = b"ZISRAWFILE\0\0\0\0\0\0";
const DEFAULT_TILE_PX: u32 = 256;
const ASSOCIATED_JPEG_PROBE_BYTES: u64 = 256 << 10;
type LevelImageCache = Mutex<LruCache<(usize, usize), Arc<CpuTile>>>;
type LocalTileCache = Mutex<LruCache<(usize, usize, i64, i64), Arc<CpuTile>>>;

static TEMP_BLOB_COUNTER: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static ZEISS_LOCAL_TILE_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static ZEISS_DIRECT_LEVEL_COMPOSE_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS: AtomicU64 = AtomicU64::new(0);

pub(crate) struct ZeissBackend {
    probe_cache: Mutex<LruCache<PathBuf, Arc<ZeissSlide>>>,
}

impl ZeissBackend {
    pub(crate) fn new() -> Self {
        Self {
            probe_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(8).unwrap())),
        }
    }

    fn cache_key(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    fn parse(&self, path: &Path) -> Result<Arc<ZeissSlide>, WsiError> {
        Ok(Arc::new(ZeissSlide::parse(path)?))
    }
}

impl Default for ZeissBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatProbe for ZeissBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        let key = Self::cache_key(path);
        if self
            .probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .is_some()
        {
            return Ok(ProbeResult {
                detected: true,
                vendor: "zeiss".into(),
                confidence: ProbeConfidence::Definite,
            });
        }

        let mut magic = [0u8; 16];
        let mut file = match fs::File::open(path) {
            Ok(file) => file,
            Err(_) => {
                return Ok(ProbeResult {
                    detected: false,
                    vendor: String::new(),
                    confidence: ProbeConfidence::Likely,
                });
            }
        };
        if std::io::Read::read_exact(&mut file, &mut magic).is_err() || &magic != FILE_MAGIC {
            return Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            });
        }

        Ok(ProbeResult {
            detected: true,
            vendor: "zeiss".into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for ZeissBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let key = Self::cache_key(path);
        let cached = self
            .probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned();
        let slide = match cached {
            Some(slide) => slide,
            None => {
                let slide = self.parse(path)?;
                self.probe_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .put(key, slide.clone());
                slide
            }
        };
        Ok(Box::new(ZeissReader { slide }))
    }
}

struct ZeissReader {
    slide: Arc<ZeissSlide>,
}

impl SlideReader for ZeissReader {
    fn dataset(&self) -> &Dataset {
        &self.slide.dataset
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let backend = (match output {
            TileOutputPreference::Cpu { backend }
            | TileOutputPreference::PreferDevice { backend, .. } => backend,
            TileOutputPreference::RequireDevice { .. } => {
                return Err(WsiError::Unsupported {
                    reason: "RequireDevice not supported for Zeiss in Phase 2".into(),
                });
            }
        })
        .to_signinum();
        reqs.iter()
            .map(|req| {
                self.read_tile_with_backend(req, backend)
                    .map(TilePixels::Cpu)
            })
            .collect()
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.read_tile_with_backend(req, BackendRequest::Auto)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.slide.read_associated(name)
    }
}

impl ZeissReader {
    fn read_tile_with_backend(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        self.slide
            .read_tile(req.scene, req.series, req.level, req.col, req.row, backend)
    }
}

struct ZeissSlide {
    dataset: Dataset,
    czi: Mutex<CziFile>,
    level_cache: LevelImageCache,
    tile_cache: LocalTileCache,
    associated_cache: Mutex<LruCache<String, Arc<CpuTile>>>,
    associated_sources: HashMap<String, czi_rs::AttachmentInfo>,
    subblock_origin: (i32, i32),
    canvas_level_subblocks: Vec<Vec<usize>>,
    canvas_level_tile_subblocks: Vec<StdHashMap<(i64, i64), Vec<usize>>>,
}

impl ZeissSlide {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let mut czi = CziFile::open(path)
            .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;

        let header = czi.file_header().clone();
        let xml = czi
            .metadata_xml()
            .map_err(|source| WsiError::DisplayConversion(source.to_string()))?
            .to_string();
        let summary = czi
            .metadata()
            .map_err(|source| WsiError::DisplayConversion(source.to_string()))?
            .clone();
        let statistics = czi.statistics().clone();
        let attachments = czi.attachments().to_vec();
        let subblocks = czi.subblocks().to_vec();

        let scene_indices = scene_indices(&statistics, &summary);
        if scene_indices.is_empty() {
            return Err(invalid_slide(path, "Zeiss slide has no scenes"));
        }

        let level_ratios = common_level_ratios(&subblocks, &scene_indices, &statistics);
        let canvas_origin = canvas_origin(&statistics);
        let subblock_origin = subblock_origin(&subblocks);
        let canvas_dimensions = canvas_dimensions(&statistics, &summary, path)?;
        let levels = build_levels(canvas_dimensions, &level_ratios);
        let mut canvas_level_subblocks = vec![Vec::new(); level_ratios.len()];
        for subblock in &subblocks {
            if !subblock_matches_default_plane(subblock, &statistics) {
                continue;
            }
            let Some(level_ratio) = subblock_ratio(subblock) else {
                continue;
            };
            let Some(level_slot) = level_ratios.iter().position(|ratio| *ratio == level_ratio)
            else {
                continue;
            };
            canvas_level_subblocks[level_slot].push(subblock.index);
        }
        let canvas_level_tile_subblocks = build_canvas_level_tile_subblocks(
            &subblocks,
            &canvas_level_subblocks,
            &levels,
            subblock_origin,
        );
        let scenes = vec![Scene {
            id: "scene_0".to_string(),
            name: Some("Canvas".to_string()),
            series: vec![Series {
                id: "series_0".to_string(),
                axes: AxesShape::default(),
                levels,
                sample_type: SampleType::Uint8,
                channels: build_channels(&summary),
            }],
        }];

        let quickhash = quickhash_for_zeiss(&header, &xml)?;
        let dataset_id = dataset_id_from_quickhash(path, &quickhash)?;

        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "zeiss");
        properties.insert("openslide.quickhash-1", quickhash.clone());
        if let Some(v) = summary.document.user_name {
            properties.insert("zeiss.document.user_name", v);
        }
        if let Some(v) = summary.document.creation_date {
            properties.insert("zeiss.document.creation_date", v);
        }
        if let Some(v) = summary.document.application_name {
            properties.insert("zeiss.document.application_name", v);
        }
        if let Some(v) = summary.document.application_version {
            properties.insert("zeiss.document.application_version", v);
        }
        if let Some(v) = summary.image.pixel_type {
            properties.insert("zeiss.image.pixel_type", v.as_str());
        }
        if let Some(x) = summary.image.sizes.get(&CziDimension::X) {
            properties.insert("zeiss.image.size_x", x.to_string());
        }
        if let Some(y) = summary.image.sizes.get(&CziDimension::Y) {
            properties.insert("zeiss.image.size_y", y.to_string());
        }
        if let Some(s) = summary.image.sizes.get(&CziDimension::S) {
            properties.insert("zeiss.image.size_s", s.to_string());
        }
        if let Some(x) = summary.scaling.x {
            let mpp_x = x * 1_000_000.0;
            properties.insert("openslide.mpp-x", format!("{mpp_x:.6}"));
            properties.insert("zeiss.scaling.x", x.to_string());
        }
        if let Some(y) = summary.scaling.y {
            let mpp_y = y * 1_000_000.0;
            properties.insert("openslide.mpp-y", format!("{mpp_y:.6}"));
            properties.insert("zeiss.scaling.y", y.to_string());
        }
        if let Some(objective) = extract_objective_magnification(&xml) {
            properties.insert("openslide.objective-power", objective);
        }

        for (idx, scene_index) in scene_indices.iter().enumerate() {
            if let Some(bounding_boxes) =
                statistics.scene_bounding_boxes.get(&(*scene_index as i32))
            {
                let region = if bounding_boxes.layer0.is_valid() {
                    bounding_boxes.layer0
                } else {
                    bounding_boxes.all
                };
                if region.is_valid() {
                    properties.insert(
                        format!("openslide.region[{idx}].x"),
                        (region.x - canvas_origin.0).to_string(),
                    );
                    properties.insert(
                        format!("openslide.region[{idx}].y"),
                        (region.y - canvas_origin.1).to_string(),
                    );
                    properties.insert(
                        format!("openslide.region[{idx}].width"),
                        region.w.to_string(),
                    );
                    properties.insert(
                        format!("openslide.region[{idx}].height"),
                        region.h.to_string(),
                    );
                }
            }
        }

        let mut associated_images = HashMap::new();
        let mut associated_sources = HashMap::new();
        for attachment in &attachments {
            let Some(name) = associated_name(&attachment.name) else {
                continue;
            };
            if let Some(metadata) = probe_associated_attachment(path, &mut czi, attachment)? {
                associated_images.insert(name.to_string(), metadata);
                associated_sources.insert(name.to_string(), attachment.clone());
            }
        }

        let dataset = Dataset {
            id: dataset_id,
            scenes,
            associated_images,
            properties,
            icc_profiles: HashMap::new(),
        };

        Ok(Self {
            dataset,
            czi: Mutex::new(czi),
            level_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(1).unwrap())),
            tile_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(8).unwrap())),
            associated_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(4).unwrap())),
            associated_sources,
            subblock_origin,
            canvas_level_subblocks,
            canvas_level_tile_subblocks,
        })
    }

    fn read_tile(
        &self,
        scene: usize,
        series: usize,
        level: u32,
        col: i64,
        row: i64,
        _backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let series_ref = self
            .dataset
            .scenes
            .get(scene)
            .and_then(|scene| scene.series.get(series))
            .ok_or(WsiError::SceneOutOfRange {
                index: scene,
                count: self.dataset.scenes.len(),
            })?;
        let level_ref = series_ref
            .levels
            .get(level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level,
                count: series_ref.levels.len() as u32,
            })?;
        let TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } = level_ref.tile_layout
        else {
            return Err(WsiError::UnsupportedFormat(
                "Zeiss levels must use regular tiles".into(),
            ));
        };
        if col < 0 || row < 0 || col >= tiles_across as i64 || row >= tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    tiles_across, tiles_down
                ),
            });
        }

        let key = (scene, level as usize, col, row);
        if let Some(cached) = self
            .tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(cached.as_ref().clone());
        }

        let buffer =
            if let Some(buffer) = self.scene_tile_image_local(scene, level as usize, col, row)? {
                #[cfg(test)]
                ZEISS_LOCAL_TILE_HITS.fetch_add(1, Ordering::Relaxed);
                buffer
            } else {
                let level_img = self.scene_level_image(scene, level as usize)?;
                let x = (col as u32).saturating_mul(tile_width);
                let y = (row as u32).saturating_mul(tile_height);
                let w = tile_width.min(level_img.width.saturating_sub(x));
                let h = tile_height.min(level_img.height.saturating_sub(y));
                crop_rgb_interleaved_u8_buffer(level_img.as_ref(), x, y, w, h)?
            };
        let arc = Arc::new(buffer);
        self.tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, arc.clone());
        Ok(arc.as_ref().clone())
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        if let Some(cached) = self
            .associated_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
        {
            return Ok(cached.as_ref().clone());
        }

        let attachment = self
            .associated_sources
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
        let buffer = {
            let mut czi = self.czi.lock().unwrap_or_else(|e| e.into_inner());
            let (_, buffer) = decode_associated_attachment(&mut czi, attachment)?
                .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
            buffer
        };
        let arc = Arc::new(buffer);
        self.associated_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(name.to_string(), arc.clone());
        Ok(arc.as_ref().clone())
    }

    fn scene_tile_image_local(
        &self,
        scene: usize,
        level: usize,
        col: i64,
        row: i64,
    ) -> Result<Option<CpuTile>, WsiError> {
        let (_tile_width, _tile_height, tile_x, tile_y, tile_w, tile_h) = {
            let series = &self.dataset.scenes[scene].series[0];
            let level_ref = &series.levels[level];
            let TileLayout::Regular {
                tile_width,
                tile_height,
                ..
            } = level_ref.tile_layout
            else {
                return Ok(None);
            };
            let tile_x = (col as u64).saturating_mul(u64::from(tile_width));
            let tile_y = (row as u64).saturating_mul(u64::from(tile_height));
            let tile_w = u32::try_from(
                level_ref
                    .dimensions
                    .0
                    .saturating_sub(tile_x)
                    .min(u64::from(tile_width)),
            )
            .map_err(|_| WsiError::DisplayConversion("Zeiss tile width overflow".into()))?;
            let tile_h = u32::try_from(
                level_ref
                    .dimensions
                    .1
                    .saturating_sub(tile_y)
                    .min(u64::from(tile_height)),
            )
            .map_err(|_| WsiError::DisplayConversion("Zeiss tile height overflow".into()))?;
            (tile_width, tile_height, tile_x, tile_y, tile_w, tile_h)
        };
        let candidate_indices = self
            .canvas_level_tile_subblocks
            .get(level)
            .and_then(|tiles| tiles.get(&(col, row)).cloned())
            .unwrap_or_default();
        if candidate_indices.is_empty() {
            return Ok(Some(CpuTile::new(
                tile_w,
                tile_h,
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(vec![0; tile_w as usize * tile_h as usize * 3]),
            )?));
        }
        let _level_ratio = self.dataset.scenes[scene].series[0].levels[level]
            .downsample
            .round()
            .max(1.0) as i32;
        let tile_origin_x = i32::try_from(tile_x)
            .map_err(|_| WsiError::DisplayConversion("Zeiss tile x overflow".into()))?;
        let tile_origin_y = i32::try_from(tile_y)
            .map_err(|_| WsiError::DisplayConversion("Zeiss tile y overflow".into()))?;

        let candidate_infos = {
            let czi = self.czi.lock().unwrap_or_else(|e| e.into_inner());
            let all = czi.subblocks();
            let mut selected = Vec::new();
            for index in candidate_indices {
                let info = all.get(index).cloned().ok_or_else(|| {
                    WsiError::DisplayConversion(format!(
                        "Zeiss subblock index {index} out of range"
                    ))
                })?;
                if info.compression != CziCompressionMode::UnCompressed {
                    #[cfg(test)]
                    eprintln!(
                        "zeiss local tile: unsupported compression {:?} for subblock {index}",
                        info.compression
                    );
                    return Ok(None);
                }
                selected.push(info);
            }
            selected
        };
        let tile_rect = IntRect::new(
            tile_origin_x,
            tile_origin_y,
            i32::try_from(tile_w)
                .map_err(|_| WsiError::DisplayConversion("Zeiss tile width overflow".into()))?,
            i32::try_from(tile_h)
                .map_err(|_| WsiError::DisplayConversion("Zeiss tile height overflow".into()))?,
        );
        let subblocks: Vec<_> = candidate_infos
            .iter()
            .filter(|&info| {
                let global_rect = IntRect::new(
                    (info.rect.x - self.subblock_origin.0).div_euclid(_level_ratio),
                    (info.rect.y - self.subblock_origin.1).div_euclid(_level_ratio),
                    i32::try_from(info.stored_size.w).unwrap_or(i32::MAX),
                    i32::try_from(info.stored_size.h).unwrap_or(i32::MAX),
                );
                global_rect.intersect(tile_rect).is_some()
            })
            .cloned()
            .collect();
        if subblocks.is_empty() {
            #[cfg(test)]
            eprintln!(
                "zeiss local tile fallback: no subblocks intersect tile ({}, {}) level {}",
                tile_origin_x, tile_origin_y, level
            );
            let pixel_type = candidate_infos
                .first()
                .map(|info| info.pixel_type)
                .ok_or_else(|| {
                    WsiError::DisplayConversion(
                        "Zeiss local tile path lost candidate pixel type".into(),
                    )
                })?;
            return czi_rs::Bitmap::zeros(pixel_type, tile_w, tile_h)
                .map_err(|source| WsiError::DisplayConversion(source.to_string()))
                .and_then(bitmap_to_sample_buffer)
                .map(Some);
        }

        let direct_uncompressed_rgb = subblocks
            .iter()
            .all(|info| matches!(info.pixel_type, CziPixelType::Bgr24 | CziPixelType::Bgra32));
        let mut czi = self.czi.lock().unwrap_or_else(|e| e.into_inner());
        if direct_uncompressed_rgb {
            let mut destination = vec![0u8; tile_w as usize * tile_h as usize * 3];
            for info in subblocks {
                let raw = czi
                    .read_subblock(info.index)
                    .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;
                blit_raw_uncompressed_rgb_subblock(
                    &mut destination,
                    tile_w,
                    tile_h,
                    &raw,
                    (info.rect.x - self.subblock_origin.0).div_euclid(_level_ratio) - tile_origin_x,
                    (info.rect.y - self.subblock_origin.1).div_euclid(_level_ratio) - tile_origin_y,
                )?;
            }
            #[cfg(test)]
            ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.fetch_add(1, Ordering::Relaxed);
            return CpuTile::new(
                tile_w,
                tile_h,
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(destination),
            )
            .map(Some);
        }

        let mut destination = vec![0u8; tile_w as usize * tile_h as usize * 3];
        for info in subblocks {
            let raw = czi
                .read_subblock(info.index)
                .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;
            let bitmap = bitmap_from_raw_uncompressed_subblock(&raw)?;
            let sample = bitmap_to_sample_buffer(bitmap)?;
            let sample_data = sample.data.as_u8().ok_or_else(|| {
                WsiError::DisplayConversion(
                    "Zeiss local tile path requires 8-bit RGB-compatible subblocks".into(),
                )
            })?;
            let blit_x =
                (info.rect.x - self.subblock_origin.0).div_euclid(_level_ratio) - tile_origin_x;
            let blit_y =
                (info.rect.y - self.subblock_origin.1).div_euclid(_level_ratio) - tile_origin_y;
            blit_rgb_sample(
                &mut destination,
                (tile_w, tile_h),
                RgbSample {
                    width: sample.width,
                    height: sample.height,
                    data: sample_data,
                },
                (blit_x, blit_y),
            )?;
        }

        CpuTile::new(
            tile_w,
            tile_h,
            3,
            ColorSpace::Rgb,
            CpuTileLayout::Interleaved,
            CpuTileData::u8(destination),
        )
        .map(Some)
    }

    fn scene_level_image(&self, scene: usize, level: usize) -> Result<Arc<CpuTile>, WsiError> {
        if let Some(cached) = self
            .level_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(scene, level))
            .cloned()
        {
            return Ok(cached);
        }

        let series = &self.dataset.scenes[scene].series[0];
        let level_ref = &series.levels[level];
        let buffer = if let Some(buffer) = self.scene_level_image_from_subblocks(scene, level)? {
            #[cfg(test)]
            ZEISS_DIRECT_LEVEL_COMPOSE_HITS.fetch_add(1, Ordering::Relaxed);
            buffer
        } else if level == 0 {
            return Err(WsiError::UnsupportedFormat(
                "Zeiss level 0 requires direct subblock composition".into(),
            ));
        } else {
            let base = self.scene_level_image(scene, 0)?;
            let rgb = base.as_ref().clone().into_rgb()?;
            let resized = imageops::resize(
                &rgb,
                level_ref.dimensions.0 as u32,
                level_ref.dimensions.1 as u32,
                FilterType::Triangle,
            );
            CpuTile::new(
                resized.width(),
                resized.height(),
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(resized.into_raw()),
            )?
        };
        let arc = Arc::new(buffer);
        self.level_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put((scene, level), arc.clone());
        Ok(arc)
    }

    fn scene_level_image_from_subblocks(
        &self,
        scene: usize,
        level: usize,
    ) -> Result<Option<CpuTile>, WsiError> {
        let candidate_indices = self
            .canvas_level_subblocks
            .get(level)
            .cloned()
            .unwrap_or_default();
        if candidate_indices.is_empty() {
            return Ok(None);
        }

        let candidate_infos = {
            let czi = self.czi.lock().unwrap_or_else(|e| e.into_inner());
            let all = czi.subblocks();
            let mut selected = Vec::with_capacity(candidate_indices.len());
            for index in candidate_indices {
                let info = all.get(index).cloned().ok_or_else(|| {
                    WsiError::DisplayConversion(format!(
                        "Zeiss subblock index {index} out of range"
                    ))
                })?;
                if info.compression != CziCompressionMode::UnCompressed {
                    return Ok(None);
                }
                selected.push(info);
            }
            selected
        };

        if candidate_infos.is_empty() {
            return Ok(None);
        }

        let series = &self.dataset.scenes[scene].series[0];
        let level_ref = &series.levels[level];

        let mut subblocks = candidate_infos;
        subblocks.sort_by_key(|info| (info.m_index.unwrap_or(i32::MIN), info.file_position));

        let direct_uncompressed_rgb = subblocks
            .iter()
            .all(|info| matches!(info.pixel_type, CziPixelType::Bgr24 | CziPixelType::Bgra32));
        let level_ratio = level_ref.downsample.round().max(1.0) as i32;
        if direct_uncompressed_rgb {
            let mut czi = self.czi.lock().unwrap_or_else(|e| e.into_inner());
            let mut destination =
                vec![0u8; level_ref.dimensions.0 as usize * level_ref.dimensions.1 as usize * 3];
            for info in subblocks {
                let raw = czi
                    .read_subblock(info.index)
                    .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;
                blit_raw_uncompressed_rgb_subblock(
                    &mut destination,
                    level_ref.dimensions.0 as u32,
                    level_ref.dimensions.1 as u32,
                    &raw,
                    (info.rect.x - self.subblock_origin.0).div_euclid(level_ratio),
                    (info.rect.y - self.subblock_origin.1).div_euclid(level_ratio),
                )?;
            }
            #[cfg(test)]
            ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.fetch_add(1, Ordering::Relaxed);
            return CpuTile::new(
                level_ref.dimensions.0 as u32,
                level_ref.dimensions.1 as u32,
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(destination),
            )
            .map(Some);
        }

        let mut destination: Option<czi_rs::Bitmap> = None;
        for info in subblocks {
            let raw = {
                let mut czi = self.czi.lock().unwrap_or_else(|e| e.into_inner());
                czi.read_subblock(info.index)
                    .map_err(|source| WsiError::DisplayConversion(source.to_string()))?
            };
            let bitmap = bitmap_from_raw_uncompressed_subblock(&raw)?;
            let blit_x = (info.rect.x - self.subblock_origin.0).div_euclid(level_ratio);
            let blit_y = (info.rect.y - self.subblock_origin.1).div_euclid(level_ratio);
            match destination.as_mut() {
                Some(destination_bitmap) => {
                    if destination_bitmap.pixel_type != bitmap.pixel_type {
                        return Ok(None);
                    }
                    blit_tile(destination_bitmap, &bitmap, blit_x, blit_y)?;
                }
                None => {
                    let mut destination_bitmap = czi_rs::Bitmap::zeros(
                        bitmap.pixel_type,
                        level_ref.dimensions.0 as u32,
                        level_ref.dimensions.1 as u32,
                    )
                    .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;
                    blit_tile(&mut destination_bitmap, &bitmap, blit_x, blit_y)?;
                    destination = Some(destination_bitmap);
                }
            }
        }

        destination.map(bitmap_to_sample_buffer).transpose()
    }
}

fn associated_name(name: &str) -> Option<&'static str> {
    match name {
        "Label" => Some("label"),
        "SlidePreview" => Some("macro"),
        "Thumbnail" => Some("thumbnail"),
        _ => None,
    }
}

fn build_channels(summary: &czi_rs::MetadataSummary) -> Vec<ChannelInfo> {
    summary
        .channels
        .first()
        .map(|channel| ChannelInfo {
            name: channel.name.clone(),
            color: channel.color.as_deref().and_then(parse_channel_color),
            excitation_nm: None,
            emission_nm: None,
        })
        .into_iter()
        .collect()
}

fn parse_channel_color(value: &str) -> Option<[u8; 3]> {
    let trimmed = value.trim().trim_start_matches('#');
    if trimmed.len() == 8 {
        let r = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
        let g = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
        let b = u8::from_str_radix(&trimmed[6..8], 16).ok()?;
        Some([r, g, b])
    } else if trimmed.len() == 6 {
        let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
        let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
        let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
        Some([r, g, b])
    } else {
        None
    }
}

fn scene_indices(
    statistics: &czi_rs::SubBlockStatistics,
    summary: &czi_rs::MetadataSummary,
) -> Vec<usize> {
    let mut indices: BTreeSet<usize> = statistics
        .scene_bounding_boxes
        .keys()
        .filter_map(|scene| (*scene >= 0).then_some(*scene as usize))
        .collect();
    if indices.is_empty() {
        let count = summary
            .image
            .sizes
            .get(&CziDimension::S)
            .copied()
            .unwrap_or(1);
        indices.extend(0..count);
    }
    indices.into_iter().collect()
}

fn scene_slot_for_subblock(
    scene_coords: &[usize],
    subblock: &czi_rs::DirectorySubBlockInfo,
) -> Option<usize> {
    match subblock.coordinate.get(CziDimension::S) {
        Some(scene) if scene >= 0 => scene_coords
            .iter()
            .position(|candidate| *candidate == scene as usize),
        Some(_) => None,
        None if scene_coords.len() == 1 => Some(0),
        None => None,
    }
}

fn subblock_matches_default_plane(
    _subblock: &czi_rs::DirectorySubBlockInfo,
    _statistics: &czi_rs::SubBlockStatistics,
) -> bool {
    true
}

fn canvas_dimensions(
    statistics: &czi_rs::SubBlockStatistics,
    summary: &czi_rs::MetadataSummary,
    path: &Path,
) -> Result<(u64, u64), WsiError> {
    let w = summary
        .image
        .sizes
        .get(&CziDimension::X)
        .copied()
        .unwrap_or(0) as u64;
    let h = summary
        .image
        .sizes
        .get(&CziDimension::Y)
        .copied()
        .unwrap_or(0) as u64;
    if w > 0 && h > 0 {
        return Ok((w, h));
    }

    let mut max_x = 0i64;
    let mut max_y = 0i64;
    let mut min_x = 0i64;
    let mut min_y = 0i64;
    let mut seen = false;
    for bounding_boxes in statistics.scene_bounding_boxes.values() {
        let rect = if bounding_boxes.layer0.is_valid() {
            bounding_boxes.layer0
        } else {
            bounding_boxes.all
        };
        if rect.is_valid() {
            if !seen {
                min_x = i64::from(rect.x);
                min_y = i64::from(rect.y);
                seen = true;
            } else {
                min_x = min_x.min(i64::from(rect.x));
                min_y = min_y.min(i64::from(rect.y));
            }
            max_x = max_x.max(i64::from(rect.x) + i64::from(rect.w));
            max_y = max_y.max(i64::from(rect.y) + i64::from(rect.h));
        }
    }

    if seen && max_x > 0 && max_y > 0 {
        return Ok(((max_x - min_x) as u64, (max_y - min_y) as u64));
    }

    Err(invalid_slide(path, "missing Zeiss canvas dimensions"))
}

fn canvas_origin(statistics: &czi_rs::SubBlockStatistics) -> (i32, i32) {
    let mut min_x = 0i32;
    let mut min_y = 0i32;
    let mut seen = false;
    for bounding_boxes in statistics.scene_bounding_boxes.values() {
        let rect = if bounding_boxes.layer0.is_valid() {
            bounding_boxes.layer0
        } else {
            bounding_boxes.all
        };
        if rect.is_valid() {
            if !seen {
                min_x = rect.x;
                min_y = rect.y;
                seen = true;
            } else {
                min_x = min_x.min(rect.x);
                min_y = min_y.min(rect.y);
            }
        }
    }
    (min_x, min_y)
}

fn subblock_origin(subblocks: &[czi_rs::DirectorySubBlockInfo]) -> (i32, i32) {
    let min_x = subblocks.iter().map(|info| info.rect.x).min().unwrap_or(0);
    let min_y = subblocks.iter().map(|info| info.rect.y).min().unwrap_or(0);
    (min_x, min_y)
}

fn common_level_ratios(
    subblocks: &[czi_rs::DirectorySubBlockInfo],
    scene_indices: &[usize],
    statistics: &czi_rs::SubBlockStatistics,
) -> Vec<u32> {
    let mut scene_sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); scene_indices.len()];
    for sb in subblocks {
        let Some(scene_slot) = scene_slot_for_subblock(scene_indices, sb) else {
            continue;
        };
        if !subblock_matches_default_plane(sb, statistics) {
            continue;
        }
        if let Some(ratio) = subblock_ratio(sb) {
            scene_sets[scene_slot].insert(ratio);
        }
    }

    let mut common: Option<BTreeSet<u32>> = None;
    for set in scene_sets.into_iter().filter(|set| !set.is_empty()) {
        common = Some(match common {
            Some(prev) => prev.intersection(&set).copied().collect(),
            None => set,
        });
    }

    let mut ratios = common.unwrap_or_else(|| {
        let mut fallback = BTreeSet::new();
        fallback.insert(1);
        fallback
    });
    ratios.insert(1);
    ratios.into_iter().collect()
}

fn build_canvas_level_tile_subblocks(
    subblocks: &[czi_rs::DirectorySubBlockInfo],
    canvas_level_subblocks: &[Vec<usize>],
    levels: &[Level],
    subblock_origin: (i32, i32),
) -> Vec<StdHashMap<(i64, i64), Vec<usize>>> {
    let mut out = vec![StdHashMap::<(i64, i64), Vec<usize>>::new(); levels.len()];
    for (level_idx, indices) in canvas_level_subblocks.iter().enumerate() {
        let Some(level) = levels.get(level_idx) else {
            continue;
        };
        let TileLayout::Regular {
            tile_width,
            tile_height,
            ..
        } = level.tile_layout
        else {
            continue;
        };
        let level_ratio = level.downsample.round().max(1.0) as i64;
        let tile_w = i64::from(tile_width);
        let tile_h = i64::from(tile_height);
        for &index in indices {
            let Some(info) = subblocks.get(index) else {
                continue;
            };
            let x = i64::from(info.rect.x - subblock_origin.0).div_euclid(level_ratio);
            let y = i64::from(info.rect.y - subblock_origin.1).div_euclid(level_ratio);
            let w = i64::from(info.stored_size.w);
            let h = i64::from(info.stored_size.h);
            if w <= 0 || h <= 0 {
                continue;
            }
            let start_col = x.div_euclid(tile_w);
            let end_col = (x + w - 1).div_euclid(tile_w);
            let start_row = y.div_euclid(tile_h);
            let end_row = (y + h - 1).div_euclid(tile_h);
            let map = &mut out[level_idx];
            for col in start_col..=end_col {
                for row in start_row..=end_row {
                    map.entry((col, row)).or_default().push(index);
                }
            }
        }
    }
    out
}

fn subblock_ratio(subblock: &czi_rs::DirectorySubBlockInfo) -> Option<u32> {
    if subblock.rect.w <= 0
        || subblock.rect.h <= 0
        || subblock.stored_size.w == 0
        || subblock.stored_size.h == 0
    {
        return None;
    }
    let width_ratio = ((subblock.rect.w as f64) / (subblock.stored_size.w as f64)).round() as u32;
    let height_ratio = ((subblock.rect.h as f64) / (subblock.stored_size.h as f64)).round() as u32;
    (width_ratio > 0 && width_ratio == height_ratio).then_some(width_ratio)
}

fn build_levels((width, height): (u64, u64), ratios: &[u32]) -> Vec<Level> {
    ratios
        .iter()
        .map(|&ratio| {
            let ratio_u64 = u64::from(ratio);
            let level_w = width / ratio_u64;
            let level_h = height / ratio_u64;
            let tile_width = DEFAULT_TILE_PX;
            let tile_height = DEFAULT_TILE_PX;
            let tiles_across = level_w.div_ceil(u64::from(tile_width));
            let tiles_down = level_h.div_ceil(u64::from(tile_height));
            Level {
                dimensions: (level_w, level_h),
                downsample: ratio as f64,
                tile_layout: TileLayout::Regular {
                    tile_width,
                    tile_height,
                    tiles_across,
                    tiles_down,
                },
            }
        })
        .collect()
}

fn quickhash_for_zeiss(header: &czi_rs::FileHeaderInfo, xml: &str) -> Result<String, WsiError> {
    let mut quickhash = Quickhash1::new();
    quickhash.update(&guid_bytes(&header.primary_file_guid)?);
    quickhash.update(&guid_bytes(&header.file_guid)?);
    quickhash.hash_string(xml);
    quickhash
        .finish()
        .ok_or_else(|| WsiError::DisplayConversion("failed to compute Zeiss quickhash".into()))
}

fn dataset_id_from_quickhash(path: &Path, quickhash: &str) -> Result<DatasetId, WsiError> {
    if quickhash.len() < 32 {
        return Err(invalid_slide(path, "quickhash too short"));
    }
    let value = u128::from_str_radix(&quickhash[..32], 16)
        .map_err(|_| invalid_slide(path, "quickhash is not valid hex"))?;
    Ok(DatasetId(value))
}

fn extract_objective_magnification(xml: &str) -> Option<String> {
    let ref_id = extract_attribute(xml, "ObjectiveRef", "Id")?;
    let objective_marker = format!(r#"<Objective Id="{}""#, ref_id);
    let objective_start = xml.find(&objective_marker)?;
    let objective_xml = &xml[objective_start..];
    extract_tag_text(objective_xml, "NominalMagnification")
}

fn extract_attribute(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let start = xml.find(&format!("<{tag} "))?;
    let rest = &xml[start..];
    let needle = format!(r#"{attr}=""#);
    let attr_start = rest.find(&needle)? + needle.len();
    let attr_end = rest[attr_start..].find('"')? + attr_start;
    Some(rest[attr_start..attr_end].to_string())
}

fn extract_tag_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    let value = xml[start..end].trim();
    (!value.is_empty()).then_some(value.to_string())
}

fn blit_tile(
    destination: &mut czi_rs::Bitmap,
    source: &czi_rs::Bitmap,
    offset_x: i32,
    offset_y: i32,
) -> Result<(), WsiError> {
    if destination.pixel_type != source.pixel_type {
        return Err(WsiError::DisplayConversion(
            "cannot compose Zeiss tiles with mismatched pixel types".into(),
        ));
    }

    let source_rect = IntRect::new(
        offset_x,
        offset_y,
        source.width as i32,
        source.height as i32,
    );
    let destination_rect = IntRect::new(0, 0, destination.width as i32, destination.height as i32);
    let Some(intersection) = source_rect.intersect(destination_rect) else {
        return Ok(());
    };

    let bytes_per_pixel = destination.pixel_type.bytes_per_pixel();
    for row in 0..intersection.h as usize {
        let src_x = (intersection.x - offset_x) as usize;
        let src_y = (intersection.y - offset_y) as usize + row;
        let dst_x = intersection.x as usize;
        let dst_y = intersection.y as usize + row;
        let row_bytes = intersection.w as usize * bytes_per_pixel;

        let src_offset = src_y
            .checked_mul(source.stride)
            .and_then(|value| value.checked_add(src_x * bytes_per_pixel))
            .ok_or_else(|| {
                WsiError::DisplayConversion("Zeiss source tile offset overflow".into())
            })?;
        let dst_offset = dst_y
            .checked_mul(destination.stride)
            .and_then(|value| value.checked_add(dst_x * bytes_per_pixel))
            .ok_or_else(|| {
                WsiError::DisplayConversion("Zeiss destination tile offset overflow".into())
            })?;

        destination.data[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&source.data[src_offset..src_offset + row_bytes]);
    }

    Ok(())
}

struct RgbSample<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

fn blit_rgb_sample(
    destination: &mut [u8],
    dest_size: (u32, u32),
    source: RgbSample<'_>,
    offset: (i32, i32),
) -> Result<(), WsiError> {
    let (dest_width, dest_height) = dest_size;
    let (offset_x, offset_y) = offset;
    let source_rect = IntRect::new(
        offset_x,
        offset_y,
        source.width as i32,
        source.height as i32,
    );
    let destination_rect = IntRect::new(0, 0, dest_width as i32, dest_height as i32);
    let Some(intersection) = source_rect.intersect(destination_rect) else {
        return Ok(());
    };

    let src_stride = source.width as usize * 3;
    let dest_stride = dest_width as usize * 3;
    for row in 0..intersection.h as usize {
        let src_x = (intersection.x - offset_x) as usize;
        let src_y = (intersection.y - offset_y) as usize + row;
        let dst_x = intersection.x as usize;
        let dst_y = intersection.y as usize + row;
        let row_bytes = intersection.w as usize * 3;

        let src_offset = src_y
            .checked_mul(src_stride)
            .and_then(|value| value.checked_add(src_x * 3))
            .ok_or_else(|| {
                WsiError::DisplayConversion("Zeiss source RGB tile offset overflow".into())
            })?;
        let dst_offset = dst_y
            .checked_mul(dest_stride)
            .and_then(|value| value.checked_add(dst_x * 3))
            .ok_or_else(|| {
                WsiError::DisplayConversion("Zeiss destination RGB tile offset overflow".into())
            })?;
        destination[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&source.data[src_offset..src_offset + row_bytes]);
    }

    Ok(())
}

fn blit_raw_uncompressed_rgb_subblock(
    destination: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    raw: &czi_rs::RawSubBlock,
    offset_x: i32,
    offset_y: i32,
) -> Result<(), WsiError> {
    let source_width = raw.info.stored_size.w;
    let source_height = raw.info.stored_size.h;
    let source_rect = IntRect::new(
        offset_x,
        offset_y,
        source_width as i32,
        source_height as i32,
    );
    let destination_rect = IntRect::new(0, 0, dest_width as i32, dest_height as i32);
    let Some(intersection) = source_rect.intersect(destination_rect) else {
        return Ok(());
    };

    let source_bytes = raw.data.as_slice();
    let source_stride = source_width as usize
        * match raw.info.pixel_type {
            CziPixelType::Bgr24 => 3,
            CziPixelType::Bgra32 => 4,
            other => {
                return Err(WsiError::DisplayConversion(format!(
                    "unsupported Zeiss direct blit pixel type {other:?}"
                )));
            }
        };
    let dest_stride = dest_width as usize * 3;
    let bytes_per_pixel = source_stride / source_width as usize;
    let source_needed = source_stride * source_height as usize;
    if source_bytes.len() < source_needed {
        return Err(WsiError::DisplayConversion(
            "Zeiss raw subblock shorter than expected".into(),
        ));
    }

    for row in 0..intersection.h as usize {
        let src_x = (intersection.x - offset_x) as usize;
        let src_y = (intersection.y - offset_y) as usize + row;
        let dst_x = intersection.x as usize;
        let dst_y = intersection.y as usize + row;
        let src_offset = src_y
            .checked_mul(source_stride)
            .and_then(|value| value.checked_add(src_x * bytes_per_pixel))
            .ok_or_else(|| {
                WsiError::DisplayConversion("Zeiss raw source offset overflow".into())
            })?;
        let dst_offset = dst_y
            .checked_mul(dest_stride)
            .and_then(|value| value.checked_add(dst_x * 3))
            .ok_or_else(|| {
                WsiError::DisplayConversion("Zeiss raw destination offset overflow".into())
            })?;
        match raw.info.pixel_type {
            CziPixelType::Bgr24 => {
                let src_row = &source_bytes[src_offset..src_offset + intersection.w as usize * 3];
                let dst_row =
                    &mut destination[dst_offset..dst_offset + intersection.w as usize * 3];
                for (src_px, dst_px) in src_row.chunks_exact(3).zip(dst_row.chunks_exact_mut(3)) {
                    dst_px[0] = src_px[2];
                    dst_px[1] = src_px[1];
                    dst_px[2] = src_px[0];
                }
            }
            CziPixelType::Bgra32 => {
                let src_row = &source_bytes[src_offset..src_offset + intersection.w as usize * 4];
                let dst_row =
                    &mut destination[dst_offset..dst_offset + intersection.w as usize * 3];
                for (src_px, dst_px) in src_row.chunks_exact(4).zip(dst_row.chunks_exact_mut(3)) {
                    dst_px[0] = src_px[2];
                    dst_px[1] = src_px[1];
                    dst_px[2] = src_px[0];
                }
            }
            other => {
                return Err(WsiError::DisplayConversion(format!(
                    "unsupported Zeiss direct blit pixel type {other:?}"
                )));
            }
        }
    }

    Ok(())
}

fn decode_associated_attachment(
    czi: &mut CziFile,
    attachment: &czi_rs::AttachmentInfo,
) -> Result<Option<(AssociatedImage, CpuTile)>, WsiError> {
    let blob: AttachmentBlob = czi
        .read_attachment(attachment.index)
        .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;

    if attachment.content_file_type.eq_ignore_ascii_case("JPG") {
        let buffer = decode_batch_jpeg(&[JpegDecodeJob {
            data: Cow::Borrowed(&blob.data),
            tables: None,
            expected_width: 0,
            expected_height: 0,
            color_transform: signinum_jpeg::ColorTransform::Auto,
            force_dimensions: false,
            requested_size: None,
        }])
        .into_iter()
        .next()
        .expect("1-element JPEG facade batch")?;
        return Ok(Some((
            AssociatedImage {
                dimensions: (buffer.width, buffer.height),
                sample_type: SampleType::Uint8,
                channels: 3,
            },
            buffer,
        )));
    }

    if attachment.content_file_type.eq_ignore_ascii_case("CZI") {
        let temp_path = temp_czi_path(attachment.index);
        fs::write(&temp_path, &blob.data).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: temp_path.clone(),
        })?;
        let result = (|| {
            let mut embedded = CziFile::open(&temp_path)
                .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;
            let bitmap = embedded
                .read_frame_2d(0, 0, 0, 0)
                .map_err(|source| WsiError::DisplayConversion(source.to_string()))?;
            let buffer = bitmap_to_sample_buffer(bitmap)?;
            Ok::<_, WsiError>((
                AssociatedImage {
                    dimensions: (buffer.width, buffer.height),
                    sample_type: buffer.data.sample_type(),
                    channels: buffer.channels,
                },
                buffer,
            ))
        })();
        let _ = fs::remove_file(&temp_path);
        return result.map(Some);
    }

    Ok(None)
}

fn probe_associated_attachment(
    path: &Path,
    czi: &mut CziFile,
    attachment: &czi_rs::AttachmentInfo,
) -> Result<Option<AssociatedImage>, WsiError> {
    if attachment.content_file_type.eq_ignore_ascii_case("JPG") {
        if let Ok(bytes) = read_attachment_prefix(path, attachment, ASSOCIATED_JPEG_PROBE_BYTES) {
            if let Ok((width, height)) = crate::decode::jpeg::jpeg_dimensions(&bytes) {
                return Ok(Some(AssociatedImage {
                    dimensions: (width, height),
                    sample_type: SampleType::Uint8,
                    channels: 3,
                }));
            }
        }
    }

    Ok(decode_associated_attachment(czi, attachment)?.map(|(metadata, _buffer)| metadata))
}

fn read_attachment_prefix(
    path: &Path,
    attachment: &czi_rs::AttachmentInfo,
    max_bytes: u64,
) -> Result<Vec<u8>, WsiError> {
    let payload_offset = attachment
        .file_position
        .checked_add(32 + 256)
        .ok_or_else(|| WsiError::DisplayConversion("Zeiss attachment offset overflow".into()))?;
    let read_len = attachment.data_size.min(max_bytes);
    let read_len_usize = usize::try_from(read_len).map_err(|_| {
        WsiError::DisplayConversion("Zeiss attachment probe length overflow".into())
    })?;
    let mut file = File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    file.seek(SeekFrom::Start(payload_offset))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let mut buffer = vec![0u8; read_len_usize];
    file.read_exact(&mut buffer)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    Ok(buffer)
}

fn temp_czi_path(index: usize) -> PathBuf {
    let counter = TEMP_BLOB_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "statumen-zeiss-{}-{}-{}.czi",
        std::process::id(),
        index,
        counter
    ))
}

fn guid_bytes(value: &str) -> Result<[u8; 16], WsiError> {
    let parts: Vec<_> = value.split('-').collect();
    if parts.len() != 5
        || parts[0].len() != 8
        || parts[1].len() != 4
        || parts[2].len() != 4
        || parts[3].len() != 4
        || parts[4].len() != 12
    {
        return Err(WsiError::DisplayConversion(format!(
            "unexpected Zeiss GUID format: {value}"
        )));
    }

    fn parse_hex_pair(value: &str, start: usize) -> Result<u8, WsiError> {
        u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| WsiError::DisplayConversion(format!("invalid GUID hex: {value}")))
    }

    let mut bytes = [0u8; 16];

    // CZI stores GUIDs with the first three fields little-endian in-file, and
    // Compatibility hashing uses those raw bytes directly.
    for (idx, start) in [6, 4, 2, 0].into_iter().enumerate() {
        bytes[idx] = parse_hex_pair(parts[0], start)?;
    }
    for (idx, start) in [2, 0].into_iter().enumerate() {
        bytes[4 + idx] = parse_hex_pair(parts[1], start)?;
        bytes[6 + idx] = parse_hex_pair(parts[2], start)?;
    }
    for (idx, start) in [0, 2].into_iter().enumerate() {
        bytes[8 + idx] = parse_hex_pair(parts[3], start)?;
    }
    for idx in 0..6 {
        bytes[10 + idx] = parse_hex_pair(parts[4], idx * 2)?;
    }
    Ok(bytes)
}

fn bitmap_to_sample_buffer(bitmap: czi_rs::Bitmap) -> Result<CpuTile, WsiError> {
    match bitmap.pixel_type {
        CziPixelType::Bgr24 => {
            let mut rgb = Vec::with_capacity(bitmap.data.len());
            for chunk in bitmap.data.chunks_exact(3) {
                rgb.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
            }
            CpuTile::new(
                bitmap.width,
                bitmap.height,
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(rgb),
            )
        }
        CziPixelType::Bgra32 => {
            let mut rgb =
                Vec::with_capacity((bitmap.width as usize) * (bitmap.height as usize) * 3);
            for chunk in bitmap.data.chunks_exact(4) {
                rgb.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
            }
            CpuTile::new(
                bitmap.width,
                bitmap.height,
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(rgb),
            )
        }
        CziPixelType::Bgr48 => {
            let values = bitmap
                .to_u16_vec()
                .map_err(|err| WsiError::DisplayConversion(err.to_string()))?;
            let mut rgb = Vec::with_capacity(values.len());
            for chunk in values.chunks_exact(3) {
                rgb.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
            }
            CpuTile::new(
                bitmap.width,
                bitmap.height,
                3,
                ColorSpace::Rgb,
                CpuTileLayout::Interleaved,
                CpuTileData::u16(rgb),
            )
        }
        other => Err(WsiError::DisplayConversion(format!(
            "unsupported Zeiss pixel type {other:?}"
        ))),
    }
}

fn bitmap_from_raw_uncompressed_subblock(
    raw: &czi_rs::RawSubBlock,
) -> Result<czi_rs::Bitmap, WsiError> {
    if raw.info.compression != CziCompressionMode::UnCompressed {
        return Err(WsiError::DisplayConversion(format!(
            "unsupported Zeiss compression {}",
            raw.info.compression.as_str()
        )));
    }
    let expected_len = (raw.info.stored_size.w as usize)
        .checked_mul(raw.info.stored_size.h as usize)
        .and_then(|value| value.checked_mul(raw.info.pixel_type.bytes_per_pixel()))
        .ok_or_else(|| WsiError::DisplayConversion("Zeiss bitmap size overflow".into()))?;
    let mut decoded = raw.data.clone();
    if decoded.len() < expected_len {
        decoded.resize(expected_len, 0);
    } else {
        decoded.truncate(expected_len);
    }
    czi_rs::Bitmap::new(
        raw.info.pixel_type,
        raw.info.stored_size.w,
        raw.info.stored_size.h,
        decoded,
    )
    .map_err(|source| WsiError::DisplayConversion(source.to_string()))
}

fn invalid_slide(path: &Path, message: impl Into<String>) -> WsiError {
    WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static ZEISS_TEST_GUARD: Mutex<()> = Mutex::new(());

    fn zeiss_uncompressed_fixture() -> Option<PathBuf> {
        if let Some(path) = env::var_os("STATUMEN_ZEISS_CZI_PATH").map(PathBuf::from) {
            return path.is_file().then_some(path);
        }

        let local = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("SlideViewer")
            .join("downloads")
            .join("openslide-testdata")
            .join("Zeiss")
            .join("Zeiss-5-Uncompressed.czi");
        local.is_file().then_some(local)
    }

    fn zeiss_fixture_or_skip() -> Option<PathBuf> {
        let path = zeiss_uncompressed_fixture();
        if path.is_none() {
            eprintln!("[zeiss] skipping: set STATUMEN_ZEISS_CZI_PATH to Zeiss-5-Uncompressed.czi");
        }
        path
    }

    #[test]
    fn uncompressed_sentinel_hits_local_tile_path() {
        let _guard = ZEISS_TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        ZEISS_LOCAL_TILE_HITS.store(0, Ordering::Relaxed);
        ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.store(0, Ordering::Relaxed);
        let Some(path) = zeiss_fixture_or_skip() else {
            return;
        };
        let handle = crate::core::registry::Slide::open(&path).expect("open Zeiss sentinel");
        assert_eq!(handle.dataset().scenes.len(), 1);
        assert_eq!(handle.dataset().scenes[0].series.len(), 1);
        assert_eq!(handle.dataset().scenes[0].series[0].levels.len(), 5);
        assert_eq!(
            handle.dataset().scenes[0].series[0].levels[0].dimensions,
            (50171, 11340)
        );
        assert_eq!(
            handle.dataset().properties.get("openslide.region[0].x"),
            Some("0")
        );
        assert_eq!(
            handle.dataset().properties.get("openslide.region[0].y"),
            Some("2")
        );
        assert_eq!(
            handle.dataset().properties.get("openslide.region[1].x"),
            Some("38866")
        );
        assert_eq!(
            handle.dataset().properties.get("openslide.region[1].y"),
            Some("0")
        );
        let req = crate::core::types::TileViewRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
            tile_width: 256,
            tile_height: 256,
        };
        let _ = handle
            .read_display_tile(&req)
            .expect("read Zeiss display tile");
        assert!(ZEISS_LOCAL_TILE_HITS.load(Ordering::Relaxed) > 0);
        assert!(ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn uncompressed_sentinel_pan_trace_l0_reads_successfully() {
        let _guard = ZEISS_TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let Some(path) = zeiss_fixture_or_skip() else {
            return;
        };
        let handle = crate::core::registry::Slide::open(&path).expect("open Zeiss sentinel");
        let dims = handle.dataset().scenes[0].series[0].levels[0].dimensions;
        let tile_px = 256i64;
        let center = ((dims.0 / 2) as i64, (dims.1 / 2) as i64);
        let coords: Vec<(i64, i64)> = (0..256)
            .map(|i| {
                let delta = (i as i64 - 128) * tile_px;
                (center.0 + delta, center.1 + delta)
            })
            .filter(|&(x, y)| {
                x >= 0 && y >= 0 && x + tile_px <= dims.0 as i64 && y + tile_px <= dims.1 as i64
            })
            .collect();

        assert!(!coords.is_empty(), "expected pan_trace_l0 coordinates");
        for &(x, y) in &coords {
            let req = crate::core::types::TileViewRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: crate::core::types::PlaneSelection::default(),
                col: x.div_euclid(tile_px),
                row: y.div_euclid(tile_px),
                tile_width: tile_px as u32,
                tile_height: tile_px as u32,
            };
            let _ = handle
                .read_display_tile(&req)
                .expect("read Zeiss pan trace tile");
        }
    }

    #[test]
    fn uncompressed_sentinel_gap_tile_is_blank() {
        let _guard = ZEISS_TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let Some(path) = zeiss_fixture_or_skip() else {
            return;
        };
        let handle = crate::core::registry::Slide::open(&path).expect("open Zeiss sentinel");
        let req = crate::core::types::TileViewRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 37,
            row: 37,
            tile_width: 256,
            tile_height: 256,
        };
        let tile = handle.read_display_tile(&req).expect("read Zeiss gap tile");
        assert!(
            tile.data.as_u8().unwrap().iter().all(|&byte| byte == 0),
            "expected the no-intersection tile to be blank"
        );
    }

    #[test]
    fn uncompressed_sentinel_top_left_tile_is_not_blank() {
        let _guard = ZEISS_TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.store(0, Ordering::Relaxed);
        let Some(path) = zeiss_fixture_or_skip() else {
            return;
        };
        let slide = ZeissSlide::parse(&path).expect("parse Zeiss sentinel");
        let candidate_indices = slide.canvas_level_subblocks[0].clone();
        assert!(
            !candidate_indices.is_empty(),
            "expected level-0 Zeiss subblocks on the shared canvas"
        );
        let handle = crate::core::registry::Slide::open(&path).expect("open Zeiss sentinel");
        let req = crate::core::types::TileViewRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: crate::core::types::PlaneSelection::default(),
            col: 0,
            row: 0,
            tile_width: 256,
            tile_height: 256,
        };
        let tile = handle
            .read_display_tile(&req)
            .expect("read Zeiss top-left tile");
        assert!(
            tile.data.as_u8().unwrap().iter().any(|&byte| byte != 0),
            "expected the top-left tile on the shared Zeiss canvas to contain visible pixels"
        );
        assert!(ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn uncompressed_sentinel_levels_use_direct_composition() {
        let _guard = ZEISS_TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        ZEISS_DIRECT_LEVEL_COMPOSE_HITS.store(0, Ordering::Relaxed);
        ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.store(0, Ordering::Relaxed);
        let Some(path) = zeiss_fixture_or_skip() else {
            return;
        };
        let slide = ZeissSlide::parse(&path).expect("parse Zeiss sentinel");
        let scene = 0;
        let level = slide.dataset.scenes[scene].series[0].levels.len() - 1;
        let image = slide
            .scene_level_image(scene, level)
            .expect("compose Zeiss level from subblocks");
        let expected = slide.dataset.scenes[scene].series[0].levels[level].dimensions;

        assert_eq!(image.width, expected.0 as u32);
        assert_eq!(image.height, expected.1 as u32);
        assert_eq!(ZEISS_DIRECT_LEVEL_COMPOSE_HITS.load(Ordering::Relaxed), 1);
        assert!(ZEISS_DIRECT_UNCOMPRESSED_BLIT_HITS.load(Ordering::Relaxed) > 0);
    }
}
