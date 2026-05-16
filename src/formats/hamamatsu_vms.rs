use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lru::LruCache;
use signinum_core::BackendRequest;
use signinum_jpeg::{
    Decoder as SigninumJpegDecoder, Downscale as SigninumDownscale,
    PixelFormat as SigninumPixelFormat, Rect as SigninumRect,
};

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::core::types::*;
use crate::decode::jpeg::{decode_batch_jpeg, JpegDecodeJob};
use crate::decode::jpeg::{jpeg_dimensions, JpegTileGeometry};
use crate::error::WsiError;
use crate::properties::Properties;

const GROUP_VMS: &str = "Virtual Microscope Specimen";
const KEY_MAP_FILE: &str = "MapFile";
const KEY_IMAGE_FILE: &str = "ImageFile";
const KEY_NUM_JPEG_COLS: &str = "NoJpegColumns";
const KEY_NUM_JPEG_ROWS: &str = "NoJpegRows";
const KEY_OPTIMISATION_FILE: &str = "OptimisationFile";
const KEY_MACRO_IMAGE: &str = "MacroImage";
const KEY_PHYSICAL_WIDTH: &str = "PhysicalWidth";
const KEY_PHYSICAL_HEIGHT: &str = "PhysicalHeight";
const KEY_SOURCE_LENS: &str = "SourceLens";
const KEY_FILE_MAX_SIZE: u64 = 64 << 10;
const VMS_SCALES: [u32; 3] = [2, 4, 8];
const JPEG_HEADER_MAX_BYTES: usize = 1 << 20;
const JPEG_SCAN_CHUNK_BYTES: usize = 64 << 10;
const VMS_DECODED_TILE_CACHE_ENTRIES: usize = 64;

pub(crate) struct HamamatsuVmsBackend {
    probe_cache: Mutex<LruCache<PathBuf, Arc<VmsSlide>>>,
}

impl HamamatsuVmsBackend {
    pub(crate) fn new() -> Self {
        Self {
            probe_cache: Mutex::new(LruCache::new(NonZeroUsize::new(16).unwrap())),
        }
    }

    fn cache_key(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    fn parse(&self, path: &Path) -> Result<Arc<VmsSlide>, WsiError> {
        let slide = Arc::new(VmsSlide::parse(path)?);
        Ok(slide)
    }
}

impl Default for HamamatsuVmsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatProbe for HamamatsuVmsBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        let ini = match parse_vms_ini(path) {
            Ok(ini) => ini,
            Err(_) => {
                return Ok(ProbeResult {
                    detected: false,
                    vendor: String::new(),
                    confidence: ProbeConfidence::Likely,
                });
            }
        };
        let Some(group) = ini.groups.get(GROUP_VMS) else {
            return Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            });
        };
        let cols = group
            .get(KEY_NUM_JPEG_COLS)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        let rows = group
            .get(KEY_NUM_JPEG_ROWS)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if cols == 0 || rows == 0 {
            return Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            });
        }

        let slide = self.parse(path)?;
        let key = Self::cache_key(path);
        self.probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, slide);

        Ok(ProbeResult {
            detected: true,
            vendor: "hamamatsu".into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for HamamatsuVmsBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let key = Self::cache_key(path);
        let cached = self
            .probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop(&key);
        let slide = match cached {
            Some(slide) => slide,
            None => self.parse(path)?,
        };
        Ok(Box::new(VmsReader { slide }))
    }
}

struct VmsReader {
    slide: Arc<VmsSlide>,
}

impl SlideReader for VmsReader {
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
                    reason: "RequireDevice not supported for VMS in Phase 2".into(),
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
        let path = self
            .slide
            .associated_paths
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
        let data = std::fs::read(path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.clone(),
        })?;
        decode_batch_jpeg(&[JpegDecodeJob {
            data: Cow::Borrowed(&data),
            tables: None,
            expected_width: 0,
            expected_height: 0,
            color_transform: signinum_jpeg::ColorTransform::Auto,
            force_dimensions: false,
            requested_size: None,
        }])
        .into_iter()
        .next()
        .expect("1-element JPEG facade batch")
    }
}

impl VmsReader {
    fn read_tile_with_backend(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let series = &self.slide.dataset.scenes[req.scene].series[req.series];
        let level_meta = &series.levels[req.level as usize];
        let level = self
            .slide
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: self.slide.levels.len() as u32,
            })?;

        let TileLayout::Regular {
            tiles_across,
            tiles_down,
            ..
        } = level_meta.tile_layout
        else {
            return Err(WsiError::UnsupportedFormat(
                "VMS levels must use regular tiles".into(),
            ));
        };

        if req.col < 0
            || req.row < 0
            || req.col >= tiles_across as i64
            || req.row >= tiles_down as i64
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

        let base_tiles_across = level.base_tiles_across;
        let base_tiles_down = level.base_tiles_down;
        let jpeg_col = req.col as u32 / base_tiles_across;
        let jpeg_row = req.row as u32 / base_tiles_down;
        let local_tile_col = req.col as u32 % base_tiles_across;
        let local_tile_row = req.row as u32 % base_tiles_down;
        let jpeg = level
            .jpegs
            .get((jpeg_row * level.jpegs_across + jpeg_col) as usize)
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "VMS tile resolved to missing JPEG shard".into(),
            })?;
        if local_tile_col >= jpeg.tiles_across || local_tile_row >= jpeg.tiles_down {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "VMS local tile coordinates out of JPEG shard bounds".into(),
            });
        }
        let tile_index = (local_tile_row * jpeg.tiles_across + local_tile_col) as usize;
        jpeg.decode_tile(tile_index, level.scale_denom, backend)
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
}

struct VmsSlide {
    dataset: Dataset,
    levels: Vec<VmsLevel>,
    associated_paths: HashMap<String, PathBuf>,
}

struct VmsLevel {
    scale_denom: u32,
    jpegs: Vec<Arc<VmsJpeg>>,
    jpegs_across: u32,
    base_tiles_across: u32,
    base_tiles_down: u32,
}

struct VmsJpeg {
    path: PathBuf,
    header: Vec<u8>,
    sof_dimensions_offset: usize,
    file_len: u64,
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    mcu_starts: Mutex<Vec<Option<u64>>>,
    unreliable_mcu_starts: Vec<Option<u64>>,
    decoded_tile_cache: Mutex<LruCache<(usize, u32), Arc<CpuTile>>>,
    comment: Option<String>,
}

impl VmsSlide {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let ini = parse_vms_ini(path)?;
        let group = ini
            .groups
            .get(GROUP_VMS)
            .ok_or_else(|| invalid_slide(path, "missing [Virtual Microscope Specimen] group"))?;
        let num_cols = parse_u32(path, group, KEY_NUM_JPEG_COLS)?;
        let num_rows = parse_u32(path, group, KEY_NUM_JPEG_ROWS)?;
        if num_cols == 0 || num_rows == 0 {
            return Err(invalid_slide(path, "VMS file has no columns or rows"));
        }

        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let mut image_paths = vec![None; (num_cols * num_rows) as usize];
        for (key, value) in group {
            if !key.starts_with(KEY_IMAGE_FILE) {
                continue;
            }
            let dims = parse_image_key_suffix(path, key)?;
            if dims.layer != 0 {
                continue;
            }
            if dims.col >= num_cols || dims.row >= num_rows {
                return Err(invalid_slide(
                    path,
                    format!("invalid VMS image coordinates ({},{})", dims.col, dims.row),
                ));
            }
            let idx = (dims.row * num_cols + dims.col) as usize;
            if image_paths[idx].is_some() {
                return Err(invalid_slide(
                    path,
                    format!("duplicate VMS image for ({},{})", dims.col, dims.row),
                ));
            }
            image_paths[idx] = Some(dir.join(value));
        }
        let image_paths: Vec<PathBuf> = image_paths
            .into_iter()
            .enumerate()
            .map(|(idx, path_opt)| {
                path_opt
                    .ok_or_else(|| invalid_slide(path, format!("missing VMS image filename {idx}")))
            })
            .collect::<Result<_, _>>()?;

        let map_path = dir.join(
            group
                .get(KEY_MAP_FILE)
                .ok_or_else(|| invalid_slide(path, "missing MapFile"))?,
        );
        let macro_path = group.get(KEY_MACRO_IMAGE).map(|value| dir.join(value));
        let opt_path = group
            .get(KEY_OPTIMISATION_FILE)
            .map(|value| dir.join(value));

        let mut quickhash = Quickhash1::new();
        quickhash.hash_file(path)?;
        quickhash.hash_file(&map_path)?;
        let quickhash = quickhash
            .finish()
            .ok_or_else(|| invalid_slide(path, "failed to compute VMS quickhash"))?;
        let dataset_id = dataset_id_from_quickhash(path, &quickhash)?;

        let opt_offsets = parse_vms_opt_offsets(opt_path.as_deref(), &image_paths)?;

        let mut base_images = Vec::with_capacity(image_paths.len());
        for (idx, image_path) in image_paths.iter().enumerate() {
            let row_starts = opt_offsets.get(idx).cloned().unwrap_or_default();
            base_images.push(Arc::new(VmsJpeg::parse(image_path, row_starts)?));
        }
        let map_image = Arc::new(VmsJpeg::parse(&map_path, Vec::new())?);

        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "hamamatsu");
        properties.insert("openslide.quickhash-1", quickhash.clone());
        for (key, value) in group {
            properties.insert(format!("hamamatsu.{key}"), value.clone());
        }
        if let Some(first_comment) = base_images
            .first()
            .and_then(|image| image.comment.as_deref())
        {
            properties.insert("openslide.comment", first_comment);
        }
        if let Some(source_lens) = group.get(KEY_SOURCE_LENS) {
            properties.insert("openslide.objective-power", source_lens.clone());
        }

        let base_level = VmsLevel::new(base_images, num_cols, num_rows, 1)?;
        let map_level = VmsLevel::new(vec![map_image], 1, 1, 1)?;
        if let Some(width_nm) = group
            .get(KEY_PHYSICAL_WIDTH)
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
        {
            properties.insert(
                "openslide.mpp-x",
                format!(
                    "{}",
                    width_nm as f64 / (1000.0 * base_level_dimensions(&base_level).0 as f64)
                ),
            );
        }
        if let Some(height_nm) = group
            .get(KEY_PHYSICAL_HEIGHT)
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
        {
            properties.insert(
                "openslide.mpp-y",
                format!(
                    "{}",
                    height_nm as f64 / (1000.0 * base_level_dimensions(&base_level).1 as f64)
                ),
            );
        }

        let levels = expanded_levels(base_level, map_level);
        let largest = base_level_dimensions(
            levels
                .first()
                .ok_or_else(|| invalid_slide(path, "VMS slide created no levels"))?,
        );
        let dataset_levels: Vec<Level> = levels
            .iter()
            .map(|level| {
                let dimensions = base_level_dimensions(level);
                Level {
                    dimensions,
                    downsample: largest.0 as f64 / dimensions.0 as f64,
                    tile_layout: TileLayout::Regular {
                        tile_width: level.jpegs[0].tile_width / level.scale_denom,
                        tile_height: level.jpegs[0].tile_height / level.scale_denom,
                        tiles_across: total_tiles_across(level),
                        tiles_down: total_tiles_down(level),
                    },
                }
            })
            .collect();

        let mut associated_images = HashMap::new();
        let mut associated_paths = HashMap::new();
        if let Some(macro_path) = macro_path.filter(|p| p.is_file()) {
            let macro_bytes =
                std::fs::read(&macro_path).map_err(|source| WsiError::IoWithPath {
                    source: Arc::new(source),
                    path: macro_path.clone(),
                })?;
            let macro_dims = jpeg_dimensions(&macro_bytes)?;
            associated_images.insert(
                "macro".into(),
                AssociatedImage {
                    dimensions: macro_dims,
                    sample_type: SampleType::Uint8,
                    channels: 3,
                },
            );
            associated_paths.insert("macro".into(), macro_path);
        }

        let dataset = Dataset {
            id: dataset_id,
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: dataset_levels,
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images,
            properties,
            icc_profiles: HashMap::new(),
        };

        Ok(Self {
            dataset,
            levels,
            associated_paths,
        })
    }
}

impl VmsLevel {
    fn new(
        jpegs: Vec<Arc<VmsJpeg>>,
        jpegs_across: u32,
        _jpegs_down: u32,
        scale_denom: u32,
    ) -> Result<Self, WsiError> {
        let first = jpegs
            .first()
            .ok_or_else(|| WsiError::InvalidSlide {
                path: PathBuf::new(),
                message: "VMS level has no JPEG shards".into(),
            })?
            .clone();
        Ok(Self {
            scale_denom,
            jpegs,
            jpegs_across,
            base_tiles_across: first.tiles_across,
            base_tiles_down: first.tiles_down,
        })
    }
}

impl VmsJpeg {
    fn parse(path: &Path, row_starts: Vec<Option<u64>>) -> Result<Self, WsiError> {
        let header = read_vms_jpeg_header(path).map_err(|err| {
            invalid_slide(
                path,
                format!("failed to derive VMS JPEG tile geometry: {err}"),
            )
        })?;
        let geometry = header.geometry;
        let tiles_across = geometry.width.div_ceil(geometry.tile_width);
        let tiles_down = geometry.height.div_ceil(geometry.tile_height);
        let tile_count = tiles_across as usize * tiles_down as usize;
        let mut unreliable_mcu_starts = vec![None; tile_count];
        for (row, offset) in row_starts.into_iter().enumerate().take(tiles_down as usize) {
            let idx = row * tiles_across as usize;
            if idx < unreliable_mcu_starts.len() {
                unreliable_mcu_starts[idx] = offset;
            }
        }
        let mut mcu_starts = vec![None; tile_count];
        if !mcu_starts.is_empty() {
            mcu_starts[0] = Some(header.scan_data_offset);
        }

        Ok(Self {
            path: path.to_path_buf(),
            header: header.header,
            sof_dimensions_offset: header.sof_dimensions_offset,
            file_len: header.file_len,
            width: geometry.width,
            height: geometry.height,
            tile_width: geometry.tile_width,
            tile_height: geometry.tile_height,
            tiles_across,
            tiles_down,
            mcu_starts: Mutex::new(mcu_starts),
            unreliable_mcu_starts,
            decoded_tile_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(VMS_DECODED_TILE_CACHE_ENTRIES).unwrap(),
            )),
            comment: header.comment,
        })
    }

    fn decode_tile(
        &self,
        tile_index: usize,
        scale_denom: u32,
        _backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let cache_key = (tile_index, scale_denom);
        if let Some(cached) = self
            .decoded_tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&cache_key)
        {
            return Ok(cached.as_ref().clone());
        }

        let scale = match scale_denom {
            1 => SigninumDownscale::None,
            2 => SigninumDownscale::Half,
            4 => SigninumDownscale::Quarter,
            8 => SigninumDownscale::Eighth,
            other => {
                return Err(WsiError::Jpeg(format!(
                    "unsupported VMS signinum downscale denominator {other}"
                )));
            }
        };
        let tile_col = tile_index as u32 % self.tiles_across;
        let tile_row = tile_index as u32 / self.tiles_across;
        let width = self
            .tile_width
            .min(self.width.saturating_sub(tile_col * self.tile_width));
        let height = self
            .tile_height
            .min(self.height.saturating_sub(tile_row * self.tile_height));
        let data = self.tile_jpeg_bytes(tile_index, width, height)?;
        let decoder =
            SigninumJpegDecoder::new(&data).map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let roi = SigninumRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        let (pixels, _outcome) = decoder
            .decode_region_scaled(SigninumPixelFormat::Rgb8, roi, scale)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let scale_denom = scale.denominator();
        let width = width.div_ceil(scale_denom);
        let height = height.div_ceil(scale_denom);
        let tile = CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(pixels),
        };
        self.decoded_tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(cache_key, Arc::new(tile.clone()));
        Ok(tile)
    }

    fn tile_jpeg_bytes(
        &self,
        tile_index: usize,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, WsiError> {
        let (start, stop) = self.tile_entropy_bounds(tile_index)?;
        if stop <= start || stop > self.file_len {
            return Err(WsiError::Jpeg(format!(
                "invalid VMS JPEG entropy bounds {}..{} for {}",
                start,
                stop,
                self.path.display()
            )));
        }
        let data_len = usize::try_from(stop - start)
            .map_err(|_| WsiError::Jpeg("VMS JPEG entropy segment is too large".into()))?;
        let mut entropy = vec![0u8; data_len];
        let mut file = File::open(&self.path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: self.path.clone(),
        })?;
        file.seek(SeekFrom::Start(start))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
        file.read_exact(&mut entropy)
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
        if entropy.len() < 2 || entropy[entropy.len() - 2] != 0xFF {
            return Err(WsiError::Jpeg(format!(
                "VMS JPEG entropy segment for {} does not end at a marker",
                self.path.display()
            )));
        }
        let last = entropy.len() - 1;
        entropy[last] = 0xD9;

        let mut header = self.header.clone();
        patch_sof_dimensions(&mut header, self.sof_dimensions_offset, width, height)?;
        header.extend_from_slice(&entropy);
        Ok(header)
    }

    fn tile_entropy_bounds(&self, tile_index: usize) -> Result<(u64, u64), WsiError> {
        let tile_count = self.tiles_across as usize * self.tiles_down as usize;
        if tile_index >= tile_count {
            return Err(WsiError::Jpeg(format!(
                "VMS JPEG tile index {tile_index} out of range {tile_count}"
            )));
        }
        let mut starts = self.mcu_starts.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure_mcu_start(&mut starts, tile_index)?;
        let start = starts[tile_index].ok_or_else(|| {
            WsiError::Jpeg(format!("missing VMS JPEG MCU start for tile {tile_index}"))
        })?;
        let stop = if tile_index + 1 == tile_count {
            self.file_len
        } else {
            self.ensure_mcu_start(&mut starts, tile_index + 1)?;
            starts[tile_index + 1].ok_or_else(|| {
                WsiError::Jpeg(format!(
                    "missing VMS JPEG MCU stop for tile {}",
                    tile_index + 1
                ))
            })?
        };
        Ok((start, stop))
    }

    fn ensure_mcu_start(&self, starts: &mut [Option<u64>], target: usize) -> Result<(), WsiError> {
        if target >= starts.len() || starts[target].is_some() {
            return Ok(());
        }
        if starts[0].is_none() {
            starts[0] = Some(self.header.len() as u64);
        }

        let mut first_good = target;
        loop {
            if starts[first_good].is_some() {
                break;
            }
            if let Some(offset) = self
                .unreliable_mcu_starts
                .get(first_good)
                .and_then(|offset| *offset)
            {
                if first_good == 0 || self.valid_recorded_restart_offset(offset)? {
                    starts[first_good] = Some(offset);
                    break;
                }
            }
            if first_good == 0 {
                starts[0] = Some(self.header.len() as u64);
                break;
            }
            first_good -= 1;
        }

        let mut offset = starts[first_good].ok_or_else(|| {
            WsiError::Jpeg(format!(
                "missing VMS JPEG known MCU start before tile {target}"
            ))
        })?;
        let mut file = File::open(&self.path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: self.path.clone(),
        })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
        for idx in first_good..target {
            offset = find_next_restart_offset(&mut file, self.file_len, &self.path)?.ok_or_else(
                || {
                    WsiError::Jpeg(format!(
                        "could not find restart marker for VMS JPEG tile {} in {}",
                        idx + 1,
                        self.path.display()
                    ))
                },
            )?;
            starts[idx + 1] = Some(offset);
            file.seek(SeekFrom::Start(offset))
                .map_err(|source| WsiError::IoWithPath {
                    source: Arc::new(source),
                    path: self.path.clone(),
                })?;
        }
        Ok(())
    }

    fn valid_recorded_restart_offset(&self, offset: u64) -> Result<bool, WsiError> {
        if offset == self.header.len() as u64 {
            return Ok(true);
        }
        if offset < 2 || offset > self.file_len {
            return Ok(false);
        }
        let mut file = File::open(&self.path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: self.path.clone(),
        })?;
        file.seek(SeekFrom::Start(offset - 2))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
        let mut marker = [0u8; 2];
        file.read_exact(&mut marker)
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
        Ok(marker[0] == 0xFF && (0xD0..=0xD7).contains(&marker[1]))
    }
}

fn expanded_levels(base_level: VmsLevel, map_level: VmsLevel) -> Vec<VmsLevel> {
    let mut levels_by_width = BTreeMap::new();
    for level in [base_level, map_level] {
        insert_scaled_levels(&mut levels_by_width, level);
    }
    levels_by_width
        .into_iter()
        .rev()
        .map(|(_, level)| level)
        .collect()
}

fn insert_scaled_levels(levels: &mut BTreeMap<u64, VmsLevel>, level: VmsLevel) {
    let width = base_level_dimensions(&level).0;
    levels.insert(width, level);
    let original = levels.get(&width).unwrap().clone_for_scale_base();
    for scale in VMS_SCALES {
        let tile_width = original.jpegs[0].tile_width;
        let tile_height = original.jpegs[0].tile_height;
        if !tile_width.is_multiple_of(scale) || !tile_height.is_multiple_of(scale) {
            continue;
        }
        levels.insert(
            base_level_dimensions(&original).0 / scale as u64,
            VmsLevel {
                scale_denom: scale,
                jpegs: original.jpegs.clone(),
                jpegs_across: original.jpegs_across,
                base_tiles_across: original.base_tiles_across,
                base_tiles_down: original.base_tiles_down,
            },
        );
    }
}

impl VmsLevel {
    fn clone_for_scale_base(&self) -> Self {
        Self {
            scale_denom: self.scale_denom,
            jpegs: self.jpegs.clone(),
            jpegs_across: self.jpegs_across,
            base_tiles_across: self.base_tiles_across,
            base_tiles_down: self.base_tiles_down,
        }
    }
}

fn base_level_dimensions(level: &VmsLevel) -> (u64, u64) {
    let row_width: u64 = level
        .jpegs
        .iter()
        .take(level.jpegs_across as usize)
        .map(|jpeg| u64::from(jpeg.width))
        .sum();
    let col_height: u64 = level
        .jpegs
        .iter()
        .step_by(level.jpegs_across as usize)
        .map(|jpeg| u64::from(jpeg.height))
        .sum();
    (
        row_width / u64::from(level.scale_denom),
        col_height / u64::from(level.scale_denom),
    )
}

fn total_tiles_across(level: &VmsLevel) -> u64 {
    level
        .jpegs
        .iter()
        .take(level.jpegs_across as usize)
        .map(|jpeg| u64::from(jpeg.tiles_across))
        .sum()
}

fn total_tiles_down(level: &VmsLevel) -> u64 {
    level
        .jpegs
        .iter()
        .step_by(level.jpegs_across as usize)
        .map(|jpeg| u64::from(jpeg.tiles_down))
        .sum()
}

#[derive(Default)]
struct ParsedIni {
    groups: HashMap<String, HashMap<String, String>>,
}

fn parse_vms_ini(path: &Path) -> Result<ParsedIni, WsiError> {
    let metadata = std::fs::metadata(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    if metadata.len() > KEY_FILE_MAX_SIZE {
        return Err(invalid_slide(path, "VMS key file too large"));
    }
    let text = std::fs::read_to_string(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    let mut parsed = ParsedIni::default();
    let mut current_group: Option<String> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(group) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            current_group = Some(group.to_string());
            parsed.groups.entry(group.to_string()).or_default();
            continue;
        }
        let Some(group) = current_group.as_ref() else {
            continue;
        };
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        parsed
            .groups
            .entry(group.clone())
            .or_default()
            .insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(parsed)
}

fn parse_u32(path: &Path, group: &HashMap<String, String>, key: &str) -> Result<u32, WsiError> {
    group
        .get(key)
        .ok_or_else(|| invalid_slide(path, format!("missing {key}")))?
        .parse::<u32>()
        .map_err(|_| invalid_slide(path, format!("invalid integer for {key}")))
}

struct ImageDims {
    layer: u32,
    col: u32,
    row: u32,
}

fn parse_image_key_suffix(path: &Path, key: &str) -> Result<ImageDims, WsiError> {
    let suffix = &key[KEY_IMAGE_FILE.len()..];
    if suffix.is_empty() {
        return Ok(ImageDims {
            layer: 0,
            col: 0,
            row: 0,
        });
    }
    let trimmed = suffix
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .ok_or_else(|| invalid_slide(path, format!("invalid VMS image key suffix: {suffix}")))?;
    let parts: Vec<&str> = trimmed.split(',').map(str::trim).collect();
    match parts.as_slice() {
        [col, row] => Ok(ImageDims {
            layer: 0,
            col: col
                .parse()
                .map_err(|_| invalid_slide(path, format!("invalid VMS col in {key}")))?,
            row: row
                .parse()
                .map_err(|_| invalid_slide(path, format!("invalid VMS row in {key}")))?,
        }),
        [layer, col, row] => Ok(ImageDims {
            layer: layer
                .parse()
                .map_err(|_| invalid_slide(path, format!("invalid VMS layer in {key}")))?,
            col: col
                .parse()
                .map_err(|_| invalid_slide(path, format!("invalid VMS col in {key}")))?,
            row: row
                .parse()
                .map_err(|_| invalid_slide(path, format!("invalid VMS row in {key}")))?,
        }),
        _ => Err(invalid_slide(
            path,
            format!("unknown VMS image coordinate arity in {key}"),
        )),
    }
}

fn parse_vms_opt_offsets(
    opt_path: Option<&Path>,
    image_paths: &[PathBuf],
) -> Result<Vec<Vec<Option<u64>>>, WsiError> {
    let Some(opt_path) = opt_path.filter(|path| path.is_file()) else {
        return Ok(vec![Vec::new(); image_paths.len()]);
    };

    let mut file = File::open(opt_path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: opt_path.to_path_buf(),
    })?;
    let mut per_image = Vec::with_capacity(image_paths.len());
    for image_path in image_paths {
        let geometry = jpeg_geometry_from_file(image_path)?;
        let tiles_down = geometry.height / geometry.tile_height;
        let mut row_starts = Vec::with_capacity(tiles_down as usize);
        let mut block = [0u8; 40];
        for _ in 0..tiles_down {
            match file.read_exact(&mut block) {
                Ok(()) => {
                    let offset = u64::from_le_bytes(block[..8].try_into().unwrap());
                    row_starts.push((offset > 0).then_some(offset));
                }
                Err(_) => {
                    return Ok(vec![Vec::new(); image_paths.len()]);
                }
            }
        }
        per_image.push(row_starts);
    }
    Ok(per_image)
}

fn jpeg_geometry_from_file(path: &Path) -> Result<JpegTileGeometry, WsiError> {
    Ok(read_vms_jpeg_header(path)?.geometry)
}

struct VmsJpegHeader {
    header: Vec<u8>,
    geometry: JpegTileGeometry,
    sof_dimensions_offset: usize,
    scan_data_offset: u64,
    file_len: u64,
    comment: Option<String>,
}

fn read_vms_jpeg_header(path: &Path) -> Result<VmsJpegHeader, WsiError> {
    let mut file = File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?
        .len();
    let mut header = Vec::with_capacity(4096);
    let mut soi = [0u8; 2];
    read_exact_header(&mut file, path, &mut header, &mut soi)?;
    if soi != [0xFF, 0xD8] {
        return Err(WsiError::Jpeg("VMS JPEG missing SOI marker".into()));
    }

    let mut dimensions = None;
    let mut sof_dimensions_offset = None;
    let mut max_h = 0u8;
    let mut max_v = 0u8;
    let mut restart_interval = None;
    let mut comment = None;

    loop {
        let marker_start = header.len();
        let marker = read_next_header_marker(&mut file, path, &mut header)?;
        match marker {
            0xD9 => return Err(WsiError::Jpeg("VMS JPEG ended before SOS".into())),
            0x01 | 0xD0..=0xD7 => continue,
            _ => {}
        }

        let mut len_bytes = [0u8; 2];
        read_exact_header(&mut file, path, &mut header, &mut len_bytes)?;
        let seg_len = u16::from_be_bytes(len_bytes) as usize;
        if seg_len < 2 {
            return Err(WsiError::Jpeg(format!(
                "invalid VMS JPEG segment length {seg_len}"
            )));
        }
        let payload_start = header.len();
        let mut payload = vec![0u8; seg_len - 2];
        read_exact_header(&mut file, path, &mut header, &mut payload)?;

        match marker {
            0xC0..=0xC2 => {
                if payload.len() < 6 {
                    return Err(WsiError::Jpeg("truncated VMS JPEG SOF segment".into()));
                }
                if payload[0] != 8 {
                    return Err(WsiError::Jpeg(format!(
                        "unsupported VMS JPEG precision {}",
                        payload[0]
                    )));
                }
                let height = u16::from_be_bytes([payload[1], payload[2]]) as u32;
                let width = u16::from_be_bytes([payload[3], payload[4]]) as u32;
                let components = payload[5] as usize;
                if components == 0 || payload.len() < 6 + components * 3 {
                    return Err(WsiError::Jpeg("truncated VMS JPEG component list".into()));
                }
                max_h = 0;
                max_v = 0;
                for idx in 0..components {
                    let sampling = payload[6 + idx * 3 + 1];
                    max_h = max_h.max(sampling >> 4);
                    max_v = max_v.max(sampling & 0x0F);
                }
                if width == 0 || height == 0 || max_h == 0 || max_v == 0 {
                    return Err(WsiError::Jpeg(
                        "invalid VMS JPEG dimensions or sampling".into(),
                    ));
                }
                dimensions = Some((width, height));
                sof_dimensions_offset = Some(payload_start + 1);
            }
            0xDD => {
                if payload.len() < 2 {
                    return Err(WsiError::Jpeg("truncated VMS JPEG DRI segment".into()));
                }
                let interval = u16::from_be_bytes([payload[0], payload[1]]);
                if interval != 0 {
                    restart_interval = Some(interval);
                }
            }
            0xFE if comment.is_none() => {
                let end = payload
                    .iter()
                    .position(|b| *b == 0)
                    .unwrap_or(payload.len());
                comment = Some(String::from_utf8_lossy(&payload[..end]).into_owned());
            }
            0xFE => {}
            0xDA => {
                let (width, height) = dimensions
                    .ok_or_else(|| WsiError::Jpeg("VMS JPEG missing SOF before SOS".into()))?;
                let restart_interval = restart_interval.ok_or_else(|| {
                    WsiError::Jpeg(
                        "VMS JPEG missing restart interval required for tile geometry".into(),
                    )
                })?;
                let mcu_width = u32::from(max_h) * 8;
                let mcu_height = u32::from(max_v) * 8;
                let mcus_per_row = width.div_ceil(mcu_width);
                let restart = u32::from(restart_interval);
                if restart > mcus_per_row {
                    return Err(WsiError::Jpeg(
                        "VMS JPEG restart interval greater than MCUs per row".into(),
                    ));
                }
                if mcus_per_row % restart != 0 {
                    return Err(WsiError::Jpeg(
                        "VMS JPEG restart interval does not align to MCU rows".into(),
                    ));
                }
                let tile_width = mcu_width
                    .checked_mul(restart)
                    .ok_or_else(|| WsiError::Jpeg("VMS JPEG tile width overflow".into()))?;
                let scan_data_offset = u64::try_from(header.len()).map_err(|_| {
                    WsiError::Jpeg("VMS JPEG header offset does not fit u64".into())
                })?;
                let sof_dimensions_offset = sof_dimensions_offset.ok_or_else(|| {
                    WsiError::Jpeg("VMS JPEG missing SOF dimensions offset".into())
                })?;
                if marker_start >= header.len() {
                    return Err(WsiError::Jpeg("invalid VMS JPEG marker accounting".into()));
                }
                return Ok(VmsJpegHeader {
                    header,
                    geometry: JpegTileGeometry {
                        width,
                        height,
                        tile_width,
                        tile_height: mcu_height,
                    },
                    sof_dimensions_offset,
                    scan_data_offset,
                    file_len,
                    comment,
                });
            }
            _ => {}
        }
    }
}

fn read_exact_header(
    file: &mut File,
    path: &Path,
    header: &mut Vec<u8>,
    buf: &mut [u8],
) -> Result<(), WsiError> {
    if header.len().saturating_add(buf.len()) > JPEG_HEADER_MAX_BYTES {
        return Err(WsiError::Jpeg(format!(
            "VMS JPEG header exceeds {} bytes: {}",
            JPEG_HEADER_MAX_BYTES,
            path.display()
        )));
    }
    file.read_exact(buf)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    header.extend_from_slice(buf);
    Ok(())
}

fn read_next_header_marker(
    file: &mut File,
    path: &Path,
    header: &mut Vec<u8>,
) -> Result<u8, WsiError> {
    let mut byte = [0u8; 1];
    loop {
        read_exact_header(file, path, header, &mut byte)?;
        if byte[0] == 0xFF {
            break;
        }
    }
    loop {
        read_exact_header(file, path, header, &mut byte)?;
        if byte[0] != 0xFF {
            return Ok(byte[0]);
        }
    }
}

fn patch_sof_dimensions(
    header: &mut [u8],
    dimensions_offset: usize,
    width: u32,
    height: u32,
) -> Result<(), WsiError> {
    let width = u16::try_from(width)
        .map_err(|_| WsiError::Jpeg(format!("VMS JPEG tile width {width} exceeds u16")))?;
    let height = u16::try_from(height)
        .map_err(|_| WsiError::Jpeg(format!("VMS JPEG tile height {height} exceeds u16")))?;
    if dimensions_offset + 4 > header.len() {
        return Err(WsiError::Jpeg(
            "VMS JPEG SOF dimensions offset is outside header".into(),
        ));
    }
    header[dimensions_offset..dimensions_offset + 2].copy_from_slice(&height.to_be_bytes());
    header[dimensions_offset + 2..dimensions_offset + 4].copy_from_slice(&width.to_be_bytes());
    Ok(())
}

fn find_next_restart_offset(
    file: &mut File,
    file_len: u64,
    path: &Path,
) -> Result<Option<u64>, WsiError> {
    let mut buf = [0u8; JPEG_SCAN_CHUNK_BYTES];
    let mut pending_ff = false;
    loop {
        let base = file
            .stream_position()
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: path.to_path_buf(),
            })?;
        if base >= file_len {
            return Ok(None);
        }
        let n = file.read(&mut buf).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
        if n == 0 {
            return Ok(None);
        }
        for (idx, byte) in buf[..n].iter().copied().enumerate() {
            if pending_ff {
                if byte == 0xFF {
                    continue;
                }
                pending_ff = false;
                if byte == 0x00 {
                    continue;
                }
                if (0xD0..=0xD7).contains(&byte) {
                    return Ok(Some(base + idx as u64 + 1));
                }
                if byte == 0xD9 {
                    return Ok(None);
                }
                return Err(WsiError::Jpeg(format!(
                    "unexpected JPEG marker FF{byte:02X} while scanning {}",
                    path.display()
                )));
            } else if byte == 0xFF {
                pending_ff = true;
            }
        }
    }
}

fn invalid_slide(path: &Path, message: impl Into<String>) -> WsiError {
    WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn dataset_id_from_quickhash(path: &Path, quickhash: &str) -> Result<DatasetId, WsiError> {
    if quickhash.len() < 32 {
        return Err(invalid_slide(path, "quickhash too short"));
    }
    let value = u128::from_str_radix(&quickhash[..32], 16)
        .map_err(|_| invalid_slide(path, "quickhash is not valid hex"))?;
    Ok(DatasetId(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_restart_jpeg(path: &Path, width: u32, height: u32) -> Vec<u8> {
        let mut pixels = vec![0u8; width as usize * height as usize * 3];
        for y in 0..height {
            for x in 0..width {
                let off = (y as usize * width as usize + x as usize) * 3;
                pixels[off] = x as u8;
                pixels[off + 1] = y as u8;
                pixels[off + 2] = x.wrapping_add(y) as u8;
            }
        }
        let encoded = signinum_jpeg::encode_jpeg_baseline(
            signinum_jpeg::JpegSamples::Rgb8 {
                data: &pixels,
                width,
                height,
            },
            signinum_jpeg::JpegEncodeOptions {
                quality: 90,
                subsampling: signinum_jpeg::JpegSubsampling::Ybr444,
                restart_interval: Some(8),
                backend: signinum_jpeg::JpegBackend::Cpu,
            },
        )
        .unwrap();
        std::fs::write(path, &encoded.data).unwrap();
        encoded.data
    }

    #[test]
    fn vms_jpeg_header_probe_reads_only_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tile.jpg");
        let width = 128u32;
        let height = 16u32;
        let mut bytes = write_restart_jpeg(&path, width, height);
        let encoded_len = bytes.len();
        bytes.extend(vec![0xA5; 1_000_000]);
        std::fs::write(&path, bytes).unwrap();

        let header = read_vms_jpeg_header(&path).unwrap();

        assert_eq!(header.geometry.width, width);
        assert_eq!(header.geometry.height, height);
        assert_eq!(header.geometry.tile_width, 64);
        assert_eq!(header.geometry.tile_height, 8);
        assert!(header.header.len() < encoded_len);
        assert!(header.header.len() < 4096);
    }

    #[test]
    fn vms_jpeg_decodes_restart_segment_tile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tile.jpg");
        let encoded = write_restart_jpeg(&path, 128, 16);
        let reference = SigninumJpegDecoder::new(&encoded)
            .unwrap()
            .decode_region_scaled(
                SigninumPixelFormat::Rgb8,
                SigninumRect {
                    x: 64,
                    y: 8,
                    w: 64,
                    h: 8,
                },
                SigninumDownscale::None,
            )
            .unwrap()
            .0;
        let restart_index = SigninumJpegDecoder::new(&encoded)
            .unwrap()
            .restart_index()
            .unwrap()
            .unwrap();
        let row_starts = vec![
            Some(restart_index.segments[0].entropy_offset as u64),
            Some(restart_index.segments[2].entropy_offset as u64),
        ];
        let jpeg = VmsJpeg::parse(&path, row_starts).unwrap();

        let tile = jpeg.decode_tile(3, 1, BackendRequest::Auto).unwrap();

        assert_eq!(tile.width, 64);
        assert_eq!(tile.height, 8);
        assert_eq!(tile.data.as_u8().unwrap(), reference.as_slice());
        assert_eq!(
            jpeg.decoded_tile_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len(),
            1
        );
    }
}
