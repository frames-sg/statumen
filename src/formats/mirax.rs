use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ashlar_core::BackendRequest;
use flate2::read::ZlibDecoder;
use lru::LruCache;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    crop_rgb_interleaved_u8_buffer, DatasetReader, FormatProbe, ProbeConfidence, ProbeResult,
    SlideReader,
};
use crate::core::types::*;
use crate::decode::jpeg::jpeg_dimensions;
use crate::decode::jpeg::{decode_batch_jpeg, JpegDecodeJob};
use crate::error::WsiError;
use crate::properties::Properties;

const MRXS_EXT: &str = "mrxs";
const SLIDEDAT_INI: &str = "Slidedat.ini";
const INDEX_VERSION: &str = "01.02";
const SLIDEDAT_MAX_SIZE: u64 = 1 << 20;
const KEY_FILE_MAX_SIZE: u64 = 1 << 20;
const SLIDE_POSITION_RECORD_SIZE: usize = 9;
const MIRAX_ASSOCIATED_DIMENSION_PROBE_BYTES: u64 = 64 << 10;

const GROUP_GENERAL: &str = "GENERAL";
const KEY_SLIDE_ID: &str = "SLIDE_ID";
const KEY_IMAGE_NUMBER_X: &str = "IMAGENUMBER_X";
const KEY_IMAGE_NUMBER_Y: &str = "IMAGENUMBER_Y";
const KEY_OBJECTIVE_MAGNIFICATION: &str = "OBJECTIVE_MAGNIFICATION";
const KEY_CAMERA_IMAGE_DIVISIONS_PER_SIDE: &str = "CameraImageDivisionsPerSide";

const GROUP_HIERARCHICAL: &str = "HIERARCHICAL";
const KEY_HIER_COUNT: &str = "HIER_COUNT";
const KEY_NONHIER_COUNT: &str = "NONHIER_COUNT";
const KEY_INDEXFILE: &str = "INDEXFILE";
const KEY_HIER_NAME: &str = "HIER_%d_NAME";
const KEY_HIER_COUNT_FMT: &str = "HIER_%d_COUNT";
const KEY_HIER_VAL_SECTION_FMT: &str = "HIER_%d_VAL_%d_SECTION";
const KEY_NONHIER_NAME: &str = "NONHIER_%d_NAME";
const KEY_NONHIER_COUNT_FMT: &str = "NONHIER_%d_COUNT";
const KEY_NONHIER_VAL_FMT: &str = "NONHIER_%d_VAL_%d";
const KEY_NONHIER_VAL_SECTION_FMT: &str = "NONHIER_%d_VAL_%d_SECTION";
const KEY_MACRO_IMAGE_TYPE: &str = "THUMBNAIL_IMAGE_TYPE";
const KEY_LABEL_IMAGE_TYPE: &str = "BARCODE_IMAGE_TYPE";
const KEY_THUMBNAIL_IMAGE_TYPE: &str = "PREVIEW_IMAGE_TYPE";
const VALUE_VIMSLIDE_POSITION_BUFFER: &str = "VIMSLIDE_POSITION_BUFFER";
const VALUE_STITCHING_INTENSITY_LAYER: &str = "StitchingIntensityLayer";
const VALUE_SCAN_DATA_LAYER: &str = "Scan data layer";
const VALUE_SCAN_DATA_LAYER_MACRO: &str = "ScanDataLayer_SlideThumbnail";
const VALUE_SCAN_DATA_LAYER_LABEL: &str = "ScanDataLayer_SlideBarcode";
const VALUE_SCAN_DATA_LAYER_THUMBNAIL: &str = "ScanDataLayer_SlidePreview";
const VALUE_SLIDE_ZOOM_LEVEL: &str = "Slide zoom level";

const GROUP_DATAFILE: &str = "DATAFILE";
const KEY_FILE_COUNT: &str = "FILE_COUNT";
const KEY_FILE_FMT: &str = "FILE_%d";

const KEY_OVERLAP_X: &str = "OVERLAP_X";
const KEY_OVERLAP_Y: &str = "OVERLAP_Y";
const KEY_MPP_X: &str = "MICROMETER_PER_PIXEL_X";
const KEY_MPP_Y: &str = "MICROMETER_PER_PIXEL_Y";
const KEY_IMAGE_FORMAT: &str = "IMAGE_FORMAT";
const KEY_IMAGE_FILL_COLOR_BGR: &str = "IMAGE_FILL_COLOR_BGR";
const KEY_DIGITIZER_WIDTH: &str = "DIGITIZER_WIDTH";
const KEY_DIGITIZER_HEIGHT: &str = "DIGITIZER_HEIGHT";
const KEY_IMAGE_CONCAT_FACTOR: &str = "IMAGE_CONCAT_FACTOR";

#[cfg(test)]
static MIRAX_ASSOCIATED_CACHE_HITS: AtomicU64 = AtomicU64::new(0);

pub(crate) struct MiraxBackend {
    probe_cache: Mutex<LruCache<PathBuf, Arc<MiraxSlide>>>,
}

impl MiraxBackend {
    pub(crate) fn new() -> Self {
        Self {
            probe_cache: Mutex::new(LruCache::new(NonZeroUsize::new(16).unwrap())),
        }
    }

    fn cache_key(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    fn parse(&self, path: &Path) -> Result<Arc<MiraxSlide>, WsiError> {
        Ok(Arc::new(MiraxSlide::parse(path)?))
    }
}

impl Default for MiraxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatProbe for MiraxBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        if !looks_like_mirax(path) {
            return Ok(not_detected());
        }
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
                vendor: "mirax".into(),
                confidence: ProbeConfidence::Definite,
            });
        }
        let slide = match self.parse(path) {
            Ok(slide) => slide,
            Err(_) => return Ok(not_detected()),
        };
        self.probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, slide);
        Ok(ProbeResult {
            detected: true,
            vendor: "mirax".into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for MiraxBackend {
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
            None => self.parse(path)?,
        };
        Ok(Box::new(MiraxReader { slide }))
    }
}

struct MiraxReader {
    slide: Arc<MiraxSlide>,
}

impl SlideReader for MiraxReader {
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
                    reason: "RequireDevice not supported for MIRAX in Phase 2".into(),
                });
            }
        })
        .to_ashlar();
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

impl MiraxReader {
    fn read_tile_with_backend(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let series = &self.slide.dataset.scenes[req.scene].series[req.series];
        let level = series
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: series.levels.len() as u32,
            })?;
        let TileLayout::Irregular { tiles, .. } = &level.tile_layout else {
            return Err(WsiError::UnsupportedFormat(
                "MIRAX levels must use irregular tiles".into(),
            ));
        };
        let entry = tiles
            .get(&(req.col, req.row))
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("no MIRAX tile at ({},{})", req.col, req.row),
            })?;
        let tile_index = entry.tiff_tile_index.ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "MIRAX tile is missing backing descriptor".into(),
        })?;
        let level_state =
            self.slide
                .levels
                .get(req.level as usize)
                .ok_or(WsiError::LevelOutOfRange {
                    level: req.level,
                    count: self.slide.levels.len() as u32,
                })?;
        let tile = level_state
            .tiles
            .get(tile_index)
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("invalid MIRAX tile descriptor index {tile_index}"),
            })?;
        let decoded = self.slide.decode_image_with_backend(&tile.image, backend)?;
        if tile.src_x == 0
            && tile.src_y == 0
            && decoded.width == entry.dimensions.0
            && decoded.height == entry.dimensions.1
        {
            return Ok(Arc::unwrap_or_clone(decoded));
        }
        crop_rgb_interleaved_u8_buffer(
            decoded.as_ref(),
            tile.src_x,
            tile.src_y,
            entry.dimensions.0,
            entry.dimensions.1,
        )
    }
}

struct MiraxSlide {
    dataset: Dataset,
    levels: Vec<MiraxLevel>,
    associated: HashMap<String, MiraxRecord>,
    decoded_images: Mutex<LruCache<u32, Arc<CpuTile>>>,
    associated_cache: Mutex<LruCache<String, Arc<CpuTile>>>,
    open_files: Mutex<HashMap<PathBuf, File>>,
}

impl MiraxSlide {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let slide_dir = slide_dir_from_entry(path)?;
        let slidedat_path = slide_dir.join(SLIDEDAT_INI);
        let slidedat = parse_mirax_ini(&slidedat_path)?;

        let general = slidedat
            .groups
            .get(GROUP_GENERAL)
            .ok_or_else(|| invalid_slide(path, "missing [GENERAL] group"))?;
        let hierarchical = slidedat
            .groups
            .get(GROUP_HIERARCHICAL)
            .ok_or_else(|| invalid_slide(path, "missing [HIERARCHICAL] group"))?;
        let datafile_group = slidedat
            .groups
            .get(GROUP_DATAFILE)
            .ok_or_else(|| invalid_slide(path, "missing [DATAFILE] group"))?;

        let slide_id = required_ini_string(path, general, KEY_SLIDE_ID)?;
        let images_x = parse_ini_u32(path, general, KEY_IMAGE_NUMBER_X)?;
        let images_y = parse_ini_u32(path, general, KEY_IMAGE_NUMBER_Y)?;
        let objective_magnification = parse_ini_i32(path, general, KEY_OBJECTIVE_MAGNIFICATION)?;
        let image_divisions = general
            .get(KEY_CAMERA_IMAGE_DIVISIONS_PER_SIDE)
            .map(|value| parse_u32_value(path, KEY_CAMERA_IMAGE_DIVISIONS_PER_SIDE, value))
            .transpose()?
            .unwrap_or(1);
        if images_x == 0 || images_y == 0 || image_divisions == 0 {
            return Err(invalid_slide(path, "MIRAX image counts must be positive"));
        }

        let hier_count = parse_ini_i32(path, hierarchical, KEY_HIER_COUNT)?;
        let nonhier_count = parse_ini_i32(path, hierarchical, KEY_NONHIER_COUNT)?;
        if hier_count <= 0 || nonhier_count < 0 {
            return Err(invalid_slide(
                path,
                "MIRAX hierarchy counts must be positive/non-negative",
            ));
        }

        let slide_zoom_level_value = (0..hier_count)
            .find(|idx| {
                hierarchical
                    .get(&fmt_key(KEY_HIER_NAME, *idx))
                    .map(|value| value == VALUE_SLIDE_ZOOM_LEVEL)
                    .unwrap_or(false)
            })
            .ok_or_else(|| invalid_slide(path, "cannot find Slide zoom level hierarchy"))?;
        if slide_zoom_level_value != 0 {
            return Err(invalid_slide(path, "Slide zoom level not HIER_0"));
        }

        let index_filename = required_ini_string(path, hierarchical, KEY_INDEXFILE)?;
        let zoom_levels = parse_ini_i32(path, hierarchical, &fmt_key(KEY_HIER_COUNT_FMT, 0))?;
        if zoom_levels <= 0 {
            return Err(invalid_slide(path, "MIRAX slide has no zoom levels"));
        }
        let zoom_sections = (0..zoom_levels)
            .map(|idx| {
                required_ini_string(
                    path,
                    hierarchical,
                    &fmt_key2(KEY_HIER_VAL_SECTION_FMT, 0, idx),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let datafile_count = parse_ini_i32(path, datafile_group, KEY_FILE_COUNT)?;
        if datafile_count <= 0 {
            return Err(invalid_slide(path, "MIRAX slide has no data files"));
        }
        let datafile_paths = (0..datafile_count)
            .map(|idx| {
                required_ini_string(path, datafile_group, &fmt_key(KEY_FILE_FMT, idx))
                    .map(|name| slide_dir.join(name))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut section_defs = Vec::with_capacity(zoom_levels as usize);
        for (idx, section_name) in zoom_sections.iter().enumerate() {
            let group = slidedat.groups.get(section_name).ok_or_else(|| {
                invalid_slide(path, format!("missing MIRAX section {section_name}"))
            })?;
            let concat_exponent = parse_ini_i32(path, group, KEY_IMAGE_CONCAT_FACTOR)?;
            if (idx == 0 && concat_exponent < 0) || (idx > 0 && concat_exponent <= 0) {
                return Err(invalid_slide(
                    path,
                    format!("invalid IMAGE_CONCAT_FACTOR on {section_name}"),
                ));
            }
            let image_w = parse_ini_u32(path, group, KEY_DIGITIZER_WIDTH)?;
            let image_h = parse_ini_u32(path, group, KEY_DIGITIZER_HEIGHT)?;
            if image_w == 0 || image_h == 0 {
                return Err(invalid_slide(
                    path,
                    format!("zero digitizer size on {section_name}"),
                ));
            }
            let bgr = parse_ini_u32(path, group, KEY_IMAGE_FILL_COLOR_BGR)?;
            section_defs.push(SlideZoomLevelSection {
                concat_exponent,
                overlap_x: parse_ini_f64(path, group, KEY_OVERLAP_X)?,
                overlap_y: parse_ini_f64(path, group, KEY_OVERLAP_Y)?,
                mpp_x: parse_ini_f64(path, group, KEY_MPP_X)?,
                mpp_y: parse_ini_f64(path, group, KEY_MPP_Y)?,
                fill_rgb: bgr_to_rgb(bgr),
                image_format: parse_image_format(
                    required_ini_string(path, group, KEY_IMAGE_FORMAT)?.as_str(),
                )?,
                image_w,
                image_h,
            });
        }

        let position_nonhier_vimslide_offset = get_nonhier_name_offset(
            path,
            &slidedat,
            nonhier_count,
            GROUP_HIERARCHICAL,
            VALUE_VIMSLIDE_POSITION_BUFFER,
        )?;
        let position_nonhier_stitching_offset = if position_nonhier_vimslide_offset.is_some() {
            None
        } else {
            get_nonhier_name_offset(
                path,
                &slidedat,
                nonhier_count,
                GROUP_HIERARCHICAL,
                VALUE_STITCHING_INTENSITY_LAYER,
            )?
        };

        let macro_nonhier_offset = get_associated_image_nonhier_offset(
            path,
            &slidedat,
            nonhier_count,
            GROUP_HIERARCHICAL,
            VALUE_SCAN_DATA_LAYER,
            VALUE_SCAN_DATA_LAYER_MACRO,
            KEY_MACRO_IMAGE_TYPE,
        )?;
        let label_nonhier_offset = get_associated_image_nonhier_offset(
            path,
            &slidedat,
            nonhier_count,
            GROUP_HIERARCHICAL,
            VALUE_SCAN_DATA_LAYER,
            VALUE_SCAN_DATA_LAYER_LABEL,
            KEY_LABEL_IMAGE_TYPE,
        )?;
        let thumbnail_nonhier_offset = get_associated_image_nonhier_offset(
            path,
            &slidedat,
            nonhier_count,
            GROUP_HIERARCHICAL,
            VALUE_SCAN_DATA_LAYER,
            VALUE_SCAN_DATA_LAYER_THUMBNAIL,
            KEY_THUMBNAIL_IMAGE_TYPE,
        )?;

        let mut quickhash = Quickhash1::new();
        quickhash.hash_file(&slidedat_path)?;

        let mut base_w = 0i64;
        let mut base_h = 0i64;
        for i in 0..images_x {
            if (i % image_divisions) != image_divisions - 1 || i == images_x - 1 {
                base_w += i64::from(section_defs[0].image_w);
            } else {
                base_w += (f64::from(section_defs[0].image_w) - section_defs[0].overlap_x) as i64;
            }
        }
        for i in 0..images_y {
            if (i % image_divisions) != image_divisions - 1 || i == images_y - 1 {
                base_h += i64::from(section_defs[0].image_h);
            } else {
                base_h += (f64::from(section_defs[0].image_h) - section_defs[0].overlap_y) as i64;
            }
        }
        if base_w <= 0 || base_h <= 0 {
            return Err(invalid_slide(path, "invalid MIRAX base dimensions"));
        }

        let mut params = Vec::with_capacity(section_defs.len());
        let mut level_builders = Vec::with_capacity(section_defs.len());
        let mut total_concat_exponent = 0i32;
        for (idx, section) in section_defs.iter().enumerate() {
            total_concat_exponent += section.concat_exponent;
            if total_concat_exponent >= 30 {
                return Err(invalid_slide(path, "MIRAX concat exponent too large"));
            }
            let image_concat = 1u32 << total_concat_exponent;
            let positions_per_image = (image_concat / image_divisions).max(1);
            let (tile_count_divisor, tiles_per_image, positions_per_tile) =
                if position_nonhier_vimslide_offset.is_some()
                    || position_nonhier_stitching_offset.is_some()
                    || section_defs[0].overlap_x != 0.0
                    || section_defs[0].overlap_y != 0.0
                {
                    (image_concat.min(image_divisions), positions_per_image, 1)
                } else {
                    (image_concat, 1, positions_per_image)
                };
            let tile_w = f64::from(section.image_w) / f64::from(tiles_per_image);
            let tile_h = f64::from(section.image_h) / f64::from(tiles_per_image);
            let images_per_position = (image_divisions / image_concat).max(1);
            let tile_advance_x = tile_w - section.overlap_x / f64::from(images_per_position);
            let tile_advance_y = tile_h - section.overlap_y / f64::from(images_per_position);
            let level_dimensions = (
                (base_w / i64::from(image_concat)) as u64,
                (base_h / i64::from(image_concat)) as u64,
            );
            let downsample =
                f64::from(image_concat) / f64::from(1u32 << section_defs[0].concat_exponent.max(0));
            params.push(SlideZoomLevelParams {
                image_concat,
                tile_count_divisor,
                tiles_per_image,
                positions_per_tile,
                tile_advance_x,
                tile_advance_y,
            });
            level_builders.push(MiraxLevelBuilder {
                dimensions: level_dimensions,
                downsample,
                image_format: section.image_format,
                raw_image_width: section.image_w,
                raw_image_height: section.image_h,
                tile_width: tile_w,
                tile_height: tile_h,
                tile_advance_x,
                tile_advance_y,
                tiles: HashMap::new(),
                descriptors: Vec::new(),
                extra_tiles: (0, 0, 0, 0),
            });
            if !tile_advance_x.is_finite()
                || !tile_advance_y.is_finite()
                || tile_advance_x <= 0.0
                || tile_advance_y <= 0.0
            {
                return Err(invalid_slide(
                    path,
                    format!("invalid MIRAX tile advance at level {idx}"),
                ));
            }
        }

        let index_path = slide_dir.join(index_filename);
        let mut index_file = File::open(&index_path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: index_path.clone(),
        })?;
        verify_index_header(path, &mut index_file, &slide_id)?;

        let associated = build_associated_records(
            path,
            &mut index_file,
            &datafile_paths,
            slide_id.len(),
            macro_nonhier_offset,
            label_nonhier_offset,
            thumbnail_nonhier_offset,
        )?;

        let slide_positions = load_slide_positions(
            path,
            &mut index_file,
            &datafile_paths,
            slide_id.len(),
            position_nonhier_vimslide_offset,
            position_nonhier_stitching_offset,
            images_x,
            images_y,
            image_divisions,
            params[0].image_concat,
            section_defs[0].image_w,
            section_defs[0].image_h,
            section_defs[0].overlap_x,
            section_defs[0].overlap_y,
        )?;

        let hier_root = (INDEX_VERSION.len() + slide_id.len()) as u64;
        index_file
            .seek(SeekFrom::Start(hier_root))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: index_path.clone(),
            })?;
        let seek_location = read_u32_le(&mut index_file, &index_path)? as u64;

        let mut quickhash_files = HashMap::new();
        process_hier_data_pages_from_indexfile(
            path,
            &mut index_file,
            &index_path,
            seek_location,
            &datafile_paths,
            images_x,
            images_y,
            image_divisions,
            &params,
            &mut level_builders,
            &slide_positions,
            &mut quickhash,
            &mut quickhash_files,
        )?;

        let quickhash = quickhash
            .finish()
            .ok_or_else(|| invalid_slide(path, "failed to compute MIRAX quickhash"))?;
        let dataset_id = dataset_id_from_quickhash(path, &quickhash)?;

        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "mirax");
        properties.insert("openslide.quickhash-1", quickhash.clone());
        properties.insert(
            "openslide.objective-power",
            objective_magnification.to_string(),
        );
        properties.insert("openslide.mpp-x", format!("{}", section_defs[0].mpp_x));
        properties.insert("openslide.mpp-y", format!("{}", section_defs[0].mpp_y));
        properties.insert(
            "openslide.background-color",
            format!("{:06X}", section_defs[0].fill_rgb),
        );

        let mut associated_metadata = HashMap::new();
        for (name, record) in &associated {
            let dimensions = read_jpeg_dimensions_from_record(path, &mut quickhash_files, record)
                .map_err(|err| {
                invalid_slide(
                    path,
                    format!("failed to read MIRAX associated image {name} dimensions: {err}"),
                )
            })?;
            associated_metadata.insert(
                name.clone(),
                AssociatedImage {
                    dimensions,
                    sample_type: SampleType::Uint8,
                    channels: 3,
                },
            );
        }

        let mut dataset_levels = Vec::with_capacity(level_builders.len());
        let mut levels = Vec::with_capacity(level_builders.len());
        for level in level_builders {
            dataset_levels.push(Level {
                dimensions: level.dimensions,
                downsample: level.downsample,
                tile_layout: TileLayout::Irregular {
                    tile_advance: (level.tile_advance_x, level.tile_advance_y),
                    extra_tiles: level.extra_tiles,
                    tiles: level.tiles,
                },
            });
            levels.push(MiraxLevel {
                tiles: level.descriptors,
            });
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
            associated_images: associated_metadata,
            properties,
            icc_profiles: HashMap::new(),
        };

        Ok(Self {
            dataset,
            levels,
            associated,
            decoded_images: Mutex::new(LruCache::new(NonZeroUsize::new(1).unwrap())),
            associated_cache: Mutex::new(LruCache::new(NonZeroUsize::new(1).unwrap())),
            open_files: Mutex::new(quickhash_files),
        })
    }

    fn decode_image_with_backend(
        &self,
        image: &Arc<MiraxImage>,
        _backend: BackendRequest,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let mut cache = self
            .decoded_images
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(buffer) = cache.get(&image.id) {
            return Ok(buffer.clone());
        }
        let decoded = Arc::new(self.decode_record_to_sample_buffer(
            &image.record,
            image.format,
            Some((image.expected_width, image.expected_height)),
            BackendRequest::Auto,
        )?);
        cache.put(image.id, decoded.clone());
        Ok(decoded)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let record = self
            .associated
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
        let mut cache = self
            .associated_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(buffer) = cache.get(name) {
            #[cfg(test)]
            {
                MIRAX_ASSOCIATED_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            }
            return Ok((**buffer).clone());
        }
        let decoded = Arc::new(self.decode_record_to_sample_buffer(
            record,
            MiraxImageFormat::Jpeg,
            None,
            BackendRequest::Auto,
        )?);
        cache.put(name.to_string(), decoded.clone());
        Ok((*decoded).clone())
    }

    fn decode_record_to_sample_buffer(
        &self,
        record: &MiraxRecord,
        format: MiraxImageFormat,
        expected_dimensions: Option<(u32, u32)>,
        _backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let bytes = self.read_record_bytes(record)?;
        match format {
            MiraxImageFormat::Jpeg => {
                let (expected_width, expected_height) = expected_dimensions.unwrap_or((0, 0));
                decode_batch_jpeg(&[JpegDecodeJob {
                    data: Cow::Borrowed(&bytes),
                    tables: None,
                    expected_width,
                    expected_height,
                    color_transform: ashlar_jpeg::ColorTransform::Auto,
                    force_dimensions: false,
                    requested_size: None,
                }])
                .into_iter()
                .next()
                .expect("1-element JPEG facade batch")
            }
            MiraxImageFormat::Png | MiraxImageFormat::Bmp24 => {
                let image = image::load_from_memory(&bytes)
                    .map_err(|err| {
                        WsiError::DisplayConversion(format!("failed to decode MIRAX image: {err}"))
                    })?
                    .to_rgb8();
                if let Some((expected_width, expected_height)) = expected_dimensions {
                    if image.width() != expected_width || image.height() != expected_height {
                        return Err(WsiError::DisplayConversion(format!(
                            "MIRAX image dimensions mismatch: expected {}x{}, got {}x{}",
                            expected_width,
                            expected_height,
                            image.width(),
                            image.height()
                        )));
                    }
                }
                Ok(rgb_image_to_sample_buffer(image))
            }
        }
    }

    fn read_record_bytes(&self, record: &MiraxRecord) -> Result<Vec<u8>, WsiError> {
        self.with_open_file(&record.path, |file| {
            read_record_bytes_from_file(file, &record.path, record.offset, record.len)
        })
    }

    fn with_open_file<T>(
        &self,
        path: &Path,
        f: impl FnOnce(&mut File) -> Result<T, WsiError>,
    ) -> Result<T, WsiError> {
        let mut files = self.open_files.lock().unwrap_or_else(|e| e.into_inner());
        if !files.contains_key(path) {
            let file = File::open(path).map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: path.to_path_buf(),
            })?;
            files.insert(path.to_path_buf(), file);
        }
        let file = files
            .get_mut(path)
            .expect("MIRAX cached file must exist after insertion");
        f(file)
    }
}

struct MiraxLevel {
    tiles: Vec<MiraxTile>,
}

struct MiraxLevelBuilder {
    dimensions: (u64, u64),
    downsample: f64,
    image_format: MiraxImageFormat,
    raw_image_width: u32,
    raw_image_height: u32,
    tile_width: f64,
    tile_height: f64,
    tile_advance_x: f64,
    tile_advance_y: f64,
    tiles: HashMap<(i64, i64), TileEntry>,
    descriptors: Vec<MiraxTile>,
    extra_tiles: (u32, u32, u32, u32),
}

#[derive(Clone)]
struct MiraxTile {
    image: Arc<MiraxImage>,
    src_x: u32,
    src_y: u32,
}

#[derive(Clone)]
struct MiraxImage {
    id: u32,
    record: MiraxRecord,
    format: MiraxImageFormat,
    expected_width: u32,
    expected_height: u32,
}

#[derive(Clone)]
struct MiraxRecord {
    path: PathBuf,
    offset: u64,
    len: u64,
}

#[derive(Clone, Copy)]
struct SlideZoomLevelSection {
    concat_exponent: i32,
    overlap_x: f64,
    overlap_y: f64,
    mpp_x: f64,
    mpp_y: f64,
    fill_rgb: u32,
    image_format: MiraxImageFormat,
    image_w: u32,
    image_h: u32,
}

#[derive(Clone, Copy)]
struct SlideZoomLevelParams {
    image_concat: u32,
    tile_count_divisor: u32,
    tiles_per_image: u32,
    positions_per_tile: u32,
    tile_advance_x: f64,
    tile_advance_y: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MiraxImageFormat {
    Jpeg,
    Png,
    Bmp24,
}

#[derive(Default)]
struct ParsedIni {
    groups: HashMap<String, HashMap<String, String>>,
}

fn not_detected() -> ProbeResult {
    ProbeResult {
        detected: false,
        vendor: String::new(),
        confidence: ProbeConfidence::Likely,
    }
}

fn looks_like_mirax(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(MRXS_EXT))
        .unwrap_or(false)
        && slide_dir_from_entry(path)
            .ok()
            .map(|dir| dir.join(SLIDEDAT_INI).is_file())
            .unwrap_or(false)
}

fn slide_dir_from_entry(path: &Path) -> Result<PathBuf, WsiError> {
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(MRXS_EXT))
        .unwrap_or(false)
    {
        return Err(WsiError::UnsupportedFormat(path.display().to_string()));
    }
    let stem = path
        .file_stem()
        .ok_or_else(|| invalid_slide(path, "MIRAX entry has no stem"))?;
    let dir = path.with_file_name(stem);
    if !dir.is_dir() {
        return Err(invalid_slide(
            path,
            format!("missing MIRAX directory {}", dir.display()),
        ));
    }
    Ok(dir)
}

fn parse_mirax_ini(path: &Path) -> Result<ParsedIni, WsiError> {
    let metadata = std::fs::metadata(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    if metadata.len() > SLIDEDAT_MAX_SIZE.max(KEY_FILE_MAX_SIZE) {
        return Err(invalid_slide(path, "MIRAX key file too large"));
    }
    let text = std::fs::read_to_string(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
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

#[allow(clippy::too_many_arguments)]
fn process_hier_data_pages_from_indexfile(
    path: &Path,
    index_file: &mut File,
    index_path: &Path,
    mut seek_location: u64,
    datafile_paths: &[PathBuf],
    images_x: u32,
    images_y: u32,
    image_divisions: u32,
    params: &[SlideZoomLevelParams],
    levels: &mut [MiraxLevelBuilder],
    slide_positions: &[i32],
    quickhash: &mut Quickhash1,
    quickhash_files: &mut HashMap<PathBuf, File>,
) -> Result<(), WsiError> {
    let mut image_number = 0u32;
    let positions_x = images_x / image_divisions;
    let positions_y = images_y / image_divisions;
    let mut active_positions = vec![false; positions_x.saturating_mul(positions_y) as usize];
    let levels_len = levels.len();
    let level0_raw_image_width = levels[0].raw_image_width;
    let level0_raw_image_height = levels[0].raw_image_height;

    for zoom_level in 0..levels.len() {
        let level = &mut levels[zoom_level];
        let params_level = params[zoom_level];

        index_file
            .seek(SeekFrom::Start(seek_location))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: index_path.to_path_buf(),
            })?;
        let ptr = read_u32_le(index_file, index_path)? as u64;
        index_file
            .seek(SeekFrom::Start(ptr))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: index_path.to_path_buf(),
            })?;
        if read_u32_le(index_file, index_path)? != 0 {
            return Err(invalid_slide(
                path,
                format!("expected initial zero for MIRAX zoom level {zoom_level}"),
            ));
        }
        let initial_data_page = read_u32_le(index_file, index_path)? as u64;
        index_file
            .seek(SeekFrom::Start(initial_data_page))
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: index_path.to_path_buf(),
            })?;

        loop {
            let page_len = read_i32_le(index_file, index_path)?;
            if page_len < 0 {
                return Err(invalid_slide(path, "negative MIRAX page length"));
            }
            let next_ptr = read_i32_le(index_file, index_path)?;
            for _ in 0..page_len {
                let image_index = read_i32_le(index_file, index_path)?;
                let offset = read_i32_le(index_file, index_path)?;
                let length = read_i32_le(index_file, index_path)?;
                let fileno = read_i32_le(index_file, index_path)?;
                if image_index < 0 || offset < 0 || length < 0 || fileno < 0 {
                    return Err(invalid_slide(path, "negative MIRAX hier record component"));
                }
                let fileno = fileno as usize;
                let x = image_index as u32 % images_x;
                let y = image_index as u32 / images_x;
                if y >= images_y {
                    return Err(invalid_slide(
                        path,
                        format!("MIRAX image row {y} outside zoom level {zoom_level}"),
                    ));
                }
                if !x.is_multiple_of(params_level.image_concat)
                    || !y.is_multiple_of(params_level.image_concat)
                {
                    return Err(invalid_slide(
                        path,
                        format!("MIRAX image coordinates ({x},{y}) not aligned for zoom level {zoom_level}"),
                    ));
                }
                let datafile_path = datafile_paths
                    .get(fileno)
                    .ok_or_else(|| {
                        invalid_slide(path, format!("invalid MIRAX data file {fileno}"))
                    })?
                    .clone();
                if zoom_level == levels_len - 1 {
                    quickhash_file_part_cached(
                        quickhash,
                        quickhash_files,
                        &datafile_path,
                        offset as u64,
                        length as u64,
                    )?;
                }

                let image = Arc::new(MiraxImage {
                    id: image_number,
                    record: MiraxRecord {
                        path: datafile_path,
                        offset: offset as u64,
                        len: length as u64,
                    },
                    format: level.image_format,
                    expected_width: level.raw_image_width,
                    expected_height: level.raw_image_height,
                });
                image_number += 1;

                for yi in 0..params_level.tiles_per_image {
                    let yy = y + yi * image_divisions;
                    if yy >= images_y {
                        break;
                    }
                    for xi in 0..params_level.tiles_per_image {
                        let xx = x + xi * image_divisions;
                        if xx >= images_x {
                            break;
                        }
                        let Some((pos0_x, pos0_y)) = get_tile_position(
                            slide_positions,
                            &mut active_positions,
                            params,
                            images_x,
                            image_divisions,
                            level0_raw_image_width,
                            level0_raw_image_height,
                            zoom_level,
                            xx,
                            yy,
                        )?
                        else {
                            continue;
                        };
                        let pos_x = f64::from(pos0_x) / f64::from(params_level.image_concat);
                        let pos_y = f64::from(pos0_y) / f64::from(params_level.image_concat);
                        let src_x = (level.tile_width * f64::from(xi)).round() as u32;
                        let src_y = (level.tile_height * f64::from(yi)).round() as u32;
                        let tile_x = x / params_level.tile_count_divisor + xi;
                        let tile_y = y / params_level.tile_count_divisor + yi;
                        insert_tile(
                            level,
                            &params_level,
                            image.clone(),
                            pos_x,
                            pos_y,
                            src_x,
                            src_y,
                            tile_x as i64,
                            tile_y as i64,
                        );
                    }
                }
            }
            if next_ptr == 0 {
                break;
            }
            index_file
                .seek(SeekFrom::Start(next_ptr as u64))
                .map_err(|source| WsiError::IoWithPath {
                    source: Arc::new(source),
                    path: index_path.to_path_buf(),
                })?;
        }

        seek_location += 4;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_tile(
    level: &mut MiraxLevelBuilder,
    params: &SlideZoomLevelParams,
    image: Arc<MiraxImage>,
    pos_x: f64,
    pos_y: f64,
    src_x: u32,
    src_y: u32,
    tile_x: i64,
    tile_y: i64,
) {
    let offset_x = pos_x - tile_x as f64 * params.tile_advance_x;
    let offset_y = pos_y - tile_y as f64 * params.tile_advance_y;
    let width = level.tile_width.ceil() as u32;
    let height = level.tile_height.ceil() as u32;
    let descriptor_index = level.descriptors.len();
    level.descriptors.push(MiraxTile {
        image,
        src_x,
        src_y,
    });
    level.tiles.insert(
        (tile_x, tile_y),
        TileEntry {
            offset: (offset_x, offset_y),
            dimensions: (width, height),
            tiff_tile_index: Some(descriptor_index),
        },
    );
    let extras = irregular_extra_tiles(
        offset_x,
        offset_y,
        params.tile_advance_x,
        params.tile_advance_y,
        level.tile_width,
        level.tile_height,
    );
    level.extra_tiles.0 = level.extra_tiles.0.max(extras.0);
    level.extra_tiles.1 = level.extra_tiles.1.max(extras.1);
    level.extra_tiles.2 = level.extra_tiles.2.max(extras.2);
    level.extra_tiles.3 = level.extra_tiles.3.max(extras.3);
}

#[allow(clippy::too_many_arguments)]
fn load_slide_positions(
    path: &Path,
    index_file: &mut File,
    datafile_paths: &[PathBuf],
    slide_id_len: usize,
    vimslide_record: Option<i32>,
    stitching_record: Option<i32>,
    images_x: u32,
    images_y: u32,
    image_divisions: u32,
    level0_image_concat: u32,
    image0_w: u32,
    image0_h: u32,
    overlap_x: f64,
    overlap_y: f64,
) -> Result<Vec<i32>, WsiError> {
    let positions_x = images_x / image_divisions;
    let positions_y = images_y / image_divisions;
    let npositions = positions_x.saturating_mul(positions_y) as usize;
    let expected_size = npositions * SLIDE_POSITION_RECORD_SIZE;
    let nonhier_root = (INDEX_VERSION.len() + slide_id_len + 4) as u64;

    if let Some(record) = vimslide_record.or(stitching_record) {
        let record = read_nonhier_record(path, index_file, datafile_paths, nonhier_root, record)?;
        let mut buffer = read_record_bytes_fields(&record.path, record.offset, record.len)?;
        if stitching_record == Some(record.index) {
            let mut decoder = ZlibDecoder::new(buffer.as_slice());
            let mut inflated = Vec::with_capacity(expected_size);
            decoder.read_to_end(&mut inflated).map_err(|err| {
                invalid_slide(
                    path,
                    format!("failed to inflate MIRAX position buffer: {err}"),
                )
            })?;
            buffer = inflated;
        }
        if buffer.len() != expected_size {
            return Err(invalid_slide(
                path,
                format!(
                    "unexpected MIRAX position buffer size {} (expected {expected_size})",
                    buffer.len()
                ),
            ));
        }
        return read_slide_position_buffer(path, &buffer, level0_image_concat);
    }

    let mut positions = Vec::with_capacity(npositions * 2);
    for i in 0..npositions {
        positions.push(
            ((i % positions_x as usize) as f64
                * (f64::from(image0_w) * f64::from(image_divisions) - overlap_x))
                as i32,
        );
        positions.push(
            ((i / positions_x as usize) as f64
                * (f64::from(image0_h) * f64::from(image_divisions) - overlap_y))
                as i32,
        );
    }
    Ok(positions)
}

fn read_slide_position_buffer(
    path: &Path,
    buffer: &[u8],
    level0_image_concat: u32,
) -> Result<Vec<i32>, WsiError> {
    if !buffer.len().is_multiple_of(SLIDE_POSITION_RECORD_SIZE) {
        return Err(invalid_slide(path, "unexpected MIRAX position buffer size"));
    }
    let mut positions = Vec::with_capacity(buffer.len() / SLIDE_POSITION_RECORD_SIZE * 2);
    let mut cursor = 0usize;
    while cursor < buffer.len() {
        let flag = buffer[cursor];
        if flag & 0xfe != 0 {
            return Err(invalid_slide(
                path,
                format!("unexpected MIRAX position flag {flag}"),
            ));
        }
        cursor += 1;
        let x = i32::from_le_bytes(buffer[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let y = i32::from_le_bytes(buffer[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        positions.push(x.saturating_mul(level0_image_concat as i32));
        positions.push(y.saturating_mul(level0_image_concat as i32));
    }
    Ok(positions)
}

#[allow(clippy::too_many_arguments)]
fn get_tile_position(
    slide_positions: &[i32],
    active_positions: &mut [bool],
    params: &[SlideZoomLevelParams],
    images_across: u32,
    image_divisions: u32,
    level0_image_width: u32,
    level0_image_height: u32,
    zoom_level: usize,
    xx: u32,
    yy: u32,
) -> Result<Option<(i32, i32)>, WsiError> {
    let params_level = params[zoom_level];
    let image0_w = level0_image_width as i32;
    let image0_h = level0_image_height as i32;
    let xp = xx / image_divisions;
    let yp = yy / image_divisions;
    let cp = (yp * (images_across / image_divisions) + xp) as usize;
    let Some(base_x) = slide_positions.get(cp * 2).copied() else {
        return Ok(None);
    };
    let Some(base_y) = slide_positions.get(cp * 2 + 1).copied() else {
        return Ok(None);
    };
    let pos0_x = base_x + image0_w * (xx as i32 - xp as i32 * image_divisions as i32);
    let pos0_y = base_y + image0_h * (yy as i32 - yp as i32 * image_divisions as i32);

    if zoom_level == 0 {
        if base_x == 0 && base_y == 0 && (xp != 0 || yp != 0) {
            return Ok(None);
        }
        active_positions[cp] = true;
        return Ok(Some((pos0_x, pos0_y)));
    }

    for ypp in yp..yp + params_level.positions_per_tile {
        for xpp in xp..xp + params_level.positions_per_tile {
            let cpp = (ypp * (images_across / image_divisions) + xpp) as usize;
            if active_positions.get(cpp).copied().unwrap_or(false) {
                return Ok(Some((pos0_x, pos0_y)));
            }
        }
    }
    Ok(None)
}

fn build_associated_records(
    path: &Path,
    index_file: &mut File,
    datafile_paths: &[PathBuf],
    slide_id_len: usize,
    macro_record: Option<i32>,
    label_record: Option<i32>,
    thumbnail_record: Option<i32>,
) -> Result<HashMap<String, MiraxRecord>, WsiError> {
    let nonhier_root = (INDEX_VERSION.len() + slide_id_len + 4) as u64;
    let mut associated = HashMap::new();
    for (name, recordno) in [
        ("macro", macro_record),
        ("label", label_record),
        ("thumbnail", thumbnail_record),
    ] {
        let Some(recordno) = recordno else {
            continue;
        };
        let record = read_nonhier_record(path, index_file, datafile_paths, nonhier_root, recordno)?;
        associated.insert(
            name.into(),
            MiraxRecord {
                path: record.path,
                offset: record.offset,
                len: record.len,
            },
        );
    }
    Ok(associated)
}

struct MiraxNonHierRecord {
    index: i32,
    path: PathBuf,
    offset: u64,
    len: u64,
}

fn read_nonhier_record(
    path: &Path,
    index_file: &mut File,
    datafile_paths: &[PathBuf],
    nonhier_root: u64,
    record_index: i32,
) -> Result<MiraxNonHierRecord, WsiError> {
    if record_index < 0 {
        return Err(invalid_slide(path, "negative MIRAX nonhier record"));
    }
    index_file
        .seek(SeekFrom::Start(nonhier_root))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let table_base = read_u32_le(index_file, path)? as u64;
    index_file
        .seek(SeekFrom::Start(table_base + 4 * record_index as u64))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let list_head = read_u32_le(index_file, path)? as u64;
    index_file
        .seek(SeekFrom::Start(list_head))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    if read_u32_le(index_file, path)? != 0 {
        return Err(invalid_slide(
            path,
            "expected zero at beginning of MIRAX data page",
        ));
    }
    let page_ptr = read_u32_le(index_file, path)? as u64;
    index_file
        .seek(SeekFrom::Start(page_ptr))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let page_len = read_i32_le(index_file, path)?;
    if page_len < 1 {
        return Err(invalid_slide(path, "expected at least one MIRAX data item"));
    }
    let _next = read_i32_le(index_file, path)?;
    if read_i32_le(index_file, path)? != 0 || read_i32_le(index_file, path)? != 0 {
        return Err(invalid_slide(path, "unexpected MIRAX data page header"));
    }
    let offset = read_i32_le(index_file, path)?;
    let len = read_i32_le(index_file, path)?;
    let fileno = read_i32_le(index_file, path)?;
    if offset < 0 || len < 0 || fileno < 0 {
        return Err(invalid_slide(path, "negative MIRAX nonhier record payload"));
    }
    let datafile = datafile_paths
        .get(fileno as usize)
        .ok_or_else(|| invalid_slide(path, format!("invalid MIRAX data file {fileno}")))?;
    Ok(MiraxNonHierRecord {
        index: record_index,
        path: datafile.clone(),
        offset: offset as u64,
        len: len as u64,
    })
}

fn verify_index_header(path: &Path, index_file: &mut File, slide_id: &str) -> Result<(), WsiError> {
    index_file
        .seek(SeekFrom::Start(0))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let version = read_exact_string(index_file, path, INDEX_VERSION.len())?;
    if version != INDEX_VERSION {
        return Err(invalid_slide(
            path,
            "Index.dat does not have expected version",
        ));
    }
    let found_uuid = read_exact_string(index_file, path, slide_id.len())?;
    if found_uuid != slide_id {
        return Err(invalid_slide(
            path,
            "Index.dat does not have matching slide identifier",
        ));
    }
    Ok(())
}

fn get_associated_image_nonhier_offset(
    path: &Path,
    ini: &ParsedIni,
    nonhier_count: i32,
    group: &str,
    target_name: &str,
    target_value: &str,
    target_format_key: &str,
) -> Result<Option<i32>, WsiError> {
    let Some((offset, section_name)) =
        get_nonhier_val_offset(path, ini, nonhier_count, group, target_name, target_value)?
    else {
        return Ok(None);
    };
    let group = ini
        .groups
        .get(&section_name)
        .ok_or_else(|| invalid_slide(path, format!("missing MIRAX section {section_name}")))?;
    let format = required_ini_string(path, group, target_format_key)?;
    if parse_image_format(format.as_str())? != MiraxImageFormat::Jpeg {
        return Err(invalid_slide(
            path,
            format!("unsupported MIRAX associated image format {format}"),
        ));
    }
    Ok(Some(offset))
}

fn get_nonhier_name_offset(
    path: &Path,
    ini: &ParsedIni,
    nonhier_count: i32,
    group: &str,
    target_name: &str,
) -> Result<Option<i32>, WsiError> {
    Ok(
        get_nonhier_name_offset_helper(path, ini, nonhier_count, group, target_name)?
            .map(|(offset, _, _)| offset),
    )
}

fn get_nonhier_val_offset(
    path: &Path,
    ini: &ParsedIni,
    nonhier_count: i32,
    group: &str,
    target_name: &str,
    target_value: &str,
) -> Result<Option<(i32, String)>, WsiError> {
    let Some((mut offset, name_count, name_index)) =
        get_nonhier_name_offset_helper(path, ini, nonhier_count, group, target_name)?
    else {
        return Ok(None);
    };
    let group_map = ini
        .groups
        .get(group)
        .ok_or_else(|| invalid_slide(path, format!("missing MIRAX group {group}")))?;
    for i in 0..name_count {
        let value = required_ini_string(
            path,
            group_map,
            &fmt_key2(KEY_NONHIER_VAL_FMT, name_index, i),
        )?;
        if value == target_value {
            let section = required_ini_string(
                path,
                group_map,
                &fmt_key2(KEY_NONHIER_VAL_SECTION_FMT, name_index, i),
            )?;
            return Ok(Some((offset, section)));
        }
        offset += 1;
    }
    Ok(None)
}

fn get_nonhier_name_offset_helper(
    path: &Path,
    ini: &ParsedIni,
    nonhier_count: i32,
    group: &str,
    target_name: &str,
) -> Result<Option<(i32, i32, i32)>, WsiError> {
    let group_map = ini
        .groups
        .get(group)
        .ok_or_else(|| invalid_slide(path, format!("missing MIRAX group {group}")))?;
    let mut offset = 0;
    for i in 0..nonhier_count {
        let name = required_ini_string(path, group_map, &fmt_key(KEY_NONHIER_NAME, i))?;
        let count = parse_ini_i32(path, group_map, &fmt_key(KEY_NONHIER_COUNT_FMT, i))?;
        if count <= 0 {
            return Err(invalid_slide(path, "MIRAX nonhier count is zero"));
        }
        if name == target_name {
            return Ok(Some((offset, count, i)));
        }
        offset += count;
    }
    Ok(None)
}

fn read_record_bytes(record: &MiraxRecord) -> Result<Vec<u8>, WsiError> {
    read_record_bytes_fields(&record.path, record.offset, record.len)
}

fn read_jpeg_dimensions_from_record(
    path: &Path,
    quickhash_files: &mut HashMap<PathBuf, File>,
    record: &MiraxRecord,
) -> Result<(u32, u32), WsiError> {
    let file = if let Some(file) = quickhash_files.get_mut(&record.path) {
        file
    } else {
        let file = File::open(&record.path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: record.path.clone(),
        })?;
        quickhash_files.insert(record.path.clone(), file);
        quickhash_files
            .get_mut(&record.path)
            .expect("MIRAX cached file must exist after insertion")
    };
    file.seek(SeekFrom::Start(record.offset))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: record.path.clone(),
        })?;

    let probe_len = record.len.min(MIRAX_ASSOCIATED_DIMENSION_PROBE_BYTES) as usize;
    let mut probe = vec![0u8; probe_len];
    file.read_exact(&mut probe)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: record.path.clone(),
        })?;
    if let Ok(dimensions) = jpeg_dimensions(&probe) {
        return Ok(dimensions);
    }

    jpeg_dimensions(&read_record_bytes(record)?).map_err(|err| {
        invalid_slide(
            path,
            format!(
                "failed to derive MIRAX associated JPEG dimensions from {}: {err}",
                record.path.display()
            ),
        )
    })
}

fn read_record_bytes_fields(path: &Path, offset: u64, len: u64) -> Result<Vec<u8>, WsiError> {
    let mut file = File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    read_record_bytes_from_file(&mut file, path, offset, len)
}

fn read_record_bytes_from_file(
    file: &mut File,
    path: &Path,
    offset: u64,
    len: u64,
) -> Result<Vec<u8>, WsiError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let mut buf = vec![0u8; len as usize];
    file.read_exact(&mut buf)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    Ok(buf)
}

fn quickhash_file_part_cached(
    quickhash: &mut Quickhash1,
    files: &mut HashMap<PathBuf, File>,
    path: &Path,
    offset: u64,
    len: u64,
) -> Result<(), WsiError> {
    let file = if let Some(file) = files.get_mut(path) {
        file
    } else {
        let file = File::open(path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
        files.insert(path.to_path_buf(), file);
        files
            .get_mut(path)
            .expect("MIRAX quickhash file must exist after insertion")
    };
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;

    let mut remaining = len;
    let mut buf = [0u8; 4096];
    while remaining > 0 {
        let to_read = (remaining as usize).min(buf.len());
        let read = file
            .read(&mut buf[..to_read])
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: path.to_path_buf(),
            })?;
        if read == 0 {
            break;
        }
        quickhash.update(&buf[..read]);
        remaining -= read as u64;
    }
    Ok(())
}

fn rgb_image_to_sample_buffer(image: image::RgbImage) -> CpuTile {
    CpuTile::new(
        image.width(),
        image.height(),
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(image.into_raw()),
    )
    .expect("RGB image dimensions must match")
}

fn parse_image_format(value: &str) -> Result<MiraxImageFormat, WsiError> {
    match value {
        "JPEG" => Ok(MiraxImageFormat::Jpeg),
        "PNG" => Ok(MiraxImageFormat::Png),
        "BMP24" => Ok(MiraxImageFormat::Bmp24),
        _ => Err(WsiError::DisplayConversion(format!(
            "unsupported MIRAX image format {value}"
        ))),
    }
}

fn bgr_to_rgb(bgr: u32) -> u32 {
    ((bgr << 16) & 0x00FF0000) | (bgr & 0x0000FF00) | ((bgr >> 16) & 0x000000FF)
}

fn irregular_extra_tiles(
    offset_x: f64,
    offset_y: f64,
    tile_advance_x: f64,
    tile_advance_y: f64,
    tile_width: f64,
    tile_height: f64,
) -> (u32, u32, u32, u32) {
    let extra_right = if offset_x < 0.0 {
        (-offset_x / tile_advance_x).ceil() as u32
    } else {
        0
    };
    let offset_xr = offset_x + (tile_width - tile_advance_x);
    let extra_left = if offset_xr > 0.0 {
        (offset_xr / tile_advance_x).ceil() as u32
    } else {
        0
    };
    let extra_bottom = if offset_y < 0.0 {
        (-offset_y / tile_advance_y).ceil() as u32
    } else {
        0
    };
    let offset_yr = offset_y + (tile_height - tile_advance_y);
    let extra_top = if offset_yr > 0.0 {
        (offset_yr / tile_advance_y).ceil() as u32
    } else {
        0
    };
    (extra_top, extra_bottom, extra_left, extra_right)
}

fn required_ini_string(
    path: &Path,
    group: &HashMap<String, String>,
    key: &str,
) -> Result<String, WsiError> {
    group
        .get(key)
        .cloned()
        .ok_or_else(|| invalid_slide(path, format!("missing MIRAX key {key}")))
}

fn parse_ini_i32(path: &Path, group: &HashMap<String, String>, key: &str) -> Result<i32, WsiError> {
    group
        .get(key)
        .ok_or_else(|| invalid_slide(path, format!("missing MIRAX key {key}")))?
        .parse::<i32>()
        .map_err(|_| invalid_slide(path, format!("invalid MIRAX integer for {key}")))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn mirax_sentinel_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../downloads/openslide-testdata-extracted/mirax/mirax-cmu1/CMU-1.mrxs")
    }

    #[test]
    fn associated_thumbnail_is_cached_after_first_read() {
        let sentinel_path = mirax_sentinel_path();
        if !sentinel_path.is_file() {
            eprintln!(
                "skipping corpus-backed MIRAX thumbnail cache test; missing {}",
                sentinel_path.display()
            );
            return;
        }
        MIRAX_ASSOCIATED_CACHE_HITS.store(0, Ordering::Relaxed);
        let slide = MiraxSlide::parse(&sentinel_path).expect("parse MIRAX sentinel");
        let first = slide
            .read_associated("thumbnail")
            .expect("read thumbnail once");
        let second = slide
            .read_associated("thumbnail")
            .expect("read thumbnail twice");
        assert_eq!(first.width, second.width);
        assert_eq!(first.height, second.height);
        assert_eq!(
            MIRAX_ASSOCIATED_CACHE_HITS.load(Ordering::Relaxed),
            1,
            "second thumbnail read should hit the cache"
        );
    }
}

fn parse_ini_u32(path: &Path, group: &HashMap<String, String>, key: &str) -> Result<u32, WsiError> {
    parse_u32_value(
        path,
        key,
        group
            .get(key)
            .ok_or_else(|| invalid_slide(path, format!("missing MIRAX key {key}")))?,
    )
}

fn parse_u32_value(path: &Path, key: &str, value: &str) -> Result<u32, WsiError> {
    value
        .parse::<u32>()
        .map_err(|_| invalid_slide(path, format!("invalid MIRAX integer for {key}")))
}

fn parse_ini_f64(path: &Path, group: &HashMap<String, String>, key: &str) -> Result<f64, WsiError> {
    group
        .get(key)
        .ok_or_else(|| invalid_slide(path, format!("missing MIRAX key {key}")))?
        .parse::<f64>()
        .map_err(|_| invalid_slide(path, format!("invalid MIRAX float for {key}")))
}

fn read_exact_string(file: &mut File, path: &Path, len: usize) -> Result<String, WsiError> {
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_i32_le(file: &mut File, path: &Path) -> Result<i32, WsiError> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32_le(file: &mut File, path: &Path) -> Result<u32, WsiError> {
    let value = read_i32_le(file, path)?;
    if value < 0 {
        return Err(invalid_slide(
            path,
            format!("negative MIRAX pointer value {value}"),
        ));
    }
    Ok(value as u32)
}

fn fmt_key(fmt: &str, value: i32) -> String {
    fmt.replacen("%d", &value.to_string(), 1)
}

fn fmt_key2(fmt: &str, value1: i32, value2: i32) -> String {
    fmt.replacen("%d", &value1.to_string(), 1)
        .replacen("%d", &value2.to_string(), 1)
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
