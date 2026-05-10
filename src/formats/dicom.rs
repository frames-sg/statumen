use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use dicom_dictionary_std::{tags, uids};
use dicom_object::{meta::FileMetaTable, DefaultDicomObject, OpenFileOptions};
use dicom_parser::dataset::{lazy_read::LazyDataSetReader, LazyDataToken};
use dicom_parser::stateful::decode::StatefulDecode;
use dicom_transfer_syntax_registry::{TransferSyntaxIndex, TransferSyntaxRegistry};
use image::imageops;
use lru::LruCache;
use signinum_core::BackendRequest;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::core::types::*;
#[cfg(feature = "metal")]
use crate::decode::jp2k::decode_batch_jp2k_pixels;
use crate::decode::jp2k::{decode_batch_jp2k, Jp2kDecodeJob};
use crate::decode::jpeg::{decode_batch_jpeg, JpegDecodeJob};
use crate::error::WsiError;
use crate::properties::Properties;

const LEVEL_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "VOLUME", "NONE"],
    &["DERIVED", "PRIMARY", "VOLUME", "NONE"],
    &["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"],
];
const LABEL_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "LABEL", "NONE"],
    &["DERIVED", "PRIMARY", "LABEL", "NONE"],
];
const OVERVIEW_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "OVERVIEW", "NONE"],
    &["DERIVED", "PRIMARY", "OVERVIEW", "NONE"],
];
const THUMBNAIL_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "THUMBNAIL", "RESAMPLED"],
    &["DERIVED", "PRIMARY", "THUMBNAIL", "RESAMPLED"],
];
const BASE_ONLY_DICOM_PYRAMID_MESSAGE: &str = "This DICOM WSI contains only a full-resolution base layer and no physical pyramid levels. Open the complete DICOM series/folder, or regenerate the DICOM with DERIVED/PRIMARY/VOLUME/RESAMPLED pyramid instances.";
const BASE_ONLY_GUARD_MIN_TILE_COUNT: u64 = 4_096;
const BASE_ONLY_GUARD_MIN_DIMENSION: u32 = 32_768;
const SUPPORTED_TRANSFER_SYNTAXES: &[&str] = &[
    uids::IMPLICIT_VR_LITTLE_ENDIAN,
    uids::EXPLICIT_VR_LITTLE_ENDIAN,
    EXPLICIT_VR_BIG_ENDIAN_TRANSFER_SYNTAX,
    uids::JPEG_BASELINE8_BIT,
    uids::JPEG2000_LOSSLESS,
    uids::JPEG2000,
    HTJ2K_LOSSLESS_TRANSFER_SYNTAX,
    HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX,
    uids::RLE_LOSSLESS,
];
const JPEG_TRANSFER_SYNTAX: &str = uids::JPEG_BASELINE8_BIT;
const RLE_TRANSFER_SYNTAX: &str = uids::RLE_LOSSLESS;
const EXPLICIT_VR_BIG_ENDIAN_TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.2";
const HTJ2K_LOSSLESS_TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.4.201";
const HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.4.202";
const JP2K_TRANSFER_SYNTAXES: &[&str] = &[
    uids::JPEG2000_LOSSLESS,
    uids::JPEG2000,
    HTJ2K_LOSSLESS_TRANSFER_SYNTAX,
    HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX,
];
#[cfg(feature = "metal")]
const DICOM_JP2K_DEVICE_DECODE_ENV: &str = "STATUMEN_JP2K_DEVICE_DECODE";

fn is_encapsulated_transfer_syntax(uid: &str) -> bool {
    uid == JPEG_TRANSFER_SYNTAX
        || uid == RLE_TRANSFER_SYNTAX
        || JP2K_TRANSFER_SYNTAXES.contains(&uid)
}

#[cfg(feature = "metal")]
fn dicom_jp2k_device_decode_enabled() -> bool {
    std::env::var(DICOM_JP2K_DEVICE_DECODE_ENV).is_ok_and(|value| {
        value.eq_ignore_ascii_case("1")
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("yes")
            || value.eq_ignore_ascii_case("on")
    })
}

pub(crate) struct DicomBackend {
    probe_cache: Mutex<LruCache<PathBuf, Arc<DicomSlide>>>,
}

impl DicomBackend {
    pub(crate) fn new() -> Self {
        Self {
            probe_cache: Mutex::new(LruCache::new(NonZeroUsize::new(4).unwrap())),
        }
    }

    fn cache_key(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    fn parse(&self, path: &Path) -> Result<Arc<DicomSlide>, WsiError> {
        Ok(Arc::new(DicomSlide::parse(path)?))
    }
}

impl Default for DicomBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatProbe for DicomBackend {
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
                vendor: "dicom".into(),
                confidence: ProbeConfidence::Definite,
            });
        }
        if path.is_dir() {
            return match self.parse(path) {
                Ok(slide) => {
                    self.probe_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .put(key, slide);
                    Ok(ProbeResult {
                        detected: true,
                        vendor: "dicom".into(),
                        confidence: ProbeConfidence::Definite,
                    })
                }
                Err(WsiError::UnsupportedFormat(_)) => Ok(ProbeResult {
                    detected: false,
                    vendor: String::new(),
                    confidence: ProbeConfidence::Likely,
                }),
                Err(err) => Err(err),
            };
        }
        match parse_metadata_object(path) {
            Ok(meta) if is_vl_wsi(meta.obj.meta().media_storage_sop_class_uid()) => {
                let slide = self.parse(path)?;
                self.probe_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .put(key, slide);
                Ok(ProbeResult {
                    detected: true,
                    vendor: "dicom".into(),
                    confidence: ProbeConfidence::Definite,
                })
            }
            Ok(_) => Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            }),
            Err(_) => Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            }),
        }
    }
}

impl DatasetReader for DicomBackend {
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
        Ok(Box::new(DicomReader { slide }))
    }
}

struct DicomReader {
    slide: Arc<DicomSlide>,
}

impl SlideReader for DicomReader {
    fn dataset(&self) -> &Dataset {
        &self.slide.dataset
    }

    fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        self.slide
            .levels
            .get(req.level as usize)
            .map(|level| level.tile_codec_kind(req))
            .unwrap_or(TileCodecKind::Other)
    }

    fn use_display_tile_cache(&self, _req: &TileViewRequest) -> bool {
        true
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let backend = output.backend().to_signinum();

        #[cfg(feature = "metal")]
        if output.prefers_device() {
            match self.read_tiles_jp2k_device_batch(reqs, &output, backend) {
                Ok(Some(tiles)) => return Ok(tiles),
                Ok(None) if output.requires_device() => {
                    return Err(WsiError::Unsupported {
                        reason: "device backend not available for DICOM tile batch".into(),
                    });
                }
                Ok(None) => {}
                Err(err) if output.requires_device() => return Err(err),
                Err(err) => {
                    tracing::debug!(
                        error = %err,
                        fallback_to_cpu = true,
                        fallback_reason = "dicom_jp2k_device_batch_failed",
                        "DICOM JP2K device batch failed; retrying through CPU output"
                    );
                }
            }
        }

        if output.requires_device() {
            return Err(WsiError::Unsupported {
                reason: "RequireDevice not supported for DICOM CPU fallback".into(),
            });
        }

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

    fn read_raw_compressed_tile(&self, req: &TileRequest) -> Result<RawCompressedTile, WsiError> {
        let image = self
            .slide
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: self.slide.levels.len() as u32,
            })?;
        image.read_raw_compressed_tile(req.col, req.row, req.level)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let image = self
            .slide
            .associated
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
        image.read_associated(name)
    }
}

#[cfg(feature = "metal")]
struct DicomDeviceDecodeJob {
    slot: usize,
    image: Arc<DicomImage>,
    frame_index: u32,
}

#[cfg(feature = "metal")]
impl DicomReader {
    fn read_tiles_jp2k_device_batch(
        &self,
        reqs: &[TileRequest],
        output: &TileOutputPreference,
        backend: BackendRequest,
    ) -> Result<Option<Vec<TilePixels>>, WsiError> {
        if reqs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if !(output.compressed_device_decode_enabled() || dicom_jp2k_device_decode_enabled()) {
            return Ok(None);
        }
        let Some(metal_sessions) = output.metal_sessions() else {
            return Ok(None);
        };

        let mut results: Vec<Option<TilePixels>> = Vec::with_capacity(reqs.len());
        results.resize_with(reqs.len(), || None);
        let mut jobs = Vec::new();
        let mut job_meta = Vec::new();
        let mut saw_device_candidate = false;

        for (slot, req) in reqs.iter().enumerate() {
            let level =
                self.slide
                    .levels
                    .get(req.level as usize)
                    .ok_or(WsiError::LevelOutOfRange {
                        level: req.level,
                        count: self.slide.levels.len() as u32,
                    })?;
            if req.col < 0
                || req.row < 0
                || req.col >= level.tiles_across as i64
                || req.row >= level.tiles_down as i64
            {
                return Err(WsiError::Unsupported {
                    reason: format!(
                        "tile ({},{}) out of range for DICOM device decode",
                        req.col, req.row
                    ),
                });
            }

            let col = req.col as u32;
            let row = req.row as u32;
            let Some(image) = level.image_for_tile(col, row) else {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason:
                            "DICOM device batch cannot return CPU black tile for sparse missing tile"
                                .into(),
                    });
                }
                let (width, height) = level.actual_tile_dimensions(col, row);
                results[slot] = Some(TilePixels::Cpu(black_sample_buffer(width, height)));
                continue;
            };
            if !JP2K_TRANSFER_SYNTAXES.contains(&image.transfer_syntax_uid.as_str()) {
                continue;
            }
            let Some(frame_index) = image.frame_index(col, row) else {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason:
                            "DICOM device batch cannot return CPU black tile for sparse missing tile"
                                .into(),
                    });
                }
                let (width, height) = level.actual_tile_dimensions(col, row);
                results[slot] = Some(TilePixels::Cpu(black_sample_buffer(width, height)));
                continue;
            };
            let (actual_width, actual_height) = level.actual_tile_dimensions(col, row);
            if actual_width != image.tile_width || actual_height != image.tile_height {
                continue;
            }
            if image.samples_per_pixel != 3 {
                continue;
            }

            saw_device_candidate = true;
            if !output.requires_device() {
                if let Some(cached) = image.cached_decoded_frame(frame_index) {
                    results[slot] = Some(TilePixels::Cpu(cached.as_ref().clone()));
                    continue;
                }
            }

            let bytes =
                image.extract_encapsulated_frame(frame_index, req.level, req.col, req.row, true)?;
            jobs.push(Jp2kDecodeJob {
                data: Cow::Owned(bytes.as_ref().clone()),
                expected_width: image.tile_width,
                expected_height: image.tile_height,
                rgb_color_space: !matches!(
                    image.photometric_interpretation.as_str(),
                    "YBR_ICT" | "YBR_RCT"
                ),
                backend,
            });
            job_meta.push(DicomDeviceDecodeJob {
                slot,
                image: image.clone(),
                frame_index,
            });
        }

        if jobs.is_empty() && !saw_device_candidate {
            return Ok(None);
        }
        if jobs.is_empty() {
            return results
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(Some)
                .ok_or_else(|| WsiError::Unsupported {
                    reason: "DICOM device batch had no decodable JP2K frames".into(),
                });
        }

        let decoded =
            decode_batch_jp2k_pixels(&jobs, output.requires_device(), Some(metal_sessions));
        if decoded.len() != job_meta.len() {
            return Err(WsiError::Jp2k(format!(
                "DICOM JP2K device batch returned {} tiles for {} jobs",
                decoded.len(),
                job_meta.len()
            )));
        }

        for (meta, decoded) in job_meta.into_iter().zip(decoded) {
            let tile = decoded?;
            if let TilePixels::Cpu(cpu) = &tile {
                meta.image
                    .cache_decoded_frame(meta.frame_index, Arc::new(cpu.clone()));
            }
            results[meta.slot] = Some(tile);
        }

        for (slot, result) in results.iter_mut().enumerate() {
            if result.is_none() {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason: "DICOM device batch contained a non-device-decodable tile".into(),
                    });
                }
                *result = Some(TilePixels::Cpu(
                    self.read_tile_with_backend(&reqs[slot], backend)?,
                ));
            }
        }

        Ok(Some(
            results
                .into_iter()
                .map(|tile| {
                    tile.ok_or_else(|| WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "DICOM device batch result was not populated".into(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

fn dicom_tile_codec_kind(transfer_syntax_uid: &str) -> TileCodecKind {
    if transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
        TileCodecKind::Jpeg
    } else if matches!(
        transfer_syntax_uid,
        HTJ2K_LOSSLESS_TRANSFER_SYNTAX | HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX
    ) {
        TileCodecKind::Htj2k
    } else if JP2K_TRANSFER_SYNTAXES.contains(&transfer_syntax_uid) {
        TileCodecKind::Jp2k
    } else {
        TileCodecKind::Other
    }
}

impl DicomReader {
    fn read_tile_with_backend(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let image = self
            .slide
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: self.slide.levels.len() as u32,
            })?;
        image.read_tile(req.col, req.row, req.level, backend)
    }
}

struct DicomSlide {
    dataset: Dataset,
    levels: Vec<DicomLevel>,
    associated: HashMap<String, Arc<DicomImage>>,
}

impl DicomSlide {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let DicomSeriesManifest {
            study_instance_uid,
            series_instance_uid,
            frame_of_reference_uid,
            container_identifier,
            specimen_identifier,
            volume_images,
            associated_images,
            source_file_count,
        } = DicomSeriesManifest::resolve(path)?;
        let level_images = volume_images
            .into_iter()
            .map(DicomImage::from_metadata)
            .map(|result| result.map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let mut associated_images = associated_images
            .into_iter()
            .map(|(kind, meta)| {
                DicomImage::from_metadata(meta)
                    .map(Arc::new)
                    .map(|image| (kind.name().to_string(), image))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if level_images.is_empty() {
            return Err(invalid_slide(path, "No pyramid levels found"));
        }

        dedupe_associated(path, &mut associated_images)?;
        let mut levels = build_levels(path, level_images)?;
        levels.sort_by(|a, b| {
            b.area()
                .cmp(&a.area())
                .then_with(|| b.width.cmp(&a.width))
                .then_with(|| b.height.cmp(&a.height))
        });
        validate_monotonic_levels(path, &levels)?;
        reject_huge_base_only_dicom(path, &levels)?;

        let level0 = levels
            .first()
            .ok_or_else(|| invalid_slide(path, "No pyramid levels found"))?
            .clone();

        let quickhash = quickhash_for_series_uid(&series_instance_uid)?;
        let dataset_id = dataset_id_from_quickhash(path, &quickhash)?;
        let largest_dimensions = (level0.width, level0.height);
        let public_levels = levels
            .iter()
            .map(|level| Level {
                dimensions: (level.width as u64, level.height as u64),
                downsample: largest_dimensions.0 as f64 / level.width as f64,
                tile_layout: TileLayout::Regular {
                    tile_width: level.tile_width,
                    tile_height: level.tile_height,
                    tiles_across: level.tiles_across as u64,
                    tiles_down: level.tiles_down as u64,
                },
            })
            .collect::<Vec<_>>();

        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "dicom");
        properties.insert("openslide.quickhash-1", quickhash);
        properties.insert("dicom.series-instance-uid", &series_instance_uid);
        if let Some(study_instance_uid) = &study_instance_uid {
            properties.insert("dicom.study-instance-uid", study_instance_uid);
        }
        if let Some(frame_of_reference_uid) = &frame_of_reference_uid {
            properties.insert("dicom.frame-of-reference-uid", frame_of_reference_uid);
        }
        if let Some(container_identifier) = &container_identifier {
            properties.insert("dicom.container-identifier", container_identifier);
        }
        if let Some(specimen_identifier) = &specimen_identifier {
            properties.insert("dicom.specimen-identifier", specimen_identifier);
        }
        properties.insert("dicom.source-file-count", source_file_count.to_string());
        let (shared_pixel_spacing, shared_objective_lens_power) =
            if level0.pixel_spacing.is_none() || level0.objective_lens_power.is_none() {
                parse_level0_properties(&level0.path).unwrap_or((None, None))
            } else {
                (None, None)
            };
        let level0_pixel_spacing = level0.pixel_spacing.or(shared_pixel_spacing);
        if let Some((mpp_x, mpp_y)) = level0_pixel_spacing {
            properties.insert("openslide.mpp-x", format!("{mpp_x}"));
            properties.insert("openslide.mpp-y", format!("{mpp_y}"));
        }
        let level0_objective_lens_power =
            level0.objective_lens_power.or(shared_objective_lens_power);
        if let Some(objective) = level0_objective_lens_power {
            properties.insert("openslide.objective-power", format!("{objective}"));
        }

        let associated_metadata = associated_images
            .iter()
            .map(|(name, image)| {
                (
                    name.clone(),
                    AssociatedImage {
                        dimensions: (image.width, image.height),
                        sample_type: SampleType::Uint8,
                        channels: 3,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let associated = associated_images.into_iter().collect::<HashMap<_, _>>();

        let dataset = Dataset {
            id: dataset_id,
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: public_levels,
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
        })
    }
}

#[derive(Clone, Debug)]
struct DicomLevel {
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    path: PathBuf,
    pixel_spacing: Option<(f64, f64)>,
    objective_lens_power: Option<f64>,
    parts: Vec<Arc<DicomImage>>,
}

impl DicomLevel {
    fn from_image(image: Arc<DicomImage>) -> Self {
        Self {
            width: image.width,
            height: image.height,
            tile_width: image.tile_width,
            tile_height: image.tile_height,
            tiles_across: image.tiles_across,
            tiles_down: image.tiles_down,
            path: image.path.clone(),
            pixel_spacing: image.pixel_spacing,
            objective_lens_power: image.objective_lens_power,
            parts: vec![image],
        }
    }

    fn area(&self) -> u64 {
        u64::from(self.width).saturating_mul(u64::from(self.height))
    }

    fn is_regular_full_tiling(&self) -> bool {
        self.parts.iter().all(|part| part.is_full_grid())
    }

    fn push_part(&mut self, path: &Path, image: Arc<DicomImage>) -> Result<(), WsiError> {
        if self
            .parts
            .iter()
            .any(|part| part.sop_instance_uid == image.sop_instance_uid)
        {
            return Ok(());
        }
        if self.tile_width != image.tile_width
            || self.tile_height != image.tile_height
            || self.tiles_across != image.tiles_across
            || self.tiles_down != image.tiles_down
            || self.samples_per_pixel() != image.samples_per_pixel
            || self.planar_configuration() != image.planar_configuration
            || self.photometric_interpretation() != image.photometric_interpretation
        {
            return Err(invalid_slide(
                path,
                format!(
                    "DICOM level {}x{} has incompatible split image {}",
                    self.width, self.height, image.sop_instance_uid
                ),
            ));
        }
        self.parts.push(image);
        Ok(())
    }

    fn samples_per_pixel(&self) -> u16 {
        self.parts[0].samples_per_pixel
    }

    fn planar_configuration(&self) -> Option<u16> {
        self.parts[0].planar_configuration
    }

    fn photometric_interpretation(&self) -> &str {
        &self.parts[0].photometric_interpretation
    }

    fn image_for_tile(&self, col: u32, row: u32) -> Option<Arc<DicomImage>> {
        self.parts
            .iter()
            .find(|image| image.frame_index(col, row).is_some())
            .cloned()
    }

    fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        if req.col < 0
            || req.row < 0
            || req.col >= self.tiles_across as i64
            || req.row >= self.tiles_down as i64
        {
            return TileCodecKind::Other;
        }
        self.image_for_tile(req.col as u32, req.row as u32)
            .map(|image| dicom_tile_codec_kind(&image.transfer_syntax_uid))
            .unwrap_or(TileCodecKind::Other)
    }

    fn read_tile(
        &self,
        col: i64,
        row: i64,
        level: u32,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        if col < 0 || row < 0 || col >= self.tiles_across as i64 || row >= self.tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    self.tiles_across, self.tiles_down
                ),
            });
        }

        let col_u32 = col as u32;
        let row_u32 = row as u32;
        if let Some(image) = self.image_for_tile(col_u32, row_u32) {
            return image.read_tile(col, row, level, backend);
        }

        let (width, height) = self.actual_tile_dimensions(col_u32, row_u32);
        Ok(black_sample_buffer(width, height))
    }

    fn read_raw_compressed_tile(
        &self,
        col: i64,
        row: i64,
        level: u32,
    ) -> Result<RawCompressedTile, WsiError> {
        if col < 0 || row < 0 || col >= self.tiles_across as i64 || row >= self.tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    self.tiles_across, self.tiles_down
                ),
            });
        }

        let col_u32 = col as u32;
        let row_u32 = row as u32;
        for image in &self.parts {
            if image.frame_index(col_u32, row_u32).is_some() {
                return image.read_raw_compressed_tile(col, row, level);
            }
        }

        Err(WsiError::Unsupported {
            reason: format!(
                "raw compressed tile access is not available for sparse missing DICOM tile ({col}, {row}) at level {level}"
            ),
        })
    }

    fn actual_tile_dimensions(&self, col: u32, row: u32) -> (u32, u32) {
        let tile_x = col * self.tile_width;
        let tile_y = row * self.tile_height;
        let width = self.width.saturating_sub(tile_x).min(self.tile_width);
        let height = self.height.saturating_sub(tile_y).min(self.tile_height);
        (width, height)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssociatedKind {
    Label,
    Macro,
    Thumbnail,
}

impl AssociatedKind {
    fn name(self) -> &'static str {
        match self {
            Self::Label => "label",
            Self::Macro => "macro",
            Self::Thumbnail => "thumbnail",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ImageRole {
    Level,
    Associated(AssociatedKind),
    Ignore,
}

struct DicomSeriesManifest {
    study_instance_uid: Option<String>,
    series_instance_uid: String,
    frame_of_reference_uid: Option<String>,
    container_identifier: Option<String>,
    specimen_identifier: Option<String>,
    volume_images: Vec<ParsedDicomMetadata>,
    associated_images: Vec<(AssociatedKind, ParsedDicomMetadata)>,
    source_file_count: usize,
}

impl DicomSeriesManifest {
    fn resolve(path: &Path) -> Result<Self, WsiError> {
        if path.is_dir() {
            Self::from_directory(path)
        } else {
            Self::from_selected_file(path)
        }
    }

    fn from_selected_file(path: &Path) -> Result<Self, WsiError> {
        let selected_meta = parse_metadata_object(path)?;
        let selected_series_uid = selected_meta.series_instance_uid.clone();
        let scan_root = path.parent().unwrap_or_else(|| Path::new("."));
        let selected_key = canonicalize_or_fallback(path);
        let mut metas = vec![selected_meta];

        for sibling_path in direct_child_files(scan_root)? {
            if canonicalize_or_fallback(&sibling_path) == selected_key {
                continue;
            }
            let meta = match parse_metadata_object(&sibling_path) {
                Ok(meta) => meta,
                Err(_) => continue,
            };
            if meta.series_instance_uid == selected_series_uid {
                metas.push(meta);
            }
        }

        Self::from_group(path, metas)
    }

    fn from_directory(path: &Path) -> Result<Self, WsiError> {
        let mut by_series = HashMap::<String, Vec<ParsedDicomMetadata>>::new();
        for child_path in direct_child_files(path)? {
            let meta = match parse_metadata_object(&child_path) {
                Ok(meta) => meta,
                Err(_) => continue,
            };
            by_series
                .entry(meta.series_instance_uid.clone())
                .or_default()
                .push(meta);
        }

        if by_series.is_empty() {
            return Err(WsiError::UnsupportedFormat(path.display().to_string()));
        }
        if by_series.len() != 1 {
            return Err(invalid_slide(
                path,
                format!(
                    "DICOM directory contains {} VL WSI series; select a directory containing exactly one series",
                    by_series.len()
                ),
            ));
        }

        let metas = by_series
            .into_values()
            .next()
            .expect("series map is known to contain one entry");
        Self::from_group(path, metas)
    }

    fn from_group(path: &Path, metas: Vec<ParsedDicomMetadata>) -> Result<Self, WsiError> {
        let first = metas
            .first()
            .ok_or_else(|| invalid_slide(path, "No DICOM VL WSI objects found"))?;
        let series_instance_uid = first.series_instance_uid.clone();
        let study_instance_uid = common_optional_value(path, "StudyInstanceUID", &metas, |meta| {
            meta.study_instance_uid.as_deref()
        })?;
        let frame_of_reference_uid =
            common_optional_value(path, "FrameOfReferenceUID", &metas, |meta| {
                meta.frame_of_reference_uid.as_deref()
            })?;
        let container_identifier =
            common_optional_value(path, "ContainerIdentifier", &metas, |meta| {
                meta.container_identifier.as_deref()
            })?;
        let specimen_identifier =
            common_optional_value(path, "SpecimenIdentifier", &metas, |meta| {
                meta.specimen_identifier.as_deref()
            })?;
        let source_file_count = metas.len();

        for meta in &metas {
            if meta.series_instance_uid != series_instance_uid {
                return Err(invalid_slide(
                    path,
                    "DICOM series resolver received mixed SeriesInstanceUID values",
                ));
            }
        }

        let mut volume_images = Vec::new();
        let mut associated_images = Vec::new();
        for meta in metas {
            match meta.classify()? {
                ImageRole::Ignore => {}
                ImageRole::Level => volume_images.push(meta),
                ImageRole::Associated(kind) => associated_images.push((kind, meta)),
            }
        }

        Ok(Self {
            study_instance_uid,
            series_instance_uid,
            frame_of_reference_uid,
            container_identifier,
            specimen_identifier,
            volume_images,
            associated_images,
            source_file_count,
        })
    }
}

fn direct_child_files(dir: &Path) -> Result<Vec<PathBuf>, WsiError> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: dir.to_path_buf(),
    })? {
        let entry = entry.map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: dir.to_path_buf(),
        })?;
        let path = entry.path();
        if path.is_file() {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn common_optional_value<F>(
    path: &Path,
    name: &str,
    metas: &[ParsedDicomMetadata],
    value: F,
) -> Result<Option<String>, WsiError>
where
    F: Fn(&ParsedDicomMetadata) -> Option<&str>,
{
    let mut common = None::<String>;
    for meta in metas {
        let Some(actual) = value(meta) else {
            continue;
        };
        match &common {
            Some(expected) if expected != actual => {
                return Err(invalid_slide(
                    path,
                    format!(
                        "DICOM series has incompatible {name} values ({expected} vs. {actual})"
                    ),
                ));
            }
            Some(_) => {}
            None => common = Some(actual.to_string()),
        }
    }
    Ok(common)
}

#[derive(Debug)]
struct DicomImage {
    path: PathBuf,
    sop_instance_uid: String,
    transfer_syntax_uid: String,
    photometric_interpretation: String,
    samples_per_pixel: u16,
    planar_configuration: Option<u16>,
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    number_of_frames: u32,
    grid: DicomGrid,
    pixel_spacing: Option<(f64, f64)>,
    objective_lens_power: Option<f64>,
    encapsulated_frames: Mutex<Option<Arc<DicomEncapsulatedFrames>>>,
    encapsulated_frame_cache: Mutex<LruCache<u32, Arc<Vec<u8>>>>,
    decoded_frame_cache: Mutex<LruCache<u32, Arc<CpuTile>>>,
    file: Mutex<Option<File>>,
}

#[derive(Debug)]
enum DicomGrid {
    Full,
    Sparse(HashMap<(u32, u32), u32>),
}

#[derive(Clone, Copy, Debug)]
struct DicomFragmentRef {
    payload_offset: u64,
    item_offset: u64,
    len: u32,
}

#[derive(Debug)]
struct DicomEncapsulatedFrames {
    fragments: Vec<DicomFragmentRef>,
    frame_ranges: Vec<std::ops::Range<usize>>,
}

impl DicomImage {
    fn from_metadata(meta: ParsedDicomMetadata) -> Result<Self, WsiError> {
        let width = meta.total_pixel_matrix_columns.unwrap_or(meta.columns);
        let height = meta.total_pixel_matrix_rows.unwrap_or(meta.rows);
        let tile_width = meta.columns;
        let tile_height = meta.rows;
        let tiles_across = width.div_ceil(tile_width);
        let tiles_down = height.div_ceil(tile_height);
        let grid = if meta.dimension_organization_type.as_deref() == Some("TILED_SPARSE") {
            DicomGrid::Sparse(parse_sparse_tile_map(&meta.obj, tile_width, tile_height)?)
        } else {
            DicomGrid::Full
        };
        let frame_cache_entries =
            if JP2K_TRANSFER_SYNTAXES.contains(&meta.transfer_syntax_uid.as_str()) {
                2
            } else {
                1
            };
        Ok(Self {
            path: meta.path,
            sop_instance_uid: meta.sop_instance_uid,
            transfer_syntax_uid: meta.transfer_syntax_uid,
            photometric_interpretation: meta.photometric_interpretation,
            samples_per_pixel: meta.samples_per_pixel,
            planar_configuration: meta.planar_configuration,
            width,
            height,
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
            number_of_frames: meta.number_of_frames,
            grid,
            pixel_spacing: meta.pixel_spacing,
            objective_lens_power: meta.objective_lens_power,
            encapsulated_frames: Mutex::new(None),
            encapsulated_frame_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(frame_cache_entries).unwrap(),
            )),
            decoded_frame_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(frame_cache_entries).unwrap(),
            )),
            file: Mutex::new(None),
        })
    }

    fn read_tile(
        &self,
        col: i64,
        row: i64,
        level: u32,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let span = tracing::info_span!(
            "dicom_read_tile",
            reader = "statumen",
            transfer_syntax = %self.transfer_syntax_uid,
        );
        let _guard = span.enter();
        if col < 0 || row < 0 || col >= self.tiles_across as i64 || row >= self.tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    self.tiles_across, self.tiles_down
                ),
            });
        }

        let col_u32 = col as u32;
        let row_u32 = row as u32;
        let Some(frame_index) = self.frame_index(col_u32, row_u32) else {
            let (width, height) = self.actual_tile_dimensions(col_u32, row_u32);
            return Ok(black_sample_buffer(width, height));
        };

        let (actual_width, actual_height) = self.actual_tile_dimensions(col_u32, row_u32);
        let buffer = self.decode_frame_sample_buffer(frame_index, level, col, row, backend)?;
        Ok(crop_sample_buffer_rgb(buffer, actual_width, actual_height))
    }

    fn read_raw_compressed_tile(
        &self,
        col: i64,
        row: i64,
        level: u32,
    ) -> Result<RawCompressedTile, WsiError> {
        if col < 0 || row < 0 || col >= self.tiles_across as i64 || row >= self.tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    self.tiles_across, self.tiles_down
                ),
            });
        }

        let col_u32 = col as u32;
        let row_u32 = row as u32;
        let Some(frame_index) = self.frame_index(col_u32, row_u32) else {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "raw compressed tile access is not available for sparse missing DICOM tile ({col}, {row}) at level {level}"
                ),
            });
        };
        let compression = raw_compression_for_transfer_syntax(
            &self.transfer_syntax_uid,
            &self.photometric_interpretation,
        )?;
        let photometric_interpretation = raw_photometric_interpretation(
            self.samples_per_pixel,
            &self.photometric_interpretation,
        )?;
        let bytes = self.extract_encapsulated_frame(frame_index, level, col, row, true)?;
        let mut data = bytes.as_ref().clone();
        trim_encapsulated_frame_padding(&mut data);

        Ok(RawCompressedTile {
            compression,
            width: self.tile_width,
            height: self.tile_height,
            bits_allocated: 8,
            samples_per_pixel: self.samples_per_pixel,
            photometric_interpretation,
            data,
        })
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let buffer = self
            .decode_frame_sample_buffer(0, 0, 0, 0, BackendRequest::Auto)
            .map_err(|err| match err {
                WsiError::TileRead { reason, .. } => {
                    WsiError::AssociatedImageNotFound(format!("{name}: {reason}"))
                }
                other => other,
            })?;
        Ok(crop_sample_buffer_rgb(buffer, self.width, self.height))
    }

    fn frame_index(&self, col: u32, row: u32) -> Option<u32> {
        match &self.grid {
            DicomGrid::Full => Some(row * self.tiles_across + col),
            DicomGrid::Sparse(map) => map.get(&(col, row)).copied(),
        }
    }

    fn is_full_grid(&self) -> bool {
        matches!(self.grid, DicomGrid::Full)
    }

    fn actual_tile_dimensions(&self, col: u32, row: u32) -> (u32, u32) {
        let tile_x = col * self.tile_width;
        let tile_y = row * self.tile_height;
        let width = self.width.saturating_sub(tile_x).min(self.tile_width);
        let height = self.height.saturating_sub(tile_y).min(self.tile_height);
        (width, height)
    }

    fn cached_decoded_frame(&self, frame_index: u32) -> Option<Arc<CpuTile>> {
        self.decoded_frame_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&frame_index)
            .cloned()
    }

    fn cache_decoded_frame(&self, frame_index: u32, tile: Arc<CpuTile>) {
        self.decoded_frame_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(frame_index, tile);
    }

    fn decode_uncompressed_frame_sample_buffer(
        &self,
        frame_index: u32,
        level: u32,
        col: i64,
        row: i64,
    ) -> Result<CpuTile, WsiError> {
        let obj = reopen_dicom_object(&self.path)?;
        let pixel_data = obj
            .element(tags::PIXEL_DATA)
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!("missing pixel data: {err}"),
            })?
            .to_bytes()
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!("failed to read DICOM pixel data: {err}"),
            })?;
        let frame_len = (self.tile_width as usize)
            .checked_mul(self.tile_height as usize)
            .and_then(|pixels| pixels.checked_mul(self.samples_per_pixel as usize))
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: "DICOM frame size overflow".into(),
            })?;
        let start = (frame_index as usize)
            .checked_mul(frame_len)
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: "DICOM frame offset overflow".into(),
            })?;
        let end = start
            .checked_add(frame_len)
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: "DICOM frame byte range overflow".into(),
            })?;
        if end > pixel_data.len() {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "DICOM frame {frame_index} byte range {}..{} exceeds pixel data length {}",
                    start,
                    end,
                    pixel_data.len()
                ),
            });
        }
        frame_bytes_to_rgb_tile(
            &pixel_data[start..end],
            self.tile_width,
            self.tile_height,
            self.samples_per_pixel,
            self.planar_configuration.unwrap_or(0),
            &self.photometric_interpretation,
        )
        .map_err(|err| WsiError::TileRead {
            col,
            row,
            level,
            reason: err.to_string(),
        })
    }

    fn decode_frame_sample_buffer(
        &self,
        frame_index: u32,
        level: u32,
        col: i64,
        row: i64,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let use_decoded_cache = is_encapsulated_transfer_syntax(&self.transfer_syntax_uid);
        if use_decoded_cache {
            if let Some(cached) = self.cached_decoded_frame(frame_index) {
                return Ok(cached.as_ref().clone());
            }
        }

        let buffer = if self.transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
            let bytes =
                self.extract_encapsulated_frame(frame_index, level, col, row, !use_decoded_cache)?;
            decode_batch_jpeg(&[JpegDecodeJob {
                data: Cow::Borrowed(bytes.as_slice()),
                tables: None,
                expected_width: self.tile_width,
                expected_height: self.tile_height,
                color_transform: signinum_jpeg::ColorTransform::Auto,
                force_dimensions: false,
                requested_size: None,
            }])
            .into_iter()
            .next()
            .expect("1-element JPEG facade batch")
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
        } else if JP2K_TRANSFER_SYNTAXES.contains(&self.transfer_syntax_uid.as_str()) {
            let bytes =
                self.extract_encapsulated_frame(frame_index, level, col, row, !use_decoded_cache)?;
            decode_batch_jp2k(&[Jp2kDecodeJob {
                data: Cow::Borrowed(bytes.as_slice()),
                expected_width: self.tile_width,
                expected_height: self.tile_height,
                rgb_color_space: !matches!(
                    self.photometric_interpretation.as_str(),
                    "YBR_ICT" | "YBR_RCT"
                ),
                backend,
            }])
            .into_iter()
            .next()
            .expect("1-element JP2K facade batch")
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
        } else if self.transfer_syntax_uid == RLE_TRANSFER_SYNTAX {
            let bytes =
                self.extract_encapsulated_frame(frame_index, level, col, row, !use_decoded_cache)?;
            decode_rle_lossless_frame(
                bytes.as_slice(),
                self.tile_width,
                self.tile_height,
                self.samples_per_pixel,
                &self.photometric_interpretation,
            )
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
        } else {
            self.decode_uncompressed_frame_sample_buffer(frame_index, level, col, row)?
        };

        let arc = Arc::new(buffer);
        if use_decoded_cache {
            self.cache_decoded_frame(frame_index, arc.clone());
        }
        Ok(arc.as_ref().clone())
    }

    fn extract_encapsulated_frame(
        &self,
        frame_index: u32,
        level: u32,
        col: i64,
        row: i64,
        cache_result: bool,
    ) -> Result<Arc<Vec<u8>>, WsiError> {
        if is_encapsulated_transfer_syntax(&self.transfer_syntax_uid) {
            if cache_result {
                if let Some(bytes) = self
                    .encapsulated_frame_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&frame_index)
                    .cloned()
                {
                    return Ok(bytes);
                }
            }
            let encapsulated_frames = self.ensure_encapsulated_frames()?;
            let frame_range = encapsulated_frames
                .frame_ranges
                .get(frame_index as usize)
                .ok_or_else(|| WsiError::TileRead {
                    col,
                    row,
                    level,
                    reason: format!(
                        "encapsulated frame {frame_index} out of range for {} frames",
                        encapsulated_frames.frame_ranges.len()
                    ),
                })?;
            let bytes = Arc::new(self.read_encapsulated_fragments(
                &encapsulated_frames.fragments[frame_range.start..frame_range.end],
            )?);
            if cache_result {
                self.encapsulated_frame_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .put(frame_index, bytes.clone());
            }
            return Ok(bytes);
        }

        let obj = reopen_dicom_object(&self.path)?;
        let pixel_data = obj
            .element(tags::PIXEL_DATA)
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!("missing pixel data: {err}"),
            })?;
        let fragments = pixel_data.fragments().ok_or_else(|| WsiError::TileRead {
            col,
            row,
            level,
            reason: "pixel data is not encapsulated".into(),
        })?;
        let number_of_frames = optional_u32(&obj, tags::NUMBER_OF_FRAMES)
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
            .unwrap_or(1);

        if number_of_frames == 1 && fragments.len() > 1 {
            let total_len = fragments.iter().map(Vec::len).sum();
            let mut data = Vec::with_capacity(total_len);
            for fragment in fragments {
                data.extend_from_slice(fragment);
            }
            return Ok(Arc::new(data));
        }

        fragments
            .get(frame_index as usize)
            .map(|fragment| Arc::new(fragment.as_slice().to_vec()))
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "encapsulated frame {frame_index} out of range for {} fragments",
                    fragments.len()
                ),
            })
    }

    fn ensure_encapsulated_frames(&self) -> Result<Arc<DicomEncapsulatedFrames>, WsiError> {
        let mut guard = self
            .encapsulated_frames
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(frames) = &*guard {
            return Ok(frames.clone());
        }
        let frames = Arc::new(scan_encapsulated_frames(
            &self.path,
            &self.transfer_syntax_uid,
            self.number_of_frames,
        )?);
        *guard = Some(frames.clone());
        Ok(frames)
    }

    fn read_encapsulated_fragments(
        &self,
        fragments: &[DicomFragmentRef],
    ) -> Result<Vec<u8>, WsiError> {
        let total_len: usize = fragments.iter().map(|fragment| fragment.len as usize).sum();
        let mut data = Vec::with_capacity(total_len);
        self.with_open_file(|file| {
            for fragment in fragments {
                file.seek(SeekFrom::Start(fragment.payload_offset))
                    .map_err(|source| WsiError::IoWithPath {
                        source: Arc::new(source),
                        path: self.path.clone(),
                    })?;
                let start = data.len();
                data.resize(start + fragment.len as usize, 0);
                file.read_exact(&mut data[start..])
                    .map_err(|source| WsiError::IoWithPath {
                        source: Arc::new(source),
                        path: self.path.clone(),
                    })?;
            }
            Ok(())
        })?;
        Ok(data)
    }

    fn with_open_file<T>(
        &self,
        f: impl FnOnce(&mut File) -> Result<T, WsiError>,
    ) -> Result<T, WsiError> {
        let mut guard = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let file = File::open(&self.path).map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
            *guard = Some(file);
        }
        let file = guard.as_mut().expect("file must be initialized");
        f(file)
    }
}

fn reopen_dicom_object(path: &Path) -> Result<DefaultDicomObject, WsiError> {
    dicom_object::open_file(path).map_err(|source| WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: format!("failed to reopen DICOM object: {source}"),
    })
}

fn scan_encapsulated_frames(
    path: &Path,
    transfer_syntax_uid: &str,
    number_of_frames: u32,
) -> Result<DicomEncapsulatedFrames, WsiError> {
    let transfer_syntax = TransferSyntaxRegistry
        .get(transfer_syntax_uid)
        .or_else(|| {
            JP2K_TRANSFER_SYNTAXES
                .contains(&transfer_syntax_uid)
                .then(|| TransferSyntaxRegistry.get(uids::EXPLICIT_VR_LITTLE_ENDIAN))
                .flatten()
        })
        .ok_or_else(|| {
            invalid_slide(
                path,
                format!("unknown transfer syntax {transfer_syntax_uid}"),
            )
        })?;
    let mut reader = BufReader::new(File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?);
    position_reader_for_dicom_magic(&mut reader, path)?;
    let _meta = FileMetaTable::from_reader(&mut reader)
        .map_err(|source| invalid_slide(path, format!("cannot parse DICOM file meta: {source}")))?;
    let mut tokens = LazyDataSetReader::new_with_ts(reader, transfer_syntax)
        .map_err(|source| invalid_slide(path, format!("cannot stream DICOM dataset: {source}")))?;

    let mut in_pixel_sequence = false;
    let mut awaiting_offset_table = false;
    let mut offset_table = Vec::new();
    let mut fragments = Vec::new();

    while let Some(token) = tokens.advance() {
        let token = token
            .map_err(|source| invalid_slide(path, format!("cannot read DICOM token: {source}")))?;
        match token {
            LazyDataToken::PixelSequenceStart => {
                in_pixel_sequence = true;
                awaiting_offset_table = true;
            }
            LazyDataToken::ItemStart { len }
                if in_pixel_sequence && awaiting_offset_table && len.0 == 0 =>
            {
                awaiting_offset_table = false;
            }
            LazyDataToken::LazyItemValue { len, decoder }
                if in_pixel_sequence && awaiting_offset_table =>
            {
                decoder
                    .read_u32_to_vec(len, &mut offset_table)
                    .map_err(|source| {
                        invalid_slide(
                            path,
                            format!("cannot read DICOM basic offset table: {source}"),
                        )
                    })?;
                awaiting_offset_table = false;
            }
            LazyDataToken::LazyItemValue { len, decoder } if in_pixel_sequence => {
                let payload_offset = decoder.position();
                let item_offset = payload_offset.saturating_sub(8);
                decoder.skip_bytes(len).map_err(|source| {
                    invalid_slide(path, format!("cannot skip DICOM fragment: {source}"))
                })?;
                fragments.push(DicomFragmentRef {
                    payload_offset,
                    item_offset,
                    len,
                });
            }
            LazyDataToken::ItemStart { len } if in_pixel_sequence && len.0 == 0 => {
                return Err(invalid_slide(
                    path,
                    "zero-length DICOM pixel fragment is not supported",
                ));
            }
            LazyDataToken::SequenceEnd if in_pixel_sequence => break,
            other => {
                other.skip().map_err(|source| {
                    invalid_slide(path, format!("cannot skip DICOM token: {source}"))
                })?;
            }
        }
    }

    if fragments.is_empty() {
        if let Some(frames) = scan_encapsulated_frames_raw_little_endian(path, number_of_frames)? {
            return Ok(frames);
        }
    }

    build_encapsulated_frame_index(path, fragments, offset_table, number_of_frames)
}

const PIXEL_DATA_TAG_LE: [u8; 4] = [0xE0, 0x7F, 0x10, 0x00];
const DICOM_ITEM_TAG_LE: [u8; 4] = [0xFE, 0xFF, 0x00, 0xE0];
const DICOM_SEQUENCE_DELIMITER_TAG_LE: [u8; 4] = [0xFE, 0xFF, 0xDD, 0xE0];
const UNDEFINED_LENGTH_LE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
const EXPLICIT_VR_LONG_HEADER_LEN: usize = 12;

fn scan_encapsulated_frames_raw_little_endian(
    path: &Path,
    number_of_frames: u32,
) -> Result<Option<DicomEncapsulatedFrames>, WsiError> {
    let mut file = File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    let Some(pixel_data_offset) = find_encapsulated_pixel_data_offset_le(&mut file, path)? else {
        return Ok(None);
    };

    let (fragments, offset_table) =
        scan_raw_encapsulated_pixel_sequence(&mut file, path, pixel_data_offset)?;
    build_encapsulated_frame_index(path, fragments, offset_table, number_of_frames).map(Some)
}

fn find_encapsulated_pixel_data_offset_le(
    file: &mut File,
    path: &Path,
) -> Result<Option<u64>, WsiError> {
    const CHUNK_LEN: usize = 64 * 1024;
    let mut chunk = [0u8; CHUNK_LEN];
    let mut overlap = Vec::new();
    let mut chunk_offset = 0u64;

    file.seek(SeekFrom::Start(0))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;

    loop {
        let read_len = file
            .read(&mut chunk)
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: path.to_path_buf(),
            })?;
        if read_len == 0 {
            return Ok(None);
        }

        let window_offset = chunk_offset.saturating_sub(overlap.len() as u64);
        let mut window = Vec::with_capacity(overlap.len() + read_len);
        window.extend_from_slice(&overlap);
        window.extend_from_slice(&chunk[..read_len]);

        for index in 0..=window.len().saturating_sub(EXPLICIT_VR_LONG_HEADER_LEN) {
            let header = &window[index..index + EXPLICIT_VR_LONG_HEADER_LEN];
            if is_encapsulated_pixel_data_header_le(header) {
                return Ok(Some(window_offset + index as u64));
            }
        }

        let keep = window.len().min(EXPLICIT_VR_LONG_HEADER_LEN - 1);
        overlap.clear();
        overlap.extend_from_slice(&window[window.len() - keep..]);
        chunk_offset = chunk_offset
            .checked_add(read_len as u64)
            .ok_or_else(|| invalid_slide(path, "DICOM raw Pixel Data scan offset overflow"))?;
    }
}

fn is_encapsulated_pixel_data_header_le(header: &[u8]) -> bool {
    header.len() >= EXPLICIT_VR_LONG_HEADER_LEN
        && header[0..4] == PIXEL_DATA_TAG_LE
        && matches!(&header[4..6], b"OB" | b"OW" | b"UN")
        && header[6..8] == [0, 0]
        && header[8..12] == UNDEFINED_LENGTH_LE
}

fn scan_raw_encapsulated_pixel_sequence(
    file: &mut File,
    path: &Path,
    pixel_data_offset: u64,
) -> Result<(Vec<DicomFragmentRef>, Vec<u32>), WsiError> {
    let mut cursor = pixel_data_offset
        .checked_add(EXPLICIT_VR_LONG_HEADER_LEN as u64)
        .ok_or_else(|| invalid_slide(path, "DICOM raw Pixel Data offset overflow"))?;
    let mut offset_table = None;
    let mut fragments = Vec::new();

    loop {
        let mut item_header = [0u8; 8];
        read_exact_at(file, path, cursor, &mut item_header)?;
        let tag = &item_header[0..4];
        let len = u32::from_le_bytes(
            item_header[4..8]
                .try_into()
                .expect("DICOM item length header is 4 bytes"),
        );
        cursor = cursor
            .checked_add(item_header.len() as u64)
            .ok_or_else(|| invalid_slide(path, "DICOM raw item offset overflow"))?;

        if tag == DICOM_SEQUENCE_DELIMITER_TAG_LE {
            return Ok((fragments, offset_table.unwrap_or_default()));
        }
        if tag != DICOM_ITEM_TAG_LE {
            return Err(invalid_slide(
                path,
                format!(
                    "unexpected DICOM pixel sequence tag {:02x?} at byte {}",
                    tag,
                    cursor - item_header.len() as u64
                ),
            ));
        }

        if offset_table.is_none() {
            offset_table = Some(read_basic_offset_table_at(file, path, cursor, len)?);
        } else {
            if len == 0 {
                return Err(invalid_slide(
                    path,
                    "zero-length DICOM pixel fragment is not supported",
                ));
            }
            fragments.push(DicomFragmentRef {
                payload_offset: cursor,
                item_offset: cursor - item_header.len() as u64,
                len,
            });
        }

        cursor = cursor
            .checked_add(len as u64)
            .ok_or_else(|| invalid_slide(path, "DICOM raw item payload offset overflow"))?;
    }
}

fn read_basic_offset_table_at(
    file: &mut File,
    path: &Path,
    offset: u64,
    len: u32,
) -> Result<Vec<u32>, WsiError> {
    if !len.is_multiple_of(4) {
        return Err(invalid_slide(
            path,
            format!("DICOM basic offset table has non-u32 length {len}"),
        ));
    }
    let len = usize::try_from(len)
        .map_err(|_| invalid_slide(path, "DICOM basic offset table length overflow"))?;
    let mut bytes = vec![0u8; len];
    read_exact_at(file, path, offset, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            u32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("DICOM basic offset table chunk is 4 bytes"),
            )
        })
        .collect())
}

fn read_exact_at(
    file: &mut File,
    path: &Path,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), WsiError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    file.read_exact(buf).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })
}

fn build_encapsulated_frame_index(
    path: &Path,
    fragments: Vec<DicomFragmentRef>,
    offset_table: Vec<u32>,
    number_of_frames: u32,
) -> Result<DicomEncapsulatedFrames, WsiError> {
    if number_of_frames == 0 {
        return Err(invalid_slide(path, "DICOM reported zero frames"));
    }
    if fragments.is_empty() {
        return Err(invalid_slide(
            path,
            "DICOM encapsulated pixel data has no fragments",
        ));
    }

    let frame_ranges = if number_of_frames == 1 {
        std::iter::once(0..fragments.len()).collect()
    } else if !offset_table.is_empty() {
        let base_item_offset = fragments[0].item_offset;
        let fragment_indices_by_offset: HashMap<u64, usize> = fragments
            .iter()
            .enumerate()
            .map(|(index, fragment)| (fragment.item_offset, index))
            .collect();
        let mut start_indices = Vec::with_capacity(offset_table.len());
        for offset in &offset_table {
            let target = base_item_offset + *offset as u64;
            let index = fragment_indices_by_offset
                .get(&target)
                .copied()
                .ok_or_else(|| {
                    invalid_slide(
                        path,
                        format!(
                            "DICOM basic offset table points to missing fragment offset {offset}"
                        ),
                    )
                })?;
            start_indices.push(index);
        }
        if start_indices.len() != number_of_frames as usize {
            return Err(invalid_slide(
                path,
                format!(
                    "DICOM basic offset table length {} does not match number_of_frames {}",
                    start_indices.len(),
                    number_of_frames
                ),
            ));
        }
        let mut ranges = Vec::with_capacity(start_indices.len());
        for (frame, start) in start_indices.iter().copied().enumerate() {
            let end = start_indices
                .get(frame + 1)
                .copied()
                .unwrap_or(fragments.len());
            ranges.push(start..end);
        }
        ranges
    } else if fragments.len() == number_of_frames as usize {
        (0..fragments.len()).map(|index| index..index + 1).collect()
    } else {
        return Err(invalid_slide(
            path,
            format!(
                "cannot map {} DICOM fragments to {} frames without a basic offset table",
                fragments.len(),
                number_of_frames
            ),
        ));
    };

    Ok(DicomEncapsulatedFrames {
        fragments,
        frame_ranges,
    })
}

fn position_reader_for_dicom_magic<R: Read + Seek>(
    reader: &mut R,
    path: &Path,
) -> Result<(), WsiError> {
    let mut preamble = [0u8; 132];
    reader
        .read_exact(&mut preamble)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let start = if &preamble[128..] == b"DICM" { 128 } else { 0 };
    reader
        .seek(SeekFrom::Start(start))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    Ok(())
}

struct ParsedDicomMetadata {
    path: PathBuf,
    obj: DefaultDicomObject,
    study_instance_uid: Option<String>,
    series_instance_uid: String,
    frame_of_reference_uid: Option<String>,
    container_identifier: Option<String>,
    specimen_identifier: Option<String>,
    sop_instance_uid: String,
    transfer_syntax_uid: String,
    photometric_interpretation: String,
    samples_per_pixel: u16,
    planar_configuration: Option<u16>,
    image_type: Vec<String>,
    rows: u32,
    columns: u32,
    number_of_frames: u32,
    total_pixel_matrix_columns: Option<u32>,
    total_pixel_matrix_rows: Option<u32>,
    dimension_organization_type: Option<String>,
    pixel_spacing: Option<(f64, f64)>,
    objective_lens_power: Option<f64>,
}

impl ParsedDicomMetadata {
    fn classify(&self) -> Result<ImageRole, WsiError> {
        let image_type_refs = self
            .image_type
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        if matches_type(&image_type_refs, LEVEL_IMAGE_TYPES) {
            validate_supported_pixel_format(self)?;
            return Ok(ImageRole::Level);
        }
        if matches_type(&image_type_refs, LABEL_IMAGE_TYPES) {
            validate_supported_pixel_format(self)?;
            return Ok(ImageRole::Associated(AssociatedKind::Label));
        }
        if matches_type(&image_type_refs, OVERVIEW_IMAGE_TYPES) {
            validate_supported_pixel_format(self)?;
            return Ok(ImageRole::Associated(AssociatedKind::Macro));
        }
        if matches_type(&image_type_refs, THUMBNAIL_IMAGE_TYPES) {
            validate_supported_pixel_format(self)?;
            return Ok(ImageRole::Associated(AssociatedKind::Thumbnail));
        }
        Ok(ImageRole::Ignore)
    }
}

fn parse_metadata_object(path: &Path) -> Result<ParsedDicomMetadata, WsiError> {
    // Stop after the top-level matrix geometry is available, but before pixel
    // data. This keeps cold-open cheap while still building the correct
    // pyramid geometry for tiled DICOM pyramids.
    let meta = parse_metadata_object_until(path, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE)?;
    if meta.dimension_organization_type.as_deref() == Some("TILED_SPARSE") {
        return parse_metadata_object_full(path);
    }
    Ok(meta)
}

fn parse_metadata_object_full(path: &Path) -> Result<ParsedDicomMetadata, WsiError> {
    parse_metadata_object_until(path, tags::PIXEL_DATA)
}

type Level0Properties = (Option<(f64, f64)>, Option<f64>);

fn parse_level0_properties(path: &Path) -> Result<Level0Properties, WsiError> {
    let obj = OpenFileOptions::new()
        .read_until(tags::PIXEL_DATA)
        .open_file(path)
        .map_err(|source| invalid_slide(path, format!("cannot parse DICOM metadata: {source}")))?;
    let pixel_spacing = optional_pixel_spacing_mpp(&obj)?;
    let objective_lens_power = optional_f64_at(
        &obj,
        (tags::OPTICAL_PATH_SEQUENCE, 0, tags::OBJECTIVE_LENS_POWER),
    )?;
    Ok((pixel_spacing, objective_lens_power))
}

#[cfg(test)]
fn parse_level0_properties_from_metadata(
    meta: &ParsedDicomMetadata,
) -> (Option<(f64, f64)>, Option<f64>) {
    let pixel_spacing = optional_pixel_spacing_mpp(&meta.obj).unwrap_or(None);
    let objective_lens_power = optional_f64_at(
        &meta.obj,
        (tags::OPTICAL_PATH_SEQUENCE, 0, tags::OBJECTIVE_LENS_POWER),
    )
    .unwrap_or(None);
    (pixel_spacing, objective_lens_power)
}

fn optional_pixel_spacing_mpp(obj: &DefaultDicomObject) -> Result<Option<(f64, f64)>, WsiError> {
    if let Some(spacing) = optional_pair_f64_at(
        obj,
        (
            tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE,
            0,
            tags::PIXEL_MEASURES_SEQUENCE,
            0,
            tags::PIXEL_SPACING,
        ),
    )? {
        return Ok(Some(spacing));
    }
    optional_pair_f64_at(obj, tags::PIXEL_SPACING)
}

fn parse_metadata_object_until(
    path: &Path,
    stop_tag: dicom_core::Tag,
) -> Result<ParsedDicomMetadata, WsiError> {
    if matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(ext) if ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff")
    ) {
        return Err(WsiError::UnsupportedFormat(format!(
            "Dual-personality DICOM-TIFF with TIFF extension: {}",
            path.display()
        )));
    }

    let obj = OpenFileOptions::new()
        .read_until(stop_tag)
        .open_file(path)
        .map_err(|source| invalid_slide(path, format!("cannot parse DICOM metadata: {source}")))?;

    if !is_vl_wsi(obj.meta().media_storage_sop_class_uid()) {
        return Err(WsiError::UnsupportedFormat(path.display().to_string()));
    }

    let series_instance_uid =
        required_string(&obj, tags::SERIES_INSTANCE_UID, "SeriesInstanceUID")?;
    let study_instance_uid = optional_string(&obj, tags::STUDY_INSTANCE_UID)?;
    let frame_of_reference_uid = optional_string(&obj, tags::FRAME_OF_REFERENCE_UID)?;
    let container_identifier = optional_string(&obj, tags::CONTAINER_IDENTIFIER)?;
    let specimen_identifier = optional_string(&obj, tags::SPECIMEN_IDENTIFIER)?;
    let sop_instance_uid = required_string(&obj, tags::SOP_INSTANCE_UID, "SOPInstanceUID")?;
    let image_type = required_multi_string(&obj, tags::IMAGE_TYPE, "ImageType")?;
    let rows = required_u32(&obj, tags::ROWS, "Rows")?;
    let columns = required_u32(&obj, tags::COLUMNS, "Columns")?;
    let number_of_frames = optional_u32(&obj, tags::NUMBER_OF_FRAMES)?.unwrap_or(1);
    let photometric_interpretation = required_string(
        &obj,
        tags::PHOTOMETRIC_INTERPRETATION,
        "PhotometricInterpretation",
    )?;
    let total_pixel_matrix_columns = optional_u32(&obj, tags::TOTAL_PIXEL_MATRIX_COLUMNS)?;
    let total_pixel_matrix_rows = optional_u32(&obj, tags::TOTAL_PIXEL_MATRIX_ROWS)?;
    let dimension_organization_type = optional_string(&obj, tags::DIMENSION_ORGANIZATION_TYPE)?;
    let pixel_spacing = if stop_tag == tags::PIXEL_DATA {
        optional_pixel_spacing_mpp(&obj)?
    } else {
        None
    };
    let samples_per_pixel = optional_u32(&obj, tags::SAMPLES_PER_PIXEL)?
        .unwrap_or(1)
        .try_into()
        .map_err(|_| WsiError::DisplayConversion("SamplesPerPixel out of range".into()))?;
    let planar_configuration = optional_u32(&obj, tags::PLANAR_CONFIGURATION)?
        .map(u16::try_from)
        .transpose()
        .map_err(|_| WsiError::DisplayConversion("PlanarConfiguration out of range".into()))?;
    let objective_lens_power = optional_f64_at(
        &obj,
        (tags::OPTICAL_PATH_SEQUENCE, 0, tags::OBJECTIVE_LENS_POWER),
    )?;

    let transfer_syntax_uid = String::from(obj.meta().transfer_syntax());

    Ok(ParsedDicomMetadata {
        path: path.to_path_buf(),
        obj,
        study_instance_uid,
        series_instance_uid,
        frame_of_reference_uid,
        container_identifier,
        specimen_identifier,
        sop_instance_uid,
        transfer_syntax_uid,
        photometric_interpretation,
        samples_per_pixel,
        planar_configuration,
        image_type,
        rows,
        columns,
        number_of_frames,
        total_pixel_matrix_columns,
        total_pixel_matrix_rows,
        dimension_organization_type,
        pixel_spacing,
        objective_lens_power,
    })
}

fn build_levels(path: &Path, images: Vec<Arc<DicomImage>>) -> Result<Vec<DicomLevel>, WsiError> {
    let mut by_dimensions = HashMap::<(u32, u32), usize>::new();
    let mut levels = Vec::<DicomLevel>::new();
    for image in images {
        let key = (image.width, image.height);
        if let Some(&level_index) = by_dimensions.get(&key) {
            levels[level_index].push_part(path, image)?;
            continue;
        }
        by_dimensions.insert(key, levels.len());
        levels.push(DicomLevel::from_image(image));
    }
    Ok(levels)
}

fn validate_monotonic_levels(path: &Path, levels: &[DicomLevel]) -> Result<(), WsiError> {
    for pair in levels.windows(2) {
        let finer = &pair[0];
        let coarser = &pair[1];
        if coarser.width > finer.width || coarser.height > finer.height {
            return Err(invalid_slide(
                path,
                format!(
                    "DICOM pyramid levels are not monotonic ({}x{} before {}x{})",
                    finer.width, finer.height, coarser.width, coarser.height
                ),
            ));
        }
    }
    Ok(())
}

fn reject_huge_base_only_dicom(path: &Path, levels: &[DicomLevel]) -> Result<(), WsiError> {
    let [level] = levels else {
        return Ok(());
    };
    if !level.is_regular_full_tiling() {
        return Ok(());
    }

    let tile_count = u64::from(level.tiles_across).saturating_mul(u64::from(level.tiles_down));
    let max_dimension = level.width.max(level.height);
    if tile_count >= BASE_ONLY_GUARD_MIN_TILE_COUNT
        || max_dimension >= BASE_ONLY_GUARD_MIN_DIMENSION
    {
        return Err(invalid_slide(path, BASE_ONLY_DICOM_PYRAMID_MESSAGE));
    }

    Ok(())
}

fn dedupe_associated(
    path: &Path,
    associated: &mut Vec<(String, Arc<DicomImage>)>,
) -> Result<(), WsiError> {
    let mut seen = HashMap::<String, Arc<DicomImage>>::new();
    let mut deduped = Vec::new();
    for (name, image) in associated.drain(..) {
        if let Some(previous) = seen.get(&name) {
            ensure_same_sop(path, &image.sop_instance_uid, &previous.sop_instance_uid)?;
            continue;
        }
        seen.insert(name.clone(), image.clone());
        deduped.push((name, image));
    }
    *associated = deduped;
    Ok(())
}

fn ensure_same_sop(path: &Path, current: &str, previous: &str) -> Result<(), WsiError> {
    if current == previous {
        Ok(())
    } else {
        Err(invalid_slide(
            path,
            format!("Slide contains unexpected image ({current} vs. {previous})"),
        ))
    }
}

fn validate_supported_pixel_format(meta: &ParsedDicomMetadata) -> Result<(), WsiError> {
    if !SUPPORTED_TRANSFER_SYNTAXES.contains(&meta.transfer_syntax_uid.as_str()) {
        return Err(invalid_slide(
            &meta.path,
            format!("Unsupported transfer syntax {}", meta.transfer_syntax_uid),
        ));
    }
    verify_required_int(
        &meta.obj,
        tags::BITS_ALLOCATED,
        8,
        "BitsAllocated",
        &meta.path,
    )?;
    verify_required_int(&meta.obj, tags::BITS_STORED, 8, "BitsStored", &meta.path)?;
    verify_required_int(&meta.obj, tags::HIGH_BIT, 7, "HighBit", &meta.path)?;
    match meta.samples_per_pixel {
        1 | 3 => {}
        value => {
            return Err(invalid_slide(
                &meta.path,
                format!("Attribute SamplesPerPixel value {value} is not supported"),
            ));
        }
    }
    verify_required_int(
        &meta.obj,
        tags::PIXEL_REPRESENTATION,
        0,
        "PixelRepresentation",
        &meta.path,
    )?;
    match (meta.samples_per_pixel, meta.planar_configuration) {
        (1, _) | (3, None | Some(0) | Some(1)) => {}
        (3, Some(value)) => {
            return Err(invalid_slide(
                &meta.path,
                format!("Attribute PlanarConfiguration value {value} is not supported"),
            ));
        }
        _ => {}
    }
    verify_optional_int(
        &meta.obj,
        tags::TOTAL_PIXEL_MATRIX_FOCAL_PLANES,
        1,
        "TotalPixelMatrixFocalPlanes",
        &meta.path,
    )?;

    let supported = if meta.samples_per_pixel == 1 {
        matches!(
            meta.photometric_interpretation.as_str(),
            "MONOCHROME1" | "MONOCHROME2"
        )
    } else if meta.transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
        meta.photometric_interpretation == "YBR_FULL_422"
            || meta.photometric_interpretation == "RGB"
    } else if JP2K_TRANSFER_SYNTAXES.contains(&meta.transfer_syntax_uid.as_str()) {
        matches!(
            meta.photometric_interpretation.as_str(),
            "YBR_ICT" | "YBR_RCT" | "RGB"
        )
    } else {
        meta.photometric_interpretation == "RGB"
    };
    if supported {
        Ok(())
    } else {
        Err(invalid_slide(
            &meta.path,
            format!(
                "Unsupported photometric interpretation {photometric} for {}",
                meta.transfer_syntax_uid,
                photometric = meta.photometric_interpretation
            ),
        ))
    }
}

fn parse_sparse_tile_map(
    obj: &DefaultDicomObject,
    tile_width: u32,
    tile_height: u32,
) -> Result<HashMap<(u32, u32), u32>, WsiError> {
    let mut map = HashMap::new();
    let items = obj
        .element(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE)
        .map_err(|_| {
            WsiError::DisplayConversion("missing PerFrameFunctionalGroupsSequence".into())
        })?
        .items()
        .ok_or_else(|| {
            WsiError::DisplayConversion("PerFrameFunctionalGroupsSequence is not a sequence".into())
        })?;

    for (frame_index, item) in items.iter().enumerate() {
        let col_position = required_u32_at_item(
            item,
            (
                tags::PLANE_POSITION_SLIDE_SEQUENCE,
                0,
                tags::COLUMN_POSITION_IN_TOTAL_IMAGE_PIXEL_MATRIX,
            ),
            "ColumnPositionInTotalImagePixelMatrix",
        )?;
        let row_position = required_u32_at_item(
            item,
            (
                tags::PLANE_POSITION_SLIDE_SEQUENCE,
                0,
                tags::ROW_POSITION_IN_TOTAL_IMAGE_PIXEL_MATRIX,
            ),
            "RowPositionInTotalImagePixelMatrix",
        )?;
        if col_position == 0 || row_position == 0 {
            return Err(WsiError::DisplayConversion(
                "DICOM sparse tile positions are 1-based and must be non-zero".into(),
            ));
        }
        let col = (col_position - 1) / tile_width;
        let row = (row_position - 1) / tile_height;
        map.insert((col, row), frame_index as u32);
    }
    Ok(map)
}

fn is_vl_wsi(sop_class_uid: &str) -> bool {
    sop_class_uid == uids::VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_STORAGE
}

fn matches_type(image_type: &[&str], allowed: &[&[&str]]) -> bool {
    allowed.contains(&image_type)
}

fn quickhash_for_series_uid(series_uid: &str) -> Result<String, WsiError> {
    let mut quickhash = Quickhash1::new();
    quickhash.hash_string(series_uid);
    quickhash
        .finish()
        .ok_or_else(|| WsiError::DisplayConversion("failed to compute DICOM quickhash".into()))
}

fn dataset_id_from_quickhash(path: &Path, quickhash: &str) -> Result<DatasetId, WsiError> {
    if quickhash.len() < 32 {
        return Err(invalid_slide(path, "quickhash too short"));
    }
    let value = u128::from_str_radix(&quickhash[..32], 16)
        .map_err(|_| invalid_slide(path, "quickhash is not valid hex"))?;
    Ok(DatasetId(value))
}

fn canonicalize_or_fallback(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn invalid_slide(path: &Path, message: impl Into<String>) -> WsiError {
    WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn required_string(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
    name: &str,
) -> Result<String, WsiError> {
    obj.element(tag)
        .map_err(|_| WsiError::DisplayConversion(format!("missing {name}")))?
        .to_str()
        .map(|value| value.trim_end_matches('\0').to_string())
        .map_err(|err| WsiError::DisplayConversion(format!("invalid {name}: {err}")))
}

fn required_multi_string(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
    name: &str,
) -> Result<Vec<String>, WsiError> {
    let raw = required_string(obj, tag, name)?;
    let values = raw
        .split('\\')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if values.len() == 4 {
        Ok(values)
    } else {
        Err(WsiError::DisplayConversion(format!(
            "{name} must have 4 values, got {}",
            values.len()
        )))
    }
}

fn optional_string(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
) -> Result<Option<String>, WsiError> {
    obj.get(tag)
        .map(|elem| {
            elem.to_str()
                .map(|value| value.trim_end_matches('\0').to_string())
                .map_err(|err| {
                    WsiError::DisplayConversion(format!("invalid DICOM string tag {tag:?}: {err}"))
                })
        })
        .transpose()
}

fn required_u32(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
    name: &str,
) -> Result<u32, WsiError> {
    obj.element(tag)
        .map_err(|_| WsiError::DisplayConversion(format!("missing {name}")))?
        .to_int::<u32>()
        .map_err(|err| WsiError::DisplayConversion(format!("invalid {name}: {err}")))
}

fn optional_u32(obj: &DefaultDicomObject, tag: dicom_core::Tag) -> Result<Option<u32>, WsiError> {
    obj.get(tag)
        .map(|elem| {
            elem.to_int::<u32>().map_err(|err| {
                WsiError::DisplayConversion(format!("invalid DICOM integer tag {tag:?}: {err}"))
            })
        })
        .transpose()
}

fn verify_required_int(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
    expected: u32,
    name: &str,
    path: &Path,
) -> Result<(), WsiError> {
    let value = required_u32(obj, tag, name)?;
    if value == expected {
        Ok(())
    } else {
        Err(invalid_slide(
            path,
            format!("Attribute {name} value {value} != {expected}"),
        ))
    }
}

fn verify_optional_int(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
    expected: u32,
    name: &str,
    path: &Path,
) -> Result<(), WsiError> {
    match optional_u32(obj, tag)? {
        Some(value) if value != expected => Err(invalid_slide(
            path,
            format!("Attribute {name} value {value} != {expected}"),
        )),
        _ => Ok(()),
    }
}

fn required_u32_at_item(
    obj: &dicom_object::InMemDicomObject,
    selector: impl Into<dicom_core::ops::AttributeSelector>,
    name: &str,
) -> Result<u32, WsiError> {
    obj.entry_at(selector)
        .map_err(|_| WsiError::DisplayConversion(format!("missing {name}")))?
        .to_int::<u32>()
        .map_err(|err| WsiError::DisplayConversion(format!("invalid {name}: {err}")))
}

fn optional_f64_at(
    obj: &DefaultDicomObject,
    selector: impl Into<dicom_core::ops::AttributeSelector>,
) -> Result<Option<f64>, WsiError> {
    match obj.entry_at(selector) {
        Ok(entry) => entry
            .to_float64()
            .map(Some)
            .map_err(|err| WsiError::DisplayConversion(format!("invalid DICOM float: {err}"))),
        Err(_) => Ok(None),
    }
}

fn optional_pair_f64_at(
    obj: &DefaultDicomObject,
    selector: impl Into<dicom_core::ops::AttributeSelector>,
) -> Result<Option<(f64, f64)>, WsiError> {
    let entry = match obj.entry_at(selector) {
        Ok(entry) => entry,
        Err(_) => return Ok(None),
    };
    let value = entry
        .to_str()
        .map_err(|err| WsiError::DisplayConversion(format!("invalid DICOM string pair: {err}")))?;
    let mut parts = value.split('\\');
    let first = parts
        .next()
        .and_then(|part| part.parse::<f64>().ok())
        .ok_or_else(|| WsiError::DisplayConversion("invalid DICOM float pair".into()))?;
    let second = parts
        .next()
        .and_then(|part| part.parse::<f64>().ok())
        .ok_or_else(|| WsiError::DisplayConversion("invalid DICOM float pair".into()))?;
    Ok(Some((second * 1000.0, first * 1000.0)))
}

fn frame_bytes_to_rgb_tile(
    frame_bytes: &[u8],
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    planar_configuration: u16,
    photometric_interpretation: &str,
) -> Result<CpuTile, WsiError> {
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| WsiError::DisplayConversion("DICOM frame dimensions overflow".into()))?;
    let rgb = match (samples_per_pixel, photometric_interpretation) {
        (3, "RGB") if planar_configuration == 0 => {
            let expected = pixel_count.checked_mul(3).ok_or_else(|| {
                WsiError::DisplayConversion("DICOM RGB frame size overflow".into())
            })?;
            if frame_bytes.len() != expected {
                return Err(WsiError::DisplayConversion(format!(
                    "DICOM RGB frame has {} bytes, expected {expected}",
                    frame_bytes.len()
                )));
            }
            frame_bytes.to_vec()
        }
        (3, "RGB") if planar_configuration == 1 => {
            let expected = pixel_count.checked_mul(3).ok_or_else(|| {
                WsiError::DisplayConversion("DICOM planar RGB frame size overflow".into())
            })?;
            if frame_bytes.len() != expected {
                return Err(WsiError::DisplayConversion(format!(
                    "DICOM planar RGB frame has {} bytes, expected {expected}",
                    frame_bytes.len()
                )));
            }
            let (r_plane, rest) = frame_bytes.split_at(pixel_count);
            let (g_plane, b_plane) = rest.split_at(pixel_count);
            let mut rgb = vec![0; expected];
            for idx in 0..pixel_count {
                let offset = idx * 3;
                rgb[offset] = r_plane[idx];
                rgb[offset + 1] = g_plane[idx];
                rgb[offset + 2] = b_plane[idx];
            }
            rgb
        }
        (1, "MONOCHROME1" | "MONOCHROME2") => {
            if frame_bytes.len() != pixel_count {
                return Err(WsiError::DisplayConversion(format!(
                    "DICOM monochrome frame has {} bytes, expected {pixel_count}",
                    frame_bytes.len()
                )));
            }
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for &gray in frame_bytes {
                // Preserve the legacy sv-slide behavior for consolidation:
                // MONOCHROME1 and MONOCHROME2 are both expanded without inversion.
                rgb.extend_from_slice(&[gray, gray, gray]);
            }
            rgb
        }
        _ => {
            return Err(WsiError::DisplayConversion(format!(
                "unsupported DICOM pixel format: samples_per_pixel={samples_per_pixel}, photometric={photometric_interpretation}, planar_configuration={planar_configuration}"
            )));
        }
    };

    CpuTile::new(
        width,
        height,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(rgb),
    )
}

fn decode_rle_lossless_frame(
    frame_bytes: &[u8],
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    photometric_interpretation: &str,
) -> Result<CpuTile, WsiError> {
    if frame_bytes.len() < 64 {
        return Err(WsiError::DisplayConversion(
            "DICOM RLE frame is shorter than its 64-byte header".into(),
        ));
    }
    let segment_count = u32::from_le_bytes(frame_bytes[0..4].try_into().unwrap()) as usize;
    if segment_count == 0 || segment_count > 15 {
        return Err(WsiError::DisplayConversion(format!(
            "DICOM RLE segment count {segment_count} is invalid"
        )));
    }
    let expected_segments = samples_per_pixel as usize;
    if segment_count < expected_segments {
        return Err(WsiError::DisplayConversion(format!(
            "DICOM RLE has {segment_count} segments, expected at least {expected_segments}"
        )));
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| WsiError::DisplayConversion("DICOM RLE dimensions overflow".into()))?;
    let mut planes = Vec::with_capacity(expected_segments);
    for segment in 0..expected_segments {
        let offset_start = 4 + segment * 4;
        let segment_start = u32::from_le_bytes(
            frame_bytes[offset_start..offset_start + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let segment_end = if segment + 1 < segment_count {
            let next_offset_start = 4 + (segment + 1) * 4;
            u32::from_le_bytes(
                frame_bytes[next_offset_start..next_offset_start + 4]
                    .try_into()
                    .unwrap(),
            ) as usize
        } else {
            frame_bytes.len()
        };
        if segment_start < 64 || segment_start > segment_end || segment_end > frame_bytes.len() {
            return Err(WsiError::DisplayConversion(format!(
                "DICOM RLE segment {segment} has invalid byte range {segment_start}..{segment_end}"
            )));
        }
        planes.push(decode_rle_segment(
            &frame_bytes[segment_start..segment_end],
            pixel_count,
        )?);
    }

    let rgb = match (samples_per_pixel, photometric_interpretation) {
        (3, "RGB") => {
            let mut rgb = vec![0; pixel_count * 3];
            for (idx, ((&red, &green), &blue)) in
                planes[0].iter().zip(&planes[1]).zip(&planes[2]).enumerate()
            {
                let offset = idx * 3;
                rgb[offset] = red;
                rgb[offset + 1] = green;
                rgb[offset + 2] = blue;
            }
            rgb
        }
        (1, "MONOCHROME1" | "MONOCHROME2") => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for &gray in &planes[0] {
                rgb.extend_from_slice(&[gray, gray, gray]);
            }
            rgb
        }
        _ => {
            return Err(WsiError::DisplayConversion(format!(
                "unsupported DICOM RLE pixel format: samples_per_pixel={samples_per_pixel}, photometric={photometric_interpretation}"
            )));
        }
    };

    CpuTile::new(
        width,
        height,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(rgb),
    )
}

fn decode_rle_segment(segment: &[u8], expected_len: usize) -> Result<Vec<u8>, WsiError> {
    let mut output = Vec::with_capacity(expected_len);
    let mut i = 0;
    while i < segment.len() && output.len() < expected_len {
        let n = segment[i] as i8;
        i += 1;
        match n {
            0..=127 => {
                let count = n as usize + 1;
                let end = i.checked_add(count).ok_or_else(|| {
                    WsiError::DisplayConversion("DICOM RLE literal run overflow".into())
                })?;
                if end > segment.len() {
                    return Err(WsiError::DisplayConversion(
                        "DICOM RLE literal run exceeds segment length".into(),
                    ));
                }
                output.extend_from_slice(&segment[i..end]);
                i = end;
            }
            -127..=-1 => {
                if i >= segment.len() {
                    return Err(WsiError::DisplayConversion(
                        "DICOM RLE repeat run missing value".into(),
                    ));
                }
                let count = 1usize + (-n as usize);
                output.extend(std::iter::repeat_n(segment[i], count));
                i += 1;
            }
            -128 => {}
        }
    }
    if output.len() != expected_len {
        return Err(WsiError::DisplayConversion(format!(
            "DICOM RLE segment decoded to {} bytes, expected {expected_len}",
            output.len()
        )));
    }
    Ok(output)
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

fn crop_sample_buffer_rgb(buffer: CpuTile, width: u32, height: u32) -> CpuTile {
    if buffer.width == width && buffer.height == height {
        return buffer;
    }
    let image = imageops::crop_imm(&buffer.into_rgb_image(), 0, 0, width, height).to_image();
    rgb_image_to_sample_buffer(image)
}

fn raw_compression_for_transfer_syntax(
    transfer_syntax_uid: &str,
    photometric_interpretation: &str,
) -> Result<Compression, WsiError> {
    if transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
        return Ok(Compression::Jpeg);
    }
    if JP2K_TRANSFER_SYNTAXES.contains(&transfer_syntax_uid) {
        return Ok(
            if matches!(photometric_interpretation, "YBR_ICT" | "YBR_RCT") {
                Compression::Jp2kYcbcr
            } else {
                Compression::Jp2kRgb
            },
        );
    }
    Err(WsiError::Unsupported {
        reason: format!(
            "raw compressed DICOM tile access requires JPEG Baseline or J2K/HTJ2K transfer syntax, got {transfer_syntax_uid}"
        ),
    })
}

fn raw_photometric_interpretation(
    samples_per_pixel: u16,
    photometric_interpretation: &str,
) -> Result<EncodedTilePhotometricInterpretation, WsiError> {
    match (samples_per_pixel, photometric_interpretation) {
        (1, "MONOCHROME1" | "MONOCHROME2") => {
            Ok(EncodedTilePhotometricInterpretation::Monochrome2)
        }
        (3, "RGB") => Ok(EncodedTilePhotometricInterpretation::Rgb),
        (3, "YBR_FULL_422" | "YBR_ICT" | "YBR_RCT") => {
            Ok(EncodedTilePhotometricInterpretation::YbrFull422)
        }
        (_, other) => Err(WsiError::Unsupported {
            reason: format!(
                "raw compressed DICOM tile access does not support photometric interpretation {other}"
            ),
        }),
    }
}

fn trim_encapsulated_frame_padding(data: &mut Vec<u8>) {
    if data.len() >= 3
        && data.last() == Some(&0)
        && data[data.len() - 3..data.len() - 1] == [0xFF, 0xD9]
    {
        data.pop();
    }
}

trait IntoRgbImage {
    fn into_rgb_image(self) -> image::RgbImage;
}

impl IntoRgbImage for CpuTile {
    fn into_rgb_image(self) -> image::RgbImage {
        match self {
            CpuTile {
                width,
                height,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::U8(bytes),
            } => image::RgbImage::from_raw(width, height, Arc::unwrap_or_clone(bytes))
                .expect("sample buffer must contain valid RGB dimensions"),
            buffer => image::DynamicImage::ImageRgba8(
                buffer
                    .to_rgba()
                    .expect("RGB crop fallback must produce RGBA"),
            )
            .into_rgb8(),
        }
    }
}

fn black_sample_buffer(width: u32, height: u32) -> CpuTile {
    CpuTile::new(
        width,
        height,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(vec![0; width as usize * height as usize * 3]),
    )
    .expect("black tile dimensions must match")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::registry::Slide;
    use dicom_core::value::fragments::Fragments;
    use dicom_core::value::DataSetSequence;
    use dicom_core::value::{PixelFragmentSequence, Value};
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_object::{FileMetaTableBuilder, InMemDicomObject};

    #[test]
    fn level0_properties_from_metadata_match_full_parse() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let path = workspace_root
            .join("downloads/openslide-testdata-extracted/dicom/dicom-cmu1-jp2k/DCM_0.dcm");
        if !path.is_file() {
            eprintln!(
                "skipping corpus-backed DICOM metadata test; missing {}",
                path.display()
            );
            return;
        }
        let meta = parse_metadata_object_full(&path).expect("full metadata parse");
        assert_eq!(
            parse_level0_properties_from_metadata(&meta),
            parse_level0_properties(&path).expect("level0 property parse")
        );
    }

    enum TestPixelData {
        Native(Vec<u8>),
        Encapsulated(Vec<u8>),
    }

    struct TestDicomOptions {
        sop_instance_uid: &'static str,
        series_instance_uid: &'static str,
        image_type: &'static str,
        transfer_syntax: &'static str,
        samples_per_pixel: u16,
        photometric_interpretation: &'static str,
        planar_configuration: Option<u16>,
        rows: u16,
        columns: u16,
        total_pixel_matrix_rows: u32,
        total_pixel_matrix_columns: u32,
        number_of_frames: u32,
        pixel_spacing: Option<&'static str>,
        shared_pixel_spacing: Option<&'static str>,
        pixel_data: TestPixelData,
    }

    impl TestDicomOptions {
        fn native(pixel_data: Vec<u8>) -> Self {
            Self {
                sop_instance_uid: "1.2.826.0.1.3680043.10.777.1",
                series_instance_uid: "1.2.826.0.1.3680043.10.777",
                image_type: "ORIGINAL\\PRIMARY\\VOLUME\\NONE",
                transfer_syntax: uids::EXPLICIT_VR_LITTLE_ENDIAN,
                samples_per_pixel: 3,
                photometric_interpretation: "RGB",
                planar_configuration: Some(0),
                rows: 2,
                columns: 2,
                total_pixel_matrix_rows: 2,
                total_pixel_matrix_columns: 2,
                number_of_frames: 1,
                pixel_spacing: Some("0.00025\\0.00025"),
                shared_pixel_spacing: None,
                pixel_data: TestPixelData::Native(pixel_data),
            }
        }
    }

    fn write_test_dicom(path: &Path, options: TestDicomOptions) {
        let mut object = InMemDicomObject::new_empty();
        object.put(DataElement::new(
            tags::SOP_CLASS_UID,
            VR::UI,
            uids::VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_STORAGE,
        ));
        object.put(DataElement::new(
            tags::SOP_INSTANCE_UID,
            VR::UI,
            options.sop_instance_uid,
        ));
        object.put(DataElement::new(
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            options.series_instance_uid,
        ));
        object.put(DataElement::new(
            tags::IMAGE_TYPE,
            VR::CS,
            options.image_type,
        ));
        object.put(DataElement::new(
            tags::ROWS,
            VR::US,
            PrimitiveValue::from(options.rows),
        ));
        object.put(DataElement::new(
            tags::COLUMNS,
            VR::US,
            PrimitiveValue::from(options.columns),
        ));
        object.put(DataElement::new(
            tags::TOTAL_PIXEL_MATRIX_ROWS,
            VR::UL,
            PrimitiveValue::from(options.total_pixel_matrix_rows),
        ));
        object.put(DataElement::new(
            tags::TOTAL_PIXEL_MATRIX_COLUMNS,
            VR::UL,
            PrimitiveValue::from(options.total_pixel_matrix_columns),
        ));
        object.put(DataElement::new(
            tags::NUMBER_OF_FRAMES,
            VR::IS,
            PrimitiveValue::from(options.number_of_frames),
        ));
        object.put(DataElement::new(
            tags::SAMPLES_PER_PIXEL,
            VR::US,
            PrimitiveValue::from(options.samples_per_pixel),
        ));
        object.put(DataElement::new(
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            options.photometric_interpretation,
        ));
        if let Some(planar_configuration) = options.planar_configuration {
            object.put(DataElement::new(
                tags::PLANAR_CONFIGURATION,
                VR::US,
                PrimitiveValue::from(planar_configuration),
            ));
        }
        object.put(DataElement::new(
            tags::BITS_ALLOCATED,
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        object.put(DataElement::new(
            tags::BITS_STORED,
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        object.put(DataElement::new(
            tags::HIGH_BIT,
            VR::US,
            PrimitiveValue::from(7u16),
        ));
        object.put(DataElement::new(
            tags::PIXEL_REPRESENTATION,
            VR::US,
            PrimitiveValue::from(0u16),
        ));
        if let Some(pixel_spacing) = options.pixel_spacing {
            object.put(DataElement::new(tags::PIXEL_SPACING, VR::DS, pixel_spacing));
        }
        if let Some(pixel_spacing) = options.shared_pixel_spacing {
            let mut pixel_measures = InMemDicomObject::new_empty();
            pixel_measures.put(DataElement::new(tags::PIXEL_SPACING, VR::DS, pixel_spacing));
            let mut shared = InMemDicomObject::new_empty();
            shared.put(DataElement::<InMemDicomObject>::new(
                tags::PIXEL_MEASURES_SEQUENCE,
                VR::SQ,
                DataSetSequence::from(vec![pixel_measures]),
            ));
            object.put(DataElement::<InMemDicomObject>::new(
                tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE,
                VR::SQ,
                DataSetSequence::from(vec![shared]),
            ));
        }
        match options.pixel_data {
            TestPixelData::Native(pixel_data) => {
                object.put(DataElement::new(
                    tags::PIXEL_DATA,
                    VR::OB,
                    PrimitiveValue::from(pixel_data),
                ));
            }
            TestPixelData::Encapsulated(frame) => {
                let pixel_sequence = PixelFragmentSequence::from(vec![Fragments::new(frame, 0)]);
                object.put(DataElement::<InMemDicomObject>::new(
                    tags::PIXEL_DATA,
                    VR::OB,
                    Value::from(pixel_sequence),
                ));
            }
        }
        object
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(uids::VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_STORAGE)
                    .media_storage_sop_instance_uid(options.sop_instance_uid)
                    .transfer_syntax(options.transfer_syntax),
            )
            .unwrap()
            .write_to_file(path)
            .unwrap();
    }

    fn read_first_tile(path: &Path) -> CpuTile {
        let slide = Slide::open(path).expect("open DICOM slide");
        match slide
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
            .expect("read first tile")
        {
            TilePixels::Cpu(tile) => tile,
            TilePixels::Device(_) => panic!("DICOM tests request CPU output"),
        }
    }

    fn read_first_raw_compressed_tile(path: &Path) -> RawCompressedTile {
        Slide::open(path)
            .expect("open DICOM slide")
            .read_raw_compressed_tile(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .expect("read first raw compressed tile")
    }

    fn test_dicom_image(sop_instance_uid: &str, grid: DicomGrid) -> Arc<DicomImage> {
        test_dicom_image_with_transfer_syntax(
            sop_instance_uid,
            grid,
            uids::EXPLICIT_VR_LITTLE_ENDIAN,
        )
    }

    fn test_dicom_image_with_transfer_syntax(
        sop_instance_uid: &str,
        grid: DicomGrid,
        transfer_syntax_uid: &str,
    ) -> Arc<DicomImage> {
        Arc::new(DicomImage {
            path: PathBuf::from(format!("{sop_instance_uid}.dcm")),
            sop_instance_uid: sop_instance_uid.into(),
            transfer_syntax_uid: transfer_syntax_uid.into(),
            photometric_interpretation: "RGB".into(),
            samples_per_pixel: 3,
            planar_configuration: Some(0),
            width: 4096,
            height: 4096,
            tile_width: 512,
            tile_height: 512,
            tiles_across: 8,
            tiles_down: 8,
            number_of_frames: 1,
            grid,
            pixel_spacing: None,
            objective_lens_power: None,
            encapsulated_frames: Mutex::new(None),
            encapsulated_frame_cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(1).unwrap(),
            )),
            decoded_frame_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(1).unwrap())),
            file: Mutex::new(None),
        })
    }

    fn empty_dataset() -> Dataset {
        Dataset {
            id: DatasetId(1),
            scenes: Vec::new(),
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        }
    }

    fn tile_request(col: i64, row: i64) -> TileRequest {
        TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col,
            row,
        }
    }

    #[cfg(feature = "metal")]
    fn test_metal_sessions() -> Option<crate::output::metal::MetalBackendSessions> {
        let device = metal::Device::system_default()?;
        Some(crate::output::metal::MetalBackendSessions::new(
            signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
            signinum_j2k_metal::MetalBackendSession::new(device),
        ))
    }

    fn rgb_bytes(tile: &CpuTile) -> Vec<u8> {
        assert_eq!(tile.width, 2);
        assert_eq!(tile.height, 2);
        assert_eq!(tile.channels, 3);
        assert_eq!(tile.color_space, ColorSpace::Rgb);
        assert_eq!(tile.layout, CpuTileLayout::Interleaved);
        tile.data.as_u8().expect("u8 RGB tile").to_vec()
    }

    fn write_series_level(
        path: &Path,
        sop_instance_uid: &'static str,
        total_rows: u32,
        total_columns: u32,
    ) {
        let mut options = TestDicomOptions::native(vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
        options.sop_instance_uid = sop_instance_uid;
        options.rows = 2;
        options.columns = 2;
        options.total_pixel_matrix_rows = total_rows;
        options.total_pixel_matrix_columns = total_columns;
        options.number_of_frames = total_rows.div_ceil(2) * total_columns.div_ceil(2);
        write_test_dicom(path, options);
    }

    fn series_level_dimensions(slide: &Slide) -> Vec<(u64, u64)> {
        slide.dataset().scenes[0].series[0]
            .levels
            .iter()
            .map(|level| level.dimensions)
            .collect()
    }

    #[test]
    fn opens_complete_sibling_series_from_any_member_file() {
        let dir = tempfile::tempdir().unwrap();
        let level0 = dir.path().join("level0.dcm");
        let level1 = dir.path().join("level1.dcm");
        let thumbnail = dir.path().join("thumbnail.dcm");

        write_series_level(&level0, "1.2.826.0.1.3680043.10.777.1", 16, 16);
        write_series_level(&level1, "1.2.826.0.1.3680043.10.777.2", 4, 4);
        let mut thumbnail_options =
            TestDicomOptions::native(vec![32, 32, 32, 64, 64, 64, 96, 96, 96, 128, 128, 128]);
        thumbnail_options.sop_instance_uid = "1.2.826.0.1.3680043.10.777.3";
        thumbnail_options.image_type = "DERIVED\\PRIMARY\\THUMBNAIL\\RESAMPLED";
        write_test_dicom(&thumbnail, thumbnail_options);

        let from_base = Slide::open(&level0).expect("open base member");
        let from_coarse = Slide::open(&level1).expect("open coarse member");
        let from_associated = Slide::open(&thumbnail).expect("open associated member");

        assert_eq!(series_level_dimensions(&from_base), vec![(16, 16), (4, 4)]);
        assert_eq!(
            series_level_dimensions(&from_coarse),
            vec![(16, 16), (4, 4)]
        );
        assert_eq!(
            series_level_dimensions(&from_associated),
            vec![(16, 16), (4, 4)]
        );
        assert!(from_associated
            .dataset()
            .associated_images
            .contains_key("thumbnail"));
    }

    #[test]
    fn opens_directory_containing_one_dicom_series() {
        let dir = tempfile::tempdir().unwrap();
        let level0 = dir.path().join("level0.dcm");
        let level1 = dir.path().join("level1.dcm");
        write_series_level(&level0, "1.2.826.0.1.3680043.10.777.1", 16, 16);
        write_series_level(&level1, "1.2.826.0.1.3680043.10.777.2", 4, 4);

        let from_file = Slide::open(&level0).expect("open DICOM member");
        let from_directory = Slide::open(dir.path()).expect("open DICOM series directory");

        assert_eq!(
            series_level_dimensions(&from_directory),
            series_level_dimensions(&from_file)
        );
    }

    #[test]
    fn opens_public_dicom_folder_and_member_with_matching_levels_when_available() {
        let bench_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let candidates = [
            bench_root
                .join("SlideViewer")
                .join("downloads/openslide-testdata-extracted/full/DICOM/CMU-1-JP2K-33005"),
            bench_root.join("downloads/openslide-testdata-extracted/full/DICOM/CMU-1-JP2K-33005"),
        ];
        let Some(folder) = candidates.iter().find(|path| path.is_dir()) else {
            eprintln!("skipping public DICOM folder test; CMU-1-JP2K-33005 not found");
            return;
        };
        let member = std::fs::read_dir(folder)
            .expect("read DICOM folder")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
            })
            .expect("public DICOM folder contains a .dcm member");

        let from_folder = Slide::open(folder).expect("open public DICOM folder");
        let from_member = Slide::open(&member).expect("open public DICOM member");

        assert!(
            series_level_dimensions(&from_folder).len() > 1,
            "public DICOM folder should expose physical pyramid levels"
        );
        assert_eq!(
            series_level_dimensions(&from_folder),
            series_level_dimensions(&from_member)
        );
    }

    #[test]
    fn rejects_huge_single_level_regular_dicom_missing_physical_pyramid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge-base-only.dcm");
        let mut options = TestDicomOptions::native(Vec::new());
        options.rows = 512;
        options.columns = 512;
        options.total_pixel_matrix_rows = 32_768;
        options.total_pixel_matrix_columns = 32_768;
        options.number_of_frames = 4_096;
        write_test_dicom(&path, options);

        let err = Slide::open(&path).expect_err("huge base-only DICOM should fail fast");
        let message = err.to_string();
        assert!(
            message.contains("contains only a full-resolution base layer"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("Open the complete DICOM series/folder"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn small_single_level_dicom_remains_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small-single-level.dcm");
        write_series_level(&path, "1.2.826.0.1.3680043.10.777.1", 16, 16);

        let slide = Slide::open(&path).expect("small single-level DICOM remains supported");
        assert_eq!(series_level_dimensions(&slide), vec![(16, 16)]);
    }

    #[test]
    fn build_levels_groups_split_sparse_instances() {
        let mut first_tiles = HashMap::new();
        first_tiles.insert((0, 0), 0);
        let mut second_tiles = HashMap::new();
        second_tiles.insert((1, 0), 0);

        let levels = build_levels(
            Path::new("split.dcm"),
            vec![
                test_dicom_image("1.2.3.1", DicomGrid::Sparse(first_tiles)),
                test_dicom_image("1.2.3.2", DicomGrid::Sparse(second_tiles)),
            ],
        )
        .expect("split sparse parts should form one logical level");

        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].parts.len(), 2);
        assert_eq!(levels[0].tiles_across, 8);
        assert_eq!(levels[0].tiles_down, 8);
    }

    #[test]
    fn tile_codec_kind_uses_actual_sparse_split_part_for_request() {
        let mut first_tiles = HashMap::new();
        first_tiles.insert((0, 0), 0);
        let mut second_tiles = HashMap::new();
        second_tiles.insert((1, 0), 0);

        let levels = build_levels(
            Path::new("split-codec.dcm"),
            vec![
                test_dicom_image_with_transfer_syntax(
                    "1.2.3.1",
                    DicomGrid::Sparse(first_tiles),
                    JPEG_TRANSFER_SYNTAX,
                ),
                test_dicom_image_with_transfer_syntax(
                    "1.2.3.2",
                    DicomGrid::Sparse(second_tiles),
                    HTJ2K_LOSSLESS_TRANSFER_SYNTAX,
                ),
            ],
        )
        .expect("split sparse parts should form one logical level");
        let reader = DicomReader {
            slide: Arc::new(DicomSlide {
                dataset: empty_dataset(),
                levels,
                associated: HashMap::new(),
            }),
        };

        assert_eq!(
            reader.tile_codec_kind(&tile_request(0, 0)),
            TileCodecKind::Jpeg
        );
        assert_eq!(
            reader.tile_codec_kind(&tile_request(1, 0)),
            TileCodecKind::Htj2k
        );
        assert_eq!(
            reader.tile_codec_kind(&tile_request(2, 0)),
            TileCodecKind::Other
        );
    }

    #[test]
    #[cfg(feature = "metal")]
    fn require_device_rejects_sparse_missing_dicom_tile_cpu_black_fallback() {
        let Some(sessions) = test_metal_sessions() else {
            return;
        };
        let mut present_tiles = HashMap::new();
        present_tiles.insert((0, 0), 0);
        let levels = build_levels(
            Path::new("sparse-device.dcm"),
            vec![test_dicom_image_with_transfer_syntax(
                "1.2.3.1",
                DicomGrid::Sparse(present_tiles),
                uids::JPEG2000_LOSSLESS,
            )],
        )
        .expect("sparse level should build");
        let reader = DicomReader {
            slide: Arc::new(DicomSlide {
                dataset: empty_dataset(),
                levels,
                associated: HashMap::new(),
            }),
        };

        let err = reader
            .read_tiles(
                &[tile_request(1, 0)],
                TileOutputPreference::require_device_auto_with_metal_and_compressed_decode(
                    sessions,
                ),
            )
            .expect_err("RequireDevice must not return CPU black sparse tile");

        assert!(matches!(err, WsiError::Unsupported { .. }));
    }

    #[test]
    fn opens_3dhistech_split_sparse_level_when_corpus_is_available() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let path =
            workspace_root.join("downloads/openslide-testdata-extracted/full/DICOM/3DHISTECH-2/2");
        if !path.exists() {
            return;
        }

        let slide = Slide::open(&path).expect("open split-level DICOM slide");
        let dataset = slide.dataset();
        assert_eq!(dataset.scenes.len(), 1);
        assert!(!dataset.scenes[0].series[0].levels.is_empty());
        let tile = slide
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
            .expect("read first split-level tile");
        assert!(matches!(tile, TilePixels::Cpu(_)));
    }

    fn literal_rle_segment(bytes: &[u8]) -> Vec<u8> {
        assert!((1..=128).contains(&bytes.len()));
        let mut encoded = Vec::with_capacity(bytes.len() + 1);
        encoded.push((bytes.len() - 1) as u8);
        encoded.extend_from_slice(bytes);
        encoded
    }

    fn rle_rgb_frame(r: &[u8], g: &[u8], b: &[u8]) -> Vec<u8> {
        let segments = [
            literal_rle_segment(r),
            literal_rle_segment(g),
            literal_rle_segment(b),
        ];
        let mut frame = vec![0; 64];
        frame[0..4].copy_from_slice(&3u32.to_le_bytes());
        let mut offset = 64u32;
        for (idx, segment) in segments.iter().enumerate() {
            let start = 4 + idx * 4;
            frame[start..start + 4].copy_from_slice(&offset.to_le_bytes());
            offset += segment.len() as u32;
        }
        for segment in segments {
            frame.extend_from_slice(&segment);
        }
        frame
    }

    fn push_explicit_vr_long_element(
        bytes: &mut Vec<u8>,
        tag: [u8; 4],
        vr: &[u8; 2],
        value: &[u8],
    ) {
        bytes.extend_from_slice(&tag);
        bytes.extend_from_slice(vr);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
        bytes.extend_from_slice(value);
    }

    fn push_pixel_fragment(bytes: &mut Vec<u8>, payload: &[u8]) -> u64 {
        let item_offset = bytes.len() as u64;
        bytes.extend_from_slice(&DICOM_ITEM_TAG_LE);
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        item_offset
    }

    #[test]
    fn raw_encapsulated_scan_handles_extended_offset_table_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw-eot-htj2k.dcm");
        let first = [0xFF, 0x4F, 0x01, 0x02];
        let second = [0xFF, 0x4F, 0x03, 0x04, 0x05, 0x06];
        let mut bytes = vec![0; 132];
        bytes[128..132].copy_from_slice(b"DICM");

        let mut eot = Vec::new();
        eot.extend_from_slice(&0u64.to_le_bytes());
        eot.extend_from_slice(&(first.len() as u64 + 8).to_le_bytes());
        push_explicit_vr_long_element(&mut bytes, [0xE0, 0x7F, 0x01, 0x00], b"OV", &eot);

        let mut eot_lengths = Vec::new();
        eot_lengths.extend_from_slice(&(first.len() as u64).to_le_bytes());
        eot_lengths.extend_from_slice(&(second.len() as u64).to_le_bytes());
        push_explicit_vr_long_element(&mut bytes, [0xE0, 0x7F, 0x02, 0x00], b"OV", &eot_lengths);

        bytes.extend_from_slice(&PIXEL_DATA_TAG_LE);
        bytes.extend_from_slice(b"OB");
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&UNDEFINED_LENGTH_LE);
        bytes.extend_from_slice(&DICOM_ITEM_TAG_LE);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let first_item_offset = push_pixel_fragment(&mut bytes, &first);
        let second_item_offset = push_pixel_fragment(&mut bytes, &second);
        bytes.extend_from_slice(&DICOM_SEQUENCE_DELIMITER_TAG_LE);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let frames = scan_encapsulated_frames_raw_little_endian(&path, 2)
            .expect("raw scan succeeds")
            .expect("Pixel Data is found");
        assert_eq!(frames.frame_ranges, vec![0..1, 1..2]);
        assert_eq!(frames.fragments.len(), 2);
        assert_eq!(frames.fragments[0].item_offset, first_item_offset);
        assert_eq!(frames.fragments[0].len, first.len() as u32);
        assert_eq!(frames.fragments[1].item_offset, second_item_offset);
        assert_eq!(frames.fragments[1].len, second.len() as u32);
    }

    #[test]
    fn large_basic_offset_table_frame_index_builds_quickly() {
        let frame_count = 25_000usize;
        let mut fragments = Vec::with_capacity(frame_count);
        let mut offset_table = Vec::with_capacity(frame_count);
        let mut item_offset = 1024u64;
        for _ in 0..frame_count {
            offset_table.push((item_offset - 1024) as u32);
            fragments.push(DicomFragmentRef {
                payload_offset: item_offset + 8,
                item_offset,
                len: 64,
            });
            item_offset += 72;
        }

        let started = std::time::Instant::now();
        let frames = build_encapsulated_frame_index(
            Path::new("large-basic-offset-table.dcm"),
            fragments,
            offset_table,
            frame_count as u32,
        )
        .expect("large basic offset table should build");

        assert_eq!(frames.frame_ranges.len(), frame_count);
        assert_eq!(frames.frame_ranges[0], 0..1);
        assert_eq!(
            frames.frame_ranges[frame_count - 1],
            frame_count - 1..frame_count
        );
        assert!(
            started.elapsed() < std::time::Duration::from_millis(250),
            "large DICOM basic offset table frame index should build in linear time"
        );
    }

    #[test]
    #[cfg(feature = "metal")]
    fn local_htj2k_dicom_full_tile_can_require_device_output() {
        let Some(path) = local_htj2k_dicom_device_fixture() else {
            return;
        };
        let Some(sessions) = test_metal_sessions() else {
            eprintln!("skipping local HTJ2K DICOM device test; no Metal device");
            return;
        };

        let slide = Slide::open(&path).expect("open local HTJ2K DICOM slide");
        let tile = slide
            .read_tile(
                &TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: 0,
                    row: 0,
                },
                TileOutputPreference::require_device_auto_with_metal_and_compressed_decode(
                    sessions,
                ),
            )
            .expect("read full HTJ2K tile with required device output");

        assert!(matches!(tile, TilePixels::Device(_)));
    }

    #[test]
    #[cfg(feature = "metal")]
    fn local_htj2k_dicom_prefer_device_batch_keeps_full_tiles_on_device() {
        let Some(path) = local_htj2k_dicom_device_fixture() else {
            return;
        };
        let Some(sessions) = test_metal_sessions() else {
            eprintln!("skipping local HTJ2K DICOM device test; no Metal device");
            return;
        };

        let slide = Slide::open(&path).expect("open local HTJ2K DICOM slide");
        let tiles = slide
            .read_tiles(
                &[
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
                ],
                TileOutputPreference::prefer_device_auto_with_metal_and_compressed_decode(sessions)
                    .without_adaptive_decode_route(),
            )
            .expect("read full HTJ2K tile batch with residency-preferred device output");

        assert!(
            tiles
                .iter()
                .any(|tile| matches!(tile, TilePixels::Device(_))),
            "prefer-device HTJ2K batch should return device tiles when full tiles are decodable"
        );
    }

    #[cfg(feature = "metal")]
    fn local_htj2k_dicom_device_fixture() -> Option<PathBuf> {
        let Some(path) = std::env::var_os("STATUMEN_LOCAL_HTJ2K_DICOM").map(PathBuf::from) else {
            eprintln!("skipping local HTJ2K DICOM device test; STATUMEN_LOCAL_HTJ2K_DICOM unset");
            return None;
        };
        if !path.is_file() {
            eprintln!(
                "skipping local HTJ2K DICOM device test; missing {}",
                path.display()
            );
            return None;
        }
        Some(path)
    }

    #[test]
    fn opens_implicit_vr_little_endian_native_rgb() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("implicit.dcm");
        let mut options =
            TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
        options.transfer_syntax = uids::IMPLICIT_VR_LITTLE_ENDIAN;
        write_test_dicom(&path, options);

        assert_eq!(
            rgb_bytes(&read_first_tile(&path)),
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]
        );
    }

    #[test]
    fn opens_explicit_vr_big_endian_native_rgb_8bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big-endian.dcm");
        let mut options =
            TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
        options.transfer_syntax = EXPLICIT_VR_BIG_ENDIAN_TRANSFER_SYNTAX;
        write_test_dicom(&path, options);

        assert_eq!(
            rgb_bytes(&read_first_tile(&path)),
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]
        );
    }

    #[test]
    fn converts_planar_rgb_native_frames_to_interleaved_rgb() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("planar.dcm");
        let mut options = TestDicomOptions::native(vec![
            255, 0, 0, 255, // R plane
            0, 255, 0, 255, // G plane
            0, 0, 255, 0, // B plane
        ]);
        options.planar_configuration = Some(1);
        write_test_dicom(&path, options);

        assert_eq!(
            rgb_bytes(&read_first_tile(&path)),
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]
        );
    }

    #[test]
    fn expands_monochrome_8bit_native_frames_to_rgb() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mono.dcm");
        let mut options = TestDicomOptions::native(vec![0, 64, 128, 255]);
        options.samples_per_pixel = 1;
        options.photometric_interpretation = "MONOCHROME2";
        options.planar_configuration = None;
        write_test_dicom(&path, options);

        assert_eq!(
            rgb_bytes(&read_first_tile(&path)),
            vec![0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255]
        );
    }

    #[test]
    fn top_level_pixel_spacing_is_mpp_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spacing.dcm");
        let mut options =
            TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
        options.pixel_spacing = Some("0.0005\\0.00025");
        write_test_dicom(&path, options);

        let slide = Slide::open(&path).expect("open DICOM slide");
        assert_eq!(
            slide.dataset().properties.get("openslide.mpp-x"),
            Some("0.25")
        );
        assert_eq!(
            slide.dataset().properties.get("openslide.mpp-y"),
            Some("0.5")
        );
    }

    #[test]
    fn shared_functional_group_pixel_spacing_is_mpp_for_start_instance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared-spacing.dcm");
        let mut options =
            TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
        options.pixel_spacing = None;
        options.shared_pixel_spacing = Some("0.0005\\0.00025");
        write_test_dicom(&path, options);

        let slide = Slide::open(&path).expect("open DICOM slide");
        assert_eq!(
            slide.dataset().properties.get("openslide.mpp-x"),
            Some("0.25")
        );
        assert_eq!(
            slide.dataset().properties.get("openslide.mpp-y"),
            Some("0.5")
        );
    }

    #[test]
    fn decodes_rle_lossless_rgb_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rle.dcm");
        write_test_dicom(
            &path,
            TestDicomOptions {
                transfer_syntax: uids::RLE_LOSSLESS,
                samples_per_pixel: 3,
                photometric_interpretation: "RGB",
                planar_configuration: Some(1),
                pixel_spacing: Some("0.00025\\0.00025"),
                shared_pixel_spacing: None,
                pixel_data: TestPixelData::Encapsulated(rle_rgb_frame(
                    &[255, 0, 0, 255],
                    &[0, 255, 0, 255],
                    &[0, 0, 255, 0],
                )),
                ..TestDicomOptions::native(Vec::new())
            },
        );

        assert_eq!(
            rgb_bytes(&read_first_tile(&path)),
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]
        );
    }

    #[test]
    fn reads_htj2k_rpcl_raw_compressed_frame_without_dicom_padding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("htj2k-rpcl.dcm");
        let codestream = vec![0xFF, 0x4F, 0x00, 0xFF, 0xD9];
        write_test_dicom(
            &path,
            TestDicomOptions {
                transfer_syntax: HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX,
                samples_per_pixel: 3,
                photometric_interpretation: "RGB",
                planar_configuration: Some(0),
                pixel_spacing: Some("0.00025\\0.00025"),
                shared_pixel_spacing: None,
                pixel_data: TestPixelData::Encapsulated(codestream.clone()),
                ..TestDicomOptions::native(Vec::new())
            },
        );

        let raw = read_first_raw_compressed_tile(&path);
        assert_eq!(raw.compression, Compression::Jp2kRgb);
        assert_eq!(raw.width, 2);
        assert_eq!(raw.height, 2);
        assert_eq!(raw.bits_allocated, 8);
        assert_eq!(raw.samples_per_pixel, 3);
        assert_eq!(
            raw.photometric_interpretation,
            EncodedTilePhotometricInterpretation::Rgb
        );
        assert_eq!(raw.data, codestream);
    }

    #[test]
    fn tile_codec_kind_classifies_dicom_transfer_syntaxes() {
        assert_eq!(
            dicom_tile_codec_kind(JPEG_TRANSFER_SYNTAX),
            TileCodecKind::Jpeg
        );
        assert_eq!(
            dicom_tile_codec_kind(uids::JPEG2000_LOSSLESS),
            TileCodecKind::Jp2k
        );
        assert_eq!(
            dicom_tile_codec_kind(HTJ2K_LOSSLESS_TRANSFER_SYNTAX),
            TileCodecKind::Htj2k
        );
        assert_eq!(
            dicom_tile_codec_kind(uids::EXPLICIT_VR_LITTLE_ENDIAN),
            TileCodecKind::Other
        );
    }

    #[test]
    fn reads_jpeg2000_ybr_rct_raw_compressed_frame_as_ycbcr() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jpeg2000-ybr-rct.dcm");
        let codestream = vec![0xFF, 0x4F, 0x00, 0xFF, 0xD9];
        write_test_dicom(
            &path,
            TestDicomOptions {
                transfer_syntax: uids::JPEG2000_LOSSLESS,
                samples_per_pixel: 3,
                photometric_interpretation: "YBR_RCT",
                planar_configuration: Some(0),
                pixel_spacing: Some("0.00025\\0.00025"),
                shared_pixel_spacing: None,
                pixel_data: TestPixelData::Encapsulated(codestream.clone()),
                ..TestDicomOptions::native(Vec::new())
            },
        );

        let raw = read_first_raw_compressed_tile(&path);
        assert_eq!(raw.compression, Compression::Jp2kYcbcr);
        assert_eq!(raw.width, 2);
        assert_eq!(raw.height, 2);
        assert_eq!(raw.bits_allocated, 8);
        assert_eq!(raw.samples_per_pixel, 3);
        assert_eq!(
            raw.photometric_interpretation,
            EncodedTilePhotometricInterpretation::YbrFull422
        );
        assert_eq!(raw.data, codestream);
    }
}
