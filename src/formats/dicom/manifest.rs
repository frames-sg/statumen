use super::*;

pub(super) struct DicomSlide {
    pub(super) dataset: Dataset,
    pub(super) levels: Vec<DicomLevel>,
    pub(super) associated: HashMap<String, Arc<DicomImage>>,
}

impl DicomSlide {
    pub(super) fn parse(path: &Path) -> Result<Self, WsiError> {
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
        preload_volume_encapsulated_frame_indexes(&levels)?;

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

fn preload_volume_encapsulated_frame_indexes(levels: &[DicomLevel]) -> Result<(), WsiError> {
    for level in levels {
        for image in &level.parts {
            image.preload_encapsulated_frame_index()?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) struct DicomLevel {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) tiles_across: u32,
    pub(super) tiles_down: u32,
    pub(super) path: PathBuf,
    pub(super) pixel_spacing: Option<(f64, f64)>,
    pub(super) objective_lens_power: Option<f64>,
    pub(super) parts: Vec<Arc<DicomImage>>,
}

impl DicomLevel {
    pub(super) fn from_image(image: Arc<DicomImage>) -> Self {
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

    pub(super) fn area(&self) -> u64 {
        u64::from(self.width).saturating_mul(u64::from(self.height))
    }

    pub(super) fn is_regular_full_tiling(&self) -> bool {
        self.parts.iter().all(|part| part.is_full_grid())
    }

    pub(super) fn push_part(
        &mut self,
        path: &Path,
        image: Arc<DicomImage>,
    ) -> Result<(), WsiError> {
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

    pub(super) fn samples_per_pixel(&self) -> u16 {
        self.parts[0].samples_per_pixel
    }

    pub(super) fn planar_configuration(&self) -> Option<u16> {
        self.parts[0].planar_configuration
    }

    pub(super) fn photometric_interpretation(&self) -> &str {
        &self.parts[0].photometric_interpretation
    }

    pub(super) fn image_for_tile(&self, col: u32, row: u32) -> Option<Arc<DicomImage>> {
        self.parts
            .iter()
            .find(|image| image.frame_index(col, row).is_some())
            .cloned()
    }

    pub(super) fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
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

    pub(super) fn read_tile(
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

    pub(super) fn read_raw_compressed_tile(
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

    pub(super) fn actual_tile_dimensions(&self, col: u32, row: u32) -> (u32, u32) {
        let tile_x = col * self.tile_width;
        let tile_y = row * self.tile_height;
        let width = self.width.saturating_sub(tile_x).min(self.tile_width);
        let height = self.height.saturating_sub(tile_y).min(self.tile_height);
        (width, height)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AssociatedKind {
    Label,
    Macro,
    Thumbnail,
}

impl AssociatedKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Label => "label",
            Self::Macro => "macro",
            Self::Thumbnail => "thumbnail",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImageRole {
    Level,
    Associated(AssociatedKind),
    Ignore,
}

pub(super) struct DicomSeriesManifest {
    pub(super) study_instance_uid: Option<String>,
    pub(super) series_instance_uid: String,
    pub(super) frame_of_reference_uid: Option<String>,
    pub(super) container_identifier: Option<String>,
    pub(super) specimen_identifier: Option<String>,
    pub(super) volume_images: Vec<ParsedDicomMetadata>,
    pub(super) associated_images: Vec<(AssociatedKind, ParsedDicomMetadata)>,
    pub(super) source_file_count: usize,
}

impl DicomSeriesManifest {
    pub(super) fn resolve(path: &Path) -> Result<Self, WsiError> {
        if path.is_dir() {
            Self::from_directory(path)
        } else {
            Self::from_selected_file(path)
        }
    }

    pub(super) fn from_selected_file(path: &Path) -> Result<Self, WsiError> {
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

    pub(super) fn from_directory(path: &Path) -> Result<Self, WsiError> {
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

    pub(super) fn from_group(
        path: &Path,
        metas: Vec<ParsedDicomMetadata>,
    ) -> Result<Self, WsiError> {
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

pub(super) fn direct_child_files(dir: &Path) -> Result<Vec<PathBuf>, WsiError> {
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

pub(super) fn common_optional_value<F>(
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
