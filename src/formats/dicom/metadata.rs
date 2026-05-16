use super::*;

pub(super) struct ParsedDicomMetadata {
    pub(super) path: PathBuf,
    pub(super) obj: DefaultDicomObject,
    pub(super) study_instance_uid: Option<String>,
    pub(super) series_instance_uid: String,
    pub(super) frame_of_reference_uid: Option<String>,
    pub(super) container_identifier: Option<String>,
    pub(super) specimen_identifier: Option<String>,
    pub(super) sop_instance_uid: String,
    pub(super) transfer_syntax_uid: String,
    pub(super) photometric_interpretation: String,
    pub(super) samples_per_pixel: u16,
    pub(super) planar_configuration: Option<u16>,
    pub(super) image_type: Vec<String>,
    pub(super) rows: u32,
    pub(super) columns: u32,
    pub(super) number_of_frames: u32,
    pub(super) total_pixel_matrix_columns: Option<u32>,
    pub(super) total_pixel_matrix_rows: Option<u32>,
    pub(super) dimension_organization_type: Option<String>,
    pub(super) pixel_spacing: Option<(f64, f64)>,
    pub(super) objective_lens_power: Option<f64>,
}

impl ParsedDicomMetadata {
    pub(super) fn classify(&self) -> Result<ImageRole, WsiError> {
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

pub(super) fn parse_metadata_object(path: &Path) -> Result<ParsedDicomMetadata, WsiError> {
    // Stop after the top-level matrix geometry is available, but before pixel
    // data. This keeps cold-open cheap while still building the correct
    // pyramid geometry for tiled DICOM pyramids.
    let meta = parse_metadata_object_until(path, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE)?;
    if meta.dimension_organization_type.as_deref() == Some("TILED_SPARSE") {
        return parse_metadata_object_full(path);
    }
    Ok(meta)
}

pub(super) fn parse_metadata_object_full(path: &Path) -> Result<ParsedDicomMetadata, WsiError> {
    parse_metadata_object_until(path, tags::PIXEL_DATA)
}

pub(super) type Level0Properties = (Option<(f64, f64)>, Option<f64>);

pub(super) fn parse_level0_properties(path: &Path) -> Result<Level0Properties, WsiError> {
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
pub(super) fn parse_level0_properties_from_metadata(
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

pub(super) fn optional_pixel_spacing_mpp(
    obj: &DefaultDicomObject,
) -> Result<Option<(f64, f64)>, WsiError> {
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

pub(super) fn parse_metadata_object_until(
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

pub(super) fn build_levels(
    path: &Path,
    images: Vec<Arc<DicomImage>>,
) -> Result<Vec<DicomLevel>, WsiError> {
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

pub(super) fn validate_monotonic_levels(
    path: &Path,
    levels: &[DicomLevel],
) -> Result<(), WsiError> {
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

pub(super) fn reject_huge_base_only_dicom(
    path: &Path,
    levels: &[DicomLevel],
) -> Result<(), WsiError> {
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

pub(super) fn dedupe_associated(
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

pub(super) fn ensure_same_sop(path: &Path, current: &str, previous: &str) -> Result<(), WsiError> {
    if current == previous {
        Ok(())
    } else {
        Err(invalid_slide(
            path,
            format!("Slide contains unexpected image ({current} vs. {previous})"),
        ))
    }
}

pub(super) fn validate_supported_pixel_format(meta: &ParsedDicomMetadata) -> Result<(), WsiError> {
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

pub(super) fn parse_sparse_tile_map(
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

pub(super) fn is_vl_wsi(sop_class_uid: &str) -> bool {
    sop_class_uid == uids::VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_STORAGE
}

pub(super) fn matches_type(image_type: &[&str], allowed: &[&[&str]]) -> bool {
    allowed.contains(&image_type)
}

pub(super) fn quickhash_for_series_uid(series_uid: &str) -> Result<String, WsiError> {
    let mut quickhash = Quickhash1::new();
    quickhash.hash_string(series_uid);
    quickhash
        .finish()
        .ok_or_else(|| WsiError::DisplayConversion("failed to compute DICOM quickhash".into()))
}

pub(super) fn dataset_id_from_quickhash(
    path: &Path,
    quickhash: &str,
) -> Result<DatasetId, WsiError> {
    if quickhash.len() < 32 {
        return Err(invalid_slide(path, "quickhash too short"));
    }
    let value = u128::from_str_radix(&quickhash[..32], 16)
        .map_err(|_| invalid_slide(path, "quickhash is not valid hex"))?;
    Ok(DatasetId(value))
}

pub(super) fn canonicalize_or_fallback(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub(super) fn invalid_slide(path: &Path, message: impl Into<String>) -> WsiError {
    WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

pub(super) fn required_string(
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

pub(super) fn required_multi_string(
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

pub(super) fn optional_string(
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

pub(super) fn required_u32(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
    name: &str,
) -> Result<u32, WsiError> {
    obj.element(tag)
        .map_err(|_| WsiError::DisplayConversion(format!("missing {name}")))?
        .to_int::<u32>()
        .map_err(|err| WsiError::DisplayConversion(format!("invalid {name}: {err}")))
}

pub(super) fn optional_u32(
    obj: &DefaultDicomObject,
    tag: dicom_core::Tag,
) -> Result<Option<u32>, WsiError> {
    obj.get(tag)
        .map(|elem| {
            elem.to_int::<u32>().map_err(|err| {
                WsiError::DisplayConversion(format!("invalid DICOM integer tag {tag:?}: {err}"))
            })
        })
        .transpose()
}

pub(super) fn verify_required_int(
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

pub(super) fn verify_optional_int(
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

pub(super) fn required_u32_at_item(
    obj: &dicom_object::InMemDicomObject,
    selector: impl Into<dicom_core::ops::AttributeSelector>,
    name: &str,
) -> Result<u32, WsiError> {
    obj.entry_at(selector)
        .map_err(|_| WsiError::DisplayConversion(format!("missing {name}")))?
        .to_int::<u32>()
        .map_err(|err| WsiError::DisplayConversion(format!("invalid {name}: {err}")))
}

pub(super) fn optional_f64_at(
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

pub(super) fn optional_pair_f64_at(
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
