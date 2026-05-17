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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    EncapsulatedFrames(Vec<Vec<u8>>),
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
        #[cfg(all(feature = "metal", target_os = "macos"))]
        TestPixelData::EncapsulatedFrames(frames) => {
            let fragments = frames
                .into_iter()
                .map(|frame| Fragments::new(frame, 0))
                .collect::<Vec<_>>();
            let pixel_sequence = PixelFragmentSequence::from(fragments);
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
    test_dicom_image_with_transfer_syntax(sop_instance_uid, grid, uids::EXPLICIT_VR_LITTLE_ENDIAN)
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn encode_test_jpeg_rgb(width: u16, height: u16, seed: u8) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            let base = seed
                .wrapping_add(x as u8)
                .wrapping_add((y as u8).wrapping_mul(3));
            rgb.extend_from_slice(&[base, base.wrapping_add(17), base.wrapping_add(31)]);
        }
    }
    let mut encoded = Vec::new();
    jpeg_encoder::Encoder::new(&mut encoded, 90)
        .encode(&rgb, width, height, jpeg_encoder::ColorType::Rgb)
        .expect("encode baseline JPEG test frame");
    encoded
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
            TileOutputPreference::require_device_auto_with_metal_and_compressed_decode(sessions),
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

fn push_explicit_vr_long_element(bytes: &mut Vec<u8>, tag: [u8; 4], vr: &[u8; 2], value: &[u8]) {
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
            TileOutputPreference::require_device_auto_with_metal_and_compressed_decode(sessions),
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

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn dicom_jpeg_require_device_batch_uses_jpeg_device_route() {
    let Some(sessions) = test_metal_sessions() else {
        eprintln!("skipping DICOM JPEG device batch test; no Metal device");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jpeg-batch.dcm");
    let mut options = TestDicomOptions::native(Vec::new());
    options.transfer_syntax = JPEG_TRANSFER_SYNTAX;
    options.rows = 16;
    options.columns = 16;
    options.total_pixel_matrix_rows = 16;
    options.total_pixel_matrix_columns = 32;
    options.number_of_frames = 2;
    options.pixel_data = TestPixelData::EncapsulatedFrames(vec![
        encode_test_jpeg_rgb(16, 16, 3),
        encode_test_jpeg_rgb(16, 16, 41),
    ]);
    write_test_dicom(&path, options);

    let slide = Slide::open(&path).expect("open generated DICOM JPEG slide");
    let tiles = slide
        .read_tiles(
            &[tile_request(0, 0), tile_request(1, 0)],
            TileOutputPreference::require_device_auto_with_metal_and_compressed_decode(sessions)
                .without_adaptive_decode_route(),
        )
        .expect("DICOM JPEG full-tile batch should support required device output");

    assert_eq!(tiles.len(), 2);
    assert!(
        tiles
            .iter()
            .all(|tile| matches!(tile, TilePixels::Device(_))),
        "DICOM JPEG batch should keep all full tiles on device"
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
    let mut options = TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
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
    let mut options = TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
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
    let mut options = TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
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
    let mut options = TestDicomOptions::native(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
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
#[cfg(feature = "metal")]
fn dicom_jp2k_device_batch_policy_is_selective() {
    let prefer_device = TileOutputPreference::prefer_device_auto_with_compressed_decode();
    let require_device = TileOutputPreference::require_device_auto_with_compressed_decode();

    assert!(dicom_jp2k_device_batch_allowed_for_output(
        HTJ2K_LOSSLESS_TRANSFER_SYNTAX,
        &prefer_device,
        false,
    ));
    assert!(!dicom_jp2k_device_batch_allowed_for_output(
        uids::JPEG2000_LOSSLESS,
        &prefer_device,
        false,
    ));
    assert!(dicom_jp2k_device_batch_allowed_for_output(
        uids::JPEG2000_LOSSLESS,
        &require_device,
        false,
    ));
    assert!(dicom_jp2k_device_batch_allowed_for_output(
        uids::JPEG2000_LOSSLESS,
        &prefer_device,
        true,
    ));
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
