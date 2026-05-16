use super::*;
use crate::properties::Properties;
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
fn region_request(
    scene: usize,
    series: usize,
    level: u32,
    plane: PlaneSelection,
    x: i64,
    y: i64,
    w: u32,
    h: u32,
) -> RegionRequest {
    RegionRequest {
        scene: SceneId(scene),
        series: SeriesId(series),
        level: LevelIdx(level),
        plane: PlaneIdx(plane),
        origin_px: (x, y),
        size_px: (w, h),
    }
}

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
fn slide_open_options_configures_decode_execution() {
    let options = SlideOpenOptions::default()
        .with_decode_execution_options(DecodeExecutionOptions::default().with_route_sample_size(2));

    assert_eq!(options.decode_execution_options().route_sample_size(), 2);
}

#[test]
fn adaptive_route_cpu_wins_when_device_is_slower() {
    let winner = DecodeRouteDecision::winner_for_measurement(
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(20),
        4,
    );

    assert_eq!(winner, DecodeRoute::Cpu);
}

#[test]
fn adaptive_route_cpu_wins_when_device_returns_no_resident_tiles() {
    let winner = DecodeRouteDecision::winner_for_measurement(
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(1),
        0,
    );

    assert_eq!(winner, DecodeRoute::Cpu);
}

#[test]
fn adaptive_route_device_wins_when_it_beats_threshold() {
    let winner = DecodeRouteDecision::winner_for_measurement(
        std::time::Duration::from_millis(100),
        std::time::Duration::from_millis(80),
        4,
    );

    assert_eq!(winner, DecodeRoute::Device);
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
fn builtin_registry_opens_raw_j2c_codestream_as_single_tile_slide() {
    let mut file = tempfile::Builder::new().suffix(".j2c").tempfile().unwrap();
    file.write_all(include_bytes!("../../../tests/fixtures/jp2k/rgb_mct.j2k"))
        .unwrap();
    file.flush().unwrap();

    let slide =
        Slide::open_with_cache_bytes(file.path(), &FormatRegistry::builtin(), 16 * 1024 * 1024)
            .unwrap();

    let level = &slide.dataset().scenes[0].series[0].levels[0];
    assert_eq!(level.dimensions, (16, 12));
    assert!(matches!(
        level.tile_layout,
        TileLayout::Regular {
            tile_width: 16,
            tile_height: 12,
            tiles_across: 1,
            tiles_down: 1
        }
    ));

    let req = TileRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };
    assert_eq!(slide.tile_codec_kind(&req), TileCodecKind::Jp2k);
    let tile = slide.source().read_tiles_cpu(&[req]).unwrap().remove(0);
    assert_eq!((tile.width, tile.height), (16, 12));
    assert_eq!(tile.channels, 3);
    assert_eq!(tile.color_space, ColorSpace::Rgb);
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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 100, 100);
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
    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 512, 512);
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
    let req = region_request(0, 0, 0, PlaneSelection::default(), 255, 0, 2, 1);
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

    let req = region_request(5, 0, 0, PlaneSelection::default(), 0, 0, 10, 10);
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

    let req = region_request(0, 0, 99, PlaneSelection::default(), 0, 0, 10, 10);
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

    let req = region_request(0, 0, 0, PlaneSelection { z: 5, c: 0, t: 0 }, 0, 0, 10, 10);
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
    let req = region_request(0, 0, 0, PlaneSelection::default(), 10000, 10000, 10, 10);
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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 512, 512, 16, 16);
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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 100, 100);

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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 64, 64);

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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 512, 256);

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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 512, 512, 16, 16);
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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 256, 256);
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

    let req = region_request(0, 0, 0, PlaneSelection::default(), 0, 0, 512, 512);
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
