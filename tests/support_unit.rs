mod support;

use std::path::PathBuf;
use std::sync::Mutex;

use statumen::{ColorSpace, CpuTile, CpuTileData, CpuTileLayout};

use support::compare::{compare_rgba, tolerance_failure, Tolerance};
use support::corpus::{
    apply_alias_filter, corpus_cache_dir, format_default_extension, parse_manifest,
    public_manifest_path,
};
use support::oracles::{
    is_reference_oracle_unsupported, read_probe, require_reference_tile, sample_buffer_to_rgba,
    top_left_probe, OpenedSlide, Oracle, ProbeKind, ReferenceOracle, ReferenceTileError,
    SigninumOracle, TileBuffer,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

const SAMPLE_MANIFEST: &str = r#"
    [[slide]]
    name             = "aperio_svs_brightfield_he_typical"
    alias            = "svs-001"
    path             = ""
    format           = "aperio"
    codecs           = ["jpeg"]
    must_decode      = ["base", "level1", "level2", "label", "macro"]
    source           = "openslide-testdata"
    license          = "CC0-1.0"
    redistributable  = true
    sha256           = "deadbeef"
    citation         = "Goode A. et al. OpenSlide..."
    phi_reviewed     = true
    url              = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
"#;

#[test]
fn compare_identical_buffers_pass_with_psnr_inf() {
    let a = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
    let report = compare_rgba(&a, &a, Tolerance::JPEG_TIGHT);

    assert!(report.passed);
    assert_eq!(report.max_abs, 0);
    assert_eq!(report.mean_abs, 0.0);
    assert!(report.psnr_db.is_infinite());
    assert_eq!(report.bytewise_equal_rate, 1.0);
}

#[test]
fn compare_off_by_one_passes_jpeg_tight() {
    let a = vec![10u8; 32];
    let mut b = a.clone();
    b[0] = 11;

    let report = compare_rgba(&a, &b, Tolerance::JPEG_TIGHT);

    assert!(report.passed);
    assert_eq!(report.max_abs, 1);
}

#[test]
fn compare_off_by_two_fails_jpeg_tight_passes_tolerant() {
    let a = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
    let b = vec![12u8, 20, 30, 255, 40, 50, 60, 255];

    let tight = compare_rgba(&a, &b, Tolerance::JPEG_TIGHT);
    let tolerant = compare_rgba(&a, &b, Tolerance::TOLERANT);

    assert!(!tight.passed);
    assert!(tolerant.passed);
}

#[test]
fn compare_tolerance_failure_formats_failed_report() {
    let a = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
    let b = vec![12u8, 20, 30, 255, 40, 50, 60, 255];
    let report = compare_rgba(&a, &b, Tolerance::JPEG_TIGHT);

    let failure = tolerance_failure("svs-001 level=0 signinum-vs-reference", &report)
        .expect("failed report should produce gate failure");

    assert!(failure.contains("svs-001 level=0 signinum-vs-reference"));
    assert!(failure.contains("max_abs=2"));
}

#[test]
fn corpus_parses_minimal_manifest() {
    let manifest = parse_manifest(SAMPLE_MANIFEST).expect("parse");
    let slide = manifest.slides.first().expect("slide");

    assert_eq!(manifest.slides.len(), 1);
    assert_eq!(slide.alias, "svs-001");
    assert_eq!(slide.format, "aperio");
    assert!(slide.redistributable);
    assert_eq!(slide.codecs, vec!["jpeg"]);
    assert_eq!(slide.must_decode.len(), 5);
}

#[test]
fn corpus_unknown_format_extension_returns_none() {
    assert!(format_default_extension("nonsense").is_none());
    assert_eq!(format_default_extension("aperio"), Some("svs"));
}

#[test]
fn corpus_cache_dir_respects_env() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _env = EnvGuard::set("STATUMEN_PARITY_CORPUS_CACHE", "/tmp/sv-corpus-test");

    let path = corpus_cache_dir();

    assert_eq!(path, PathBuf::from("/tmp/sv-corpus-test"));
}

#[test]
fn corpus_public_manifest_parses() {
    let path = public_manifest_path();
    let text = std::fs::read_to_string(&path).expect("read public manifest");
    let manifest = parse_manifest(&text).expect("parse public manifest");

    assert!(!manifest.slides.is_empty(), "public manifest has no slides");
    for slide in &manifest.slides {
        assert!(
            slide.redistributable,
            "public entry {} not redistributable",
            slide.alias
        );
        assert!(!slide.alias.is_empty());
        assert!(!slide.format.is_empty());
        assert!(!slide.codecs.is_empty());
    }
}

#[test]
fn corpus_must_decode_level_matches_base_and_numbered_levels() {
    let mut manifest = parse_manifest(SAMPLE_MANIFEST).expect("parse");
    let entry = manifest.slides.first_mut().expect("slide");
    entry.must_decode = vec!["base".into(), "level1".into(), "level12".into()];

    assert!(entry.must_decode_level(0));
    assert!(entry.must_decode_level(1));
    assert!(entry.must_decode_level(12));
    assert!(!entry.must_decode_level(2));
    assert!(!entry.must_decode_level(10));
}

#[test]
fn corpus_expected_failure_matches_pair_and_level_aliases() {
    let mut manifest = parse_manifest(SAMPLE_MANIFEST).expect("parse");
    let entry = manifest.slides.first_mut().expect("slide");
    entry.expected_failures = vec![
        "signinum-vs-reference:base".into(),
        "reference-vs-openslide:level2".into(),
    ];

    assert!(entry.expected_failure("signinum-vs-reference", 0));
    assert!(entry.expected_failure("reference-vs-openslide", 2));
    assert!(!entry.expected_failure("signinum-vs-reference", 1));
    assert!(!entry.expected_failure("signinum-vs-openslide", 0));
}

#[test]
fn corpus_alias_filter_keeps_requested_manifest_entries() {
    let mut manifest = parse_manifest(SAMPLE_MANIFEST).expect("parse");
    let second = manifest.slides[0].clone();
    manifest.slides.push(support::corpus::CorpusEntry {
        alias: "ndpi-001".into(),
        format: "ndpi".into(),
        name: "hamamatsu_ndpi".into(),
        ..second
    });

    apply_alias_filter(&mut manifest, Some("ndpi-001,missing-001"));

    assert_eq!(manifest.slides.len(), 1);
    assert_eq!(manifest.slides[0].alias, "ndpi-001");
}

#[test]
fn oracle_names_are_stable() {
    assert_eq!(SigninumOracle.name(), "signinum");
    assert_eq!(ReferenceOracle.name(), "reference");
}

#[test]
fn oracle_reference_unsupported_decode_is_an_error() {
    let err = require_reference_tile(
        Err(ReferenceTileError::unsupported(
            "fixture format is not TIFF JPEG",
        )),
        "fixture level=0 tile=(0,0)",
    )
    .expect_err("unsupported reference decode must not fall back to production");

    assert!(err.contains("reference oracle unsupported"));
    assert!(err.contains("fixture format is not TIFF JPEG"));
    assert!(is_reference_oracle_unsupported(&err));
}

#[test]
fn oracle_sample_buffer_to_rgba_respects_planar_rgb_layout() {
    let tile = CpuTile {
        width: 2,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Planar,
        data: CpuTileData::u8(vec![10, 40, 20, 50, 30, 60]),
    };

    let out = sample_buffer_to_rgba(tile).expect("convert");

    assert_eq!(out.pixels_rgba, vec![10, 20, 30, 255, 40, 50, 60, 255]);
}

#[test]
fn oracle_top_left_probe_falls_back_to_region_for_irregular_layout() {
    let slide = OpenedSlide {
        path: PathBuf::from("fixture.bif"),
        oracle_name: "fixture",
        level_count: 1,
        level_dimensions: vec![(123, 45)],
        tile_sizes: vec![None],
        reader: Box::new(|_, _, _, _, _| Err("tile reader should not be used".into())),
        region_reader: Box::new(|level, x, y, width, height| {
            assert_eq!(level, 0);
            assert_eq!((x, y), (0, 0));
            assert_eq!((width, height), (123, 45));
            Ok(TileBuffer {
                pixels_rgba: vec![0; width as usize * height as usize * 4],
                width,
                height,
            })
        }),
    };

    let probe = top_left_probe(&slide, 0).expect("probe");

    assert_eq!(probe.kind, ProbeKind::Region);
    assert_eq!((probe.width, probe.height), (123, 45));
    let tile = read_probe(&slide, probe).expect("read probe");
    assert_eq!((tile.width, tile.height), (123, 45));
}

#[cfg(feature = "parity-openslide")]
#[test]
fn openslide_shim_missing_env_path_does_not_panic() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _env = EnvGuard::set(
        "OPENSLIDE_LIB_PATH",
        "/definitely/does/not/exist/libopenslide.dylib",
    );

    let result = support::openslide_shim::try_load();

    let _ = result;
}

#[cfg(feature = "parity-openslide")]
#[test]
fn openslide_shim_bounds_parser_reads_canvas_origin() {
    let props = [
        ("openslide.bounds-x", "10778"),
        ("openslide.bounds-y", "35096"),
        ("openslide.bounds-width", "36832"),
        ("openslide.bounds-height", "38432"),
    ];

    let bounds = support::openslide_shim::parse_bounds_from_properties(|name| {
        props
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_string())
    })
    .expect("bounds");

    assert_eq!(bounds.x, 10778);
    assert_eq!(bounds.y, 35096);
    assert_eq!(bounds.width, 36832);
    assert_eq!(bounds.height, 38432);
}

#[cfg(feature = "parity-openslide")]
#[test]
fn openslide_shim_bounds_parser_rejects_missing_or_empty_bounds() {
    assert!(support::openslide_shim::parse_bounds_from_properties(|_| None).is_none());

    let props = [
        ("openslide.bounds-x", "0"),
        ("openslide.bounds-y", "0"),
        ("openslide.bounds-width", "0"),
        ("openslide.bounds-height", "100"),
    ];

    assert!(
        support::openslide_shim::parse_bounds_from_properties(|name| {
            props
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        })
        .is_none()
    );
}

struct EnvGuard {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(value) = &self.prev {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}
