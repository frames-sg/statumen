//! Cross-format brightfield color regression tests.
//!
//! These checks intentionally operate on decoded RGBA buffers instead of the
//! GUI. They catch gross color regressions such as channel swaps, grayscale
//! decode, missing YCbCr conversion, or unexpectedly desaturated output.

mod support;

use std::path::{Path, PathBuf};

use statumen::{LevelIdx, PlaneIdx, PlaneSelection, RegionRequest, SceneId, SeriesId, Slide};
#[cfg(feature = "parity-openslide")]
use support::compare::{compare_rgba, tolerance_failure, Tolerance};

#[derive(Clone, Copy, Debug)]
struct ColorThreshold {
    min_tissue_fraction: f64,
    min_tissue_chroma: f64,
    min_red_green_gap: f64,
}

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

#[derive(Debug)]
struct ColorCase {
    label: &'static str,
    alias: Option<&'static str>,
    env_path: Option<&'static str>,
    relative_paths: &'static [&'static str],
    absolute_paths: &'static [&'static str],
    threshold: ColorThreshold,
}

#[derive(Clone, Debug)]
struct RgbaRegion {
    level: u32,
    x: i64,
    y: i64,
    width: u32,
    height: u32,
    pixels_rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct ColorStats {
    opaque_fraction: f64,
    tissue_fraction: f64,
    mean_rgb: [f64; 3],
    tissue_mean_rgb: [f64; 3],
    tissue_mean_chroma: f64,
}

#[test]
fn public_brightfield_color_sanity_svs_ndpi_dicom() {
    let cases = color_cases();
    let mut checked = 0usize;
    let mut missing = Vec::new();

    for case in &cases {
        let Some(path) = resolve_case_path(case) else {
            missing.push(case.label);
            eprintln!("[color] {}: missing sample; skipping", case.label);
            continue;
        };
        let region = read_overview_region(&path)
            .unwrap_or_else(|err| panic!("[color] {} read failed: {err}", case.label));
        let stats = color_stats(&region.pixels_rgba);
        eprintln!(
            "[color] {} path={} level={} origin=({}, {}) size={}x{} opaque={:.4} tissue={:.4} mean_rgb=({:.2},{:.2},{:.2}) tissue_rgb=({:.2},{:.2},{:.2}) tissue_chroma={:.2}",
            case.label,
            path.display(),
            region.level,
            region.x,
            region.y,
            region.width,
            region.height,
            stats.opaque_fraction,
            stats.tissue_fraction,
            stats.mean_rgb[0],
            stats.mean_rgb[1],
            stats.mean_rgb[2],
            stats.tissue_mean_rgb[0],
            stats.tissue_mean_rgb[1],
            stats.tissue_mean_rgb[2],
            stats.tissue_mean_chroma,
        );
        assert_color_sane(case.label, stats, case.threshold);
        checked += 1;
    }

    if checked == 0 {
        eprintln!(
            "[color] no public color samples were available; skipped {:?}",
            missing
        );
    }
}

#[cfg(feature = "parity-openslide")]
#[test]
fn svs_and_ndpi_overview_color_matches_openslide() {
    let Some(lib) = support::openslide_shim::try_load() else {
        eprintln!("[color-openslide] libopenslide unavailable; skipping");
        return;
    };
    let cases: Vec<_> = color_cases()
        .into_iter()
        .filter(|case| matches!(case.label, "svs" | "ndpi"))
        .collect();
    let mut checked = 0usize;

    for case in &cases {
        let Some(path) = resolve_case_path(case) else {
            eprintln!("[color-openslide] {}: missing sample; skipping", case.label);
            continue;
        };
        let ours = read_overview_region(&path)
            .unwrap_or_else(|err| panic!("[color-openslide] {} statumen read: {err}", case.label));
        let openslide = lib
            .open(&path)
            .unwrap_or_else(|err| panic!("[color-openslide] {} OpenSlide open: {err}", case.label));
        let theirs = openslide
            .read_region(ours.x, ours.y, ours.level, ours.width, ours.height)
            .unwrap_or_else(|err| panic!("[color-openslide] {} OpenSlide read: {err}", case.label));
        let report = compare_rgba(&ours.pixels_rgba, &theirs, Tolerance::TOLERANT);
        eprintln!(
            "[color-openslide] {} max_abs={} mean_abs={:.4} psnr={:.2}dB equal_rate={:.4}",
            case.label, report.max_abs, report.mean_abs, report.psnr_db, report.bytewise_equal_rate,
        );
        if let Some(failure) = tolerance_failure(case.label, &report) {
            panic!("[color-openslide] {failure}");
        }
        checked += 1;
    }

    if checked == 0 {
        eprintln!("[color-openslide] no SVS/NDPI samples were available; skipped");
    }
}

fn color_cases() -> Vec<ColorCase> {
    vec![
        ColorCase {
            label: "svs",
            alias: Some("svs-001"),
            env_path: Some("STATUMEN_COLOR_SVS_PATH"),
            relative_paths: &["downloads/openslide-testdata/Aperio/CMU-1.svs"],
            absolute_paths: &[],
            threshold: ColorThreshold {
                min_tissue_fraction: 0.005,
                min_tissue_chroma: 18.0,
                min_red_green_gap: 8.0,
            },
        },
        ColorCase {
            label: "ndpi",
            alias: Some("ndpi-001"),
            env_path: Some("STATUMEN_COLOR_NDPI_PATH"),
            relative_paths: &["downloads/openslide-testdata/Hamamatsu/CMU-1.ndpi"],
            absolute_paths: &[],
            threshold: ColorThreshold {
                min_tissue_fraction: 0.02,
                min_tissue_chroma: 25.0,
                min_red_green_gap: 15.0,
            },
        },
        ColorCase {
            label: "dicom-jp2k",
            alias: Some("dicom-jp2k-001"),
            env_path: Some("STATUMEN_COLOR_DICOM_PATH"),
            relative_paths: &[
                "downloads/openslide-testdata-extracted/dicom/dicom-cmu1-jp2k/DCM_0.dcm",
                "downloads/openslide-testdata-extracted/full/DICOM/CMU-1-JP2K-33005/DCM_0.dcm",
            ],
            absolute_paths: &[],
            threshold: ColorThreshold {
                min_tissue_fraction: 0.005,
                min_tissue_chroma: 14.0,
                min_red_green_gap: 4.0,
            },
        },
        ColorCase {
            label: "dicom-htj2k",
            alias: None,
            env_path: Some("STATUMEN_COLOR_HTJ2K_PATH"),
            relative_paths: &[],
            absolute_paths: &[
                "/private/tmp/wsi-dicom-htj2k-cmu1-full/level-0000-z0000-c0000-t0000.dcm",
                "/private/tmp/wsi-dicom-htj2k-test-cmu-small-20260505115912/level-0000-z0000-c0000-t0000.dcm",
            ],
            threshold: ColorThreshold {
                min_tissue_fraction: 0.005,
                min_tissue_chroma: 14.0,
                min_red_green_gap: 4.0,
            },
        },
    ]
}

fn resolve_case_path(case: &ColorCase) -> Option<PathBuf> {
    if let Some(env_path) = case.env_path {
        if let Some(path) = std::env::var_os(env_path).map(PathBuf::from) {
            if path.is_file() {
                return Some(path);
            }
        }
    }

    if let Some(alias) = case.alias {
        if let Some(path) = support::corpus::find_slide_by_alias(alias) {
            return Some(path);
        }
    }

    for relative in case.relative_paths {
        for root in candidate_roots() {
            let path = root.join(relative);
            if path.is_file() {
                return Some(path);
            }
        }
    }

    for absolute in case.absolute_paths {
        let path = PathBuf::from(absolute);
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

fn candidate_roots() -> Vec<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bench = manifest_dir.parent().unwrap_or(manifest_dir);
    let slideviewer = bench.join("SlideViewer");
    vec![manifest_dir.to_path_buf(), bench.to_path_buf(), slideviewer]
}

fn read_overview_region(path: &Path) -> Result<RgbaRegion, String> {
    let slide = Slide::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let series = slide
        .dataset()
        .scenes
        .first()
        .and_then(|scene| scene.series.first())
        .ok_or_else(|| "slide has no scene/series".to_string())?;
    let (level, dimensions) = series
        .levels
        .iter()
        .enumerate()
        .rev()
        .find(|(_, level)| level.dimensions.0 <= 2048 && level.dimensions.1 <= 2048)
        .or_else(|| series.levels.iter().enumerate().next_back())
        .map(|(idx, level)| (idx as u32, level.dimensions))
        .ok_or_else(|| "slide has no levels".to_string())?;
    let width = u32::try_from(dimensions.0.min(1024)).map_err(|_| "overview width overflow")?;
    let height = u32::try_from(dimensions.1.min(1024)).map_err(|_| "overview height overflow")?;
    let req = region_request(0, 0, level, PlaneSelection::default(), 0, 0, width, height);
    let pixels_rgba = slide
        .read_region_rgba(&req)
        .map_err(|err| format!("read overview level {level}: {err}"))?
        .into_raw();
    Ok(RgbaRegion {
        level,
        x: 0,
        y: 0,
        width,
        height,
        pixels_rgba,
    })
}

fn color_stats(rgba: &[u8]) -> ColorStats {
    assert!(rgba.len().is_multiple_of(4), "RGBA buffer must be aligned");
    let mut opaque = 0usize;
    let mut rgb_sum = [0f64; 3];
    let mut tissue = 0usize;
    let mut tissue_rgb_sum = [0f64; 3];
    let mut tissue_chroma_sum = 0f64;

    for pixel in rgba.chunks_exact(4) {
        if pixel[3] == 0 {
            continue;
        }
        opaque += 1;
        let rgb = [
            f64::from(pixel[0]),
            f64::from(pixel[1]),
            f64::from(pixel[2]),
        ];
        for channel in 0..3 {
            rgb_sum[channel] += rgb[channel];
        }
        let max = rgb[0].max(rgb[1]).max(rgb[2]);
        let min = rgb[0].min(rgb[1]).min(rgb[2]);
        let brightness = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        let chroma = max - min;
        if brightness < 235.0 && chroma > 12.0 {
            tissue += 1;
            for channel in 0..3 {
                tissue_rgb_sum[channel] += rgb[channel];
            }
            tissue_chroma_sum += chroma;
        }
    }

    let denom = opaque.max(1) as f64;
    let tissue_denom = tissue.max(1) as f64;
    ColorStats {
        opaque_fraction: opaque as f64 / (rgba.len() / 4).max(1) as f64,
        tissue_fraction: tissue as f64 / denom,
        mean_rgb: rgb_sum.map(|sum| sum / denom),
        tissue_mean_rgb: tissue_rgb_sum.map(|sum| sum / tissue_denom),
        tissue_mean_chroma: tissue_chroma_sum / tissue_denom,
    }
}

fn assert_color_sane(label: &str, stats: ColorStats, threshold: ColorThreshold) {
    assert!(
        stats.opaque_fraction > 0.95,
        "[color] {label}: expected mostly opaque overview, got {:.4}",
        stats.opaque_fraction
    );
    assert!(
        stats.tissue_fraction >= threshold.min_tissue_fraction,
        "[color] {label}: tissue fraction {:.4} below {:.4}",
        stats.tissue_fraction,
        threshold.min_tissue_fraction
    );
    assert!(
        stats.tissue_mean_chroma >= threshold.min_tissue_chroma,
        "[color] {label}: tissue chroma {:.2} below {:.2}; possible desaturation/grayscale decode",
        stats.tissue_mean_chroma,
        threshold.min_tissue_chroma
    );
    assert!(
        stats.tissue_mean_rgb[0] >= stats.tissue_mean_rgb[1] + threshold.min_red_green_gap,
        "[color] {label}: red channel does not dominate green enough in H&E tissue: tissue_rgb=({:.2},{:.2},{:.2}) min_gap={:.2}",
        stats.tissue_mean_rgb[0],
        stats.tissue_mean_rgb[1],
        stats.tissue_mean_rgb[2],
        threshold.min_red_green_gap
    );
}
