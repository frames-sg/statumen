use std::env;
use std::path::{Path, PathBuf};

use statumen::{
    CpuTileData, LevelIdx, PlaneIdx, PlaneSelection, RegionRequest, SceneId, SeriesId, Slide,
};

const ZVI_FIXTURES: &[&str] = &[
    "Zeiss-1-Merged.zvi",
    "Zeiss-1-Stacked.zvi",
    "Zeiss-2-Merged.zvi",
    "Zeiss-2-Stacked.zvi",
    "Zeiss-3-Mosaic.zvi",
    "Zeiss-4-Mosaic.zvi",
];

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

fn zeiss_zvi_root() -> Option<PathBuf> {
    if let Some(path) = env::var_os("STATUMEN_ZVI_ROOT").map(PathBuf::from) {
        return path.is_dir().then_some(path);
    }

    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("SlideViewer")
        .join("downloads")
        .join("openslide-testdata")
        .join("Zeiss");
    local.is_dir().then_some(local)
}

fn fixture_paths() -> Option<Vec<PathBuf>> {
    let root = zeiss_zvi_root()?;
    let paths = ZVI_FIXTURES
        .iter()
        .map(|name| root.join(name))
        .collect::<Vec<_>>();
    paths.iter().all(|path| path.is_file()).then_some(paths)
}

fn assert_decodes_u16_region(
    slide: &Slide,
    plane: PlaneSelection,
    path: &Path,
    require_nonzero: bool,
) {
    let region = slide
        .read_region(&region_request(0, 0, 0, plane, 0, 0, 64, 64))
        .expect("read Zeiss ZVI region");
    assert_eq!((region.width, region.height), (64, 64));
    assert_eq!(region.channels, 1);
    let CpuTileData::U16(samples) = &region.data else {
        panic!("ZVI region should decode as U16");
    };
    if require_nonzero {
        assert!(
            samples.iter().any(|&sample| sample != 0),
            "ZVI region should contain image samples for {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "requires STATUMEN_ZVI_ROOT or local Zeiss ZVI testdata"]
fn builtin_registry_opens_zeiss_zvi_variants_and_reads_u16_regions() {
    let paths = fixture_paths().expect("set STATUMEN_ZVI_ROOT to the Zeiss ZVI fixture directory");

    for path in paths {
        let slide = Slide::open(&path).expect("open Zeiss ZVI through builtin registry");
        let dataset = slide.dataset();
        assert_eq!(dataset.properties.vendor(), Some("zeiss"));
        assert_eq!(dataset.properties.get("zeiss.format"), Some("zvi"));
        assert_eq!(dataset.scenes.len(), 1);
        assert_eq!(dataset.scenes[0].series.len(), 1);

        let series = &dataset.scenes[0].series[0];
        assert_eq!(series.sample_type, statumen::SampleType::Uint16);
        assert!(series.axes.c >= 1, "ZVI should expose channel planes");
        assert!(!series.levels.is_empty());

        assert_decodes_u16_region(&slide, PlaneSelection::default(), &path, true);
        assert_decodes_u16_region(
            &slide,
            PlaneSelection {
                c: series.axes.c - 1,
                ..PlaneSelection::default()
            },
            &path,
            false,
        );
        assert_decodes_u16_region(
            &slide,
            PlaneSelection {
                z: series.axes.z - 1,
                ..PlaneSelection::default()
            },
            &path,
            false,
        );

        assert!(dataset.associated_images.contains_key("thumbnail"));
        let thumbnail = slide
            .read_associated("thumbnail")
            .expect("read ZVI thumbnail");
        assert!(thumbnail.width > 0 && thumbnail.height > 0);
        assert_eq!(thumbnail.channels, 3);
        assert!(matches!(thumbnail.data, CpuTileData::U8(_)));

        match path.file_name().and_then(|name| name.to_str()) {
            Some("Zeiss-1-Merged.zvi") => {
                assert_eq!(series.axes.c, 3);
                assert_eq!(series.axes.z, 1);
                assert_eq!(series.levels[0].dimensions, (1480, 1132));
            }
            Some("Zeiss-1-Stacked.zvi") => {
                assert_eq!(series.axes.c, 3);
                assert_eq!(series.axes.z, 13);
                assert_eq!(series.levels[0].dimensions, (1388, 1040));
            }
            Some("Zeiss-3-Mosaic.zvi") => {
                assert_eq!(series.axes.c, 3);
                assert_eq!(series.axes.z, 1);
                assert_eq!(series.levels[0].dimensions, (13882, 21631));
            }
            _ => {}
        }
    }
}
