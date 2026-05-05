use std::env;
use std::path::{Path, PathBuf};

use statumen::{PlaneSelection, Slide, TileViewRequest};

fn zeiss_czi_fixture() -> Option<PathBuf> {
    if let Some(path) = env::var_os("STATUMEN_ZEISS_CZI_PATH").map(PathBuf::from) {
        return path.is_file().then_some(path);
    }

    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("SlideViewer")
        .join("downloads")
        .join("openslide-testdata")
        .join("Zeiss")
        .join("Zeiss-5-Uncompressed.czi");
    local.is_file().then_some(local)
}

#[test]
#[ignore = "requires STATUMEN_ZEISS_CZI_PATH or local Zeiss CZI testdata"]
fn builtin_registry_opens_zeiss_czi_and_reads_display_tile() {
    let path = zeiss_czi_fixture().expect("set STATUMEN_ZEISS_CZI_PATH to a Zeiss CZI fixture");

    let slide = Slide::open(&path).expect("open Zeiss CZI through builtin registry");
    let dataset = slide.dataset();
    assert_eq!(dataset.properties.vendor(), Some("zeiss"));
    assert!(!dataset.scenes.is_empty(), "Zeiss CZI should expose scenes");
    assert!(
        !dataset.scenes[0].series[0].levels.is_empty(),
        "Zeiss CZI should expose at least one level"
    );

    let tile = slide
        .read_display_tile(&TileViewRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
            tile_width: 256,
            tile_height: 256,
        })
        .expect("read Zeiss display tile");
    assert_eq!((tile.width, tile.height), (256, 256));
    assert!(
        tile.as_u8().is_some(),
        "Zeiss display tile should decode to CPU u8 samples"
    );
}
