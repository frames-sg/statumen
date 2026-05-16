use std::collections::HashSet;
use std::path::{Path, PathBuf};

use statumen::{
    Compression, CpuTile, EncodedTilePhotometricInterpretation, FormatRegistry, LevelIdx, PlaneIdx,
    PlaneSelection, RegionRequest, SceneId, SeriesId, Slide, TileLayout, TileRequest,
};

mod support;

#[derive(Clone, Copy, Debug)]
struct RegularLevel {
    index: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u64,
    tiles_down: u64,
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

fn require_corpus_slide(alias: &str) -> PathBuf {
    match support::corpus::find_slide_by_alias(alias) {
        Some(path) => path,
        None => {
            eprintln!(
                "[real_wsi] corpus slide '{alias}' not found; run scripts/parity-corpus-fetch.sh or set STATUMEN_PARITY_CORPUS_CACHE"
            );
            panic!("corpus slide missing: {alias}");
        }
    }
}

fn aperio_jpeg_slide() -> PathBuf {
    require_corpus_slide("svs-001")
}

fn aperio_jp2k_slide() -> PathBuf {
    require_corpus_slide("svs-jp2k-001")
}

fn open_with_large_cache(path: &Path) -> Slide {
    let registry = FormatRegistry::builtin();
    Slide::open_with_cache_bytes(path, &registry, 128 * 1024 * 1024).expect("open real WSI fixture")
}

fn regular_level_with_min_tiles(handle: &Slide, min_across: u64, min_down: u64) -> RegularLevel {
    let levels = &handle.dataset().scenes[0].series[0].levels;
    for (index, level) in levels.iter().enumerate() {
        if let TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } = &level.tile_layout
        {
            if *tiles_across >= min_across && *tiles_down >= min_down {
                return RegularLevel {
                    index: index as u32,
                    tile_width: *tile_width,
                    tile_height: *tile_height,
                    tiles_across: *tiles_across,
                    tiles_down: *tiles_down,
                };
            }
        }
    }
    panic!("no regular level has at least {min_across}x{min_down} tiles");
}

fn tile_requests(
    level: RegularLevel,
    start_col: i64,
    start_row: i64,
    cols: i64,
    rows: i64,
) -> Vec<TileRequest> {
    let mut reqs = Vec::with_capacity((cols * rows) as usize);
    for row in start_row..start_row + rows {
        for col in start_col..start_col + cols {
            reqs.push(TileRequest {
                scene: 0,
                series: 0,
                level: level.index,
                plane: PlaneSelection::default(),
                col,
                row,
            });
        }
    }
    reqs
}

fn assert_distinct_requests(reqs: &[TileRequest], expected_len: usize) {
    let distinct = reqs
        .iter()
        .map(|req| (req.level, req.col, req.row))
        .collect::<HashSet<_>>();
    assert_eq!(reqs.len(), expected_len);
    assert_eq!(distinct.len(), expected_len);
}

fn assert_same_buffers(lhs: &CpuTile, rhs: &CpuTile) {
    assert_eq!(lhs.width, rhs.width);
    assert_eq!(lhs.height, rhs.height);
    assert_eq!(lhs.channels, rhs.channels);
    assert_eq!(lhs.color_space, rhs.color_space);
    assert_eq!(lhs.layout, rhs.layout);
    assert_eq!(lhs.data.as_u8(), rhs.data.as_u8());
}

fn tile_request(level: u32, col: i64, row: i64) -> TileRequest {
    TileRequest {
        scene: 0,
        series: 0,
        level,
        plane: PlaneSelection::default(),
        col,
        row,
    }
}

fn region_hits(handle: &Slide, req: &RegionRequest) -> HashSet<(i64, i64)> {
    let level = &handle.dataset().scenes[0].series[0].levels[req.level.0 as usize];
    level
        .tile_layout
        .tiles_for_region(
            req.origin_px.0,
            req.origin_px.1,
            req.size_px.0,
            req.size_px.1,
        )
        .into_iter()
        .map(|hit| (hit.col, hit.row))
        .collect()
}

#[test]
#[ignore = "requires public parity corpus; run after scripts/parity-corpus-fetch.sh"]
fn aperio_jpeg_distinct_tile_batch_matches_sequential_tile_reads() {
    let handle = Slide::open(aperio_jpeg_slide()).expect("open Aperio JPEG slide");
    let level = regular_level_with_min_tiles(&handle, 8, 8);
    let reqs = tile_requests(level, 0, 0, 8, 8);
    assert_distinct_requests(&reqs, 64);

    let batched = handle
        .source()
        .read_tiles_cpu(&reqs)
        .expect("batched read_tiles");
    let sequential = reqs
        .iter()
        .map(|req| handle.source().read_tile_cpu(req))
        .collect::<Result<Vec<_>, _>>()
        .expect("sequential read_tile");

    assert_eq!(batched.len(), sequential.len());
    for (batched, sequential) in batched.iter().zip(sequential.iter()) {
        assert_same_buffers(batched, sequential);
    }
}

#[test]
#[ignore = "requires public parity corpus; run after scripts/parity-corpus-fetch.sh"]
fn aperio_jpeg_viewport_pan_populates_and_reuses_distinct_tile_cache_entries() {
    let handle = open_with_large_cache(&aperio_jpeg_slide());
    let level = regular_level_with_min_tiles(&handle, 10, 8);
    assert!(level.tiles_across >= 10 && level.tiles_down >= 8);

    let first = region_request(
        0,
        0,
        level.index,
        PlaneSelection::default(),
        0,
        0,
        level.tile_width * 8,
        level.tile_height * 8,
    );
    let second = RegionRequest {
        origin_px: (i64::from(level.tile_width) * 2, first.origin_px.1),
        ..first.clone()
    };

    let first_hits = region_hits(&handle, &first);
    assert_eq!(first_hits.len(), 64);
    let first_image = handle.read_region(&first).expect("read first viewport");
    assert_eq!(
        (first_image.width, first_image.height),
        (first.size_px.0, first.size_px.1)
    );

    for &(col, row) in &first_hits {
        let req = tile_request(level.index, col, row);
        assert!(
            handle.cached_tile_present(&req),
            "missing cached tile {col},{row}"
        );
    }

    let second_hits = region_hits(&handle, &second);
    assert_eq!(second_hits.len(), 64);
    let overlap = first_hits.intersection(&second_hits).count();
    assert_eq!(overlap, 48);

    let second_image = handle.read_region(&second).expect("read panned viewport");
    assert_eq!(
        (second_image.width, second_image.height),
        (second.size_px.0, second.size_px.1)
    );
    for &(col, row) in &second_hits {
        let req = tile_request(level.index, col, row);
        assert!(
            handle.cached_tile_present(&req),
            "missing panned cached tile {col},{row}"
        );
    }
}

#[test]
#[ignore = "requires public parity corpus; run after scripts/parity-corpus-fetch.sh"]
fn aperio_jp2k_distinct_tile_batch_matches_sequential_tile_reads() {
    let handle = Slide::open(aperio_jp2k_slide()).expect("open Aperio JP2K slide");
    let level = regular_level_with_min_tiles(&handle, 4, 4);
    let reqs = tile_requests(level, 0, 0, 4, 4);
    assert_distinct_requests(&reqs, 16);

    let batched = handle
        .source()
        .read_tiles_cpu(&reqs)
        .expect("batched JP2K read_tiles");
    let sequential = reqs
        .iter()
        .map(|req| handle.source().read_tile_cpu(req))
        .collect::<Result<Vec<_>, _>>()
        .expect("sequential JP2K read_tile");

    assert_eq!(batched.len(), sequential.len());
    for (batched, sequential) in batched.iter().zip(sequential.iter()) {
        assert_same_buffers(batched, sequential);
    }
}

#[test]
#[ignore = "requires public parity corpus; run after scripts/parity-corpus-fetch.sh"]
fn aperio_jp2k_exposes_raw_compressed_codestream_tile() {
    let handle = Slide::open(aperio_jp2k_slide()).expect("open Aperio JP2K slide");
    let level = regular_level_with_min_tiles(&handle, 1, 1);

    let raw = handle
        .read_raw_compressed_tile(&tile_request(level.index, 0, 0))
        .expect("read raw compressed JP2K tile");

    assert!(matches!(
        raw.compression,
        Compression::Jp2kRgb | Compression::Jp2kYcbcr
    ));
    assert_eq!(
        (raw.width, raw.height),
        (level.tile_width, level.tile_height)
    );
    assert_eq!(raw.bits_allocated, 8);
    assert_eq!(raw.samples_per_pixel, 3);
    assert!(matches!(
        raw.photometric_interpretation,
        EncodedTilePhotometricInterpretation::Rgb
            | EncodedTilePhotometricInterpretation::YbrFull422
    ));
    assert!(raw.data.starts_with(&[0xFF, 0x4F]));
    assert!(raw.data.windows(2).any(|marker| marker == [0xFF, 0x51]));
}
