use std::env;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::RgbImage;
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
use statumen::{
    LevelIdx, PlaneIdx, PlaneSelection, RegionRequest, SceneId, SeriesId, Slide, TileLayout,
    TileRequest, TileViewRequest,
};
use tempfile::NamedTempFile;

const TILE_CACHE_BYTES_ENV: &str = "STATUMEN_TILE_CACHE_BYTES";
const DEFAULT_REAL_WSI_ROOT_RELATIVE: &[&str] = &["downloads", "openslide-testdata"];
const DEFAULT_SLIDEVIEWER_WSI_ROOT_RELATIVE: &[&str] =
    &["..", "SlideViewer", "downloads", "openslide-testdata"];

struct EnvVarGuard {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = env::var_os(key);
        env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            env::set_var(self.key, previous);
        } else {
            env::remove_var(self.key);
        }
    }
}

fn centered_level0_region(handle: &Slide, side: u32) -> RegionRequest {
    let dims = handle.dataset().scenes[0].series[0].levels[0].dimensions;
    let w = side.min(dims.0 as u32);
    let h = side.min(dims.1 as u32);
    RegionRequest {
        scene: SceneId(0),
        series: SeriesId(0),
        level: LevelIdx(0),
        plane: PlaneIdx(PlaneSelection::default()),
        origin_px: (
            ((dims.0 as i64 - i64::from(w)) / 2).max(0),
            ((dims.1 as i64 - i64::from(h)) / 2).max(0),
        ),
        size_px: (w, h),
    }
}

#[derive(Clone, Copy)]
struct RegularLevel {
    index: u32,
    tiles_across: u64,
    tiles_down: u64,
}

fn external_slide_path(env_var: &str, relative_default: &[&str]) -> Option<PathBuf> {
    if let Some(path) = env::var_os(env_var) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    for root in [
        DEFAULT_REAL_WSI_ROOT_RELATIVE,
        DEFAULT_SLIDEVIEWER_WSI_ROOT_RELATIVE,
    ] {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for part in root {
            path.push(part);
        }
        for part in relative_default {
            path.push(part);
        }
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

fn external_slide_path_env_only(env_var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env::var_os(env_var)?);
    path.is_file().then_some(path)
}

fn external_sample_region(handle: &Slide) -> RegionRequest {
    centered_level0_region(handle, 512)
}

fn external_sample_display_tile(handle: &Slide) -> TileViewRequest {
    let dims = handle.dataset().scenes[0].series[0].levels[0].dimensions;
    TileViewRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
        tile_width: 256.min(dims.0 as u32),
        tile_height: 256.min(dims.1 as u32),
    }
}

fn regular_level_with_min_tiles(
    handle: &Slide,
    min_across: u64,
    min_down: u64,
) -> Option<RegularLevel> {
    handle.dataset().scenes[0].series[0]
        .levels
        .iter()
        .enumerate()
        .find_map(|(index, level)| {
            let TileLayout::Regular {
                tiles_across,
                tiles_down,
                ..
            } = &level.tile_layout
            else {
                return None;
            };
            (*tiles_across >= min_across && *tiles_down >= min_down).then_some(RegularLevel {
                index: index as u32,
                tiles_across: *tiles_across,
                tiles_down: *tiles_down,
            })
        })
}

fn tile_requests(level: RegularLevel, cols: i64, rows: i64) -> Vec<TileRequest> {
    assert!(level.tiles_across >= cols as u64);
    assert!(level.tiles_down >= rows as u64);
    let mut reqs = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
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

struct BenchSlide {
    _file: NamedTempFile,
    handle: Slide,
}

impl BenchSlide {
    fn open(file: NamedTempFile) -> Self {
        let handle = Slide::open(file.path()).expect("open benchmark slide");
        Self {
            _file: file,
            handle,
        }
    }
}

fn le_u16(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}

fn le_u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

fn short_in_u32(v: u16) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    bytes[..2].copy_from_slice(&le_u16(v));
    bytes
}

fn encode_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut image = RgbImage::new(width, height);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8]);
    }

    let mut encoded = Vec::new();
    JpegEncoder::new(&mut encoded, 80)
        .encode(
            image.as_raw(),
            image.width() as u16,
            image.height() as u16,
            JpegColorType::Rgb,
        )
        .expect("encode jpeg");
    encoded
}

fn build_aperio_tiled_tiff(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    compression_tag: u16,
    tiles: &[Vec<u8>],
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let mut tile_offsets = Vec::with_capacity(tiles.len());
    let mut tile_byte_counts = Vec::with_capacity(tiles.len());
    for tile in tiles {
        tile_offsets.push(buf.len() as u32);
        tile_byte_counts.push(tile.len() as u32);
        buf.extend_from_slice(tile);
    }

    let desc = b"Aperio Image Library|AppMag = 20|MPP = 0.250000\0";
    let desc_offset = buf.len() as u32;
    buf.extend_from_slice(desc);

    let tile_offsets_offset = if tile_offsets.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_offsets {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let tile_byte_counts_offset = if tile_byte_counts.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_byte_counts {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (258u16, 3u16, 1u32, short_in_u32(8)),
        (259u16, 3u16, 1u32, short_in_u32(compression_tag)),
        (262u16, 3u16, 1u32, short_in_u32(2)),
        (270u16, 2u16, desc.len() as u32, le_u32(desc_offset)),
        (277u16, 3u16, 1u32, short_in_u32(3)),
        (322u16, 4u16, 1u32, le_u32(tile_width)),
        (323u16, 4u16, 1u32, le_u32(tile_height)),
        (
            324u16,
            4u16,
            tile_offsets.len() as u32,
            tile_offsets_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_offsets[0])),
        ),
        (
            325u16,
            4u16,
            tile_byte_counts.len() as u32,
            tile_byte_counts_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().expect("create aperio tempfile");
    file.write_all(&buf).expect("write aperio tempfile");
    file.flush().expect("flush aperio tempfile");
    file
}

fn build_ndpi_full_decode_tiff(width: u32, height: u32) -> NamedTempFile {
    let jpeg = encode_test_jpeg(width, height);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let strip_offset = buf.len() as u32;
    let strip_byte_count = jpeg.len() as u32;
    buf.extend_from_slice(&jpeg);

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (259u16, 3u16, 1u32, short_in_u32(7)),
        (273u16, 4u16, 1u32, le_u32(strip_offset)),
        (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
        (65420u16, 4u16, 1u32, le_u32(1)),
        (65421u16, 11u16, 1u32, 40.0f32.to_le_bytes()),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&0u64.to_le_bytes());

    let mut file = NamedTempFile::new().expect("create ndpi tempfile");
    file.write_all(&buf).expect("write ndpi tempfile");
    file.flush().expect("flush ndpi tempfile");
    file
}

fn synthetic_aperio_jpeg() -> BenchSlide {
    let tile = encode_test_jpeg(128, 128);
    let file = build_aperio_tiled_tiff(
        256,
        256,
        128,
        128,
        7,
        &[tile.clone(), tile.clone(), tile.clone(), tile],
    );
    BenchSlide::open(file)
}

fn synthetic_aperio_jp2k() -> BenchSlide {
    let tile = include_bytes!("../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
    let file = build_aperio_tiled_tiff(
        16,
        16,
        8,
        8,
        33004,
        &[tile.clone(), tile.clone(), tile.clone(), tile],
    );
    BenchSlide::open(file)
}

fn synthetic_ndpi_full_decode() -> BenchSlide {
    let file = build_ndpi_full_decode_tiff(256, 256);
    BenchSlide::open(file)
}

fn benchmark_handle_reads(c: &mut Criterion) {
    let aperio_jpeg = synthetic_aperio_jpeg();
    let aperio_jp2k = synthetic_aperio_jp2k();
    let ndpi = synthetic_ndpi_full_decode();

    let tile_req = TileViewRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
        tile_width: 128,
        tile_height: 128,
    };
    let region_req = RegionRequest {
        scene: SceneId(0),
        series: SeriesId(0),
        level: LevelIdx(0),
        plane: PlaneIdx(PlaneSelection::default()),
        origin_px: (32, 32),
        size_px: (160, 160),
    };
    let small_region_req = RegionRequest {
        scene: SceneId(0),
        series: SeriesId(0),
        level: LevelIdx(0),
        plane: PlaneIdx(PlaneSelection::default()),
        origin_px: (2, 2),
        size_px: (12, 12),
    };

    let mut group = c.benchmark_group("synthetic_read_paths");
    group.bench_function("aperio_jpeg_display_tile", |b| {
        b.iter(|| aperio_jpeg.handle.read_display_tile(&tile_req).unwrap())
    });
    group.bench_function("aperio_jpeg_region", |b| {
        b.iter(|| aperio_jpeg.handle.read_region(&region_req).unwrap())
    });
    group.bench_function("aperio_jp2k_region", |b| {
        b.iter(|| aperio_jp2k.handle.read_region(&small_region_req).unwrap())
    });
    group.bench_function("ndpi_full_decode_display_tile", |b| {
        b.iter(|| ndpi.handle.read_display_tile(&tile_req).unwrap())
    });
    group.bench_function("ndpi_full_decode_region", |b| {
        b.iter(|| ndpi.handle.read_region(&region_req).unwrap())
    });
    group.finish();
}

fn benchmark_external_samples(c: &mut Criterion) {
    let sample_vars = [
        ("aperio_jpeg", "STATUMEN_BENCH_APERIO_JPEG", None),
        ("aperio_jp2k", "STATUMEN_BENCH_APERIO_JP2K", None),
        ("ndpi", "STATUMEN_BENCH_NDPI", None),
        (
            "zeiss_zvi_merged",
            "STATUMEN_BENCH_ZEISS_ZVI",
            Some(&["Zeiss", "Zeiss-1-Merged.zvi"][..]),
        ),
        (
            "zeiss_zvi_mosaic",
            "STATUMEN_BENCH_ZEISS_ZVI_MOSAIC",
            Some(&["Zeiss", "Zeiss-3-Mosaic.zvi"][..]),
        ),
        (
            "zeiss_czi",
            "STATUMEN_BENCH_ZEISS_CZI",
            Some(&["Zeiss", "Zeiss-5-Uncompressed.czi"][..]),
        ),
    ];

    let mut group = c.benchmark_group("external_samples");
    for (label, env_var, relative_default) in sample_vars {
        let path = relative_default
            .and_then(|relative_default| external_slide_path(env_var, relative_default))
            .or_else(|| external_slide_path_env_only(env_var));
        let Some(path) = path else { continue };

        let handle = Slide::open(&path).expect("open external benchmark slide");
        let tile_req = external_sample_display_tile(&handle);
        let region_req = external_sample_region(&handle);

        group.bench_with_input(
            BenchmarkId::new(label, "display_tile"),
            &handle,
            |b, handle| b.iter(|| black_box(handle.read_display_tile(&tile_req).unwrap())),
        );
        group.bench_with_input(BenchmarkId::new(label, "region"), &handle, |b, handle| {
            b.iter(|| black_box(handle.read_region(&region_req).unwrap()))
        });
    }
    group.finish();
}

fn benchmark_external_jp2k_backends(c: &mut Criterion) {
    let Some(path) = env::var_os("STATUMEN_BENCH_APERIO_JP2K") else {
        return;
    };
    let path = Path::new(&path);
    if !path.is_file() {
        return;
    }

    let _cache_guard = EnvVarGuard::set(TILE_CACHE_BYTES_ENV, "0");
    let mut group = c.benchmark_group("external_jp2k_production_policy");
    let handle = Slide::open(path).expect("open external JP2K benchmark slide");
    let region_req = centered_level0_region(&handle, 2048);
    group.bench_with_input(
        BenchmarkId::new("aperio_jp2k_region", "signinum_auto"),
        &(handle, region_req),
        |b, (handle, region_req)| b.iter(|| handle.read_region(region_req).unwrap()),
    );

    group.finish();
}

fn benchmark_external_wsi_tile_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_wsi_tile_batches");

    if let Some(path) = external_slide_path("STATUMEN_BENCH_APERIO_JPEG", &["Aperio", "CMU-1.svs"])
    {
        let handle = Slide::open(&path).expect("open external Aperio JPEG benchmark slide");
        if let Some(level) = regular_level_with_min_tiles(&handle, 8, 8) {
            let reqs = tile_requests(level, 8, 8);
            group.bench_function("aperio_jpeg_64_distinct_read_tiles", |b| {
                b.iter(|| handle.source().read_tiles_cpu(&reqs).unwrap())
            });
            group.bench_function("aperio_jpeg_64_distinct_read_tile_loop", |b| {
                b.iter(|| {
                    reqs.iter()
                        .map(|req| handle.source().read_tile_cpu(req))
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap()
                })
            });
        }
    }

    if let Some(path) = external_slide_path(
        "STATUMEN_BENCH_APERIO_JP2K",
        &["Aperio", "JP2K-33003-1.svs"],
    ) {
        let handle = Slide::open(&path).expect("open external Aperio JP2K benchmark slide");
        if let Some(level) = regular_level_with_min_tiles(&handle, 4, 4) {
            let reqs = tile_requests(level, 4, 4);
            group.bench_function("aperio_jp2k_16_distinct_read_tiles", |b| {
                b.iter(|| handle.source().read_tiles_cpu(&reqs).unwrap())
            });
            group.bench_function("aperio_jp2k_16_distinct_read_tile_loop", |b| {
                b.iter(|| {
                    reqs.iter()
                        .map(|req| handle.source().read_tile_cpu(req))
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap()
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_handle_reads,
    benchmark_external_samples,
    benchmark_external_jp2k_backends,
    benchmark_external_wsi_tile_batches
);
criterion_main!(benches);
