use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use statumen::{
    Compression, DecodeExecutionOptions, PlaneSelection, Slide, SlideOpenOptions, TileCodecKind,
    TileLayout, TileRequest,
};

const DEFAULT_BATCH_SIZES: &[usize] = &[1, 16, 512, 1024];
const DEFAULT_REPEATS: usize = 3;

#[derive(Serialize)]
struct BatchMeasurement {
    batch_size: usize,
    repeats: usize,
    sample_ms: Vec<f64>,
    median_ms: f64,
    mean_ms: f64,
    tiles_per_second_median: f64,
    decoded_bytes_per_repeat: usize,
}

#[derive(Serialize)]
struct BenchReport {
    slide_path: String,
    level: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u64,
    tiles_down: u64,
    available_tiles: u64,
    codec: String,
    jp2k_cpu_threads: Option<usize>,
    measurements: Vec<BatchMeasurement>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(
            "usage: jp2k_batch_bench <slide-path-or-raw.j2c> [batch-size ...]\n       jp2k_batch_bench --export-raw-tiles <slide-path> <output-dir> [count]"
                .to_string(),
        );
    }
    if args[0] == "--export-raw-tiles" {
        return export_raw_tiles(&args[1..]);
    }

    let slide_path = PathBuf::from(&args[0]);
    if !slide_path.is_file() {
        return Err(format!(
            "slide path is not a file: {}",
            slide_path.display()
        ));
    }

    let batch_sizes = if args.len() > 1 {
        args[1..]
            .iter()
            .map(|value| parse_positive_usize(value, "batch size"))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        DEFAULT_BATCH_SIZES.to_vec()
    };
    let repeats = std::env::var("STATUMEN_JP2K_BATCH_BENCH_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "STATUMEN_JP2K_BATCH_BENCH_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);

    let jp2k_cpu_threads = std::env::var("STATUMEN_JP2K_CPU_THREADS")
        .ok()
        .map(|value| parse_positive_usize(&value, "STATUMEN_JP2K_CPU_THREADS"))
        .transpose()?;
    let mut decode_options = DecodeExecutionOptions::default();
    if let Some(threads) = jp2k_cpu_threads {
        let threads = NonZeroUsize::new(threads)
            .ok_or_else(|| "STATUMEN_JP2K_CPU_THREADS must be > 0".to_string())?;
        decode_options = decode_options.with_jp2k_cpu_threads(threads);
    }
    let slide = Slide::open_with_options(
        &slide_path,
        SlideOpenOptions::default().with_decode_execution_options(decode_options),
    )
    .map_err(|err| format!("open failed: {err}"))?;

    let (level, tile_width, tile_height, tiles_across, tiles_down) = select_jp2k_level(&slide)?;
    let available_tiles = tiles_across
        .checked_mul(tiles_down)
        .ok_or_else(|| "tile grid overflow".to_string())?;
    let first_req = tile_request(level, 0, 0);
    let codec = format!("{:?}", slide.tile_codec_kind(&first_req));

    let mut measurements = Vec::with_capacity(batch_sizes.len());
    for batch_size in batch_sizes {
        if batch_size as u64 > available_tiles {
            return Err(format!(
                "batch size {batch_size} exceeds available tile count {available_tiles}"
            ));
        }
        let requests = build_requests(level, tiles_across, batch_size)?;

        let warm = slide
            .source()
            .read_tiles_cpu(&requests)
            .map_err(|err| format!("warmup batch {batch_size} failed: {err}"))?;
        std::hint::black_box(decoded_bytes(&warm));

        let mut samples = Vec::with_capacity(repeats);
        let mut decoded_bytes_per_repeat = 0usize;
        for _ in 0..repeats {
            let started = Instant::now();
            let decoded = slide
                .source()
                .read_tiles_cpu(&requests)
                .map_err(|err| format!("batch {batch_size} failed: {err}"))?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            decoded_bytes_per_repeat = decoded_bytes(&decoded);
            std::hint::black_box(decoded_bytes_per_repeat);
            samples.push(elapsed);
        }
        let median_ms = median(samples.clone());
        let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
        measurements.push(BatchMeasurement {
            batch_size,
            repeats,
            sample_ms: samples,
            median_ms,
            mean_ms,
            tiles_per_second_median: batch_size as f64 / (median_ms / 1000.0),
            decoded_bytes_per_repeat,
        });
    }

    let report = BenchReport {
        slide_path: slide_path.display().to_string(),
        level,
        tile_width,
        tile_height,
        tiles_across,
        tiles_down,
        available_tiles,
        codec,
        jp2k_cpu_threads,
        measurements,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|err| err.to_string())?
    );
    Ok(())
}

fn export_raw_tiles(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args.len() > 3 {
        return Err(
            "usage: jp2k_batch_bench --export-raw-tiles <slide-path> <output-dir> [count]"
                .to_string(),
        );
    }
    let slide_path = PathBuf::from(&args[0]);
    if !slide_path.is_file() {
        return Err(format!(
            "slide path is not a file: {}",
            slide_path.display()
        ));
    }
    let output_dir = PathBuf::from(&args[1]);
    let requested_count = args
        .get(2)
        .map(|value| parse_positive_usize(value, "export count"))
        .transpose()?;

    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("create output dir {}: {err}", output_dir.display()))?;

    let slide = Slide::open_with_options(
        &slide_path,
        SlideOpenOptions::default()
            .with_decode_execution_options(DecodeExecutionOptions::default()),
    )
    .map_err(|err| format!("open failed: {err}"))?;
    let (level, tile_width, tile_height, tiles_across, tiles_down) = select_jp2k_level(&slide)?;
    let available_tiles = tiles_across
        .checked_mul(tiles_down)
        .ok_or_else(|| "tile grid overflow".to_string())?;
    let count = requested_count
        .unwrap_or(usize::try_from(available_tiles).map_err(|_| "tile count overflow")?);
    if count as u64 > available_tiles {
        return Err(format!(
            "export count {count} exceeds available tile count {available_tiles}"
        ));
    }

    let requests = build_requests(level, tiles_across, count)?;
    let mut skipped = 0usize;
    let mut exported = 0usize;
    for (index, req) in requests.iter().enumerate() {
        let raw = slide
            .read_raw_compressed_tile(req)
            .map_err(|err| format!("read raw tile {index}: {err}"))?;
        let extension = match raw.compression {
            Compression::Jp2kRgb | Compression::Jp2kYcbcr => "j2k",
            _ => {
                skipped += 1;
                continue;
            }
        };
        let path = output_dir.join(format!(
            "tile_{index:06}_r{row:05}_c{col:05}.{extension}",
            row = req.row,
            col = req.col
        ));
        fs::write(&path, raw.data)
            .map_err(|err| format!("write raw tile {}: {err}", path.display()))?;
        exported += 1;
    }

    println!(
        "{}",
        serde_json::json!({
            "slide_path": slide_path.display().to_string(),
            "output_dir": output_dir.display().to_string(),
            "level": level,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_across": tiles_across,
            "tiles_down": tiles_down,
            "available_tiles": available_tiles,
            "requested_tiles": count,
            "exported_tiles": exported,
            "skipped_non_jp2k_tiles": skipped,
        })
    );
    Ok(())
}

fn select_jp2k_level(slide: &Slide) -> Result<(u32, u32, u32, u64, u64), String> {
    let series = &slide.dataset().scenes[0].series[0];
    for (level_index, level) in series.levels.iter().enumerate() {
        let (tile_width, tile_height, tiles_across, tiles_down) = match &level.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across,
                tiles_down,
            } => (*tile_width, *tile_height, *tiles_across, *tiles_down),
            _ => continue,
        };
        let level = u32::try_from(level_index).map_err(|_| "level index overflow".to_string())?;
        let codec = slide.tile_codec_kind(&tile_request(level, 0, 0));
        if matches!(codec, TileCodecKind::Jp2k | TileCodecKind::Htj2k) {
            return Ok((level, tile_width, tile_height, tiles_across, tiles_down));
        }
    }
    Err("no regular JP2K/HTJ2K level found".into())
}

fn build_requests(
    level: u32,
    tiles_across: u64,
    batch_size: usize,
) -> Result<Vec<TileRequest>, String> {
    (0..batch_size)
        .map(|index| {
            let index = u64::try_from(index).map_err(|_| "batch index overflow".to_string())?;
            let col = i64::try_from(index % tiles_across)
                .map_err(|_| "tile column overflow".to_string())?;
            let row =
                i64::try_from(index / tiles_across).map_err(|_| "tile row overflow".to_string())?;
            Ok(tile_request(level, col, row))
        })
        .collect()
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

fn decoded_bytes(tiles: &[statumen::CpuTile]) -> usize {
    tiles.iter().map(|tile| tile.data.byte_size()).sum()
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid {label} {value:?}: {err}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be > 0"));
    }
    Ok(parsed)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}
