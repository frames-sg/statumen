#![cfg_attr(
    not(feature = "metal"),
    allow(dead_code, unreachable_code, unused_variables)
)]

use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use statumen::{PlaneSelection, Slide, TileLayout, TileOutputPreference, TilePixels, TileRequest};

#[cfg(feature = "metal")]
fn metal_sessions() -> Result<statumen::output::metal::MetalBackendSessions, String> {
    let device = metal::Device::system_default().ok_or("no system Metal device")?;
    Ok(statumen::output::metal::MetalBackendSessions::new(
        signinum_jpeg_metal::MetalBackendSession::new(device.clone()),
        signinum_j2k_metal::MetalBackendSession::new(device),
    ))
}

#[cfg(not(feature = "metal"))]
fn metal_sessions() -> Result<(), String> {
    Err("bench_dicom_tile_batch requires --features metal".to_string())
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn summarize(samples: &[Duration]) -> (f64, f64, f64) {
    let mut ms = samples.iter().copied().map(elapsed_ms).collect::<Vec<_>>();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = ms[ms.len() / 2];
    let p95 = ms[((ms.len() - 1) as f64 * 0.95).round() as usize];
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    (p50, p95, mean)
}

fn tile_requests(slide: &Slide, max_tiles: usize) -> Result<Vec<TileRequest>, String> {
    let series = &slide.dataset().scenes[0].series[0];
    for (level_index, level) in series.levels.iter().enumerate() {
        match &level.tile_layout {
            TileLayout::Regular {
                tiles_across,
                tiles_down,
                ..
            } => {
                let usable_cols = tiles_across.saturating_sub(1).max(1);
                let usable_rows = tiles_down.saturating_sub(1).max(1);
                let cols = usable_cols.min(max_tiles as u64).max(1);
                let rows = (usable_rows.min((max_tiles as u64).div_ceil(cols))).max(1);
                let mut reqs = Vec::new();
                for row in 0..rows {
                    for col in 0..cols {
                        if reqs.len() == max_tiles {
                            return Ok(reqs);
                        }
                        reqs.push(TileRequest {
                            scene: 0,
                            series: 0,
                            level: level_index as u32,
                            plane: PlaneSelection::default(),
                            col: col as i64,
                            row: row as i64,
                        });
                    }
                }
                if !reqs.is_empty() {
                    return Ok(reqs);
                }
            }
            TileLayout::Irregular { tiles, .. } => {
                let mut reqs = tiles
                    .keys()
                    .take(max_tiles)
                    .map(|&(col, row)| TileRequest {
                        scene: 0,
                        series: 0,
                        level: level_index as u32,
                        plane: PlaneSelection::default(),
                        col,
                        row,
                    })
                    .collect::<Vec<_>>();
                reqs.sort_by_key(|req| (req.level, req.row, req.col));
                if !reqs.is_empty() {
                    return Ok(reqs);
                }
            }
            TileLayout::WholeLevel { .. } => {}
        }
    }
    Err("slide has no tile-addressable level".to_string())
}

fn count_device(tiles: &[TilePixels]) -> usize {
    tiles
        .iter()
        .filter(|tile| matches!(tile, TilePixels::Device(_)))
        .count()
}

fn run_batch(
    slide: &Slide,
    reqs: &[TileRequest],
    output: &TileOutputPreference,
) -> Result<(Duration, usize), String> {
    let started = Instant::now();
    let tiles = slide
        .source()
        .read_tiles(reqs, output.clone())
        .map_err(|err| err.to_string())?;
    if tiles.len() != reqs.len() {
        return Err(format!(
            "read_tiles returned {} tiles for {} requests",
            tiles.len(),
            reqs.len()
        ));
    }
    Ok((started.elapsed(), count_device(&tiles)))
}

fn run_loop(
    slide: &Slide,
    reqs: &[TileRequest],
    output: &TileOutputPreference,
) -> Result<(Duration, usize), String> {
    let started = Instant::now();
    let mut device = 0;
    for req in reqs {
        let tile = slide
            .source()
            .read_tile(req, output.clone())
            .map_err(|err| err.to_string())?;
        if matches!(tile, TilePixels::Device(_)) {
            device += 1;
        }
    }
    Ok((started.elapsed(), device))
}

fn main() {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("usage: cargo run --example bench_dicom_tile_batch --features metal -- <dicom-path> [tile-count] [repeats]");
            std::process::exit(2);
        });
    let max_tiles = env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16);
    let repeats = env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(5);

    let slide = Slide::open(&path).unwrap_or_else(|err| {
        eprintln!("open {}: {err}", path.display());
        std::process::exit(1);
    });
    let reqs = tile_requests(&slide, max_tiles).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    #[cfg(feature = "metal")]
    let output = TileOutputPreference::require_device_auto_with_metal_and_compressed_decode(
        metal_sessions().unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        }),
    );

    #[cfg(not(feature = "metal"))]
    let output = {
        eprintln!("{}", metal_sessions().unwrap_err());
        std::process::exit(1);
    };

    let codec = slide.source().tile_codec_kind(&reqs[0]);
    println!("slide={}", path.display());
    println!("codec={codec:?}");
    println!("tile_count={}", reqs.len());
    println!("repeats={repeats}");

    let mut batch_samples = Vec::with_capacity(repeats);
    let mut loop_samples = Vec::with_capacity(repeats);
    let mut batch_device = 0;
    let mut loop_device = 0;

    for _ in 0..repeats {
        let (elapsed, device) = run_batch(&slide, &reqs, &output).unwrap_or_else(|err| {
            eprintln!("batch read failed: {err}");
            std::process::exit(1);
        });
        batch_samples.push(elapsed);
        batch_device = device;

        let (elapsed, device) = run_loop(&slide, &reqs, &output).unwrap_or_else(|err| {
            eprintln!("loop read failed: {err}");
            std::process::exit(1);
        });
        loop_samples.push(elapsed);
        loop_device = device;
    }

    let (batch_p50, batch_p95, batch_mean) = summarize(&batch_samples);
    let (loop_p50, loop_p95, loop_mean) = summarize(&loop_samples);
    println!("batch_device_tiles={batch_device}");
    println!("loop_device_tiles={loop_device}");
    println!("read_tiles_batch p50={batch_p50:.3}ms p95={batch_p95:.3}ms mean={batch_mean:.3}ms");
    println!("read_tile_loop p50={loop_p50:.3}ms p95={loop_p95:.3}ms mean={loop_mean:.3}ms");
    println!("loop_over_batch_mean_ratio={:.3}x", loop_mean / batch_mean);
}
