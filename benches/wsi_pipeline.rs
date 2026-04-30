//! Custom timing harness mirroring viewer prefetch waves.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use ziggurat::{FormatRegistry, PlaneSelection, Slide, TileLayout, TileRequest};

#[path = "../tests/support/mod.rs"]
mod support;

use support::corpus::{load_public, resolve_entry_path};

#[derive(Serialize)]
struct Sample {
    alias: String,
    level: u32,
    cold_first_tile_ms: f64,
    batch_p50_ms: f64,
    batch_p95_ms: f64,
    batch_p99_ms: f64,
    sustained_tiles_per_sec: f64,
    tile_count: u32,
    rss_peak_kb: Option<u64>,
}

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::from_secs(0);
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct).round() as usize;
    sorted[idx]
}

fn rss_peak_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest.split_whitespace().next().and_then(|s| s.parse().ok());
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id().to_string();
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn run_wave(alias: &str, handle: &Slide, level: u32, tile_count: u32) -> Option<Sample> {
    let level_meta = &handle.dataset().scenes[0].series[0].levels[level as usize];
    let tiles_down = match &level_meta.tile_layout {
        TileLayout::Regular { tiles_down, .. } => *tiles_down,
        TileLayout::WholeLevel {
            virtual_tile_height,
            height,
            ..
        } => height.div_ceil(u64::from(*virtual_tile_height)),
        TileLayout::Irregular { .. } => return None,
    };

    let mut reqs = Vec::with_capacity(tile_count as usize);
    let mut col = 0i64;
    let mut row = 0i64;
    for _ in 0..tile_count {
        reqs.push(TileRequest {
            scene: 0,
            series: 0,
            level,
            plane: PlaneSelection::default(),
            col,
            row,
        });
        col += 1;
        if col >= 4 {
            col = 0;
            row = (row + 1).min(i64::try_from(tiles_down.saturating_sub(1)).unwrap_or(0));
        }
    }

    let start = Instant::now();
    let _ = handle.source().read_tile_cpu(&reqs[0]).ok()?;
    let cold_first = start.elapsed();

    let mut latencies = Vec::with_capacity(reqs.len());
    let wave_start = Instant::now();
    for req in &reqs {
        let started = Instant::now();
        let _ = handle.source().read_tile_cpu(req).ok()?;
        latencies.push(started.elapsed());
    }
    let wave_total = wave_start.elapsed();
    latencies.sort();

    Some(Sample {
        alias: alias.to_string(),
        level,
        cold_first_tile_ms: cold_first.as_secs_f64() * 1e3,
        batch_p50_ms: percentile(&latencies, 0.50).as_secs_f64() * 1e3,
        batch_p95_ms: percentile(&latencies, 0.95).as_secs_f64() * 1e3,
        batch_p99_ms: percentile(&latencies, 0.99).as_secs_f64() * 1e3,
        sustained_tiles_per_sec: tile_count as f64 / wave_total.as_secs_f64().max(1e-9),
        tile_count,
        rss_peak_kb: rss_peak_kb(),
    })
}

fn main() {
    let manifest = match load_public() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("[wsi_pipeline] no corpus: {err}");
            println!("{{\"samples\":[]}}");
            return;
        }
    };

    let mut samples = Vec::new();
    for entry in manifest.slides {
        let path: PathBuf = resolve_entry_path(&entry);
        if !path.is_file() {
            continue;
        }
        let registry = FormatRegistry::builtin();
        let handle = match Slide::open_with_cache_bytes(&path, &registry, 256 * 1024 * 1024) {
            Ok(handle) => handle,
            Err(_) => continue,
        };
        let levels = handle.dataset().scenes[0].series[0].levels.len() as u32;
        for level in 0..levels.min(2) {
            if let Some(sample) = run_wave(&entry.alias, &handle, level, 32) {
                samples.push(sample);
            }
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "samples": samples }))
            .unwrap_or_else(|_| "{}".into())
    );
}
