//! Bench harness driving the OpenSlide compatibility oracle.

#![cfg(feature = "parity-openslide")]

use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use statumen::{FormatRegistry, PlaneSelection, Slide, TileLayout, TileRequest};

#[path = "../tests/support/mod.rs"]
mod support;

use support::corpus::{load_public, resolve_entry_path};
use support::openslide_shim;

#[derive(Serialize)]
struct Sample {
    alias: String,
    level: u32,
    tile_count: u32,
    statumen_total_ms: f64,
    openslide_total_ms: f64,
    ratio_wsirs_over_openslide: f64,
}

fn main() {
    let manifest = match load_public() {
        Ok(manifest) => manifest,
        Err(_) => {
            println!("{{\"samples\":[]}}");
            return;
        }
    };
    let openslide = match openslide_shim::try_load() {
        Some(openslide) => openslide,
        None => {
            eprintln!("[openslide_parity] libopenslide not found; emitting empty result");
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
        let handle = match Slide::open_with_cache_bytes(&path, &registry, 128 * 1024 * 1024) {
            Ok(handle) => handle,
            Err(_) => continue,
        };
        let osr = match openslide.open(&path) {
            Ok(osr) => osr,
            Err(_) => continue,
        };
        let openslide_origin = osr.bounds().map_or((0, 0), |bounds| (bounds.x, bounds.y));
        let level0 = &handle.dataset().scenes[0].series[0].levels[0];
        let (tile_width, tile_height, tiles_across, tiles_down) = match &level0.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across,
                tiles_down,
            } => (*tile_width, *tile_height, *tiles_across, *tiles_down),
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                width,
                height,
            } => (
                *virtual_tile_width,
                *virtual_tile_height,
                width.div_ceil(u64::from(*virtual_tile_width)),
                height.div_ceil(u64::from(*virtual_tile_height)),
            ),
            TileLayout::Irregular { .. } => continue,
        };
        if tiles_across == 0 || tiles_down == 0 {
            continue;
        }
        let count = 32u32;
        let reqs: Vec<TileRequest> = (0..count as i64)
            .map(|i| TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: i % i64::try_from(tiles_across).unwrap_or(1),
                row: (i / i64::try_from(tiles_across).unwrap_or(1))
                    .min(i64::try_from(tiles_down.saturating_sub(1)).unwrap_or(0)),
            })
            .collect();

        let started = Instant::now();
        for req in &reqs {
            let _ = handle
                .source()
                .read_tile_cpu(req)
                .expect("statumen read_tile");
        }
        let statumen_ms = started.elapsed().as_secs_f64() * 1e3;

        let started = Instant::now();
        for req in &reqs {
            let _ = osr
                .read_region(
                    openslide_origin.0 + req.col * i64::from(tile_width),
                    openslide_origin.1 + req.row * i64::from(tile_height),
                    0,
                    tile_width,
                    tile_height,
                )
                .expect("openslide read_region");
        }
        let openslide_ms = started.elapsed().as_secs_f64() * 1e3;

        samples.push(Sample {
            alias: entry.alias,
            level: 0,
            tile_count: count,
            statumen_total_ms: statumen_ms,
            openslide_total_ms: openslide_ms,
            ratio_wsirs_over_openslide: statumen_ms / openslide_ms.max(1e-9),
        });
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "samples": samples }))
            .unwrap_or_else(|_| "{}".into())
    );
}
