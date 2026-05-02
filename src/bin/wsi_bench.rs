//! `wsi_bench` — runs the audit workload set against a single slide via statumen.
//!
//! Usage:
//!   `wsi_bench SLIDE_PATH`
//!
//! Prints a single JSON object to stdout describing the run.

#[allow(dead_code)]
mod bench_common;

use bench_common::{
    host_string, repeat_index, run_workload, selected_workload, should_run_workload,
    tile_top_left_at_pixel, BenchRun, WorkloadPlan, WorkloadResult,
};
use statumen::{PlaneSelection, RegionRequest, Slide, TileViewRequest};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: wsi_bench <slide-path>");
        std::process::exit(2);
    }
    let slide_path = PathBuf::from(&args[1]);
    if !slide_path.is_file() {
        eprintln!("slide path is not a file: {}", slide_path.display());
        std::process::exit(2);
    }
    let selected_workload = match selected_workload() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };
    let repeat_index = match repeat_index() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };
    let slide_path_json = slide_path.display().to_string();
    let host = host_string();

    let mut workloads = Vec::new();

    if should_run_workload(selected_workload.as_deref(), "cold_open") {
        let mut cold_open = WorkloadResult::new("cold_open");
        for _ in 0..10 {
            let start = std::time::Instant::now();
            match Slide::open(&slide_path) {
                Ok(_handle) => cold_open.samples.push(start.elapsed()),
                Err(e) => {
                    cold_open.error = Some(format!("open failed: {e}"));
                    break;
                }
            }
        }
        let cold_open_failed = cold_open.error.is_some();
        workloads.push(cold_open);

        if selected_workload.as_deref() == Some("cold_open") || cold_open_failed {
            let run = BenchRun::new(
                "statumen",
                slide_path_json,
                host,
                selected_workload,
                repeat_index,
                workloads,
            );
            println!("{}", run.to_json());
            std::process::exit(if cold_open_failed { 1 } else { 0 });
        }
    }

    let handle = match Slide::open(&slide_path) {
        Ok(handle) => handle,
        Err(err) => {
            let workload_name = selected_workload
                .clone()
                .unwrap_or_else(|| "single_tile_l0".to_string());
            let run = BenchRun::new(
                "statumen",
                slide_path_json,
                host,
                selected_workload,
                repeat_index,
                vec![WorkloadResult::with_error(
                    &workload_name,
                    format!("open failed: {err}"),
                )],
            );
            println!("{}", run.to_json());
            std::process::exit(1);
        }
    };
    let series = &handle.dataset().scenes[0].series[0];
    let level_dims: Vec<(u64, u64)> = series.levels.iter().map(|l| l.dimensions).collect();
    let plan = WorkloadPlan::compute(&level_dims);

    if should_run_workload(selected_workload.as_deref(), "single_tile_l0") {
        // Read the center tile of level 0 enough times that p99 has signal.
        let top_left = tile_top_left_at_pixel(plan.center_l0, plan.tile_px);
        workloads.push(run_workload::<_, statumen::WsiError>(
            "single_tile_l0",
            200,
            || {
                let req = display_tile_at_top_left(0, top_left, plan.tile_px);
                handle.read_display_tile(&req)
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "pan_trace_l0") {
        let pan_l0_coords = plan.pan_trace_l0();
        let mut idx = 0;
        workloads.push(run_workload::<_, statumen::WsiError>(
            "pan_trace_l0",
            pan_l0_coords.len(),
            || {
                let (x, y) = pan_l0_coords[idx];
                idx += 1;
                let req = display_tile_at_top_left(0, (x, y), plan.tile_px);
                handle.read_display_tile(&req)
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "pan_trace_l2") {
        let pan_l2_coords = plan.pan_trace_l2();
        let mut idx = 0;
        workloads.push(run_workload::<_, statumen::WsiError>(
            "pan_trace_l2",
            pan_l2_coords.len(),
            || {
                let (x, y) = pan_l2_coords[idx];
                idx += 1;
                let req = display_tile_at_top_left(plan.level2_idx, (x, y), plan.tile_px);
                handle.read_display_tile(&req)
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "pan_trace_l2_dense") {
        let dense_l2_coords = plan.pan_trace_l2_dense();
        let mut idx = 0;
        workloads.push(run_workload::<_, statumen::WsiError>(
            "pan_trace_l2_dense",
            dense_l2_coords.len(),
            || {
                let (x, y) = dense_l2_coords[idx];
                idx += 1;
                let req = display_tile_at_top_left(plan.level2_idx, (x, y), plan.tile_px);
                handle.read_display_tile(&req)
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "region_2k") {
        workloads.push(run_workload::<_, statumen::WsiError>(
            "region_2k",
            30,
            || {
                let req = RegionRequest::legacy_xywh(
                    0,
                    0,
                    0,
                    PlaneSelection::default(),
                    plan.center_l0.0 - 1024,
                    plan.center_l0.1 - 1024,
                    2048,
                    2048,
                );
                handle.read_region(&req)
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "viewport_region_l2") {
        let viewport = plan.centered_viewport_l2(1024);
        workloads.push(run_workload::<_, statumen::WsiError>(
            "viewport_region_l2",
            30,
            || {
                let req = RegionRequest::legacy_xywh(
                    0,
                    0,
                    plan.level2_idx,
                    PlaneSelection::default(),
                    viewport.level2_top_left.0,
                    viewport.level2_top_left.1,
                    viewport.side_px,
                    viewport.side_px,
                );
                handle.read_region(&req)
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "thumbnail") {
        let thumbnail = if handle.dataset().associated_images.contains_key("thumbnail") {
            run_workload::<_, statumen::WsiError>("thumbnail", 30, || {
                handle.read_associated("thumbnail")
            })
        } else {
            let deepest = plan.level_count - 1;
            let dims = level_dims[deepest as usize];
            run_workload::<_, statumen::WsiError>("thumbnail", 30, || {
                let req = RegionRequest::legacy_xywh(
                    0,
                    0,
                    deepest,
                    PlaneSelection::default(),
                    0,
                    0,
                    dims.0 as u32,
                    dims.1 as u32,
                );
                handle.read_region(&req)
            })
        };
        workloads.push(thumbnail);
    }

    let run = BenchRun::new(
        "statumen",
        slide_path_json,
        host,
        selected_workload,
        repeat_index,
        workloads,
    );
    println!("{}", run.to_json());
}

fn display_tile_at_top_left(level: u32, top_left: (i64, i64), tile_px: u32) -> TileViewRequest {
    let col = top_left.0.div_euclid(tile_px as i64);
    let row = top_left.1.div_euclid(tile_px as i64);
    TileViewRequest {
        scene: 0,
        series: 0,
        level,
        plane: PlaneSelection::default(),
        col,
        row,
        tile_width: tile_px,
        tile_height: tile_px,
    }
}
