//! `openslide_bench` — runs the audit workload set against a single slide via libopenslide.
//!
//! Usage:
//!   `openslide_bench SLIDE_PATH`
//!
//! Prints a single JSON object to stdout in the same shape as wsi_bench.

#![allow(non_camel_case_types)]

#[allow(dead_code)]
mod bench_common;

use bench_common::{
    host_string, repeat_index, run_workload, selected_workload, should_run_workload,
    tile_top_left_at_pixel, BenchRun, WorkloadPlan, WorkloadResult,
};
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::time::Instant;

#[repr(C)]
pub struct openslide_t {
    _private: [u8; 0],
}

#[link(name = "openslide")]
extern "C" {
    fn openslide_open(filename: *const std::os::raw::c_char) -> *mut openslide_t;
    fn openslide_close(osr: *mut openslide_t);
    fn openslide_get_error(osr: *mut openslide_t) -> *const std::os::raw::c_char;
    fn openslide_get_level_count(osr: *mut openslide_t) -> i32;
    fn openslide_get_level_dimensions(osr: *mut openslide_t, level: i32, w: *mut i64, h: *mut i64);
    fn openslide_read_region(
        osr: *mut openslide_t,
        dest: *mut u32,
        x: i64,
        y: i64,
        level: i32,
        w: i64,
        h: i64,
    );
    fn openslide_get_associated_image_names(
        osr: *mut openslide_t,
    ) -> *const *const std::os::raw::c_char;
    fn openslide_get_property_value(
        osr: *mut openslide_t,
        name: *const std::os::raw::c_char,
    ) -> *const std::os::raw::c_char;
    fn openslide_get_associated_image_dimensions(
        osr: *mut openslide_t,
        name: *const std::os::raw::c_char,
        w: *mut i64,
        h: *mut i64,
    );
    fn openslide_read_associated_image(
        osr: *mut openslide_t,
        name: *const std::os::raw::c_char,
        dest: *mut u32,
    );
}

/// Safe wrapper handle. Drops by calling openslide_close.
struct Slide {
    raw: *mut openslide_t,
}

impl Slide {
    fn open(path: &std::path::Path) -> Result<Self, String> {
        let cpath = CString::new(path.to_str().ok_or("path is not utf-8")?.as_bytes())
            .map_err(|e| e.to_string())?;
        let raw = unsafe { openslide_open(cpath.as_ptr()) };
        if raw.is_null() {
            return Err("openslide_open returned NULL".into());
        }
        // openslide reports errors via get_error after open succeeds.
        let err = unsafe { openslide_get_error(raw) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { openslide_close(raw) };
            return Err(format!("openslide error: {msg}"));
        }
        Ok(Self { raw })
    }

    fn level_count(&self) -> i32 {
        unsafe { openslide_get_level_count(self.raw) }
    }

    fn level_dims(&self, level: i32) -> (u64, u64) {
        let mut w: i64 = 0;
        let mut h: i64 = 0;
        unsafe { openslide_get_level_dimensions(self.raw, level, &mut w, &mut h) };
        (w as u64, h as u64)
    }

    fn read_region(&self, x: i64, y: i64, level: i32, w: i64, h: i64) -> Result<Vec<u32>, String> {
        let mut buf = vec![0u32; (w * h) as usize];
        unsafe { openslide_read_region(self.raw, buf.as_mut_ptr(), x, y, level, w, h) };
        let err = unsafe { openslide_get_error(self.raw) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            return Err(msg);
        }
        Ok(buf)
    }

    fn property(&self, name: &str) -> Option<String> {
        let cname = CString::new(name).ok()?;
        let value = unsafe { openslide_get_property_value(self.raw, cname.as_ptr()) };
        if value.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn bounds(&self) -> Option<SlideBounds> {
        let x = self.property("openslide.bounds-x")?.parse::<i64>().ok()?;
        let y = self.property("openslide.bounds-y")?.parse::<i64>().ok()?;
        let width = self
            .property("openslide.bounds-width")?
            .parse::<u64>()
            .ok()?;
        let height = self
            .property("openslide.bounds-height")?
            .parse::<u64>()
            .ok()?;
        (width > 0 && height > 0).then_some(SlideBounds {
            x,
            y,
            width,
            height,
        })
    }

    fn associated_names(&self) -> Vec<String> {
        let names_ptr = unsafe { openslide_get_associated_image_names(self.raw) };
        if names_ptr.is_null() {
            return vec![];
        }
        let mut out = Vec::new();
        let mut i = 0;
        loop {
            let p = unsafe { *names_ptr.add(i) };
            if p.is_null() {
                break;
            }
            out.push(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
            i += 1;
        }
        out
    }

    fn read_associated(&self, name: &str) -> Result<Vec<u32>, String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let mut w: i64 = 0;
        let mut h: i64 = 0;
        unsafe {
            openslide_get_associated_image_dimensions(self.raw, cname.as_ptr(), &mut w, &mut h)
        };
        if w == 0 || h == 0 {
            return Err("associated image has zero dimensions".into());
        }
        let mut buf = vec![0u32; (w * h) as usize];
        unsafe { openslide_read_associated_image(self.raw, cname.as_ptr(), buf.as_mut_ptr()) };
        Ok(buf)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlideBounds {
    x: i64,
    y: i64,
    width: u64,
    height: u64,
}

fn comparable_level_dims(
    level_dims: &[(u64, u64)],
    bounds: Option<SlideBounds>,
) -> Vec<(u64, u64)> {
    let Some(bounds) = bounds else {
        return level_dims.to_vec();
    };
    let Some(&(level0_w, level0_h)) = level_dims.first() else {
        return level_dims.to_vec();
    };
    if bounds.width >= level0_w && bounds.height >= level0_h {
        return level_dims.to_vec();
    }
    level_dims
        .iter()
        .map(|&(level_w, level_h)| {
            let dx = level0_w as f64 / level_w.max(1) as f64;
            let dy = level0_h as f64 / level_h.max(1) as f64;
            (
                ((bounds.width as f64 / dx).ceil() as u64)
                    .max(1)
                    .min(level_w),
                ((bounds.height as f64 / dy).ceil() as u64)
                    .max(1)
                    .min(level_h),
            )
        })
        .collect()
}

impl Drop for Slide {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { openslide_close(self.raw) };
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: openslide_bench <slide-path>");
        std::process::exit(2);
    }
    let slide_path = PathBuf::from(&args[1]);
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
            let start = Instant::now();
            match Slide::open(&slide_path) {
                Ok(_s) => cold_open.samples.push(start.elapsed()),
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
                "openslide",
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

    let slide = match Slide::open(&slide_path) {
        Ok(slide) => slide,
        Err(err) => {
            let workload_name = selected_workload
                .clone()
                .unwrap_or_else(|| "single_tile_l0".to_string());
            let run = BenchRun::new(
                "openslide",
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
    let level_count = slide.level_count();
    let level_dims: Vec<(u64, u64)> = (0..level_count).map(|l| slide.level_dims(l)).collect();
    let bounds = slide.bounds();
    let plan_dims = comparable_level_dims(&level_dims, bounds);
    let plan = WorkloadPlan::compute(&plan_dims);
    let origin = bounds.map_or((0, 0), |bounds| (bounds.x, bounds.y));

    if should_run_workload(selected_workload.as_deref(), "single_tile_l0") {
        let (x, y) = tile_top_left_at_pixel(plan.center_l0, plan.tile_px);
        workloads.push(run_workload::<_, String>("single_tile_l0", 200, || {
            slide.read_region(
                origin.0 + x,
                origin.1 + y,
                0,
                plan.tile_px as i64,
                plan.tile_px as i64,
            )
        }));
    }

    if should_run_workload(selected_workload.as_deref(), "pan_trace_l0") {
        let coords = plan.pan_trace_l0();
        let mut idx = 0;
        workloads.push(run_workload::<_, String>(
            "pan_trace_l0",
            coords.len(),
            || {
                let (x, y) = coords[idx];
                idx += 1;
                slide.read_region(
                    origin.0 + x,
                    origin.1 + y,
                    0,
                    plan.tile_px as i64,
                    plan.tile_px as i64,
                )
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "pan_trace_l2") {
        let coords = plan.pan_trace_l2_world_l0();
        let mut idx = 0;
        workloads.push(run_workload::<_, String>(
            "pan_trace_l2",
            coords.len(),
            || {
                let (x, y) = coords[idx];
                idx += 1;
                slide.read_region(
                    origin.0 + x,
                    origin.1 + y,
                    plan.level2_idx as i32,
                    plan.tile_px as i64,
                    plan.tile_px as i64,
                )
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "pan_trace_l2_dense") {
        let dense_coords = plan.pan_trace_l2_dense_world_l0();
        let mut idx = 0;
        workloads.push(run_workload::<_, String>(
            "pan_trace_l2_dense",
            dense_coords.len(),
            || {
                let (x, y) = dense_coords[idx];
                idx += 1;
                slide.read_region(
                    origin.0 + x,
                    origin.1 + y,
                    plan.level2_idx as i32,
                    plan.tile_px as i64,
                    plan.tile_px as i64,
                )
            },
        ));
    }

    if should_run_workload(selected_workload.as_deref(), "region_2k") {
        workloads.push(run_workload::<_, String>("region_2k", 30, || {
            let (cx, cy) = plan.center_l0;
            slide.read_region(origin.0 + cx - 1024, origin.1 + cy - 1024, 0, 2048, 2048)
        }));
    }

    if should_run_workload(selected_workload.as_deref(), "viewport_region_l2") {
        let viewport = plan.centered_viewport_l2(1024);
        workloads.push(run_workload::<_, String>("viewport_region_l2", 30, || {
            slide.read_region(
                origin.0 + viewport.level0_top_left.0,
                origin.1 + viewport.level0_top_left.1,
                plan.level2_idx as i32,
                viewport.side_px as i64,
                viewport.side_px as i64,
            )
        }));
    }

    if should_run_workload(selected_workload.as_deref(), "thumbnail") {
        let assoc_names = slide.associated_names();
        let thumbnail = if assoc_names.iter().any(|n| n == "thumbnail") {
            run_workload::<_, String>("thumbnail", 30, || slide.read_associated("thumbnail"))
        } else {
            let deepest = level_count - 1;
            let dims = slide.level_dims(deepest);
            run_workload::<_, String>("thumbnail", 30, || {
                slide.read_region(0, 0, deepest, dims.0 as i64, dims.1 as i64)
            })
        };
        workloads.push(thumbnail);
    }

    let run = BenchRun::new(
        "openslide",
        slide_path_json,
        host,
        selected_workload,
        repeat_index,
        workloads,
    );
    println!("{}", run.to_json());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparable_level_dims_uses_bounds_canvas() {
        let dims = [(1000, 2000), (250, 500), (125, 250)];
        let bounds = SlideBounds {
            x: 100,
            y: 200,
            width: 400,
            height: 800,
        };

        assert_eq!(
            comparable_level_dims(&dims, Some(bounds)),
            vec![(400, 800), (100, 200), (50, 100)]
        );
    }
}
