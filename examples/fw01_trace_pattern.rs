use std::path::PathBuf;

use statumen::{Slide, TileLayout};

const NDPI_BAND_HEIGHT: u32 = 256;
const TILE_PX: i64 = 256;
const SAMPLE_COUNT: usize = 32;
const PAN_STEPS: i64 = 256;

#[derive(Debug, Clone)]
struct WorkloadPlan {
    center_l0: (i64, i64),
    level0_dims: (u64, u64),
    level2_dims: (u64, u64),
}

impl WorkloadPlan {
    fn compute(level_dims: &[(u64, u64)]) -> Self {
        assert!(!level_dims.is_empty(), "no levels");
        let level0_dims = level_dims[0];
        let level2_dims = level_dims[(2usize).min(level_dims.len() - 1)];
        Self {
            center_l0: ((level0_dims.0 / 2) as i64, (level0_dims.1 / 2) as i64),
            level0_dims,
            level2_dims,
        }
    }

    fn pan_trace_l2(&self) -> Vec<(i64, i64)> {
        let w = self.level2_dims.0 as i64;
        let h = self.level2_dims.1 as i64;
        let dx = self.level0_dims.0 as f64 / self.level2_dims.0 as f64;
        let dy = self.level0_dims.1 as f64 / self.level2_dims.1 as f64;
        self.diagonal_coords(self.center_l0)
            .into_iter()
            .map(|(x_l0, y_l0)| ((x_l0 as f64 / dx) as i64, (y_l0 as f64 / dy) as i64))
            .filter(|&(x, y)| x >= 0 && y >= 0 && x + TILE_PX <= w && y + TILE_PX <= h)
            .collect()
    }

    fn diagonal_coords(&self, center: (i64, i64)) -> Vec<(i64, i64)> {
        let half = PAN_STEPS / 2;
        (0..PAN_STEPS)
            .map(|i| {
                let delta = (i - half) * TILE_PX;
                (center.0 + delta, center.1 + delta)
            })
            .collect()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: fw01_trace_pattern <slide-path>");
        std::process::exit(2);
    }

    let slide_path = PathBuf::from(&args[1]);
    let handle = Slide::open(&slide_path).unwrap_or_else(|err| {
        eprintln!("failed to open {}: {err}", slide_path.display());
        std::process::exit(1);
    });
    let series = &handle.dataset().scenes[0].series[0];
    let level_dims: Vec<(u64, u64)> = series.levels.iter().map(|l| l.dimensions).collect();
    let plan = WorkloadPlan::compute(&level_dims);
    let level = &series.levels[(2usize).min(series.levels.len() - 1)];
    let TileLayout::WholeLevel {
        virtual_tile_width,
        virtual_tile_height,
        ..
    } = level.tile_layout
    else {
        eprintln!("level 2 is not WholeLevel/NDPI");
        std::process::exit(1);
    };

    let strip_rows_per_band = NDPI_BAND_HEIGHT.div_ceil(virtual_tile_height).max(1);
    let trace = plan.pan_trace_l2();

    println!(
        "# slide={} level2_dims={}x{} virtual_tile={}x{} strip_rows_per_band={}",
        slide_path.display(),
        plan.level2_dims.0,
        plan.level2_dims.1,
        virtual_tile_width,
        virtual_tile_height,
        strip_rows_per_band
    );
    println!("idx\tx2\ty2\tcol\trow\tband_col\tband_row\tnew_tile\tnew_band");

    let mut prev_tile: Option<(i64, i64)> = None;
    let mut prev_band: Option<(i64, u32)> = None;
    for (idx, (x, y)) in trace.into_iter().take(SAMPLE_COUNT).enumerate() {
        let col = x.div_euclid(TILE_PX);
        let row = y.div_euclid(TILE_PX);
        let band_col = col;
        let band_row = (row as u32) / strip_rows_per_band;
        let new_tile = prev_tile != Some((col, row));
        let new_band = prev_band != Some((band_col, band_row));
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            idx,
            x,
            y,
            col,
            row,
            band_col,
            band_row,
            if new_tile { "yes" } else { "no" },
            if new_band { "yes" } else { "no" }
        );
        prev_tile = Some((col, row));
        prev_band = Some((band_col, band_row));
    }
}
