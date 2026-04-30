//! Shared bench library for `wsi_bench` and `openslide_bench`.
//!
//! Included via `mod bench_common;` from each bin's main.rs. Each binary
//! gets its own compiled copy — that is fine, this module is small.

use std::env::VarError;
use std::time::Duration;
use std::time::Instant;

pub const SCHEMA_VERSION: u32 = 2;
pub const RUN_MODE_FULL_SUITE: &str = "full_suite";
pub const RUN_MODE_SINGLE_WORKLOAD: &str = "single_workload";
const SELECTED_WORKLOAD_ENV: &str = "WSI_BENCH_ONLY";
const REPEAT_INDEX_ENV: &str = "WSI_BENCH_REPEAT_INDEX";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Percentiles {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    pub mean_us: u64,
    pub n: usize,
}

impl Percentiles {
    pub fn from_durations(samples: &[Duration]) -> Self {
        assert!(!samples.is_empty(), "empty samples");
        let mut us: Vec<u64> = samples.iter().map(|d| d.as_micros() as u64).collect();
        us.sort_unstable();
        let n = us.len();
        let pick = |q: f64| -> u64 {
            // Nearest-rank, clamped. p50 of 100 samples → index 49 (1-based 50).
            let rank = (q * n as f64).ceil() as usize;
            let idx = rank.saturating_sub(1).min(n - 1);
            us[idx]
        };
        let sum: u64 = us.iter().sum();
        Self {
            p50_us: pick(0.50),
            p95_us: pick(0.95),
            p99_us: pick(0.99),
            max_us: us[n - 1],
            mean_us: sum / n as u64,
            n,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadSpec {
    pub name: &'static str,
    pub target_n: usize,
    pub gate_mode: &'static str,
    pub comparability: &'static str,
    pub comparability_note: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CenteredViewportRegion {
    pub level2_top_left: (i64, i64),
    pub level0_top_left: (i64, i64),
    pub side_px: u32,
}

const WORKLOAD_SPECS: [WorkloadSpec; 8] = [
    WorkloadSpec {
        name: "cold_open",
        target_n: 10,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: None,
    },
    WorkloadSpec {
        name: "single_tile_l0",
        target_n: 200,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: None,
    },
    WorkloadSpec {
        name: "pan_trace_l0",
        target_n: 256,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: None,
    },
    WorkloadSpec {
        name: "pan_trace_l2",
        target_n: 256,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: Some(
            "OpenSlide consumes the same accepted level-2 tile top-lefts mapped back into level-0 world coordinates",
        ),
    },
    WorkloadSpec {
        name: "pan_trace_l2_dense",
        target_n: 16,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: None,
    },
    WorkloadSpec {
        name: "region_2k",
        target_n: 30,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: None,
    },
    WorkloadSpec {
        name: "viewport_region_l2",
        target_n: 30,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: Some(
            "OpenSlide reads the same centered level-2 viewport mapped into level-0 world coordinates",
        ),
    },
    WorkloadSpec {
        name: "thumbnail",
        target_n: 30,
        gate_mode: "gating",
        comparability: "exact",
        comparability_note: None,
    },
];

pub fn workload_specs() -> &'static [WorkloadSpec] {
    &WORKLOAD_SPECS
}

pub fn workload_spec(name: &str) -> Option<&'static WorkloadSpec> {
    WORKLOAD_SPECS.iter().find(|spec| spec.name == name)
}

pub fn valid_workload_names() -> String {
    workload_specs()
        .iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn selected_workload() -> Result<Option<String>, String> {
    match std::env::var(SELECTED_WORKLOAD_ENV) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            if workload_spec(trimmed).is_none() {
                return Err(format!(
                    "invalid {} value {:?}; valid workloads: {}",
                    SELECTED_WORKLOAD_ENV,
                    trimmed,
                    valid_workload_names()
                ));
            }
            Ok(Some(trimmed.to_string()))
        }
        Err(VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("failed to read {}: {err}", SELECTED_WORKLOAD_ENV)),
    }
}

pub fn repeat_index() -> Result<Option<u32>, String> {
    match std::env::var(REPEAT_INDEX_ENV) {
        Ok(value) => value
            .parse::<u32>()
            .map(Some)
            .map_err(|err| format!("invalid {} value {:?}: {err}", REPEAT_INDEX_ENV, value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("failed to read {}: {err}", REPEAT_INDEX_ENV)),
    }
}

pub fn run_mode(selected_workload: Option<&str>) -> &'static str {
    match selected_workload {
        Some(_) => RUN_MODE_SINGLE_WORKLOAD,
        None => RUN_MODE_FULL_SUITE,
    }
}

pub fn should_run_workload(selected_workload: Option<&str>, workload_name: &str) -> bool {
    match selected_workload {
        Some(selected) => selected == workload_name,
        None => true,
    }
}

#[derive(Debug, Clone)]
pub struct WorkloadResult {
    pub name: String,
    pub target_n: usize,
    pub gate_mode: &'static str,
    pub comparability: &'static str,
    pub comparability_note: Option<&'static str>,
    pub samples: Vec<Duration>,
    pub error: Option<String>,
}

impl WorkloadResult {
    pub fn new(name: &str) -> Self {
        let spec = workload_spec(name);
        Self {
            name: name.to_string(),
            target_n: spec.map_or(0, |spec| spec.target_n),
            gate_mode: spec.map_or("gating", |spec| spec.gate_mode),
            comparability: spec.map_or("exact", |spec| spec.comparability),
            comparability_note: spec.and_then(|spec| spec.comparability_note),
            samples: Vec::new(),
            error: None,
        }
    }

    pub fn with_error(name: &str, error: impl Into<String>) -> Self {
        let mut result = Self::new(name);
        result.error = Some(error.into());
        result
    }
}

#[derive(Debug, Clone)]
pub struct BenchRun {
    pub schema_version: u32,
    pub library: String, // "ziggurat" or "openslide"
    pub slide_path: String,
    pub host: String, // uname -a output
    pub run_mode: &'static str,
    pub selected_workload: Option<String>,
    pub repeat_index: Option<u32>,
    pub peak_rss_bytes: Option<u64>,
    pub rss_method: Option<String>,
    pub workloads: Vec<WorkloadResult>,
}

impl BenchRun {
    pub fn new(
        library: &str,
        slide_path: String,
        host: String,
        selected_workload: Option<String>,
        repeat_index: Option<u32>,
        workloads: Vec<WorkloadResult>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            library: library.to_string(),
            slide_path,
            host,
            run_mode: run_mode(selected_workload.as_deref()),
            selected_workload,
            repeat_index,
            peak_rss_bytes: None,
            rss_method: None,
            workloads,
        }
    }

    /// Render this run as a single JSON object string.
    /// Format is intentionally hand-rolled (no serde) to keep dev-deps light.
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"schema_version\": {},\n", self.schema_version));
        s.push_str(&format!("  \"library\": {},\n", json_str(&self.library)));
        s.push_str(&format!(
            "  \"slide_path\": {},\n",
            json_str(&self.slide_path)
        ));
        s.push_str(&format!("  \"host\": {},\n", json_str(&self.host)));
        s.push_str(&format!("  \"run_mode\": {},\n", json_str(self.run_mode)));
        s.push_str(&format!(
            "  \"selected_workload\": {},\n",
            self.selected_workload
                .as_deref()
                .map(json_str)
                .unwrap_or_else(|| "null".into())
        ));
        s.push_str(&format!(
            "  \"repeat_index\": {},\n",
            self.repeat_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "null".into())
        ));
        s.push_str(&format!(
            "  \"peak_rss_bytes\": {},\n",
            self.peak_rss_bytes
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "null".into())
        ));
        s.push_str(&format!(
            "  \"rss_method\": {},\n",
            self.rss_method
                .as_deref()
                .map(json_str)
                .unwrap_or_else(|| "null".into())
        ));
        s.push_str("  \"workloads\": [\n");
        for (i, w) in self.workloads.iter().enumerate() {
            let p = if w.samples.is_empty() {
                None
            } else {
                Some(Percentiles::from_durations(&w.samples))
            };
            s.push_str("    {\n");
            s.push_str(&format!("      \"name\": {},\n", json_str(&w.name)));
            s.push_str(&format!("      \"target_n\": {},\n", w.target_n));
            s.push_str(&format!("      \"n\": {},\n", w.samples.len()));
            if let Some(p) = p {
                s.push_str(&format!("      \"p50_us\": {},\n", p.p50_us));
                s.push_str(&format!("      \"p95_us\": {},\n", p.p95_us));
                s.push_str(&format!("      \"p99_us\": {},\n", p.p99_us));
                s.push_str(&format!("      \"max_us\": {},\n", p.max_us));
                s.push_str(&format!("      \"mean_us\": {},\n", p.mean_us));
            }
            s.push_str(&format!(
                "      \"gate_mode\": {},\n",
                json_str(w.gate_mode)
            ));
            s.push_str(&format!(
                "      \"comparability\": {},\n",
                json_str(w.comparability)
            ));
            s.push_str(&format!(
                "      \"comparability_note\": {},\n",
                w.comparability_note
                    .map(json_str)
                    .unwrap_or_else(|| "null".into())
            ));
            let samples_us: Vec<String> = w
                .samples
                .iter()
                .map(|d| (d.as_micros() as u64).to_string())
                .collect();
            s.push_str(&format!(
                "      \"samples_us\": [{}],\n",
                samples_us.join(",")
            ));
            s.push_str(&format!(
                "      \"error\": {}\n",
                w.error
                    .as_deref()
                    .map(json_str)
                    .unwrap_or_else(|| "null".into())
            ));
            s.push_str(if i + 1 == self.workloads.len() {
                "    }\n"
            } else {
                "    },\n"
            });
        }
        s.push_str("  ]\n}\n");
        s
    }
}

/// Per-(level, slide) coordinates derived from the slide's actual dimensions.
/// Computed once at startup so the workload list is fixed and reproducible.
#[derive(Debug, Clone)]
pub struct WorkloadPlan {
    /// Tile size used by single_tile and pan_trace workloads.
    pub tile_px: u32,
    /// Center pixel at level 0, in level-0 coordinates.
    pub center_l0: (i64, i64),
    /// Total levels in the pyramid.
    pub level_count: u32,
    /// Level-0 dimensions in pixels.
    pub level0_dims: (u64, u64),
    /// Level-2 dimensions in pixels (or the deepest level if <3 exist).
    pub level2_idx: u32,
    pub level2_dims: (u64, u64),
    /// Pan trace step in tiles (how far each pan_trace step advances).
    pub pan_step_tiles: u32,
    /// Number of pan_trace steps.
    pub pan_steps: u32,
}

impl WorkloadPlan {
    /// Compute the plan from level dimensions only — no library calls.
    /// Both wsi_bench and openslide_bench feed this the same numbers so the
    /// trace is identical between libraries.
    pub fn compute(level_dims: &[(u64, u64)]) -> Self {
        assert!(!level_dims.is_empty(), "no levels");
        let level_count = level_dims.len() as u32;
        let level0_dims = level_dims[0];
        let level2_idx = (2u32).min(level_count - 1);
        let level2_dims = level_dims[level2_idx as usize];

        let tile_px: u32 = 256;
        let center_l0 = ((level0_dims.0 / 2) as i64, (level0_dims.1 / 2) as i64);

        Self {
            tile_px,
            center_l0,
            level_count,
            level0_dims,
            level2_idx,
            level2_dims,
            pan_step_tiles: 1,
            pan_steps: 256,
        }
    }

    /// Returns a centered square viewport on level 2, clamped to the actual
    /// level dimensions and converted back into level-0 world coordinates for
    /// OpenSlide.
    pub fn centered_viewport_l2(&self, desired_side_px: u32) -> CenteredViewportRegion {
        let side_px = desired_side_px
            .min(self.level2_dims.0 as u32)
            .min(self.level2_dims.1 as u32)
            .max(1);
        let x_l2 = ((self.level2_dims.0 as i64 - side_px as i64) / 2).max(0);
        let y_l2 = ((self.level2_dims.1 as i64 - side_px as i64) / 2).max(0);
        let dx = self.level0_dims.0 as f64 / self.level2_dims.0 as f64;
        let dy = self.level0_dims.1 as f64 / self.level2_dims.1 as f64;
        CenteredViewportRegion {
            level2_top_left: (x_l2, y_l2),
            level0_top_left: ((x_l2 as f64 * dx) as i64, (y_l2 as f64 * dy) as i64),
            side_px,
        }
    }

    /// Returns up to `pan_steps` in-bounds top-left tile coordinates along a
    /// diagonal centered on the slide, **at level 0**, in level-0 pixel units.
    /// Coordinates are filtered so each tile of size `tile_px × tile_px` fits
    /// entirely inside the level-0 image.
    ///
    /// On large slides this returns all `pan_steps` coordinates. On small
    /// slides it returns however many fit (the trace simply has fewer tiles).
    pub fn pan_trace_l0(&self) -> Vec<(i64, i64)> {
        let tile_px = self.tile_px as i64;
        let w = self.level0_dims.0 as i64;
        let h = self.level0_dims.1 as i64;
        self.diagonal_coords(self.center_l0)
            .into_iter()
            .filter(|&(x, y)| x >= 0 && y >= 0 && x + tile_px <= w && y + tile_px <= h)
            .collect()
    }

    /// Returns the **level-2** in-bounds coordinates derived from the SAME
    /// level-0 trace by downsampling, then filtered against level-2 bounds.
    /// The downsampling preserves the trace shape across levels so the ziggurat
    /// and openslide bench binaries always read the same regions.
    pub fn pan_trace_l2(&self) -> Vec<(i64, i64)> {
        let tile_px = self.tile_px as i64;
        let w = self.level2_dims.0 as i64;
        let h = self.level2_dims.1 as i64;
        let dx = self.level0_dims.0 as f64 / self.level2_dims.0 as f64;
        let dy = self.level0_dims.1 as f64 / self.level2_dims.1 as f64;
        self.diagonal_coords(self.center_l0)
            .into_iter()
            .map(|(x_l0, y_l0)| ((x_l0 as f64 / dx) as i64, (y_l0 as f64 / dy) as i64))
            .filter(|&(x, y)| x >= 0 && y >= 0 && x + tile_px <= w && y + tile_px <= h)
            .collect()
    }

    /// Returns the same accepted level-2 trace as `pan_trace_l2`, but mapped
    /// back into level-0 world-space top-lefts for OpenSlide's `read_region`.
    pub fn pan_trace_l2_world_l0(&self) -> Vec<(i64, i64)> {
        let dx = self.level0_dims.0 as f64 / self.level2_dims.0 as f64;
        let dy = self.level0_dims.1 as f64 / self.level2_dims.1 as f64;
        self.pan_trace_l2()
            .into_iter()
            .map(|(x_l2, y_l2)| ((x_l2 as f64 * dx) as i64, (y_l2 as f64 * dy) as i64))
            .collect()
    }

    /// Returns a dense 4x4 tile cluster centered near the level-2 center tile.
    /// Used as a regression guard for bounded look-ahead behavior.
    pub fn pan_trace_l2_dense(&self) -> Vec<(i64, i64)> {
        let w = self.level2_dims.0 as i64;
        let h = self.level2_dims.1 as i64;
        let dx = self.level0_dims.0 as f64 / self.level2_dims.0 as f64;
        let dy = self.level0_dims.1 as f64 / self.level2_dims.1 as f64;
        let (cx_l2, cy_l2) = (
            (self.center_l0.0 as f64 / dx) as i64,
            (self.center_l0.1 as f64 / dy) as i64,
        );
        let base_col = cx_l2.div_euclid(self.tile_px as i64).saturating_sub(1);
        let base_row = cy_l2.div_euclid(self.tile_px as i64).saturating_sub(1);
        let mut coords = Vec::new();
        for row in 0..4 {
            for col in 0..4 {
                let x = (base_col + col) * self.tile_px as i64;
                let y = (base_row + row) * self.tile_px as i64;
                if x >= 0 && y >= 0 && x + self.tile_px as i64 <= w && y + self.tile_px as i64 <= h
                {
                    coords.push((x, y));
                }
            }
        }
        coords
    }

    /// Returns the same dense cluster as `pan_trace_l2_dense`, but converted
    /// into level-0 world-space top-lefts for OpenSlide's `read_region`.
    pub fn pan_trace_l2_dense_world_l0(&self) -> Vec<(i64, i64)> {
        let dx = self.level0_dims.0 as f64 / self.level2_dims.0 as f64;
        let dy = self.level0_dims.1 as f64 / self.level2_dims.1 as f64;
        self.pan_trace_l2_dense()
            .into_iter()
            .map(|(x_l2, y_l2)| ((x_l2 as f64 * dx) as i64, (y_l2 as f64 * dy) as i64))
            .collect()
    }

    /// Internal: generate the unfiltered diagonal trace from a center point.
    /// Used by both pan_trace_l0 and pan_trace_l2 (after downsampling).
    fn diagonal_coords(&self, center: (i64, i64)) -> Vec<(i64, i64)> {
        let step_l0 = (self.tile_px as i64) * (self.pan_step_tiles as i64);
        let half = self.pan_steps as i64 / 2;
        (0..self.pan_steps as i64)
            .map(|i| {
                let dx = (i - half) * step_l0;
                let dy = (i - half) * step_l0;
                (center.0 + dx, center.1 + dy)
            })
            .collect()
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    #[test]
    fn plan_computes_levels() {
        let dims = [(40000, 30000), (20000, 15000), (10000, 7500), (5000, 3750)];
        let p = WorkloadPlan::compute(&dims);
        assert_eq!(p.level_count, 4);
        assert_eq!(p.level0_dims, (40000, 30000));
        assert_eq!(p.level2_idx, 2);
        assert_eq!(p.level2_dims, (10000, 7500));
        assert_eq!(p.center_l0, (20000, 15000));
    }

    #[test]
    fn plan_handles_short_pyramid() {
        let dims = [(100, 100), (50, 50)];
        let p = WorkloadPlan::compute(&dims);
        assert_eq!(p.level2_idx, 1); // clamped to deepest
        assert_eq!(p.level2_dims, (50, 50));
    }

    #[test]
    fn pan_trace_has_expected_step_count_and_center() {
        let dims = [(100_000, 100_000)];
        let p = WorkloadPlan::compute(&dims);
        let coords = p.pan_trace_l0();
        assert_eq!(coords.len(), 256);
        let mid = coords[128];
        assert_eq!(mid, (50_000, 50_000));
    }

    #[test]
    fn pan_trace_filters_out_of_bounds_on_small_slides() {
        // 31744 x 64256 — the user's smallest NDPI. Diagonal extends ±32768
        // px from center, so X dimension cannot fit the whole trace.
        let dims = [(31744, 64256), (15872, 32128), (7936, 16064)];
        let p = WorkloadPlan::compute(&dims);
        let l0 = p.pan_trace_l0();
        let l2 = p.pan_trace_l2();
        // Both should be non-empty (slide is large enough for SOME trace)
        // and strictly less than the planned 256.
        assert!(
            !l0.is_empty(),
            "pan_trace_l0 should have some in-bounds tiles"
        );
        assert!(
            l0.len() < 256,
            "pan_trace_l0 should be filtered down from 256"
        );
        assert!(
            !l2.is_empty(),
            "pan_trace_l2 should have some in-bounds tiles"
        );
        // All returned coordinates must be in bounds for their level.
        let tile_px = p.tile_px as i64;
        for &(x, y) in &l0 {
            assert!(x >= 0 && y >= 0, "l0 coord ({x},{y}) below zero");
            assert!(
                x + tile_px <= p.level0_dims.0 as i64,
                "l0 coord ({x},{y}) past right edge"
            );
            assert!(
                y + tile_px <= p.level0_dims.1 as i64,
                "l0 coord ({x},{y}) past bottom edge"
            );
        }
        for &(x, y) in &l2 {
            assert!(x >= 0 && y >= 0, "l2 coord ({x},{y}) below zero");
            assert!(
                x + tile_px <= p.level2_dims.0 as i64,
                "l2 coord past l2 right edge"
            );
            assert!(
                y + tile_px <= p.level2_dims.1 as i64,
                "l2 coord past l2 bottom edge"
            );
        }
    }

    #[test]
    fn pan_trace_l2_dense_returns_a_dense_4x4_cluster_near_center() {
        let dims = [(8192, 8192), (4096, 4096), (2048, 2048)];
        let p = WorkloadPlan::compute(&dims);

        let coords = p.pan_trace_l2_dense();

        assert_eq!(coords.len(), 16);
        assert_eq!(
            coords,
            vec![
                (768, 768),
                (1024, 768),
                (1280, 768),
                (1536, 768),
                (768, 1024),
                (1024, 1024),
                (1280, 1024),
                (1536, 1024),
                (768, 1280),
                (1024, 1280),
                (1280, 1280),
                (1536, 1280),
                (768, 1536),
                (1024, 1536),
                (1280, 1536),
                (1536, 1536),
            ]
        );
        assert!(coords.iter().all(|&(x, y)| x >= 0
            && y >= 0
            && x + p.tile_px as i64 <= p.level2_dims.0 as i64
            && y + p.tile_px as i64 <= p.level2_dims.1 as i64));
    }

    #[test]
    fn pan_trace_l2_dense_world_coords_map_to_level0_for_openslide() {
        let dims = [(8192, 8192), (4096, 4096), (2048, 2048)];
        let p = WorkloadPlan::compute(&dims);

        let coords = p.pan_trace_l2_dense_world_l0();

        assert_eq!(
            coords,
            vec![
                (3072, 3072),
                (4096, 3072),
                (5120, 3072),
                (6144, 3072),
                (3072, 4096),
                (4096, 4096),
                (5120, 4096),
                (6144, 4096),
                (3072, 5120),
                (4096, 5120),
                (5120, 5120),
                (6144, 5120),
                (3072, 6144),
                (4096, 6144),
                (5120, 6144),
                (6144, 6144),
            ]
        );
    }

    #[test]
    fn pan_trace_l2_world_coords_map_back_to_level0() {
        let dims = [(8192, 8192), (4096, 4096), (2048, 2048)];
        let p = WorkloadPlan::compute(&dims);

        let l2 = p.pan_trace_l2();
        let world = p.pan_trace_l2_world_l0();

        assert_eq!(l2.len(), world.len());
        for ((x_l2, y_l2), (x_l0, y_l0)) in l2.into_iter().zip(world.into_iter()) {
            assert_eq!(x_l0, x_l2 * 4);
            assert_eq!(y_l0, y_l2 * 4);
        }
    }

    #[test]
    fn centered_viewport_l2_returns_centered_square_region() {
        let dims = [(8192, 8192), (4096, 4096), (2048, 2048)];
        let p = WorkloadPlan::compute(&dims);

        let viewport = p.centered_viewport_l2(1024);

        assert_eq!(viewport.side_px, 1024);
        assert_eq!(viewport.level2_top_left, (512, 512));
        assert_eq!(viewport.level0_top_left, (2048, 2048));
    }

    #[test]
    fn centered_viewport_l2_clamps_to_small_levels() {
        let dims = [(900, 600), (450, 300), (225, 150)];
        let p = WorkloadPlan::compute(&dims);

        let viewport = p.centered_viewport_l2(1024);

        assert_eq!(viewport.side_px, 150);
        assert_eq!(viewport.level2_top_left, (37, 0));
        assert_eq!(viewport.level0_top_left, (148, 0));
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn tile_top_left_at_pixel(pixel_xy: (i64, i64), tile_px: u32) -> (i64, i64) {
    let col = pixel_xy.0.div_euclid(tile_px as i64);
    let row = pixel_xy.1.div_euclid(tile_px as i64);
    (col * tile_px as i64, row * tile_px as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_basic() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_micros).collect();
        let p = Percentiles::from_durations(&samples);
        assert_eq!(p.n, 100);
        assert_eq!(p.p50_us, 50);
        assert_eq!(p.p95_us, 95);
        assert_eq!(p.p99_us, 99);
        assert_eq!(p.max_us, 100);
        assert_eq!(p.mean_us, 50); // (1+...+100)/100 = 50.5 → 50
    }

    #[test]
    fn percentiles_handles_unsorted_input() {
        let samples: Vec<Duration> = [50, 10, 99, 1, 100, 95]
            .iter()
            .map(|&n| Duration::from_micros(n))
            .collect();
        let p = Percentiles::from_durations(&samples);
        assert_eq!(p.max_us, 100);
        assert_eq!(p.p99_us, 100);
    }

    #[test]
    #[should_panic(expected = "empty samples")]
    fn percentiles_panics_on_empty() {
        Percentiles::from_durations(&[]);
    }

    #[test]
    fn json_round_trips_through_serde_json() {
        let mut run = BenchRun::new(
            "ziggurat",
            "/tmp/x.svs".into(),
            "test-host".into(),
            Some("single_tile_l0".into()),
            Some(3),
            vec![
                {
                    let mut result = WorkloadResult::new("single_tile_l0");
                    result.samples = vec![Duration::from_micros(1200), Duration::from_micros(1500)];
                    result
                },
                WorkloadResult::with_error("pan_trace_l2", "file not found"),
            ],
        );
        run.peak_rss_bytes = Some(1234);
        run.rss_method = Some("macos:/usr/bin/time -l".into());
        let json = run.to_json();
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["schema_version"], SCHEMA_VERSION);
        assert_eq!(value["run_mode"], RUN_MODE_SINGLE_WORKLOAD);
        assert_eq!(value["selected_workload"], "single_tile_l0");
        assert_eq!(value["repeat_index"], 3);
        assert_eq!(value["peak_rss_bytes"], 1234);
        assert_eq!(value["rss_method"], "macos:/usr/bin/time -l");
        assert_eq!(value["workloads"][0]["target_n"], 200);
        assert_eq!(value["workloads"][1]["gate_mode"], "gating");
        assert_eq!(value["workloads"][1]["comparability"], "exact");
    }

    #[test]
    fn pan_trace_l2_spec_is_gating_and_exact() {
        let spec = workload_spec("pan_trace_l2").expect("pan_trace_l2 spec");
        assert_eq!(spec.target_n, 256);
        assert_eq!(spec.gate_mode, "gating");
        assert_eq!(spec.comparability, "exact");
        assert!(spec.comparability_note.is_some());
    }

    #[test]
    fn viewport_region_l2_spec_is_gating_and_exact() {
        let spec = workload_spec("viewport_region_l2").expect("viewport_region_l2 spec");
        assert_eq!(spec.target_n, 30);
        assert_eq!(spec.gate_mode, "gating");
        assert_eq!(spec.comparability, "exact");
        assert!(spec.comparability_note.is_some());
    }

    #[test]
    fn tile_top_left_snaps_to_containing_tile() {
        assert_eq!(tile_top_left_at_pixel((513, 700), 256), (512, 512));
        assert_eq!(tile_top_left_at_pixel((0, 0), 256), (0, 0));
    }
}

/// Returns a one-line host description for the bench JSON. Uses `uname -a`
/// when available, otherwise just the OS name.
pub fn host_string() -> String {
    if let Ok(out) = std::process::Command::new("uname").arg("-a").output() {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    std::env::consts::OS.to_string()
}

/// Times a single call. Discards the success value (we only need the duration);
/// errors propagate so callers can record them in WorkloadResult.error.
pub fn time_call<T, E>(f: impl FnOnce() -> Result<T, E>) -> Result<Duration, String>
where
    E: std::fmt::Display,
{
    let start = Instant::now();
    match f() {
        Ok(_) => Ok(start.elapsed()),
        Err(e) => Err(e.to_string()),
    }
}

/// Repeatedly times `f`, capturing samples until either `n` successful calls
/// have been collected or the first error occurs (workload aborts on first
/// failure to keep the report comparable across libraries).
pub fn run_workload<T, E>(
    name: &str,
    n: usize,
    mut f: impl FnMut() -> Result<T, E>,
) -> WorkloadResult
where
    E: std::fmt::Display,
{
    let mut result = WorkloadResult::new(name);
    result.samples = Vec::with_capacity(n);
    for _ in 0..n {
        match time_call(&mut f) {
            Ok(d) => result.samples.push(d),
            Err(msg) => {
                result.error = Some(msg);
                return result;
            }
        }
    }
    result
}

#[cfg(test)]
mod runner_tests {
    use super::*;

    #[test]
    fn run_workload_collects_n_samples_when_no_error() {
        let mut count = 0;
        let result = run_workload::<(), &str>("noop", 5, || {
            count += 1;
            Ok(())
        });
        assert_eq!(result.samples.len(), 5);
        assert!(result.error.is_none());
        assert_eq!(count, 5);
    }

    #[test]
    fn run_workload_aborts_on_first_error_and_records_message() {
        let mut count = 0;
        let result = run_workload::<(), String>("flaky", 5, || {
            count += 1;
            if count == 3 {
                Err("boom".to_string())
            } else {
                Ok(())
            }
        });
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.error.as_deref(), Some("boom"));
    }
}
