use crate::core::registry::SlideReader;
use crate::core::types::{
    CpuTile, Dataset, Level, OutputBackendRequest, TileCodecKind, TileLayout, TileOutputPreference,
    TilePixels, TileRequest,
};
use crate::error::WsiError;
use rayon::ThreadPool;
use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const DEFAULT_ROUTE_SAMPLE_SIZE: usize = 4;
const DEVICE_WIN_RATIO: f64 = 0.85;

thread_local! {
    static CURRENT_DECODE_RUNTIME: RefCell<Option<Arc<DecodeRuntime>>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeExecutionOptions {
    jp2k_cpu_threads: Option<NonZeroUsize>,
    route_sample_size: usize,
}

impl DecodeExecutionOptions {
    pub fn with_jp2k_cpu_threads(mut self, threads: NonZeroUsize) -> Self {
        self.jp2k_cpu_threads = Some(threads);
        self
    }

    pub fn with_route_sample_size(mut self, sample_size: usize) -> Self {
        self.route_sample_size = sample_size.max(1);
        self
    }

    pub fn jp2k_cpu_threads(&self) -> Option<NonZeroUsize> {
        self.jp2k_cpu_threads
    }

    pub fn route_sample_size(&self) -> usize {
        self.route_sample_size
    }
}

impl Default for DecodeExecutionOptions {
    fn default() -> Self {
        Self {
            jp2k_cpu_threads: None,
            route_sample_size: DEFAULT_ROUTE_SAMPLE_SIZE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRoute {
    Cpu,
    Device,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeRouteDecision {
    pub winner: DecodeRoute,
    pub sample_tile_count: usize,
    pub cpu_elapsed: Duration,
    pub device_elapsed: Duration,
    pub device_tile_count: usize,
}

impl DecodeRouteDecision {
    pub fn measured(
        sample_tile_count: usize,
        cpu_elapsed: Duration,
        device_elapsed: Duration,
        device_tile_count: usize,
    ) -> Self {
        Self {
            winner: Self::winner_for_measurement(cpu_elapsed, device_elapsed, device_tile_count),
            sample_tile_count,
            cpu_elapsed,
            device_elapsed,
            device_tile_count,
        }
    }

    pub fn winner_for_measurement(
        cpu_elapsed: Duration,
        device_elapsed: Duration,
        device_tile_count: usize,
    ) -> DecodeRoute {
        let cpu_ms = cpu_elapsed.as_secs_f64() * 1000.0;
        let device_ms = device_elapsed.as_secs_f64() * 1000.0;
        if device_tile_count > 0 && cpu_ms > 0.0 && device_ms <= cpu_ms * DEVICE_WIN_RATIO {
            DecodeRoute::Device
        } else {
            DecodeRoute::Cpu
        }
    }
}

#[derive(Debug)]
pub(crate) struct DecodeRuntime {
    options: DecodeExecutionOptions,
    jp2k_cpu_pool: ThreadPool,
    route_cache: Mutex<HashMap<DecodeRouteKey, DecodeRouteDecision>>,
}

impl DecodeRuntime {
    pub(crate) fn new(options: DecodeExecutionOptions) -> Result<Self, WsiError> {
        let threads = options
            .jp2k_cpu_threads
            .map_or_else(default_jp2k_cpu_threads, NonZeroUsize::get);
        let jp2k_cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("statumen-jp2k-cpu-{index}"))
            .build()
            .map_err(|err| WsiError::Unsupported {
                reason: format!("failed to initialize JP2K CPU decode pool: {err}"),
            })?;
        Ok(Self {
            options,
            jp2k_cpu_pool,
            route_cache: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn default_arc() -> Arc<Self> {
        static DEFAULT_RUNTIME: OnceLock<Arc<DecodeRuntime>> = OnceLock::new();
        DEFAULT_RUNTIME
            .get_or_init(|| {
                Arc::new(
                    Self::new(DecodeExecutionOptions::default()).expect("default decode runtime"),
                )
            })
            .clone()
    }

    pub(crate) fn jp2k_cpu_pool(&self) -> &ThreadPool {
        &self.jp2k_cpu_pool
    }

    pub(crate) fn options(&self) -> DecodeExecutionOptions {
        self.options
    }

    pub(crate) fn with_current<T>(self: &Arc<Self>, f: impl FnOnce() -> T) -> T {
        struct Restore(Option<Arc<DecodeRuntime>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                let previous = self.0.take();
                CURRENT_DECODE_RUNTIME.with(|slot| {
                    *slot.borrow_mut() = previous;
                });
            }
        }

        let previous = CURRENT_DECODE_RUNTIME.with(|slot| slot.replace(Some(self.clone())));
        let _restore = Restore(previous);
        f()
    }

    fn cached_route(&self, key: &DecodeRouteKey) -> Option<DecodeRouteDecision> {
        self.route_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(key)
            .cloned()
    }

    fn store_route(&self, key: DecodeRouteKey, decision: DecodeRouteDecision) {
        self.route_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(key, decision);
    }
}

pub(crate) fn current_decode_runtime() -> Option<Arc<DecodeRuntime>> {
    CURRENT_DECODE_RUNTIME.with(|slot| slot.borrow().clone())
}

fn default_jp2k_cpu_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .max(1)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DecodeRouteKey {
    dataset_id: u128,
    scene: usize,
    series: usize,
    level: u32,
    tile_grid: RouteTileGrid,
    codec_kind: TileCodecKind,
    output_backend: OutputBackendRequest,
    device_backend_identity: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RouteTileGrid {
    tile_width: u32,
    tile_height: u32,
    tiles_across: u64,
    tiles_down: u64,
}

pub(crate) struct AdaptiveDecodeReader {
    inner: Box<dyn SlideReader>,
    runtime: Arc<DecodeRuntime>,
}

impl AdaptiveDecodeReader {
    pub(crate) fn new(inner: Box<dyn SlideReader>, runtime: Arc<DecodeRuntime>) -> Self {
        Self { inner, runtime }
    }

    fn read_tiles_adaptive(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        if !should_adapt_output(&output) {
            return self
                .runtime
                .with_current(|| self.inner.read_tiles(reqs, output));
        }
        let Some(key) = route_key_for_batch(self.inner.as_ref(), reqs, &output) else {
            return self
                .runtime
                .with_current(|| self.inner.read_tiles(reqs, output));
        };
        let route = match self.runtime.cached_route(&key) {
            Some(decision) => decision.winner,
            None => {
                let decision = self.measure_route(reqs, output.clone())?;
                let winner = decision.winner;
                self.runtime.store_route(key, decision);
                winner
            }
        };
        let routed_output = match route {
            DecodeRoute::Cpu => TileOutputPreference::cpu(),
            DecodeRoute::Device => output,
        };
        self.runtime
            .with_current(|| self.inner.read_tiles(reqs, routed_output))
    }

    fn measure_route(
        &self,
        reqs: &[TileRequest],
        device_output: TileOutputPreference,
    ) -> Result<DecodeRouteDecision, WsiError> {
        let sample_len = reqs.len().min(self.runtime.options.route_sample_size());
        let sample = &reqs[..sample_len];

        let device_started = Instant::now();
        let device_result = self
            .runtime
            .with_current(|| self.inner.read_tiles(sample, device_output));
        let device_elapsed = device_started.elapsed();
        let device_tile_count = device_result
            .as_ref()
            .map(|tiles| {
                tiles
                    .iter()
                    .filter(|tile| matches!(tile, TilePixels::Device(_)))
                    .count()
            })
            .unwrap_or(0);

        let cpu_started = Instant::now();
        let cpu_tiles = self
            .runtime
            .with_current(|| self.inner.read_tiles(sample, TileOutputPreference::cpu()))?;
        let cpu_elapsed = cpu_started.elapsed();

        Ok(DecodeRouteDecision::measured(
            cpu_tiles.len(),
            cpu_elapsed,
            device_elapsed,
            device_tile_count,
        ))
    }
}

impl SlideReader for AdaptiveDecodeReader {
    fn dataset(&self) -> &Dataset {
        self.inner.dataset()
    }

    fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        self.inner.tile_codec_kind(req)
    }

    fn level_source_kind(
        &self,
        scene: usize,
        series: usize,
        level: u32,
    ) -> Result<crate::core::types::LevelSourceKind, WsiError> {
        self.inner.level_source_kind(scene, series, level)
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        self.read_tiles_adaptive(reqs, output)
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.runtime.with_current(|| self.inner.read_tile_cpu(req))
    }

    fn read_raw_compressed_tile(
        &self,
        req: &TileRequest,
    ) -> Result<crate::core::types::RawCompressedTile, WsiError> {
        self.inner.read_raw_compressed_tile(req)
    }

    fn read_tiles_cpu(&self, reqs: &[TileRequest]) -> Result<Vec<CpuTile>, WsiError> {
        self.runtime
            .with_current(|| self.inner.read_tiles_cpu(reqs))
    }

    fn use_display_tile_cache(&self, req: &crate::core::types::TileViewRequest) -> bool {
        self.inner.use_display_tile_cache(req)
    }

    fn read_region_fastpath(
        &self,
        ctx: &mut crate::core::registry::SlideReadContext<'_>,
        req: &crate::core::types::RegionRequest,
    ) -> Option<Result<CpuTile, WsiError>> {
        self.runtime
            .with_current(|| self.inner.read_region_fastpath(ctx, req))
    }

    fn read_region(
        &self,
        req: &crate::core::types::RegionRequest,
        output: TileOutputPreference,
    ) -> Result<TilePixels, WsiError> {
        self.runtime
            .with_current(|| self.inner.read_region(req, output))
    }

    fn read_display_tile(
        &self,
        req: &crate::core::types::TileViewRequest,
    ) -> Result<CpuTile, WsiError> {
        self.runtime
            .with_current(|| self.inner.read_display_tile(req))
    }

    fn associated_image(&self, name: &str) -> Result<Option<CpuTile>, WsiError> {
        self.inner.associated_image(name)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.inner.read_associated(name)
    }

    fn recommended_shared_cache_bytes(&self) -> Option<u64> {
        self.inner.recommended_shared_cache_bytes()
    }
}

fn should_adapt_output(output: &TileOutputPreference) -> bool {
    matches!(output, TileOutputPreference::PreferDevice { .. })
        && output.compressed_device_decode_enabled()
        && output.adaptive_decode_route_enabled()
}

fn route_key_for_batch(
    reader: &dyn SlideReader,
    reqs: &[TileRequest],
    output: &TileOutputPreference,
) -> Option<DecodeRouteKey> {
    let first = reqs.first()?;
    if !reqs.iter().all(|req| {
        req.scene == first.scene && req.series == first.series && req.level == first.level
    }) {
        return None;
    }
    let codec_kind = reader.tile_codec_kind(first);
    if !matches!(codec_kind, TileCodecKind::Jp2k | TileCodecKind::Htj2k) {
        return None;
    }
    if !reqs
        .iter()
        .all(|req| reader.tile_codec_kind(req) == codec_kind)
    {
        return None;
    }
    let level = dataset_level(reader.dataset(), first.scene, first.series, first.level)?;
    let tile_grid = route_tile_grid(level)?;
    Some(DecodeRouteKey {
        dataset_id: reader.dataset().id.0,
        scene: first.scene,
        series: first.series,
        level: first.level,
        tile_grid,
        codec_kind,
        output_backend: output.backend(),
        device_backend_identity: device_backend_identity(output),
    })
}

fn dataset_level(dataset: &Dataset, scene: usize, series: usize, level: u32) -> Option<&Level> {
    dataset
        .scenes
        .get(scene)?
        .series
        .get(series)?
        .levels
        .get(level as usize)
}

fn route_tile_grid(level: &Level) -> Option<RouteTileGrid> {
    match &level.tile_layout {
        TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } => Some(RouteTileGrid {
            tile_width: *tile_width,
            tile_height: *tile_height,
            tiles_across: *tiles_across,
            tiles_down: *tiles_down,
        }),
        _ => None,
    }
}

fn device_backend_identity(output: &TileOutputPreference) -> String {
    #[cfg(feature = "metal")]
    if let Some(metal) = output.metal_sessions() {
        return format!("{:?}:{}", output.backend(), metal.device_identity());
    }
    format!("{:?}", output.backend())
}
