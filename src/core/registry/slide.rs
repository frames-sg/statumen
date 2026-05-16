use super::*;

// ── Slide ──────────────────────────────────────────────────

/// Top-level handle. Owns the SlideReader + shared cache.
pub struct Slide {
    source: Box<dyn SlideReader>,
    cache: Arc<TileCache>,
    display_cache: Arc<TileCache>,
    max_region_pixels: u64,
    decode_runtime: Arc<DecodeRuntime>,
}

impl std::fmt::Debug for Slide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slide")
            .field("dataset_id", &self.source.dataset().id)
            .finish()
    }
}

impl Slide {
    /// Construct from an already-opened source and cache.
    pub(crate) fn from_source(source: Box<dyn SlideReader>, cache: Arc<TileCache>) -> Self {
        let decode_runtime = DecodeRuntime::default_arc();
        Self {
            source: Box::new(AdaptiveDecodeReader::new(source, decode_runtime.clone())),
            cache,
            display_cache: Arc::new(TileCache::display_default()),
            max_region_pixels: DEFAULT_MAX_REGION_PIXELS,
            decode_runtime,
        }
    }

    pub(crate) fn from_source_with_config_and_runtime(
        source: Box<dyn SlideReader>,
        cache_config: CacheConfig,
        max_region_pixels: u64,
        decode_runtime: Arc<DecodeRuntime>,
    ) -> Self {
        let source_hint = source.recommended_shared_cache_bytes();
        Self {
            source: Box::new(AdaptiveDecodeReader::new(source, decode_runtime.clone())),
            cache: Arc::new(TileCache::shared_with_config(cache_config, source_hint)),
            display_cache: Arc::new(TileCache::display_with_config(cache_config)),
            max_region_pixels,
            decode_runtime,
        }
    }

    /// Construct from an already-opened source with an internal cache budget.
    pub fn from_source_with_cache_bytes(source: Box<dyn SlideReader>, cache_bytes: u64) -> Self {
        Self::from_source(source, Arc::new(TileCache::new(cache_bytes)))
    }

    /// Zero-config entry point: builtin registry + source-aware default cache.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WsiError> {
        Self::open_with_options(path, SlideOpenOptions::default())
    }

    pub fn open_with_options(
        path: impl AsRef<Path>,
        options: SlideOpenOptions,
    ) -> Result<Self, WsiError> {
        let resolved_path = crate::formats::svcache::resolve_open_path_with_policy(
            path.as_ref(),
            options.svcache_policy,
        )?;
        let source = options.registry.open(&resolved_path)?;
        let decode_runtime = Arc::new(DecodeRuntime::new(options.decode_execution_options)?);
        Ok(Self::from_source_with_config_and_runtime(
            source,
            options.cache_config,
            options.max_region_pixels,
            decode_runtime,
        ))
    }

    /// Open with the given registry and cache.
    ///
    /// Reusing the same [`TileCache`] across multiple handles allows decoded
    /// tiles from one handle to satisfy later reads from another handle that
    /// targets the same dataset and plane.
    pub(crate) fn open_with(
        path: impl AsRef<Path>,
        registry: &FormatRegistry,
        cache: Arc<TileCache>,
    ) -> Result<Self, WsiError> {
        let source = registry.open(path.as_ref())?;
        let mut slide = Self::from_source(source, cache);
        slide.max_region_pixels = DEFAULT_MAX_REGION_PIXELS;
        Ok(slide)
    }

    /// Open with the given registry and an internal cache budget.
    pub fn open_with_cache_bytes(
        path: impl AsRef<Path>,
        registry: &FormatRegistry,
        cache_bytes: u64,
    ) -> Result<Self, WsiError> {
        Self::open_with(path, registry, Arc::new(TileCache::new(cache_bytes)))
    }

    pub fn dataset(&self) -> &Dataset {
        self.source.dataset()
    }

    pub fn decode_execution_options(&self) -> DecodeExecutionOptions {
        self.decode_runtime.options()
    }

    pub fn level_source_kind(
        &self,
        scene: usize,
        series: usize,
        level: u32,
    ) -> Result<LevelSourceKind, WsiError> {
        self.source.level_source_kind(scene, series, level)
    }

    pub fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        self.source.tile_codec_kind(req)
    }

    pub fn cached_tile_present(&self, req: &TileRequest) -> bool {
        let key = CacheKey {
            dataset_id: self.dataset().id,
            scene: req.scene as u32,
            series: req.series as u32,
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
            tile_col: req.col,
            tile_row: req.row,
        };
        self.cache.get(&key).is_some()
    }

    pub fn source(&self) -> &dyn SlideReader {
        self.source.as_ref()
    }

    pub fn read_tile(
        &self,
        req: &TileRequest,
        output: TileOutputPreference,
    ) -> Result<TilePixels, WsiError> {
        let device_decode_attempted = matches!(
            output,
            TileOutputPreference::PreferDevice { .. } | TileOutputPreference::RequireDevice { .. }
        );
        let span = tracing::debug_span!(
            "wsi_read_tile",
            device_decode_attempted,
            fallback_to_cpu = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
            device_decoded_host_resident = tracing::field::Empty,
        );
        let _guard = span.enter();
        let result = self.source.read_tile(req, output);
        let mut fallback_to_cpu = false;
        let mut fallback_reason = "none";
        let device_decoded_host_resident = false;
        match &result {
            Ok(TilePixels::Cpu(_)) if device_decode_attempted => {
                fallback_to_cpu = true;
                fallback_reason = "signinum_auto_chose_cpu";
                span.record("fallback_to_cpu", true);
                span.record("fallback_reason", fallback_reason);
                span.record("device_decoded_host_resident", false);
            }
            Ok(TilePixels::Cpu(_)) => {
                span.record("fallback_to_cpu", false);
                span.record("fallback_reason", "none");
                span.record("device_decoded_host_resident", false);
            }
            Ok(TilePixels::Device(_)) => {
                span.record("fallback_to_cpu", false);
                span.record("fallback_reason", "none");
                span.record("device_decoded_host_resident", false);
            }
            Err(WsiError::Unsupported { .. }) if device_decode_attempted => {
                fallback_to_cpu = true;
                fallback_reason = "no_device_backend_for_codec";
                span.record("fallback_to_cpu", true);
                span.record("fallback_reason", fallback_reason);
                span.record("device_decoded_host_resident", false);
            }
            Err(_) => {
                span.record("fallback_to_cpu", false);
                span.record("fallback_reason", "none");
                span.record("device_decoded_host_resident", false);
            }
        }
        tracing::debug!(
            device_decode_attempted,
            fallback_to_cpu,
            fallback_reason,
            device_decoded_host_resident,
            "wsi tile output preference resolved"
        );
        result
    }

    pub fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        self.source.read_tiles(reqs, output)
    }

    pub fn read_raw_compressed_tile(
        &self,
        req: &TileRequest,
    ) -> Result<RawCompressedTile, WsiError> {
        self.source.read_raw_compressed_tile(req)
    }

    /// Read a pixel region, compositing from cached or freshly-decoded tiles.
    ///
    /// Validates all indices (scene, series, level, plane axes) before reading.
    /// Output buffer metadata (color_space, channels, sample_type, layout) is
    /// inherited from the first decoded tile -- no hardcoded assumptions.
    ///
    /// Only `CpuTileLayout::Interleaved` is supported for compositing. Planar
    /// tiles return `WsiError::DisplayConversion`.
    pub fn read_region(&self, req: &RegionRequest) -> Result<CpuTile, WsiError> {
        let mut ctx = SlideReadContext::new(
            Some(self.cache.as_ref()),
            TileOutputPreference::cpu(),
            self.max_region_pixels,
        );
        if let Some(result) = self.source.read_region_fastpath(&mut ctx, req) {
            return result;
        }
        composite_region_from_source(self.source.as_ref(), Some(self.cache.as_ref()), req)
    }

    pub fn read_display_tile(&self, req: &TileViewRequest) -> Result<CpuTile, WsiError> {
        // For Regular tile layouts, route through the generic composition path
        // with cache so intermediate tile reads are reused. For WholeLevel and
        // Irregular layouts, delegate to the source's override which may have
        // format-specific fast paths (e.g. NDPI MCU-level JPEG access).
        let is_regular = self
            .source
            .dataset()
            .scenes
            .get(req.scene)
            .and_then(|s| s.series.get(req.series))
            .and_then(|s| s.levels.get(req.level as usize))
            .is_some_and(|level| matches!(level.tile_layout, TileLayout::Regular { .. }));
        if is_regular {
            let display_cache = self
                .source
                .use_display_tile_cache(req)
                .then_some(self.display_cache.as_ref());
            read_display_tile_from_source(
                self.source.as_ref(),
                display_cache,
                req,
                TileOutputPreference::cpu(),
            )
        } else {
            self.source.read_display_tile(req)
        }
    }

    pub fn read_display_tile_with_output(
        &self,
        req: &TileViewRequest,
        output: TileOutputPreference,
    ) -> Result<CpuTile, WsiError> {
        let is_regular = self
            .source
            .dataset()
            .scenes
            .get(req.scene)
            .and_then(|s| s.series.get(req.series))
            .and_then(|s| s.levels.get(req.level as usize))
            .is_some_and(|level| matches!(level.tile_layout, TileLayout::Regular { .. }));
        if is_regular {
            let display_cache = self
                .source
                .use_display_tile_cache(req)
                .then_some(self.display_cache.as_ref());
            read_display_tile_from_source(self.source.as_ref(), display_cache, req, output)
        } else if matches!(output, TileOutputPreference::RequireDevice { .. }) {
            Err(WsiError::Unsupported {
                reason: "format-specific display tile fast paths return CPU pixels in Phase 2"
                    .into(),
            })
        } else {
            self.source.read_display_tile(req)
        }
    }

    /// Convenience: read a region and convert to RgbaImage.
    /// Only works for Uint8 data (brightfield). For Uint16/Float32,
    /// use read_region() + to_rgba_windowed() with an explicit DisplayWindow.
    pub fn read_region_rgba(&self, req: &RegionRequest) -> Result<image::RgbaImage, WsiError> {
        self.read_region(req)?.to_rgba()
    }

    /// Read a region and convert to RgbaImage with explicit windowing.
    /// For Uint16/Float32 data (fluorescence, computed images).
    pub fn read_region_rgba_windowed(
        &self,
        req: &RegionRequest,
        window: &DisplayWindow,
    ) -> Result<image::RgbaImage, WsiError> {
        self.read_region(req)?.to_rgba_windowed(window)
    }

    /// Read an associated image (label, macro, thumbnail).
    /// Direct delegation to the underlying SlideReader. No caching.
    pub fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.source.read_associated(name)
    }
}
