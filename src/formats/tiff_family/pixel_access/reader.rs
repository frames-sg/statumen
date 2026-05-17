use super::*;

// ── TiffPixelReader ───────────────────────────────────────────────

/// Implements SlideReader by dispatching tile reads based on TileSource type.
/// Holds an Arc<TiffContainer> for concurrent pread access and the layout
/// produced by a TiffLayoutInterpreter.
pub(crate) struct TiffPixelReader {
    pub(super) container: Arc<TiffContainer>,
    pub(super) layout: DatasetLayout,
    pub(super) full_decode_cache: Mutex<FullDecodeCache>,
    pub(super) full_decode_flights: Mutex<HashMap<IfdId, FullDecodeFlight>>,
    pub(super) full_decode_ready: Condvar,
    pub(super) ndpi_strip_cache: Mutex<NdpiStripCache>,
    pub(super) ndpi_mcu_starts_cache: Mutex<NdpiMcuStartsCache>,
    pub(super) ndpi_strip_flights: Mutex<HashMap<NdpiStripKey, NdpiStripFlight>>,
    pub(super) ndpi_strip_ready: Condvar,
    pub(super) synthetic_level_cache: Mutex<SyntheticLevelCache>,
    pub(super) synthetic_region_cache: Mutex<SyntheticLevelCache>,
    pub(super) synthetic_level_flights: Mutex<HashMap<SyntheticLevelKey, SyntheticLevelFlight>>,
    pub(super) synthetic_level_ready: Condvar,
    pub(super) synthetic_prime_once: OnceLock<()>,
    pub(super) stitched_component_tile_cache: Mutex<StitchedComponentTileCache>,
}

impl TiffPixelReader {
    pub(super) fn stripped_associated_decode_pool() -> Result<&'static rayon::ThreadPool, WsiError>
    {
        static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
        match POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .use_current_thread()
                .stack_size(2 * 1024 * 1024)
                .thread_name(|idx| format!("wsi-strips-{idx}"))
                .build()
                .map_err(|err| err.to_string())
        }) {
            Ok(pool) => Ok(pool),
            Err(reason) => Err(WsiError::DisplayConversion(format!(
                "failed to build stripped associated decode pool: {reason}"
            ))),
        }
    }

    pub(super) fn full_decode_cache_bytes() -> u64 {
        std::env::var(FULL_DECODE_CACHE_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_FULL_DECODE_CACHE_BYTES)
    }

    pub(super) fn ndpi_strip_cache_bytes() -> u64 {
        std::env::var(NDPI_STRIP_CACHE_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_NDPI_STRIP_CACHE_BYTES)
    }

    pub(super) fn synthetic_level_cache_bytes() -> u64 {
        std::env::var(SYNTHETIC_LEVEL_CACHE_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_SYNTHETIC_LEVEL_CACHE_BYTES)
    }

    pub(super) fn get_cached_ndpi_strip(&self, strip_key: NdpiStripKey) -> Option<Arc<CpuTile>> {
        self.ndpi_strip_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&strip_key)
    }

    pub(super) fn synthetic_level_key_for_region(
        req: &RegionRequest,
        base_level: u32,
    ) -> SyntheticLevelKey {
        let plane = req.plane.0;
        SyntheticLevelKey {
            scene: req.scene.0,
            series: req.series.0,
            base_level,
            target_level: req.level.0,
            z: plane.z,
            c: plane.c,
            t: plane.t,
        }
    }

    pub(super) fn synthetic_level_key_for_tile(
        req: &TileRequest,
        base_level: u32,
    ) -> SyntheticLevelKey {
        SyntheticLevelKey {
            scene: req.scene,
            series: req.series,
            base_level,
            target_level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
        }
    }

    pub(super) fn get_cached_synthetic_level(
        &self,
        key: &SyntheticLevelKey,
    ) -> Option<Arc<CpuTile>> {
        self.synthetic_region_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }

    pub(super) fn put_synthetic_level_cache(&self, key: SyntheticLevelKey, image: Arc<CpuTile>) {
        self.synthetic_region_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, image);
    }

    pub(super) fn try_decode_synthetic_level_with_signinum(
        &self,
        req: &TileRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<Option<CpuTile>, WsiError> {
        let Some(scale) = signinum_downscale_for_factor(factor) else {
            return Ok(None);
        };
        let target =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let base_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: base_level,
            plane: req.plane,
            col: 0,
            row: 0,
        };
        let TileSource::NdpiFullDecode {
            ifd_id,
            strip_offset,
            strip_byte_count,
            ..
        } = self.tile_source_for(&base_req)?
        else {
            return Ok(None);
        };

        let jpeg = self
            .container
            .pread(*strip_offset, *strip_byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let options = signinum_decode_options(
            self.tiff_jpeg_decode_options_for_data(*ifd_id, false, &jpeg, None)
                .color_transform,
        );
        let decoder = SigninumJpegDecoder::new_with_options(&jpeg, options)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let source_dims = decoder.info().dimensions;
        let scale_denom = scale.denominator();
        let scaled_width = source_dims.0.div_ceil(scale_denom);
        let scaled_height = source_dims.1.div_ceil(scale_denom);
        let (pixels, _outcome) = decoder
            .decode_scaled(SigninumPixelFormat::Rgb8, scale)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let scaled = cpu_tile_from_rgb_pixels(scaled_width, scaled_height, pixels)?;

        if scaled.width == target.dimensions.0 as u32 && scaled.height == target.dimensions.1 as u32
        {
            Ok(Some(scaled))
        } else {
            Ok(None)
        }
    }

    pub(super) fn prime_deepest_synthetic_levels_best_effort(&self) {
        let mut deepest: HashMap<SyntheticDeepestKey, SyntheticDeepestValue> = HashMap::new();
        for (key, source) in &self.layout.tile_sources {
            let TileSource::SyntheticDownsample { base_level, factor } = source else {
                continue;
            };
            deepest
                .entry((key.scene, key.series, key.z, key.c, key.t))
                .and_modify(|current| {
                    if key.level > current.0 {
                        *current = (key.level, *base_level, *factor);
                    }
                })
                .or_insert((key.level, *base_level, *factor));
        }

        for ((scene, series, z, c, t), (target_level, base_level, factor)) in deepest {
            let req = TileRequest {
                scene,
                series,
                level: target_level,
                plane: PlaneSelection { z, c, t },
                col: 0,
                row: 0,
            };
            let key = Self::synthetic_level_key_for_tile(&req, base_level);
            if self.get_cached_synthetic_level(&key).is_some() {
                continue;
            }
            if let Ok(Some(image)) =
                self.try_decode_synthetic_level_with_signinum(&req, base_level, factor)
            {
                self.put_synthetic_level_cache(key, Arc::new(image));
            }
        }
    }

    pub(super) fn clamp_ndpi_strip_crop(
        src_x: u32,
        src_y: u32,
        width: u32,
        height: u32,
        strip_width: u32,
        strip_height: u32,
    ) -> Option<(u32, u32)> {
        if src_x >= strip_width || src_y >= strip_height {
            return None;
        }

        let clamped_width = width.min(strip_width - src_x);
        let clamped_height = height.min(strip_height - src_y);
        if clamped_width == 0 || clamped_height == 0 {
            return None;
        }

        Some((clamped_width, clamped_height))
    }

    pub fn new(container: Arc<TiffContainer>, layout: DatasetLayout) -> Self {
        Self {
            container,
            layout,
            full_decode_cache: Mutex::new(FullDecodeCache::new(Self::full_decode_cache_bytes())),
            full_decode_flights: Mutex::new(HashMap::new()),
            full_decode_ready: Condvar::new(),
            ndpi_strip_cache: Mutex::new(NdpiStripCache::new(Self::ndpi_strip_cache_bytes())),
            ndpi_mcu_starts_cache: Mutex::new(HashMap::new()),
            ndpi_strip_flights: Mutex::new(HashMap::new()),
            ndpi_strip_ready: Condvar::new(),
            synthetic_level_cache: Mutex::new(SyntheticLevelCache::new(
                Self::synthetic_level_cache_bytes(),
            )),
            synthetic_region_cache: Mutex::new(SyntheticLevelCache::new(
                Self::synthetic_level_cache_bytes(),
            )),
            synthetic_level_flights: Mutex::new(HashMap::new()),
            synthetic_level_ready: Condvar::new(),
            synthetic_prime_once: OnceLock::new(),
            stitched_component_tile_cache: Mutex::new(StitchedComponentTileCache::default()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_or_decode_stitched_component_tile(
        &self,
        ifd_id: IfdId,
        tile_idx: usize,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        width: u32,
        height: u32,
        offsets: &[u64],
        byte_counts: &[u64],
    ) -> Result<Arc<CpuTile>, WsiError> {
        let key = StitchedComponentTileKey {
            ifd_id,
            tile_idx,
            width,
            height,
        };
        if let Some(cached) = self
            .stitched_component_tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Ok(cached);
        }

        let tile = Arc::new(self.decode_tiled_ifd_tile_index(
            ifd_id,
            tile_idx,
            jpeg_tables,
            compression,
            width,
            height,
            offsets,
            byte_counts,
            BackendRequest::Auto,
        )?);
        self.stitched_component_tile_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(key, tile.clone());
        Ok(tile)
    }

    pub(super) fn ndpi_full_decode_error(req: &TileRequest, reason: impl Into<String>) -> WsiError {
        WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: reason.into(),
        }
    }

    pub(super) fn read_stripped_data(
        &self,
        name: &str,
        strip_offsets: &[u64],
        strip_byte_counts: &[u64],
    ) -> Result<Vec<u8>, WsiError> {
        if strip_offsets.len() != strip_byte_counts.len() {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has mismatched strip metadata ({} offsets vs {} byte counts)",
                name,
                strip_offsets.len(),
                strip_byte_counts.len()
            )));
        }

        let total_bytes = strip_byte_counts.iter().try_fold(0usize, |acc, count| {
            acc.checked_add(usize::try_from(*count).ok()?)
        });
        let total_bytes = total_bytes.ok_or_else(|| {
            WsiError::UnsupportedFormat(format!(
                "associated image '{}' strip byte counts exceed addressable memory",
                name
            ))
        })?;

        let mut data = Vec::with_capacity(total_bytes);
        for (&offset, &byte_count) in strip_offsets.iter().zip(strip_byte_counts.iter()) {
            if byte_count == 0 {
                continue;
            }
            let bytes = self
                .container
                .pread(offset, byte_count)
                .map_err(|e| e.into_wsi_error(self.container.path()))?;
            data.extend_from_slice(&bytes);
        }
        Ok(data)
    }

    pub(super) fn read_stripped_jpeg_image(
        &self,
        name: &str,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        dimensions: (u32, u32),
        strip_offsets: &[u64],
        strip_byte_counts: &[u64],
    ) -> Result<CpuTile, WsiError> {
        if strip_offsets.len() != strip_byte_counts.len() {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has mismatched strip metadata ({} offsets vs {} byte counts)",
                name,
                strip_offsets.len(),
                strip_byte_counts.len()
            )));
        }

        let (width, height) = dimensions;
        let rows_per_strip = self
            .container
            .get_u32(ifd_id, tags::ROWS_PER_STRIP)
            .unwrap_or(height)
            .max(1);
        let total_bytes = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| {
                WsiError::UnsupportedFormat(format!(
                    "associated image '{}' dimensions overflow RGB buffer size",
                    name
                ))
            })?;
        let mut composed = vec![0u8; total_bytes];
        let dst_stride = width as usize * 3;
        let strip_count = height.div_ceil(rows_per_strip) as usize;
        if strip_offsets.len() < strip_count || strip_byte_counts.len() < strip_count {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' expected at least {} strips for {} rows, found offsets={} byte_counts={}",
                name,
                strip_count,
                height,
                strip_offsets.len(),
                strip_byte_counts.len()
            )));
        }
        let strip_chunk_bytes =
            dst_stride
                .checked_mul(rows_per_strip as usize)
                .ok_or_else(|| {
                    WsiError::UnsupportedFormat(format!(
                        "associated image '{}' rows_per_strip overflow for width {}",
                        name, width
                    ))
                })?;

        Self::stripped_associated_decode_pool()?.install(|| {
            composed
                .par_chunks_mut(strip_chunk_bytes)
                .zip(
                    strip_offsets[..strip_count]
                        .par_iter()
                        .zip(strip_byte_counts[..strip_count].par_iter())
                        .enumerate(),
                )
                .try_for_each(|(dst_chunk, (strip_idx, (&offset, &byte_count)))| {
                    if byte_count == 0 {
                        return Ok(());
                    }

                    let strip_y = rows_per_strip.saturating_mul(strip_idx as u32);
                    let strip_height = rows_per_strip.min(height - strip_y);
                    let expected_len = strip_height as usize * dst_stride;
                    if dst_chunk.len() != expected_len {
                        return Err(WsiError::UnsupportedFormat(format!(
                            "associated image '{}' destination chunk for strip {} has {} bytes, expected {}",
                            name,
                            strip_idx,
                            dst_chunk.len(),
                            expected_len
                        )));
                    }

                    let data = self
                        .container
                        .pread(offset, byte_count)
                        .map_err(|e| e.into_wsi_error(self.container.path()))?;
                    let decode_options = self.tiff_jpeg_decode_options_for_data(
                        ifd_id,
                        true,
                        &data,
                        jpeg_tables,
                    );
                    let decoded = decode_one_jpeg(
                        JpegDecodeJob {
                            data: Cow::Borrowed(&data),
                            tables: jpeg_tables.map(Cow::Borrowed),
                            expected_width: width,
                            expected_height: strip_height,
                            color_transform: decode_options.color_transform,
                            force_dimensions: decode_options.force_dimensions,
                            requested_size: None,
                        }
                    )
                    .map_err(|err| WsiError::TileRead {
                        col: strip_idx as i64,
                        row: i64::from(strip_y),
                        level: 0,
                        reason: format!(
                            "associated image '{}' JPEG strip {} decode failed (offset={}, bytes={}, dims={}x{}): {}",
                            name, strip_idx, offset, byte_count, width, strip_height, err
                        ),
                    })?;
                    let CpuTileData::U8(decoded_rows) = decoded.data else {
                        return Err(WsiError::DisplayConversion(
                            "stripped JPEG decode expected U8 RGB data".into(),
                        ));
                    };
                    if decoded.width != width || decoded.height != strip_height {
                        return Err(WsiError::UnsupportedFormat(format!(
                            "associated image '{}' decoded strip {} as {}x{} but expected {}x{}",
                            name, strip_idx, decoded.width, decoded.height, width, strip_height
                        )));
                    }
                    if decoded_rows.len() != expected_len {
                        return Err(WsiError::UnsupportedFormat(format!(
                            "associated image '{}' decoded strip {} produced {} bytes, expected {}",
                            name,
                            strip_idx,
                            decoded_rows.len(),
                            expected_len
                        )));
                    }

                    dst_chunk.copy_from_slice(&decoded_rows);
                    Ok(())
                })
        })?;

        Ok(CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(composed),
        })
    }

    pub(super) fn tiff_jpeg_decode_options_for_data(
        &self,
        ifd_id: IfdId,
        force_dimensions: bool,
        data: &[u8],
        tables: Option<&[u8]>,
    ) -> TiffJpegDecodeOptions {
        self.tiff_jpeg_decode_options_with_hint(
            ifd_id,
            force_dimensions,
            jpeg_bitstream_color_hint(data, tables),
        )
    }

    pub(super) fn tiff_jpeg_decode_options_with_hint(
        &self,
        ifd_id: IfdId,
        force_dimensions: bool,
        bitstream_hint: JpegBitstreamColorHint,
    ) -> TiffJpegDecodeOptions {
        if self.layout.dataset.properties.vendor() == Some("philips") {
            return TiffJpegDecodeOptions {
                force_dimensions,
                color_transform: SigninumColorTransform::Auto,
            };
        }

        let photometric = self
            .container
            .get_u32(ifd_id, tags::PHOTOMETRIC)
            .unwrap_or(2);
        let samples_per_pixel = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(3);
        let color_transform =
            tiff_jpeg_color_transform(photometric, samples_per_pixel, bitstream_hint);
        TiffJpegDecodeOptions {
            force_dimensions,
            color_transform,
        }
    }

    pub(super) fn ndpi_mcu_starts(
        &self,
        ifd_id: IfdId,
        mcu_starts_tag: u16,
    ) -> Result<Arc<Vec<u64>>, WsiError> {
        if let Some(starts) = self
            .ndpi_mcu_starts_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(ifd_id, mcu_starts_tag))
            .cloned()
        {
            return Ok(starts);
        }

        let starts = Arc::new(
            self.container
                .get_u64_array(ifd_id, mcu_starts_tag)
                .map_err(|e| e.into_wsi_error(self.container.path()))?
                .to_vec(),
        );
        self.ndpi_mcu_starts_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((ifd_id, mcu_starts_tag), starts.clone());
        Ok(starts)
    }

    pub(super) fn decode_ndpi_full_image(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let data = self
            .container
            .pread(strip_offset, strip_byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;

        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let options = signinum_decode_options(
            self.tiff_jpeg_decode_options_for_data(ifd_id, false, &data, None)
                .color_transform,
        );
        let decoder = SigninumJpegDecoder::new_with_options(&data, options)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let (pixels, outcome) = decoder
            .decode(SigninumPixelFormat::Rgb8)
            .map_err(|err| WsiError::Jpeg(err.to_string()))?;
        let decoded = cpu_tile_from_rgb_pixels(outcome.decoded.w, outcome.decoded.h, pixels)?;
        let decoded = if decoded.width > level_w as u32 || decoded.height > level_h as u32 {
            crop_rgb_interleaved_u8_buffer(&decoded, 0, 0, level_w as u32, level_h as u32)?
        } else {
            decoded
        };

        Ok(Arc::new(decoded))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_ndpi_display_tile(
        &self,
        req: &TileViewRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NdpiJpeg display tile expects WholeLevel layout".into(),
                });
            }
        };

        let tile_origin_x = req.col.saturating_mul(i64::from(req.tile_width));
        let tile_origin_y = req.row.saturating_mul(i64::from(req.tile_height));
        let level_w = level_w as i64;
        let level_h = level_h as i64;
        if tile_origin_x < 0
            || tile_origin_y < 0
            || tile_origin_x >= level_w
            || tile_origin_y >= level_h
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "display tile origin out of bounds".into(),
            });
        }

        let content_width = req.tile_width.min((level_w - tile_origin_x) as u32);
        let content_height = req.tile_height.min((level_h - tile_origin_y) as u32);
        let tile_end_x = tile_origin_x as u32 + content_width;
        let tile_end_y = tile_origin_y as u32 + content_height;
        let native_col_start = tile_origin_x as u32 / vtw;
        let native_col_end = tile_end_x.saturating_sub(1) / vtw;
        let tile_origin_y_u32 = tile_origin_y as u32;
        let native_row_start = tile_origin_y_u32 / vth;
        let native_row_end = tile_end_y.saturating_sub(1) / vth;

        struct NeededNdpiStrip {
            strip_key: NdpiStripKey,
            strip_req: TileRequest,
            copy_start_x: u32,
            copy_start_y: u32,
            copy_width: u32,
            copy_height: u32,
            dest_x: u32,
            dest_y: u32,
            strip: Option<Arc<CpuTile>>,
        }

        let mut needed_strips = Vec::new();
        for native_row in native_row_start..=native_row_end {
            let strip_origin_y = native_row * vth;

            for native_col in native_col_start..=native_col_end {
                let strip_origin_x = native_col * vtw;
                let strip_width = vtw.min((level_w as u32).saturating_sub(strip_origin_x));
                let strip_height = vth.min((level_h as u32).saturating_sub(strip_origin_y));
                let copy_start_x = (tile_origin_x as u32).saturating_sub(strip_origin_x);
                let copy_start_y = tile_origin_y_u32.saturating_sub(strip_origin_y);
                let copy_end_x = tile_end_x.min(strip_origin_x + strip_width);
                let copy_end_y = tile_end_y.min(strip_origin_y + strip_height);
                let desired_width = copy_end_x.saturating_sub(strip_origin_x + copy_start_x);
                let desired_height = copy_end_y.saturating_sub(strip_origin_y + copy_start_y);
                let Some((copy_width, copy_height)) = Self::clamp_ndpi_strip_crop(
                    copy_start_x,
                    copy_start_y,
                    desired_width,
                    desired_height,
                    strip_width,
                    strip_height,
                ) else {
                    continue;
                };

                let strip_key = NdpiStripKey {
                    ifd_id,
                    col: native_col,
                    native_row,
                };
                let dest_x = strip_origin_x
                    .saturating_add(copy_start_x)
                    .saturating_sub(tile_origin_x as u32);
                let dest_y = strip_origin_y
                    .saturating_add(copy_start_y)
                    .saturating_sub(tile_origin_y as u32);
                needed_strips.push(NeededNdpiStrip {
                    strip_key,
                    strip_req: TileRequest {
                        scene: req.scene,
                        series: req.series,
                        level: req.level,
                        plane: req.plane,
                        col: i64::from(native_col),
                        row: i64::from(native_row),
                    },
                    copy_start_x,
                    copy_start_y,
                    copy_width,
                    copy_height,
                    dest_x,
                    dest_y,
                    strip: self.get_cached_ndpi_strip(strip_key),
                });
            }
        }

        let missing_indices: Vec<usize> = needed_strips
            .iter()
            .enumerate()
            .filter_map(|(idx, needed)| needed.strip.is_none().then_some(idx))
            .collect();
        if !missing_indices.is_empty() {
            let decode_batch = if vtw > NDPI_DISPLAY_WIDE_STRIP_WIDTH {
                NDPI_DISPLAY_WIDE_STRIP_BATCH
            } else {
                NDPI_DISPLAY_NARROW_STRIP_BATCH
            };
            let decoded_missing: Result<Vec<(usize, Arc<CpuTile>)>, WsiError> =
                if missing_indices.len() == 1 {
                    let idx = missing_indices[0];
                    let needed = &needed_strips[idx];
                    Ok(vec![(
                        idx,
                        self.get_or_decode_ndpi_strip(
                            &needed.strip_req,
                            ifd_id,
                            jpeg_header,
                            mcu_starts_tag,
                            tiles_across,
                            tiles_down,
                            strip_offset,
                            strip_byte_count,
                            needed.strip_key,
                            vtw,
                            vth,
                            level_w as u32,
                            level_h as u32,
                        )?,
                    )])
                } else {
                    let mut decoded = Vec::with_capacity(missing_indices.len());
                    for batch in missing_indices.chunks(decode_batch) {
                        let mut decoded_batch: Vec<(usize, Arc<CpuTile>)> = batch
                            .par_iter()
                            .map(|idx| {
                                let needed = &needed_strips[*idx];
                                let strip = self.get_or_decode_ndpi_strip(
                                    &needed.strip_req,
                                    ifd_id,
                                    jpeg_header,
                                    mcu_starts_tag,
                                    tiles_across,
                                    tiles_down,
                                    strip_offset,
                                    strip_byte_count,
                                    needed.strip_key,
                                    vtw,
                                    vth,
                                    level_w as u32,
                                    level_h as u32,
                                )?;
                                Ok::<(usize, Arc<CpuTile>), WsiError>((*idx, strip))
                            })
                            .collect::<Result<_, _>>()?;
                        decoded.append(&mut decoded_batch);
                    }
                    Ok(decoded)
                };
            for (idx, strip) in decoded_missing? {
                needed_strips[idx].strip = Some(strip);
            }
        }

        let mut tile_data = vec![255u8; (content_width * content_height * 3) as usize];
        let dst_stride = content_width as usize * 3;

        for needed in needed_strips {
            let strip = needed.strip.ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("missing decoded NDPI strip {:?}", needed.strip_key),
            })?;

            let copy_start_x = needed.copy_start_x;
            let copy_start_y = needed.copy_start_y;
            let copy_width = needed.copy_width;
            let copy_height = needed.copy_height;
            let dest_x = needed.dest_x;
            let dest_y = needed.dest_y;

            if strip.layout != CpuTileLayout::Interleaved || strip.channels != 3 {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI display tile expected interleaved RGB strips".into(),
                });
            }
            let CpuTileData::U8(strip_rgb) = &strip.data else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI display tile expected U8 RGB strip data".into(),
                });
            };
            let src_stride = strip.width as usize * 3;
            let copy_row_bytes = copy_width as usize * 3;
            for row in 0..copy_height as usize {
                let src_off =
                    (copy_start_y as usize + row) * src_stride + copy_start_x as usize * 3;
                let dst_off = (dest_y as usize + row) * dst_stride + dest_x as usize * 3;
                tile_data[dst_off..dst_off + copy_row_bytes]
                    .copy_from_slice(&strip_rgb[src_off..src_off + copy_row_bytes]);
            }
        }

        Ok(CpuTile {
            width: content_width,
            height: content_height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(tile_data),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ndpi_jpeg_tile_payload(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
        strip_key: NdpiStripKey,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
        level_width: u32,
        level_height: u32,
    ) -> Result<NdpiJpegTilePayload, WsiError> {
        if strip_key.native_row >= tiles_down {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("NDPI strip row {} out of range", strip_key.native_row),
            });
        }
        if strip_key.col >= tiles_across {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("NDPI strip column {} out of range", strip_key.col),
            });
        }

        let strip_origin_y = strip_key.native_row * virtual_tile_height;
        let strip_height = virtual_tile_height.min(level_height.saturating_sub(strip_origin_y));
        let strip_width =
            virtual_tile_width.min(level_width.saturating_sub(strip_key.col * virtual_tile_width));

        let mcu_starts = self.ndpi_mcu_starts(ifd_id, mcu_starts_tag)?;

        let idx =
            (strip_key.native_row as u64 * tiles_across as u64 + strip_key.col as u64) as usize;
        if idx >= mcu_starts.len() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI MCU-starts index {} out of range (len={})",
                    idx,
                    mcu_starts.len(),
                ),
            });
        }

        if idx + 1 < mcu_starts.len() && mcu_starts[idx + 1] <= mcu_starts[idx] {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI MCU-starts table is not strictly increasing at index {}",
                    idx
                ),
            });
        }

        let segment_start = *mcu_starts.get(idx).ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: format!("NDPI MCU-starts index {idx} out of range"),
        })?;
        let next_segment_start = if idx + 1 < mcu_starts.len() {
            Some(mcu_starts[idx + 1])
        } else {
            None
        };
        let segment_end = next_segment_start.unwrap_or(strip_byte_count);
        if segment_start >= strip_byte_count
            || segment_end > strip_byte_count
            || segment_end <= segment_start
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI MCU segment [{segment_start}, {segment_end}) exceeds strip byte count {strip_byte_count}"
                ),
            });
        }
        if jpeg_header.is_empty() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "NDPI JPEG header is empty".into(),
            });
        }

        let segment_len = segment_end.saturating_sub(segment_start);
        let read_offset =
            strip_offset
                .checked_add(segment_start)
                .ok_or_else(|| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI strip offset overflow".into(),
                })?;
        let segment = self
            .container
            .pread(read_offset, segment_len)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let entropy = strip_trailing_eoi_marker(strip_trailing_restart_marker(
            strip_leading_restart_marker(&segment),
        ));
        let mut tile_jpeg = Vec::with_capacity(jpeg_header.len() + entropy.len() + 2);
        tile_jpeg.extend_from_slice(jpeg_header);
        disable_jpeg_restart_interval(&mut tile_jpeg);
        tile_jpeg.extend_from_slice(entropy);
        tile_jpeg.extend_from_slice(&[0xFF, 0xD9]);

        Ok(NdpiJpegTilePayload {
            jpeg: tile_jpeg,
            width: strip_width,
            height: strip_height,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_ndpi_strip(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
        strip_key: NdpiStripKey,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
        level_width: u32,
        level_height: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let payload = self.ndpi_jpeg_tile_payload(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            strip_key,
            virtual_tile_width,
            virtual_tile_height,
            level_width,
            level_height,
        )?;
        let decoded = decode_jpeg_rgb_with_size_override(
            &payload.jpeg,
            None,
            payload.width,
            payload.height,
            None,
            None,
            self.tiff_jpeg_decode_options_for_data(ifd_id, false, &payload.jpeg, None)
                .color_transform,
        )?;
        let decoded = cpu_tile_from_rgb_pixels(decoded.width, decoded.height, decoded.pixels)?;

        Ok(Arc::new(decoded))
    }

    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ndpi_jpeg_decode_job<'a>(
        &'a self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<JpegDecodeJob<'a>, WsiError> {
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NdpiJpeg device decode expects WholeLevel tile layout".into(),
                });
            }
        };
        let (col, row) = validate_tile_coords(req.col, req.row, req.level)?;
        let payload = self.ndpi_jpeg_tile_payload(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            NdpiStripKey {
                ifd_id,
                col,
                native_row: row,
            },
            vtw,
            vth,
            level_w as u32,
            level_h as u32,
        )?;
        let color_transform = self
            .tiff_jpeg_decode_options_for_data(ifd_id, false, &payload.jpeg, None)
            .color_transform;
        Ok(JpegDecodeJob {
            data: Cow::Owned(payload.jpeg),
            tables: None,
            expected_width: payload.width,
            expected_height: payload.height,
            color_transform,
            force_dimensions: true,
            requested_size: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_or_decode_ndpi_strip(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        strip_offset: u64,
        strip_byte_count: u64,
        strip_key: NdpiStripKey,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
        level_width: u32,
        level_height: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        if let Some(strip) = {
            let mut cache = self
                .ndpi_strip_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.get(&strip_key)
        } {
            return Ok(strip);
        }

        let mut flights = self
            .ndpi_strip_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut registered_waiter = false;
        loop {
            match flights.get_mut(&strip_key) {
                Some(flight) => {
                    if !registered_waiter {
                        flight.waiters += 1;
                        registered_waiter = true;
                    }
                    if let Some(result) = flight.result.clone() {
                        flight.waiters -= 1;
                        if flight.waiters == 0 {
                            flights.remove(&strip_key);
                        }
                        return result.map_err(|reason| Self::ndpi_full_decode_error(req, reason));
                    }
                    flights = self
                        .ndpi_strip_ready
                        .wait(flights)
                        .unwrap_or_else(|e| e.into_inner());
                }
                None if registered_waiter => {
                    return Err(Self::ndpi_full_decode_error(
                        req,
                        format!("NDPI strip decode flight for {:?} disappeared", strip_key),
                    ));
                }
                None => {
                    flights.insert(strip_key, NdpiStripFlight::default());
                    break;
                }
            }
        }
        drop(flights);

        let decode_result = self
            .decode_ndpi_strip(
                req,
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                strip_offset,
                strip_byte_count,
                strip_key,
                virtual_tile_width,
                virtual_tile_height,
                level_width,
                level_height,
            )
            .map_err(|err| err.to_string());

        if let Ok(strip) = decode_result.as_ref() {
            let mut cache = self
                .ndpi_strip_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.put(strip_key, strip.clone());
        }

        let mut flights = self
            .ndpi_strip_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(flight) = flights.get_mut(&strip_key) {
            flight.result = Some(decode_result.clone());
            if flight.waiters == 0 {
                flights.remove(&strip_key);
            }
        }
        drop(flights);
        self.ndpi_strip_ready.notify_all();

        decode_result.map_err(|reason| Self::ndpi_full_decode_error(req, reason))
    }

    pub(super) fn get_or_decode_ndpi_full_image(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<Arc<CpuTile>, WsiError> {
        if let Some(img) = {
            let mut cache = self
                .full_decode_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.get(&ifd_id)
        } {
            return Ok(img);
        }

        let mut flights = self
            .full_decode_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut registered_waiter = false;
        loop {
            match flights.get_mut(&ifd_id) {
                Some(flight) => {
                    if !registered_waiter {
                        flight.waiters += 1;
                        registered_waiter = true;
                    }
                    if let Some(result) = flight.result.clone() {
                        flight.waiters -= 1;
                        let should_remove = flight.waiters == 0;
                        if should_remove {
                            flights.remove(&ifd_id);
                        }
                        return result.map_err(|reason| Self::ndpi_full_decode_error(req, reason));
                    }
                    flights = self
                        .full_decode_ready
                        .wait(flights)
                        .unwrap_or_else(|e| e.into_inner());
                }
                None if registered_waiter => {
                    return Err(Self::ndpi_full_decode_error(
                        req,
                        format!("NDPI full decode flight for {ifd_id} disappeared"),
                    ));
                }
                None => {
                    flights.insert(ifd_id, FullDecodeFlight::default());
                    break;
                }
            }
        }
        drop(flights);

        let decode_result = self
            .decode_ndpi_full_image(req, ifd_id, strip_offset, strip_byte_count)
            .map_err(|err| err.to_string());
        if let Ok(image) = decode_result.as_ref() {
            let mut cache = self
                .full_decode_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.put(ifd_id, image.clone());
        }

        let mut flights = self
            .full_decode_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(flight) = flights.get_mut(&ifd_id) {
            flight.result = Some(decode_result.clone());
            if flight.waiters == 0 {
                flights.remove(&ifd_id);
            }
        }
        drop(flights);
        self.full_decode_ready.notify_all();

        decode_result.map_err(|reason| Self::ndpi_full_decode_error(req, reason))
    }

    pub(super) fn decode_synthetic_level(
        &self,
        req: &TileRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        if !factor.is_power_of_two() || factor < 2 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!("invalid synthetic NDPI factor {factor}"),
            });
        }

        if let Some(image) =
            self.try_decode_synthetic_level_with_signinum(req, base_level, factor)?
        {
            return Ok(Arc::new(image));
        }

        let base =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[base_level as usize];
        let target =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let base_tile_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: base_level,
            plane: req.plane,
            col: 0,
            row: 0,
        };
        let mut current = if matches!(
            self.tile_source_for(&base_tile_req),
            Ok(TileSource::NdpiFullDecode { .. })
        ) {
            self.read_tile_cpu(&base_tile_req)?
        } else {
            composite_region_from_source(
                self,
                None,
                &RegionRequest {
                    scene: SceneId(req.scene),
                    series: SeriesId(req.series),
                    level: LevelIdx(base_level),
                    plane: PlaneIdx(req.plane),
                    origin_px: (0, 0),
                    size_px: (
                        u32::try_from(base.dimensions.0).unwrap_or(u32::MAX),
                        u32::try_from(base.dimensions.1).unwrap_or(u32::MAX),
                    ),
                },
            )?
        };

        if current.layout != CpuTileLayout::Interleaved
            || current.channels != 3
            || current.color_space != ColorSpace::Rgb
            || current.data.as_u8().is_none()
        {
            current = rgba_image_to_sample_buffer(current.to_rgba()?);
        }

        current = fit_synthetic_rgb_tile_to_dimensions(
            downsample_rgb_pow2_box(&current, factor)?,
            target.dimensions.0 as u32,
            target.dimensions.1 as u32,
        )?;

        Ok(Arc::new(current))
    }

    pub(super) fn read_full_synthetic_region_fastpath(
        &self,
        cache: Option<&crate::core::cache::TileCache>,
        req: &RegionRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<CpuTile, WsiError> {
        if !factor.is_power_of_two() || !(2..=8).contains(&factor) {
            return composite_region_from_source(self, cache, req);
        }

        let (x, y) = req.origin_px;
        let (w, h) = req.size_px;
        let plane = req.plane.0;
        let level = &self.layout.dataset.scenes[req.scene.0].series[req.series.0].levels
            [req.level.0 as usize];
        if x != 0
            || y != 0
            || u64::from(w) != level.dimensions.0
            || u64::from(h) != level.dimensions.1
        {
            return self.read_synthetic_subregion_fastpath(
                cache,
                req,
                base_level,
                factor,
                level.dimensions.0,
                level.dimensions.1,
            );
        }

        let key = CacheKey {
            dataset_id: self.layout.dataset.id,
            scene: req.scene.0 as u32,
            series: req.series.0 as u32,
            level: req.level.0,
            z: plane.z,
            c: plane.c,
            t: plane.t,
            tile_col: 0,
            tile_row: 0,
        };
        if let Some(cache) = cache {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.as_ref().clone());
            }
        }

        let synthetic_key = Self::synthetic_level_key_for_region(req, base_level);
        if let Some(cached) = self.get_cached_synthetic_level(&synthetic_key) {
            if let Some(cache) = cache {
                cache.put(key, cached.clone());
            }
            return Ok(cached.as_ref().clone());
        }

        let base_req = TileRequest {
            scene: req.scene.0,
            series: req.series.0,
            level: base_level,
            plane,
            col: 0,
            row: 0,
        };
        let TileSource::NdpiFullDecode {
            ifd_id,
            strip_offset,
            strip_byte_count,
            ..
        } = self.tile_source_for(&base_req)?
        else {
            return composite_region_from_source(self, cache, req);
        };

        let tile_req = TileRequest {
            scene: req.scene.0,
            series: req.series.0,
            level: req.level.0,
            plane: req.plane.0,
            col: 0,
            row: 0,
        };
        let scaled = if let Some(image) =
            self.try_decode_synthetic_level_with_signinum(&tile_req, base_level, factor)?
        {
            image
        } else {
            let full = self.get_or_decode_ndpi_full_image(
                &base_req,
                *ifd_id,
                *strip_offset,
                *strip_byte_count,
            )?;
            downsample_rgb_pow2_box(full.as_ref(), factor)?
        };
        let image = Arc::new(fit_synthetic_rgb_tile_to_dimensions(scaled, w, h)?);
        if image.width != w || image.height != h {
            return composite_region_from_source(self, cache, req);
        }
        self.put_synthetic_level_cache(synthetic_key, image.clone());
        if let Some(cache) = cache {
            cache.put(key, image.clone());
        }
        Ok(image.as_ref().clone())
    }

    pub(super) fn read_synthetic_subregion_fastpath(
        &self,
        cache: Option<&crate::core::cache::TileCache>,
        req: &RegionRequest,
        base_level: u32,
        factor: u32,
        target_width: u64,
        target_height: u64,
    ) -> Result<CpuTile, WsiError> {
        let (x, y) = req.origin_px;
        let (w, h) = req.size_px;
        if w == 0 || h == 0 {
            return zero_rgb_interleaved_u8_tile(w, h);
        }

        let x0 = i128::from(x);
        let y0 = i128::from(y);
        let x1 = x0 + i128::from(w);
        let y1 = y0 + i128::from(h);
        let target_w = i128::from(target_width);
        let target_h = i128::from(target_height);
        let clipped_x0 = x0.clamp(0, target_w);
        let clipped_y0 = y0.clamp(0, target_h);
        let clipped_x1 = x1.clamp(0, target_w);
        let clipped_y1 = y1.clamp(0, target_h);

        if clipped_x1 <= clipped_x0 || clipped_y1 <= clipped_y0 {
            return zero_rgb_interleaved_u8_tile(w, h);
        }

        let valid_w = u32::try_from(clipped_x1 - clipped_x0).map_err(|_| {
            WsiError::DisplayConversion(format!(
                "synthetic NDPI ROI width exceeds region API bounds: {}",
                clipped_x1 - clipped_x0
            ))
        })?;
        let valid_h = u32::try_from(clipped_y1 - clipped_y0).map_err(|_| {
            WsiError::DisplayConversion(format!(
                "synthetic NDPI ROI height exceeds region API bounds: {}",
                clipped_y1 - clipped_y0
            ))
        })?;
        let dst_x = u32::try_from(clipped_x0 - x0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI destination x overflow".into())
        })?;
        let dst_y = u32::try_from(clipped_y0 - y0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI destination y overflow".into())
        })?;

        let base_tile_req = TileRequest {
            scene: req.scene.0,
            series: req.series.0,
            level: base_level,
            plane: req.plane.0,
            col: 0,
            row: 0,
        };
        if matches!(
            self.tile_source_for(&base_tile_req)?,
            TileSource::NdpiFullDecode { .. }
        ) {
            let tile_req = TileRequest {
                scene: req.scene.0,
                series: req.series.0,
                level: req.level.0,
                plane: req.plane.0,
                col: 0,
                row: 0,
            };
            if let Some(scaled) =
                self.try_decode_synthetic_level_with_signinum(&tile_req, base_level, factor)?
            {
                let crop_x0 = u32::try_from(clipped_x0).map_err(|_| {
                    WsiError::DisplayConversion(
                        "synthetic NDPI ROI source x exceeds crop bounds".into(),
                    )
                })?;
                let crop_y0 = u32::try_from(clipped_y0).map_err(|_| {
                    WsiError::DisplayConversion(
                        "synthetic NDPI ROI source y exceeds crop bounds".into(),
                    )
                })?;
                let cropped =
                    crop_rgb_interleaved_u8_buffer(&scaled, crop_x0, crop_y0, valid_w, valid_h)?;
                return paste_rgb_interleaved_u8_tile(&cropped, w, h, dst_x, dst_y);
            }
        }

        let series = self
            .layout
            .dataset
            .scenes
            .get(req.scene.0)
            .and_then(|scene| scene.series.get(req.series.0))
            .ok_or_else(|| WsiError::SeriesOutOfRange {
                index: req.series.0,
                count: self
                    .layout
                    .dataset
                    .scenes
                    .get(req.scene.0)
                    .map_or(0, |scene| scene.series.len()),
            })?;
        let base =
            series
                .levels
                .get(base_level as usize)
                .ok_or_else(|| WsiError::LevelOutOfRange {
                    level: base_level,
                    count: series.levels.len() as u32,
                })?;
        let clipped_x0 = u128::try_from(clipped_x0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source x is negative".into())
        })?;
        let clipped_y0 = u128::try_from(clipped_y0).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source y is negative".into())
        })?;
        let clipped_x1 = u128::try_from(clipped_x1).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source right is negative".into())
        })?;
        let clipped_y1 = u128::try_from(clipped_y1).map_err(|_| {
            WsiError::DisplayConversion("synthetic NDPI ROI source bottom is negative".into())
        })?;
        let factor = u128::from(factor);
        let base_x0 = clipped_x0.checked_mul(factor).ok_or_else(|| {
            WsiError::DisplayConversion("synthetic NDPI base ROI x overflow".into())
        })?;
        let base_y0 = clipped_y0.checked_mul(factor).ok_or_else(|| {
            WsiError::DisplayConversion("synthetic NDPI base ROI y overflow".into())
        })?;
        let base_x1 = clipped_x1
            .checked_mul(factor)
            .ok_or_else(|| {
                WsiError::DisplayConversion("synthetic NDPI base ROI right overflow".into())
            })?
            .min(u128::from(base.dimensions.0));
        let base_y1 = clipped_y1
            .checked_mul(factor)
            .ok_or_else(|| {
                WsiError::DisplayConversion("synthetic NDPI base ROI bottom overflow".into())
            })?
            .min(u128::from(base.dimensions.1));
        if base_x1 <= base_x0 || base_y1 <= base_y0 {
            return zero_rgb_interleaved_u8_tile(w, h);
        }

        let base_req = RegionRequest {
            scene: req.scene,
            series: req.series,
            level: LevelIdx(base_level),
            plane: req.plane,
            origin_px: (
                i64::try_from(base_x0).map_err(|_| {
                    WsiError::DisplayConversion("synthetic NDPI base ROI x exceeds i64".into())
                })?,
                i64::try_from(base_y0).map_err(|_| {
                    WsiError::DisplayConversion("synthetic NDPI base ROI y exceeds i64".into())
                })?,
            ),
            size_px: (
                u32::try_from(base_x1 - base_x0).map_err(|_| {
                    WsiError::DisplayConversion(
                        "synthetic NDPI base ROI width exceeds region API bounds".into(),
                    )
                })?,
                u32::try_from(base_y1 - base_y0).map_err(|_| {
                    WsiError::DisplayConversion(
                        "synthetic NDPI base ROI height exceeds region API bounds".into(),
                    )
                })?,
            ),
        };
        let base_source = TiffPixelReaderNoSyntheticPrime { inner: self };
        let base_region = ensure_interleaved_rgb_u8(composite_region_from_source(
            &base_source,
            cache,
            &base_req,
        )?)?;
        let downsampled = fit_synthetic_rgb_tile_to_dimensions(
            downsample_rgb_pow2_box(&base_region, factor as u32)?,
            valid_w,
            valid_h,
        )?;
        paste_rgb_interleaved_u8_tile(&downsampled, w, h, dst_x, dst_y)
    }

    pub(super) fn read_synthetic_display_tile(
        &self,
        req: &TileViewRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<CpuTile, WsiError> {
        let series = self
            .layout
            .dataset
            .scenes
            .get(req.scene)
            .and_then(|scene| scene.series.get(req.series))
            .ok_or_else(|| WsiError::SeriesOutOfRange {
                index: req.series,
                count: self
                    .layout
                    .dataset
                    .scenes
                    .get(req.scene)
                    .map_or(0, |scene| scene.series.len()),
            })?;
        let level =
            series
                .levels
                .get(req.level as usize)
                .ok_or_else(|| WsiError::LevelOutOfRange {
                    level: req.level,
                    count: series.levels.len() as u32,
                })?;

        let origin_x = req.col.saturating_mul(i64::from(req.tile_width));
        let origin_y = req.row.saturating_mul(i64::from(req.tile_height));
        let level_w = i64::try_from(level.dimensions.0).unwrap_or(i64::MAX);
        let level_h = i64::try_from(level.dimensions.1).unwrap_or(i64::MAX);
        if origin_x >= level_w || origin_y >= level_h {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "display tile origin out of bounds".into(),
            });
        }

        let clipped = RegionRequest {
            scene: SceneId(req.scene),
            series: SeriesId(req.series),
            level: LevelIdx(req.level),
            plane: PlaneIdx(req.plane),
            origin_px: (origin_x, origin_y),
            size_px: (
                req.tile_width.min((level_w - origin_x) as u32),
                req.tile_height.min((level_h - origin_y) as u32),
            ),
        };
        self.read_full_synthetic_region_fastpath(None, &clipped, base_level, factor)
    }

    pub(super) fn get_or_decode_synthetic_level(
        &self,
        req: &TileRequest,
        base_level: u32,
        factor: u32,
    ) -> Result<Arc<CpuTile>, WsiError> {
        let key = Self::synthetic_level_key_for_tile(req, base_level);

        if let Some(image) = {
            let mut cache = self
                .synthetic_level_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.get(&key)
        } {
            return Ok(image);
        }

        let mut flights = self
            .synthetic_level_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut registered_waiter = false;
        loop {
            match flights.get_mut(&key) {
                Some(flight) => {
                    if !registered_waiter {
                        flight.waiters += 1;
                        registered_waiter = true;
                    }
                    if let Some(result) = flight.result.clone() {
                        flight.waiters -= 1;
                        if flight.waiters == 0 {
                            flights.remove(&key);
                        }
                        return result.map_err(|reason| Self::ndpi_full_decode_error(req, reason));
                    }
                    flights = self
                        .synthetic_level_ready
                        .wait(flights)
                        .unwrap_or_else(|e| e.into_inner());
                }
                None if registered_waiter => {
                    return Err(Self::ndpi_full_decode_error(
                        req,
                        format!(
                            "synthetic NDPI level decode flight for {:?} disappeared",
                            key
                        ),
                    ));
                }
                None => {
                    flights.insert(key, SyntheticLevelFlight::default());
                    break;
                }
            }
        }
        drop(flights);

        let decode_result = self
            .decode_synthetic_level(req, base_level, factor)
            .map_err(|err| err.to_string());
        if let Ok(image) = decode_result.as_ref() {
            let mut cache = self
                .synthetic_level_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.put(key, image.clone());
        }

        let mut flights = self
            .synthetic_level_flights
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(flight) = flights.get_mut(&key) {
            flight.result = Some(decode_result.clone());
            if flight.waiters == 0 {
                flights.remove(&key);
            }
        }
        drop(flights);
        self.synthetic_level_ready.notify_all();

        decode_result.map_err(|reason| Self::ndpi_full_decode_error(req, reason))
    }

    /// Look up the TileSource for a given tile request.
    pub(super) fn tile_source_for(&self, req: &TileRequest) -> Result<&TileSource, WsiError> {
        let key = TileSourceKey {
            scene: req.scene,
            series: req.series,
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
        };
        self.layout
            .tile_sources
            .get(&key)
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "no tile source for scene={}, series={}, level={}, z={}, c={}, t={}",
                    req.scene, req.series, req.level, req.plane.z, req.plane.c, req.plane.t,
                ),
            })
    }

    /// Read a tile from an NdpiJpeg source (MCU extraction fast path).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_ndpi_restart_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        _restart_interval: u16,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let col = req.col;
        let row = req.row;

        // Bounds check
        if col < 0 || col >= tiles_across as i64 || row < 0 || row >= tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level: req.level,
                reason: format!(
                    "tile ({},{}) out of range ({}x{})",
                    col, row, tiles_across, tiles_down,
                ),
            });
        }

        // Compute tile dimensions first (needed for empty-tile fallback and decode)
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: "NdpiJpeg expects WholeLevel tile layout".into(),
                });
            }
        };

        let strip_key = NdpiStripKey {
            ifd_id,
            col: col as u32,
            native_row: row as u32,
        };
        let strip = self.get_or_decode_ndpi_strip(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            strip_key,
            vtw,
            vth,
            level_w as u32,
            level_h as u32,
        )?;

        Ok(strip.as_ref().clone())
    }

    pub(super) fn tiled_ifd_tile_index_and_dimensions(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
    ) -> Result<(usize, u32, u32), WsiError> {
        let col = req.col;
        let row = req.row;

        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];

        let tile_idx = match &level.tile_layout {
            TileLayout::Regular {
                tiles_across,
                tiles_down,
                ..
            } => {
                if col < 0 || col >= *tiles_across as i64 || row < 0 || row >= *tiles_down as i64 {
                    return Err(WsiError::TileRead {
                        col,
                        row,
                        level: req.level,
                        reason: format!(
                            "tile ({},{}) out of range ({}x{})",
                            col, row, tiles_across, tiles_down,
                        ),
                    });
                }
                (row as u64 * *tiles_across + col as u64) as usize
            }
            TileLayout::Irregular { tiles, .. } => {
                let entry = tiles.get(&(col, row)).ok_or_else(|| WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: format!("no irregular tile at ({},{})", col, row),
                })?;
                if let Some(tile_idx) = entry.tiff_tile_index {
                    tile_idx
                } else {
                    let image_width =
                        self.container
                            .get_u64(ifd_id, tags::IMAGE_WIDTH)
                            .map_err(|err| WsiError::TileRead {
                                col,
                                row,
                                level: req.level,
                                reason: format!("failed to read tiled IFD image width: {err}"),
                            })?;
                    let tile_width =
                        self.container
                            .get_u32(ifd_id, tags::TILE_WIDTH)
                            .map_err(|err| WsiError::TileRead {
                                col,
                                row,
                                level: req.level,
                                reason: format!("failed to read tiled IFD tile width: {err}"),
                            })?;
                    let tiles_across = image_width.div_ceil(tile_width as u64);
                    if col < 0 || row < 0 {
                        return Err(WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: "irregular tile row/col out of range for TIFF tile grid".into(),
                        });
                    }
                    (row as u64 * tiles_across + col as u64) as usize
                }
            }
            TileLayout::WholeLevel { .. } => {
                return Err(WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: "TiledIfd does not use WholeLevel layout".into(),
                });
            }
        };

        let (level_w, level_h) = level.dimensions;
        let (tw, th) = match &level.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                ..
            } => {
                let tw =
                    (*tile_width).min((level_w as u32).saturating_sub(col as u32 * *tile_width));
                let th =
                    (*tile_height).min((level_h as u32).saturating_sub(row as u32 * *tile_height));
                (tw, th)
            }
            TileLayout::Irregular { .. } => {
                let image_width =
                    self.container
                        .get_u64(ifd_id, tags::IMAGE_WIDTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF image width: {err}"),
                        })?;
                let image_height =
                    self.container
                        .get_u64(ifd_id, tags::IMAGE_LENGTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF image height: {err}"),
                        })?;
                let tile_width =
                    self.container
                        .get_u32(ifd_id, tags::TILE_WIDTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF tile width: {err}"),
                        })?;
                let tile_height =
                    self.container
                        .get_u32(ifd_id, tags::TILE_LENGTH)
                        .map_err(|err| WsiError::TileRead {
                            col,
                            row,
                            level: req.level,
                            reason: format!("failed to read irregular TIFF tile height: {err}"),
                        })?;
                let tw = tile_width.min(
                    image_width
                        .saturating_sub(col.max(0) as u64 * tile_width as u64)
                        .try_into()
                        .unwrap_or(u32::MAX),
                );
                let th = tile_height.min(
                    image_height
                        .saturating_sub(row.max(0) as u64 * tile_height as u64)
                        .try_into()
                        .unwrap_or(u32::MAX),
                );
                (tw, th)
            }
            _ => {
                return Err(WsiError::TileRead {
                    col,
                    row,
                    level: req.level,
                    reason: "unexpected tile layout for tiled IFD read".into(),
                });
            }
        };

        Ok((tile_idx, tw, th))
    }

    /// Read a tile from a TiledIfd source (standard TIFF tiled IFDs).
    pub(super) fn read_tiled_ifd_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let (tile_idx, tw, th) = self.tiled_ifd_tile_index_and_dimensions(req, ifd_id)?;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        self.decode_tiled_ifd_tile_index(
            ifd_id,
            tile_idx,
            jpeg_tables,
            compression,
            tw,
            th,
            offsets,
            byte_counts,
            backend,
        )
        .map_err(|err| match err {
            WsiError::TileRead { .. } => err,
            other => WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: other.to_string(),
            },
        })
    }

    pub(super) fn tiled_ifd_offsets_and_byte_counts(
        &self,
        ifd_id: IfdId,
    ) -> Result<(&[u64], &[u64]), WsiError> {
        let offsets = self
            .container
            .get_u64_array(ifd_id, tags::TILE_OFFSETS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let byte_counts = self
            .container
            .get_u64_array(ifd_id, tags::TILE_BYTE_COUNTS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        Ok((offsets, byte_counts))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tiled_ifd_tile_index(
        &self,
        ifd_id: IfdId,
        tile_idx: usize,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        width: u32,
        height: u32,
        offsets: &[u64],
        byte_counts: &[u64],
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
            return Err(WsiError::UnsupportedFormat(format!(
                "tile index {} out of range (offsets={}, byte_counts={})",
                tile_idx,
                offsets.len(),
                byte_counts.len(),
            )));
        }

        let offset = offsets[tile_idx];
        let byte_count = byte_counts[tile_idx];
        if byte_count == 0 {
            let pixel_count = (width * height * 3) as usize;
            return Ok(CpuTile {
                width,
                height,
                channels: 3,
                color_space: ColorSpace::Rgb,
                layout: CpuTileLayout::Interleaved,
                data: CpuTileData::u8(vec![0u8; pixel_count]),
            });
        }

        let tile_data = self
            .container
            .pread(offset, byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        match compression {
            Compression::Jpeg => {
                self.decode_tiled_ifd_jpeg_tile_data(ifd_id, jpeg_tables, &tile_data, width, height)
            }
            Compression::Jp2kRgb => decode_one_jp2k(Jp2kDecodeJob {
                data: Cow::Borrowed(&tile_data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: true,
                backend,
            }),
            Compression::Jp2kYcbcr => decode_one_jp2k(Jp2kDecodeJob {
                data: Cow::Borrowed(&tile_data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: false,
                backend,
            }),
            Compression::None => {
                // Uncompressed: interpret raw bytes using TIFF metadata
                self.decode_uncompressed_tile(ifd_id, &tile_data, width, height)
            }
            Compression::Lzw | Compression::Deflate | Compression::Zstd => self
                .decode_compressed_tiff_tile_data(ifd_id, compression, &tile_data, width, height),
            other => Err(WsiError::UnsupportedFormat(format!(
                "unsupported TiledIfd compression: {:?}",
                other,
            ))),
        }
    }

    pub(super) fn decode_tiled_ifd_jpeg_tile_data(
        &self,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        tile_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        let options = self.tiff_jpeg_decode_options_for_data(ifd_id, false, tile_data, jpeg_tables);
        decode_one_jpeg(JpegDecodeJob {
            data: Cow::Borrowed(tile_data),
            tables: jpeg_tables.map(Cow::Borrowed),
            expected_width: width,
            expected_height: height,
            color_transform: options.color_transform,
            force_dimensions: options.force_dimensions,
            requested_size: None,
        })
    }

    pub(super) fn read_tiled_ifd_raw_jpeg_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
    ) -> Result<RawCompressedTile, WsiError> {
        let (tile_idx, _, _) = self.tiled_ifd_tile_index_and_dimensions(req, ifd_id)?;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile index {} out of range (offsets={}, byte_counts={})",
                    tile_idx,
                    offsets.len(),
                    byte_counts.len()
                ),
            });
        }
        let byte_count = byte_counts[tile_idx];
        if byte_count == 0 {
            return Err(WsiError::Unsupported {
                reason: "JPEG passthrough does not support empty TIFF tiles".into(),
            });
        }
        let tile_data = self
            .container
            .pread(offsets[tile_idx], byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let (data, info) = standalone_jpeg_frame(&tile_data, jpeg_tables)?;
        Ok(RawCompressedTile {
            compression: Compression::Jpeg,
            width: info.width,
            height: info.height,
            bits_allocated: info.bits_allocated,
            samples_per_pixel: info.samples_per_pixel,
            photometric_interpretation: info.photometric_interpretation,
            data,
        })
    }

    pub(super) fn read_tiled_ifd_raw_jp2k_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        compression: Compression,
    ) -> Result<RawCompressedTile, WsiError> {
        let (tile_idx, width, height) = self.tiled_ifd_tile_index_and_dimensions(req, ifd_id)?;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile index {} out of range (offsets={}, byte_counts={})",
                    tile_idx,
                    offsets.len(),
                    byte_counts.len()
                ),
            });
        }
        let byte_count = byte_counts[tile_idx];
        if byte_count == 0 {
            return Err(WsiError::Unsupported {
                reason: "J2K passthrough does not support empty TIFF tiles".into(),
            });
        }

        let data = self
            .container
            .pread(offsets[tile_idx], byte_count)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let samples_per_pixel = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(3);
        if samples_per_pixel == 0 || samples_per_pixel > u32::from(u16::MAX) {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "J2K passthrough requires samples per pixel to fit in u16, got {samples_per_pixel}"
                ),
            });
        }
        let bits_allocated = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8);
        if bits_allocated == 0 || bits_allocated > u32::from(u16::MAX) {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "J2K passthrough requires bits per sample to fit in u16, got {bits_allocated}"
                ),
            });
        }
        let photometric = self.container.get_u32(ifd_id, tags::PHOTOMETRIC).unwrap_or(
            match (compression, samples_per_pixel) {
                (_, 1) => 1,
                (Compression::Jp2kYcbcr, _) => 6,
                _ => 2,
            },
        );
        let photometric_interpretation = match samples_per_pixel {
            1 => EncodedTilePhotometricInterpretation::Monochrome2,
            3 => match compression {
                Compression::Jp2kRgb => EncodedTilePhotometricInterpretation::Rgb,
                Compression::Jp2kYcbcr => EncodedTilePhotometricInterpretation::YbrFull422,
                _ if photometric == 2 => EncodedTilePhotometricInterpretation::Rgb,
                _ if photometric == 6 => EncodedTilePhotometricInterpretation::YbrFull422,
                _ => {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "J2K passthrough does not support photometric interpretation {photometric}"
                        ),
                    });
                }
            },
            other => {
                return Err(WsiError::Unsupported {
                    reason: format!("J2K passthrough supports 1 or 3 samples, got {other}"),
                });
            }
        };

        Ok(RawCompressedTile {
            compression,
            width,
            height,
            bits_allocated: bits_allocated as u16,
            samples_per_pixel: samples_per_pixel as u16,
            photometric_interpretation,
            data,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_ndpi_raw_jpeg_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        jpeg_header: &[u8],
        mcu_starts_tag: u16,
        tiles_across: u32,
        tiles_down: u32,
        restart_interval: u16,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<RawCompressedTile, WsiError> {
        let (col, row) = validate_tile_coords(req.col, req.row, req.level)?;
        if col >= tiles_across || row >= tiles_down {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "NDPI raw JPEG tile ({},{}) out of range ({}x{})",
                    req.col, req.row, tiles_across, tiles_down
                ),
            });
        }

        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (virtual_tile_width, virtual_tile_height) = match level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } if virtual_tile_width > 0 && virtual_tile_height > 0 => {
                (virtual_tile_width, virtual_tile_height)
            }
            TileLayout::WholeLevel { .. } => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI raw JPEG passthrough requires nonzero WholeLevel virtual tile dimensions"
                        .into(),
                });
            }
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI raw JPEG passthrough expects WholeLevel tile layout".into(),
                });
            }
        };
        if !ndpi_restart_segments_align_to_rows(level_w, virtual_tile_width, restart_interval) {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "NDPI raw JPEG passthrough requires restart segments to align to image rows (level width {level_w}, virtual tile width {virtual_tile_width}, restart interval {restart_interval})"
                ),
            });
        }

        let payload = self.ndpi_jpeg_tile_payload(
            req,
            ifd_id,
            jpeg_header,
            mcu_starts_tag,
            tiles_across,
            tiles_down,
            strip_offset,
            strip_byte_count,
            NdpiStripKey {
                ifd_id,
                col,
                native_row: row,
            },
            virtual_tile_width,
            virtual_tile_height,
            u32::try_from(level_w).map_err(|_| WsiError::Unsupported {
                reason: "NDPI raw JPEG passthrough requires level width to fit in u32".into(),
            })?,
            u32::try_from(level_h).map_err(|_| WsiError::Unsupported {
                reason: "NDPI raw JPEG passthrough requires level height to fit in u32".into(),
            })?,
        )?;
        let mut data = payload.jpeg;
        patch_jpeg_sof0_dimensions(&mut data, virtual_tile_width, virtual_tile_height)?;
        let info = parse_baseline_jpeg_frame_info(&data)?;
        if info.width != virtual_tile_width || info.height != virtual_tile_height {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "NDPI raw JPEG passthrough SOF dimensions {}x{} do not match virtual tile {}x{}",
                    info.width, info.height, virtual_tile_width, virtual_tile_height
                ),
            });
        }

        Ok(RawCompressedTile {
            compression: Compression::Jpeg,
            width: info.width,
            height: info.height,
            bits_allocated: info.bits_allocated,
            samples_per_pixel: info.samples_per_pixel,
            photometric_interpretation: info.photometric_interpretation,
            data,
        })
    }

    pub(super) fn empty_rgb_tile(width: u32, height: u32) -> CpuTile {
        let pixel_count = (width * height * 3) as usize;
        CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(vec![0u8; pixel_count]),
        }
    }

    pub(super) fn tiled_ifd_batch_compression(
        &self,
        reqs: &[TileRequest],
    ) -> Result<Option<Compression>, WsiError> {
        let mut batch_compression = None;
        for req in reqs {
            let TileSource::TiledIfd { compression, .. } = self.tile_source_for(req)? else {
                return Ok(None);
            };
            if !matches!(
                compression,
                Compression::Jpeg | Compression::Jp2kRgb | Compression::Jp2kYcbcr
            ) {
                return Ok(None);
            }
            match batch_compression {
                Some(existing) if existing != *compression => return Ok(None),
                Some(_) => {}
                None => batch_compression = Some(*compression),
            }
        }
        Ok(batch_compression)
    }

    #[cfg(feature = "metal")]
    pub(super) fn ndpi_jpeg_batchable(&self, reqs: &[TileRequest]) -> Result<bool, WsiError> {
        if reqs.is_empty() {
            return Ok(false);
        }
        for req in reqs {
            if !matches!(self.tile_source_for(req)?, TileSource::NdpiJpeg { .. }) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn decode_tiled_ifd_mixed_batch(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
    ) -> Result<Option<Vec<CpuTile>>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } = source
            else {
                return Ok(None);
            };
            if !matches!(
                compression,
                Compression::Jpeg | Compression::Jp2kRgb | Compression::Jp2kYcbcr
            ) {
                return Ok(None);
            }

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return Ok(None);
            }
            let data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;

            let job = match compression {
                Compression::Jpeg => {
                    let options = self.tiff_jpeg_decode_options_for_data(
                        *ifd_id,
                        false,
                        &data,
                        jpeg_tables.as_deref(),
                    );
                    CodecBatchJob::Jpeg(JpegDecodeJob {
                        data: Cow::Owned(data),
                        tables: jpeg_tables.as_deref().map(Cow::Borrowed),
                        expected_width: width,
                        expected_height: height,
                        color_transform: options.color_transform,
                        force_dimensions: options.force_dimensions,
                        requested_size: None,
                    })
                }
                Compression::Jp2kRgb | Compression::Jp2kYcbcr => {
                    CodecBatchJob::Jp2k(Jp2kDecodeJob {
                        data: Cow::Owned(data),
                        expected_width: width,
                        expected_height: height,
                        rgb_color_space: matches!(compression, Compression::Jp2kRgb),
                        backend,
                    })
                }
                _ => unreachable!("filtered above"),
            };
            jobs.push(job);
        }

        decode_mixed_batch(jobs)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    pub(super) fn decode_tiled_ifd_jpeg_batch(
        &self,
        reqs: &[TileRequest],
        _backend: BackendRequest,
    ) -> Result<Vec<CpuTile>, WsiError> {
        reqs.par_iter()
            .map(|req| {
                let source = self.tile_source_for(req)?;
                let TileSource::TiledIfd {
                    ifd_id,
                    jpeg_tables,
                    compression: Compression::Jpeg,
                } = source
                else {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "JPEG tiled batch received a non-JPEG tile source".into(),
                    });
                };

                let (tile_idx, width, height) =
                    self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
                let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
                if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: format!(
                            "tile index {} out of range (offsets={}, byte_counts={})",
                            tile_idx,
                            offsets.len(),
                            byte_counts.len()
                        ),
                    });
                }

                let byte_count = byte_counts[tile_idx];
                if byte_count == 0 {
                    return Ok(Self::empty_rgb_tile(width, height));
                }

                let tile_data = self
                    .container
                    .pread(offsets[tile_idx], byte_count)
                    .map_err(|err| err.into_wsi_error(self.container.path()))?;
                let options = self.tiff_jpeg_decode_options_for_data(
                    *ifd_id,
                    false,
                    &tile_data,
                    jpeg_tables.as_deref(),
                );
                decode_one_jpeg(JpegDecodeJob {
                    data: Cow::Borrowed(&tile_data),
                    tables: jpeg_tables.as_deref().map(Cow::Borrowed),
                    expected_width: width,
                    expected_height: height,
                    color_transform: options.color_transform,
                    force_dimensions: options.force_dimensions,
                    requested_size: None,
                })
                .map_err(|err| match err {
                    WsiError::TileRead { .. } => err,
                    other => WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: other.to_string(),
                    },
                })
            })
            .collect()
    }

    #[cfg(feature = "metal")]
    pub(super) fn decode_tiled_ifd_jpeg_pixels(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
        require_device: bool,
        metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let jobs = self.collect_tiled_ifd_jpeg_jobs(reqs)?;
        decode_batch_jpeg_pixels(&jobs, backend, require_device, metal_sessions)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect()
    }

    #[cfg(feature = "metal")]
    pub(super) fn decode_ndpi_jpeg_pixels(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
        require_device: bool,
        metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                strip_offset,
                strip_byte_count,
                ..
            } = source
            else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NDPI JPEG device batch received a non-NDPI tile source".into(),
                });
            };
            jobs.push(self.ndpi_jpeg_decode_job(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *strip_offset,
                *strip_byte_count,
            )?);
        }
        decode_batch_jpeg_pixels(&jobs, backend, require_device, metal_sessions)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect()
    }

    #[cfg(feature = "metal")]
    pub(super) fn collect_tiled_ifd_jpeg_jobs<'a>(
        &'a self,
        reqs: &[TileRequest],
    ) -> Result<Vec<JpegDecodeJob<'a>>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression: Compression::Jpeg,
            } = source
            else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "JPEG tiled device batch received a non-JPEG tile source".into(),
                });
            };

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return Err(WsiError::Unsupported {
                    reason: "device backend not available for empty jpeg tile".into(),
                });
            }

            let tile_data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;
            let options = self.tiff_jpeg_decode_options_for_data(
                *ifd_id,
                false,
                &tile_data,
                jpeg_tables.as_deref(),
            );
            jobs.push(JpegDecodeJob {
                data: Cow::Owned(tile_data),
                tables: jpeg_tables.as_deref().map(Cow::Borrowed),
                expected_width: width,
                expected_height: height,
                color_transform: options.color_transform,
                force_dimensions: options.force_dimensions,
                requested_size: None,
            });
        }
        Ok(jobs)
    }

    #[cfg(feature = "metal")]
    pub(super) fn decode_tiled_ifd_jp2k_pixels(
        &self,
        reqs: &[TileRequest],
        compression: Compression,
        backend: BackendRequest,
        require_device: bool,
        metal_sessions: Option<&crate::output::metal::MetalBackendSessions>,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let mut jobs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                compression: actual_compression,
                ..
            } = source
            else {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "JP2K tiled device batch received a non-tiled tile source".into(),
                });
            };
            if *actual_compression != compression {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "JP2K tiled device batch received mixed compression".into(),
                });
            }

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return Err(WsiError::Unsupported {
                    reason: "device backend not available for empty jp2k tile".into(),
                });
            }
            let data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;
            jobs.push(Jp2kDecodeJob {
                data: Cow::Owned(data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: matches!(compression, Compression::Jp2kRgb),
                backend,
            });
        }

        decode_batch_jp2k_pixels(&jobs, require_device, metal_sessions)
            .into_iter()
            .zip(reqs.iter())
            .map(|(result, req)| {
                result.map_err(|err| WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: err.to_string(),
                })
            })
            .collect()
    }

    pub(super) fn read_stitched_level_tile(
        &self,
        req: &TileRequest,
        components: &[StitchedLevelComponent],
        direct_tiles: &HashMap<(i64, i64), usize>,
    ) -> Result<CpuTile, WsiError> {
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } = &level.tile_layout
        else {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "stitched level expects regular public tile layout".into(),
            });
        };

        if req.col < 0
            || req.row < 0
            || req.col >= *tiles_across as i64
            || req.row >= *tiles_down as i64
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile ({},{}) out of range ({}x{})",
                    req.col, req.row, tiles_across, tiles_down
                ),
            });
        }

        let public_x = req.col * i64::from(*tile_width);
        let public_y = req.row * i64::from(*tile_height);
        let out_width = (*tile_width)
            .min((level.dimensions.0 as u32).saturating_sub(req.col as u32 * *tile_width));
        let out_height = (*tile_height)
            .min((level.dimensions.1 as u32).saturating_sub(req.row as u32 * *tile_height));

        if let Some(tile) = self.try_read_stitched_level_direct_tile(
            req.col,
            req.row,
            public_x,
            public_y,
            out_width,
            out_height,
            components,
            direct_tiles,
        )? {
            return Ok(tile);
        }

        let mut out = vec![0u8; out_width as usize * out_height as usize * 3];
        let out_stride = out_width as usize * 3;

        for component in components {
            let comp_left = component.origin_x;
            let comp_top = component.origin_y;
            let comp_right = comp_left + component.width as i64;
            let comp_bottom = comp_top + component.height as i64;
            let tile_right = public_x + i64::from(out_width);
            let tile_bottom = public_y + i64::from(out_height);

            let inter_left = public_x.max(comp_left);
            let inter_top = public_y.max(comp_top);
            let inter_right = tile_right.min(comp_right);
            let inter_bottom = tile_bottom.min(comp_bottom);
            if inter_left >= inter_right || inter_top >= inter_bottom {
                continue;
            }

            let local_x = (inter_left - comp_left) as u32;
            let local_y = (inter_top - comp_top) as u32;
            let inter_width = (inter_right - inter_left) as u32;
            let inter_height = (inter_bottom - inter_top) as u32;
            let region = self.read_tiled_ifd_component_region(
                component,
                local_x,
                local_y,
                inter_width,
                inter_height,
            )?;

            let region_data = region.data.as_u8().ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "stitched component produced non-U8 data".into(),
            })?;
            let dst_x = (inter_left - public_x) as usize;
            let dst_y = (inter_top - public_y) as usize;
            let src_stride = inter_width as usize * 3;
            for row in 0..inter_height as usize {
                let src_off = row * src_stride;
                let dst_off = (dst_y + row) * out_stride + dst_x * 3;
                out[dst_off..dst_off + src_stride]
                    .copy_from_slice(&region_data[src_off..src_off + src_stride]);
            }
        }

        Ok(CpuTile {
            width: out_width,
            height: out_height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(out),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_read_stitched_level_direct_tile(
        &self,
        public_col: i64,
        public_row: i64,
        public_x: i64,
        public_y: i64,
        out_width: u32,
        out_height: u32,
        components: &[StitchedLevelComponent],
        direct_tiles: &HashMap<(i64, i64), usize>,
    ) -> Result<Option<CpuTile>, WsiError> {
        let tile_right = public_x + i64::from(out_width);
        let tile_bottom = public_y + i64::from(out_height);
        let component = if let Some(&index) = direct_tiles.get(&(public_col, public_row)) {
            components.get(index)
        } else {
            let mut covering_component: Option<&StitchedLevelComponent> = None;

            for component in components {
                let comp_left = component.origin_x;
                let comp_top = component.origin_y;
                let comp_right = comp_left + component.width as i64;
                let comp_bottom = comp_top + component.height as i64;

                let inter_left = public_x.max(comp_left);
                let inter_top = public_y.max(comp_top);
                let inter_right = tile_right.min(comp_right);
                let inter_bottom = tile_bottom.min(comp_bottom);
                if inter_left >= inter_right || inter_top >= inter_bottom {
                    continue;
                }
                if inter_left != public_x
                    || inter_top != public_y
                    || inter_right != tile_right
                    || inter_bottom != tile_bottom
                {
                    return Ok(None);
                }
                if covering_component.is_some() {
                    return Ok(None);
                }
                covering_component = Some(component);
            }
            covering_component
        };

        let Some(component) = component else {
            return Ok(None);
        };

        let local_x = (public_x - component.origin_x) as u32;
        let local_y = (public_y - component.origin_y) as u32;
        let tile_col = local_x / component.tile_width;
        let tile_row = local_y / component.tile_height;
        if u64::from(tile_col) >= component.tiles_across
            || u64::from(tile_row) >= component.tiles_down
        {
            return Ok(None);
        }

        let tile_left = tile_col.saturating_mul(component.tile_width);
        let tile_top = tile_row.saturating_mul(component.tile_height);
        let decoded_width = component.tile_width.min(
            (component.width as u32).saturating_sub(tile_col.saturating_mul(component.tile_width)),
        );
        let decoded_height = component.tile_height.min(
            (component.height as u32)
                .saturating_sub(tile_row.saturating_mul(component.tile_height)),
        );
        if local_x < tile_left
            || local_y < tile_top
            || local_x.saturating_add(out_width) > tile_left.saturating_add(decoded_width)
            || local_y.saturating_add(out_height) > tile_top.saturating_add(decoded_height)
        {
            return Ok(None);
        }

        let tile_idx =
            (u64::from(tile_row) * component.tiles_across + u64::from(tile_col)) as usize;
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(component.ifd_id)?;
        let tile = self.get_or_decode_stitched_component_tile(
            component.ifd_id,
            tile_idx,
            component.jpeg_tables.as_deref(),
            component.compression,
            decoded_width,
            decoded_height,
            offsets,
            byte_counts,
        )?;
        let crop_x = local_x - tile_left;
        let crop_y = local_y - tile_top;
        if crop_x == 0 && crop_y == 0 && decoded_width == out_width && decoded_height == out_height
        {
            Ok(Some(tile.as_ref().clone()))
        } else {
            Ok(Some(crop_rgb_interleaved_u8_buffer(
                tile.as_ref(),
                crop_x,
                crop_y,
                out_width,
                out_height,
            )?))
        }
    }

    pub(super) fn read_tiled_ifd_component_region(
        &self,
        component: &StitchedLevelComponent,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        #[derive(Clone, Copy)]
        struct ComponentTileJob {
            decoded_width: u32,
            decoded_height: u32,
            tile_origin_x: u32,
            tile_origin_y: u32,
            inter_left: u32,
            inter_top: u32,
            inter_right: u32,
            inter_bottom: u32,
        }

        let offsets = self
            .container
            .get_u64_array(component.ifd_id, tags::TILE_OFFSETS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let byte_counts = self
            .container
            .get_u64_array(component.ifd_id, tags::TILE_BYTE_COUNTS)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;

        let mut out = vec![0u8; width as usize * height as usize * 3];
        let out_stride = width as usize * 3;
        let tile_width = component.tile_width;
        let tile_height = component.tile_height;
        let start_col = x / tile_width;
        let end_col = (x + width - 1) / tile_width;
        let start_row = y / tile_height;
        let end_row = (y + height - 1) / tile_height;
        let mut tile_jobs = Vec::with_capacity(
            ((end_col - start_col + 1) as usize).saturating_mul((end_row - start_row + 1) as usize),
        );

        for row in start_row..=end_row {
            for col in start_col..=end_col {
                if u64::from(col) >= component.tiles_across
                    || u64::from(row) >= component.tiles_down
                {
                    continue;
                }

                let decoded_width = tile_width
                    .min((component.width as u32).saturating_sub(col.saturating_mul(tile_width)));
                let decoded_height = tile_height
                    .min((component.height as u32).saturating_sub(row.saturating_mul(tile_height)));
                let tile_origin_x = col * tile_width;
                let tile_origin_y = row * tile_height;
                let inter_left = x.max(tile_origin_x);
                let inter_top = y.max(tile_origin_y);
                let inter_right = (x + width).min(tile_origin_x + decoded_width);
                let inter_bottom = (y + height).min(tile_origin_y + decoded_height);
                if inter_left >= inter_right || inter_top >= inter_bottom {
                    continue;
                }

                tile_jobs.push(ComponentTileJob {
                    decoded_width,
                    decoded_height,
                    tile_origin_x,
                    tile_origin_y,
                    inter_left,
                    inter_top,
                    inter_right,
                    inter_bottom,
                });
            }
        }

        let decoded_tiles: Vec<_> = if tile_jobs.len() <= 1 {
            tile_jobs
                .into_iter()
                .map(|job| {
                    let tile_idx = (u64::from(job.tile_origin_y / tile_height)
                        * component.tiles_across
                        + u64::from(job.tile_origin_x / tile_width))
                        as usize;
                    let tile = self.get_or_decode_stitched_component_tile(
                        component.ifd_id,
                        tile_idx,
                        component.jpeg_tables.as_deref(),
                        component.compression,
                        job.decoded_width,
                        job.decoded_height,
                        offsets,
                        byte_counts,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        } else {
            tile_jobs
                .into_par_iter()
                .map(|job| {
                    let tile_idx = (u64::from(job.tile_origin_y / tile_height)
                        * component.tiles_across
                        + u64::from(job.tile_origin_x / tile_width))
                        as usize;
                    let tile = self.get_or_decode_stitched_component_tile(
                        component.ifd_id,
                        tile_idx,
                        component.jpeg_tables.as_deref(),
                        component.compression,
                        job.decoded_width,
                        job.decoded_height,
                        offsets,
                        byte_counts,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        };

        for (job, tile) in decoded_tiles {
            let tile_data = tile.data.as_u8().ok_or_else(|| {
                WsiError::DisplayConversion(
                    "stitched Leica level requires interleaved U8 RGB tiles".into(),
                )
            })?;

            let src_x = (job.inter_left - job.tile_origin_x) as usize;
            let src_y = (job.inter_top - job.tile_origin_y) as usize;
            let dst_x = (job.inter_left - x) as usize;
            let dst_y = (job.inter_top - y) as usize;
            let copy_width = (job.inter_right - job.inter_left) as usize;
            let copy_height = (job.inter_bottom - job.inter_top) as usize;
            let src_stride = job.decoded_width as usize * 3;

            for copy_row in 0..copy_height {
                let src_off = (src_y + copy_row) * src_stride + src_x * 3;
                let dst_off = (dst_y + copy_row) * out_stride + dst_x * 3;
                let len = copy_width * 3;
                out[dst_off..dst_off + len].copy_from_slice(&tile_data[src_off..src_off + len]);
            }
        }

        Ok(CpuTile {
            width,
            height,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(out),
        })
    }

    pub(super) fn read_tiled_associated_image(
        &self,
        name: &str,
        ifd_id: IfdId,
        jpeg_tables: Option<&[u8]>,
        compression: Compression,
        dimensions: (u32, u32),
    ) -> Result<CpuTile, WsiError> {
        #[derive(Clone, Copy)]
        struct AssociatedTileJob {
            tile_w: u32,
            tile_h: u32,
            dest_x: u32,
            dest_y: u32,
            tile_idx: usize,
        }

        let tile_width = self
            .container
            .get_u32(ifd_id, tags::TILE_WIDTH)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        let tile_height = self
            .container
            .get_u32(ifd_id, tags::TILE_LENGTH)
            .map_err(|e| e.into_wsi_error(self.container.path()))?;
        if tile_width == 0 || tile_height == 0 {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has invalid tile size {}x{}",
                name, tile_width, tile_height,
            )));
        }

        let tiles_across = dimensions.0.div_ceil(tile_width);
        let tiles_down = dimensions.1.div_ceil(tile_height);
        let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(ifd_id)?;
        let required_tiles = tiles_across as usize * tiles_down as usize;
        if offsets.len() < required_tiles || byte_counts.len() < required_tiles {
            return Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' expected {} tiles, found offsets={} byte_counts={}",
                name,
                required_tiles,
                offsets.len(),
                byte_counts.len(),
            )));
        }

        let mut composed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * 3];
        let composed_stride = dimensions.0 as usize * 3;
        let mut tile_jobs = Vec::with_capacity(tiles_across as usize * tiles_down as usize);
        for row in 0..tiles_down {
            for col in 0..tiles_across {
                tile_jobs.push(AssociatedTileJob {
                    tile_w: tile_width.min(dimensions.0.saturating_sub(col * tile_width)),
                    tile_h: tile_height.min(dimensions.1.saturating_sub(row * tile_height)),
                    dest_x: col * tile_width,
                    dest_y: row * tile_height,
                    tile_idx: (row * tiles_across + col) as usize,
                });
            }
        }

        let decoded_tiles: Vec<_> = if tile_jobs.len() <= 1 {
            tile_jobs
                .into_iter()
                .map(|job| {
                    let tile = self.decode_tiled_ifd_tile_index(
                        ifd_id,
                        job.tile_idx,
                        jpeg_tables,
                        compression,
                        job.tile_w,
                        job.tile_h,
                        offsets,
                        byte_counts,
                        BackendRequest::Auto,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        } else {
            tile_jobs
                .into_par_iter()
                .map(|job| {
                    let tile = self.decode_tiled_ifd_tile_index(
                        ifd_id,
                        job.tile_idx,
                        jpeg_tables,
                        compression,
                        job.tile_w,
                        job.tile_h,
                        offsets,
                        byte_counts,
                        BackendRequest::Auto,
                    )?;
                    Ok((job, tile))
                })
                .collect::<Result<_, WsiError>>()?
        };

        for (job, tile) in decoded_tiles {
            match (&tile.data, tile.layout, tile.channels, &tile.color_space) {
                (CpuTileData::U8(tile_rgb), CpuTileLayout::Interleaved, 3, ColorSpace::Rgb) => {
                    let tile_src_stride = job.tile_w as usize * 3;
                    for y in 0..job.tile_h as usize {
                        let src_row = y * tile_src_stride;
                        let dst_row =
                            (job.dest_y as usize + y) * composed_stride + job.dest_x as usize * 3;
                        composed[dst_row..dst_row + tile_src_stride]
                            .copy_from_slice(&tile_rgb[src_row..src_row + tile_src_stride]);
                    }
                }
                _ => {
                    let tile_rgba = tile.to_rgba()?;
                    let tile_rgba_raw = tile_rgba.as_raw();
                    let tile_src_stride = job.tile_w as usize * 4;
                    for y in 0..job.tile_h as usize {
                        let src_row = y * tile_src_stride;
                        let dst_row =
                            (job.dest_y as usize + y) * composed_stride + job.dest_x as usize * 3;
                        let src_pixels = &tile_rgba_raw[src_row..src_row + tile_src_stride];
                        let dst_pixels = &mut composed[dst_row..dst_row + job.tile_w as usize * 3];
                        for (src_px, dst_px) in src_pixels
                            .chunks_exact(4)
                            .zip(dst_pixels.chunks_exact_mut(3))
                        {
                            dst_px.copy_from_slice(&src_px[..3]);
                        }
                    }
                }
            }
        }

        Ok(CpuTile {
            width: dimensions.0,
            height: dimensions.1,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(composed),
        })
    }

    /// Decode an uncompressed TIFF tile using IFD metadata.
    pub(super) fn decode_uncompressed_tile(
        &self,
        ifd_id: IfdId,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        use crate::formats::tiff_family::container::Endian;

        // Resolve TIFF metadata from container
        let spp = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(1);
        let bps_val = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8);
        // Tag 339 SAMPLE_FORMAT: 1=unsigned int (default), 2=signed int, 3=float
        let sample_format = self.container.get_u32(ifd_id, 339).unwrap_or(1);
        // Tag 262 PHOTOMETRIC: 0=MinIsWhite, 1=MinIsBlack, 2=RGB, 3=Palette, 6=YCbCr.
        // When the tag is absent, prefer grayscale for single-sample images and
        // RGB otherwise. Real NDPI associated thumbnails omit PHOTOMETRIC while
        // still storing 8-bit grayscale strips.
        let photometric = self
            .container
            .get_u32(ifd_id, tags::PHOTOMETRIC)
            .unwrap_or(if spp == 1 { 1 } else { 2 });
        // Tag 284 PLANAR_CONFIGURATION: 1=chunky (default), 2=planar
        let planar = self.container.get_u32(ifd_id, 284).unwrap_or(1);

        let endian = self.container.endian();

        if planar == 2 {
            return Err(WsiError::UnsupportedFormat(
                "planar TIFF tiles not supported".into(),
            ));
        }

        let effective_photometric = if spp == 1 && photometric == 2 {
            1
        } else {
            photometric
        };

        // Determine sample type and color space. Some NDPI associated images
        // report RGB photometric with a single 8-bit sample plane; treat those
        // contradictory tags as grayscale because the byte layout is 1 channel.
        let (sample_type, color_space) = match (bps_val, sample_format, spp, effective_photometric)
        {
            (8, 1, 3, 2) => (SampleType::Uint8, ColorSpace::Rgb), // RGB u8
            (8, 1, 1, 0) => (SampleType::Uint8, ColorSpace::Grayscale), // MinIsWhite (inverted below)
            (8, 1, 1, 1) => (SampleType::Uint8, ColorSpace::Grayscale), // MinIsBlack
            (8, 1, 3, 6) => (SampleType::Uint8, ColorSpace::YCbCr),     // YCbCr u8
            (16, 1, 1, 0) | (16, 1, 1, 1) => (SampleType::Uint16, ColorSpace::Grayscale),
            (16, 1, 3, 2) => (SampleType::Uint16, ColorSpace::Rgb), // RGB u16
            (32, 3, 1, _) => (SampleType::Float32, ColorSpace::Grayscale), // Float32 grayscale
            _ => {
                return Err(WsiError::UnsupportedFormat(format!(
                    "unsupported uncompressed format: bps={}, format={}, spp={}, photometric={}",
                    bps_val, sample_format, spp, photometric,
                )));
            }
        };

        let expected_bytes =
            width as usize * height as usize * spp as usize * sample_type.byte_size();
        if data.len() < expected_bytes {
            return Err(WsiError::TileRead {
                col: 0,
                row: 0,
                level: 0,
                reason: format!(
                    "uncompressed tile data too short: {} < {}",
                    data.len(),
                    expected_bytes,
                ),
            });
        }

        let sample_data = match sample_type {
            SampleType::Uint8 => {
                let mut bytes = data[..expected_bytes].to_vec();
                // MinIsWhite: invert grayscale values
                if effective_photometric == 0 {
                    for b in &mut bytes {
                        *b = 255 - *b;
                    }
                }
                CpuTileData::u8(bytes)
            }
            SampleType::Uint16 => {
                let mut samples: Vec<u16> = data[..expected_bytes]
                    .chunks_exact(2)
                    .map(|c| match endian {
                        Endian::Little => u16::from_le_bytes([c[0], c[1]]),
                        Endian::Big => u16::from_be_bytes([c[0], c[1]]),
                    })
                    .collect();
                // MinIsWhite: invert
                if effective_photometric == 0 {
                    for s in &mut samples {
                        *s = u16::MAX - *s;
                    }
                }
                CpuTileData::u16(samples)
            }
            SampleType::Float32 => {
                let samples: Vec<f32> = data[..expected_bytes]
                    .chunks_exact(4)
                    .map(|c| match endian {
                        Endian::Little => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                        Endian::Big => f32::from_be_bytes([c[0], c[1], c[2], c[3]]),
                    })
                    .collect();
                CpuTileData::f32(samples)
            }
        };

        // After MinIsWhite inversion, report as standard Grayscale
        // (the inversion already happened in the sample data)
        if effective_photometric == 0 && color_space == ColorSpace::Grayscale {
            // Already inverted above — color_space stays Grayscale
        }

        Ok(CpuTile {
            width,
            height,
            channels: spp as u16,
            color_space,
            layout: CpuTileLayout::Interleaved,
            data: sample_data,
        })
    }

    pub(super) fn expected_uncompressed_tile_bytes(
        &self,
        ifd_id: IfdId,
        width: u32,
        height: u32,
    ) -> Result<usize, WsiError> {
        let spp = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(1);
        let bps = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8);
        if bps == 0 || !bps.is_multiple_of(8) {
            return Err(WsiError::UnsupportedFormat(format!(
                "unsupported compressed TIFF bits per sample: {bps}"
            )));
        }
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|value| value.checked_mul(spp as usize))
            .and_then(|value| value.checked_mul((bps / 8) as usize))
            .ok_or_else(|| WsiError::UnsupportedFormat("compressed TIFF tile size overflow".into()))
    }

    pub(super) fn decompress_tiff_payload(
        &self,
        ifd_id: IfdId,
        compression: Compression,
        input: &[u8],
        expected_bytes: usize,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, WsiError> {
        let mut out = vec![0_u8; expected_bytes];
        let written = match compression {
            Compression::Lzw => {
                let mut pool = LzwPool::new();
                LzwCodec::decompress_into(&mut pool, input, &mut out)
            }
            Compression::Deflate => {
                let mut pool = DeflatePool::new();
                DeflateCodec::decompress_into(&mut pool, input, &mut out)
            }
            Compression::Zstd => {
                let mut pool = ZstdPool::new();
                ZstdCodec::decompress_into(&mut pool, input, &mut out)
            }
            other => {
                return Err(WsiError::UnsupportedFormat(format!(
                    "compression {:?} is not a tilecodec payload",
                    other
                )));
            }
        }
        .map_err(|err| WsiError::Codec {
            codec: match compression {
                Compression::Lzw => "tiff-lzw",
                Compression::Deflate => "tiff-deflate",
                Compression::Zstd => "tiff-zstd",
                _ => "tiff-tilecodec",
            },
            source: Box::new(err),
        })?;
        out.truncate(written);
        self.apply_tiff_predictor(ifd_id, width, height, &mut out)?;
        Ok(out)
    }

    pub(super) fn apply_tiff_predictor(
        &self,
        ifd_id: IfdId,
        width: u32,
        height: u32,
        data: &mut [u8],
    ) -> Result<(), WsiError> {
        use crate::formats::tiff_family::container::Endian;

        let predictor = self.container.get_u32(ifd_id, tags::PREDICTOR).unwrap_or(1);
        if predictor == 1 {
            return Ok(());
        }
        if predictor != 2 {
            return Err(WsiError::UnsupportedFormat(format!(
                "unsupported TIFF predictor: {predictor}"
            )));
        }

        let spp = self
            .container
            .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
            .unwrap_or(1) as usize;
        let bps = self
            .container
            .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
            .unwrap_or(8) as usize;
        let width = width as usize;
        let height = height as usize;
        if width == 0 || height == 0 || spp == 0 {
            return Ok(());
        }

        match bps {
            8 => {
                let row_stride = width.checked_mul(spp).ok_or_else(|| {
                    WsiError::UnsupportedFormat("TIFF predictor row stride overflow".into())
                })?;
                if data.len() < row_stride.saturating_mul(height) {
                    return Err(WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "TIFF predictor payload is shorter than expected".into(),
                    });
                }
                for row in data.chunks_exact_mut(row_stride).take(height) {
                    for idx in spp..row_stride {
                        let prior = row[idx - spp];
                        row[idx] = row[idx].wrapping_add(prior);
                    }
                }
                Ok(())
            }
            16 => {
                let row_samples = width.checked_mul(spp).ok_or_else(|| {
                    WsiError::UnsupportedFormat("TIFF predictor row sample overflow".into())
                })?;
                let row_stride = row_samples.checked_mul(2).ok_or_else(|| {
                    WsiError::UnsupportedFormat("TIFF predictor row stride overflow".into())
                })?;
                if data.len() < row_stride.saturating_mul(height) {
                    return Err(WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "TIFF predictor payload is shorter than expected".into(),
                    });
                }
                for row in data.chunks_exact_mut(row_stride).take(height) {
                    for sample_idx in spp..row_samples {
                        let byte_idx = sample_idx * 2;
                        let prior_idx = (sample_idx - spp) * 2;
                        let current = match self.container.endian() {
                            Endian::Little => {
                                u16::from_le_bytes([row[byte_idx], row[byte_idx + 1]])
                            }
                            Endian::Big => u16::from_be_bytes([row[byte_idx], row[byte_idx + 1]]),
                        };
                        let prior = match self.container.endian() {
                            Endian::Little => {
                                u16::from_le_bytes([row[prior_idx], row[prior_idx + 1]])
                            }
                            Endian::Big => u16::from_be_bytes([row[prior_idx], row[prior_idx + 1]]),
                        };
                        let value = current.wrapping_add(prior);
                        let bytes = match self.container.endian() {
                            Endian::Little => value.to_le_bytes(),
                            Endian::Big => value.to_be_bytes(),
                        };
                        row[byte_idx..byte_idx + 2].copy_from_slice(&bytes);
                    }
                }
                Ok(())
            }
            _ => Err(WsiError::UnsupportedFormat(format!(
                "unsupported TIFF predictor bits per sample: {bps}"
            ))),
        }
    }

    pub(super) fn decode_compressed_tiff_tile_data(
        &self,
        ifd_id: IfdId,
        compression: Compression,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CpuTile, WsiError> {
        let expected_bytes = self.expected_uncompressed_tile_bytes(ifd_id, width, height)?;
        let decoded = self.decompress_tiff_payload(
            ifd_id,
            compression,
            input,
            expected_bytes,
            width,
            height,
        )?;
        self.decode_uncompressed_tile(ifd_id, &decoded, width, height)
    }

    /// Read a tile from an NdpiFullDecode source (full JPEG decode fallback).
    ///
    /// Decodes the entire JPEG strip and extracts the requested tile region.
    /// Caches the decoded image in FullDecodeCache for subsequent tile requests
    /// from the same level. Oversize images (larger than cache max) are decoded
    /// per-request without caching.
    pub(super) fn read_ndpi_full_decode_tile(
        &self,
        req: &TileRequest,
        ifd_id: IfdId,
        _jpeg_header: &[u8],
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let full_image =
            self.get_or_decode_ndpi_full_image(req, ifd_id, strip_offset, strip_byte_count)?;

        // Extract the requested tile from the full image
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = level.dimensions;
        let (vtw, vth) = match &level.tile_layout {
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            _ => {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "NdpiFullDecode expects WholeLevel tile layout".into(),
                });
            }
        };

        let (col_u32, row_u32) = validate_tile_coords(req.col, req.row, req.level)?;
        let src_x = col_u32 * vtw;
        let src_y = row_u32 * vth;
        let tile_w = vtw.min((level_w as u32).saturating_sub(src_x));
        let tile_h = vth.min((level_h as u32).saturating_sub(src_y));

        if tile_w == 0 || tile_h == 0 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "tile has zero dimensions".into(),
            });
        }

        // Extract sub-region from the full interleaved RGB image
        let full_w = full_image.width as usize;
        let channels = full_image.channels as usize;
        let src_data = full_image.data.as_u8().ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "expected U8 data in full decode cache".into(),
        })?;

        let mut tile_data = Vec::with_capacity(tile_w as usize * tile_h as usize * channels);
        for y in 0..tile_h {
            let row_start = ((src_y + y) as usize * full_w + src_x as usize) * channels;
            let row_end = row_start + tile_w as usize * channels;
            if row_end > src_data.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "decoded image smaller than expected".into(),
                });
            }
            tile_data.extend_from_slice(&src_data[row_start..row_end]);
        }

        Ok(CpuTile {
            width: tile_w,
            height: tile_h,
            channels: full_image.channels,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(tile_data),
        })
    }

    pub(super) fn read_ndpi_full_display_tile(
        &self,
        req: &TileViewRequest,
        ifd_id: IfdId,
        strip_offset: u64,
        strip_byte_count: u64,
    ) -> Result<CpuTile, WsiError> {
        let tile_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: req.level,
            plane: req.plane,
            col: req.col,
            row: req.row,
        };
        let full_image =
            self.get_or_decode_ndpi_full_image(&tile_req, ifd_id, strip_offset, strip_byte_count)?;
        let level =
            &self.layout.dataset.scenes[req.scene].series[req.series].levels[req.level as usize];
        let (level_w, level_h) = (level.dimensions.0 as i64, level.dimensions.1 as i64);
        let tile_origin_x = req.col.saturating_mul(i64::from(req.tile_width));
        let tile_origin_y = req.row.saturating_mul(i64::from(req.tile_height));
        if tile_origin_x < 0
            || tile_origin_y < 0
            || tile_origin_x >= level_w
            || tile_origin_y >= level_h
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "display tile origin out of bounds".into(),
            });
        }

        let tile_w = req.tile_width.min((level_w - tile_origin_x) as u32);
        let tile_h = req.tile_height.min((level_h - tile_origin_y) as u32);
        let full_w = full_image.width as usize;
        let channels = full_image.channels as usize;
        let src_data = full_image.data.as_u8().ok_or_else(|| WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "expected U8 data in full decode cache".into(),
        })?;

        let mut tile_data = Vec::with_capacity(tile_w as usize * tile_h as usize * channels);
        for y in 0..tile_h {
            let row_start =
                ((tile_origin_y as u32 + y) as usize * full_w + tile_origin_x as usize) * channels;
            let row_end = row_start + tile_w as usize * channels;
            if row_end > src_data.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: "decoded image smaller than expected".into(),
                });
            }
            tile_data.extend_from_slice(&src_data[row_start..row_end]);
        }

        Ok(CpuTile {
            width: tile_w,
            height: tile_h,
            channels: full_image.channels,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(tile_data),
        })
    }

    pub(super) fn read_tiles_cpu_with_backend(
        &self,
        reqs: &[TileRequest],
        backend: BackendRequest,
    ) -> Result<Vec<CpuTile>, WsiError> {
        if reqs.is_empty() {
            return Ok(Vec::new());
        }

        let first_source = self.tile_source_for(&reqs[0])?;
        if matches!(
            first_source,
            TileSource::TiledIfd {
                compression: Compression::Jpeg | Compression::Jp2kRgb | Compression::Jp2kYcbcr,
                ..
            }
        ) {
            if self.tiled_ifd_batch_compression(reqs)? == Some(Compression::Jpeg) {
                return self.decode_tiled_ifd_jpeg_batch(reqs, backend);
            }
            if let Some(tiles) = self.decode_tiled_ifd_mixed_batch(reqs, backend)? {
                return Ok(tiles);
            }
        }

        let mut decode_reqs = Vec::with_capacity(reqs.len());
        for req in reqs {
            let source = self.tile_source_for(req)?;
            let TileSource::TiledIfd {
                ifd_id,
                compression,
                ..
            } = source
            else {
                return reqs
                    .iter()
                    .map(|req| self.read_tile_cpu_with_backend_request(req, backend))
                    .collect();
            };
            let colorspace = match compression {
                Compression::Jp2kRgb => Jp2kColorSpace::Rgb,
                Compression::Jp2kYcbcr => Jp2kColorSpace::YCbCr,
                _ => {
                    return reqs
                        .iter()
                        .map(|req| self.read_tile_cpu_with_backend_request(req, backend))
                        .collect();
                }
            };

            let (tile_idx, width, height) =
                self.tiled_ifd_tile_index_and_dimensions(req, *ifd_id)?;
            let (offsets, byte_counts) = self.tiled_ifd_offsets_and_byte_counts(*ifd_id)?;
            if tile_idx >= offsets.len() || tile_idx >= byte_counts.len() {
                return Err(WsiError::TileRead {
                    col: req.col,
                    row: req.row,
                    level: req.level,
                    reason: format!(
                        "tile index {} out of range (offsets={}, byte_counts={})",
                        tile_idx,
                        offsets.len(),
                        byte_counts.len()
                    ),
                });
            }
            let byte_count = byte_counts[tile_idx];
            if byte_count == 0 {
                return reqs
                    .iter()
                    .map(|req| self.read_tile_cpu_with_backend_request(req, backend))
                    .collect();
            }
            let data = self
                .container
                .pread(offsets[tile_idx], byte_count)
                .map_err(|err| err.into_wsi_error(self.container.path()))?;
            decode_reqs.push(Jp2kDecodeJob {
                data: Cow::Owned(data),
                expected_width: width,
                expected_height: height,
                rgb_color_space: matches!(colorspace, Jp2kColorSpace::Rgb),
                backend,
            });
        }

        decode_batch_jp2k(&decode_reqs)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                let first = &reqs[0];
                WsiError::TileRead {
                    col: first.col,
                    row: first.row,
                    level: first.level,
                    reason: err.to_string(),
                }
            })
    }

    pub(super) fn read_tile_cpu_with_backend_request(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        match self.tile_source_for(req)? {
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => self.read_tiled_ifd_tile(
                req,
                *ifd_id,
                jpeg_tables.as_deref(),
                *compression,
                backend,
            ),
            _ => self.read_tile_cpu(req),
        }
    }
}

struct TiffPixelReaderNoSyntheticPrime<'a> {
    inner: &'a TiffPixelReader,
}

impl SlideReader for TiffPixelReaderNoSyntheticPrime<'_> {
    fn dataset(&self) -> &Dataset {
        &self.inner.layout.dataset
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        <TiffPixelReader as SlideReader>::read_tiles(self.inner, reqs, output)
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.inner.read_tile_cpu(req)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.inner.read_associated(name)
    }
}

impl SlideReader for TiffPixelReader {
    fn dataset(&self) -> &Dataset {
        let _ = self
            .synthetic_prime_once
            .get_or_init(|| self.prime_deepest_synthetic_levels_best_effort());
        &self.layout.dataset
    }

    fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        match self.tile_source_for(req) {
            Ok(TileSource::NdpiJpeg { .. } | TileSource::NdpiFullDecode { .. }) => {
                TileCodecKind::Jpeg
            }
            Ok(TileSource::TiledIfd { compression, .. }) => {
                TileCodecKind::from_compression(*compression)
            }
            Ok(TileSource::SyntheticDownsample { base_level, .. }) => {
                let mut base_req = req.clone();
                base_req.level = *base_level;
                self.tile_codec_kind(&base_req)
            }
            Ok(_) | Err(_) => TileCodecKind::Other,
        }
    }

    fn level_source_kind(
        &self,
        scene: usize,
        series: usize,
        level: u32,
    ) -> Result<LevelSourceKind, WsiError> {
        let scene_ref = self
            .layout
            .dataset
            .scenes
            .get(scene)
            .ok_or(WsiError::SceneOutOfRange {
                index: scene,
                count: self.layout.dataset.scenes.len(),
            })?;
        let series_ref = scene_ref
            .series
            .get(series)
            .ok_or(WsiError::SeriesOutOfRange {
                index: series,
                count: scene_ref.series.len(),
            })?;
        if level as usize >= series_ref.levels.len() {
            return Err(WsiError::LevelOutOfRange {
                level,
                count: series_ref.levels.len() as u32,
            });
        }

        let synthetic = self.layout.tile_sources.iter().any(|(key, source)| {
            key.scene == scene
                && key.series == series
                && key.level == level
                && matches!(source, TileSource::SyntheticDownsample { .. })
        });
        if synthetic {
            Ok(LevelSourceKind::SyntheticDownsample)
        } else {
            Ok(LevelSourceKind::Physical)
        }
    }

    fn read_raw_compressed_tile(&self, req: &TileRequest) -> Result<RawCompressedTile, WsiError> {
        match self.tile_source_for(req)? {
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression: Compression::Jpeg,
            } => self.read_tiled_ifd_raw_jpeg_tile(req, *ifd_id, jpeg_tables.as_deref()),
            TileSource::TiledIfd {
                ifd_id,
                compression: compression @ (Compression::Jp2kRgb | Compression::Jp2kYcbcr),
                ..
            } => self.read_tiled_ifd_raw_jp2k_tile(req, *ifd_id, *compression),
            TileSource::TiledIfd { compression, .. } => Err(WsiError::Unsupported {
                reason: format!(
                    "compressed passthrough requires TIFF JPEG or J2K compression, got {:?}",
                    compression
                ),
            }),
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                restart_interval,
                strip_offset,
                strip_byte_count,
                ..
            } => self.read_ndpi_raw_jpeg_tile(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *restart_interval,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::NdpiFullDecode { .. } => Err(WsiError::Unsupported {
                reason: "NDPI JPEG passthrough is not available for whole-level full-decode JPEG sources".into(),
            }),
            TileSource::SyntheticDownsample { .. } => Err(WsiError::Unsupported {
                reason: "JPEG passthrough is not available for synthetic downsample levels".into(),
            }),
            TileSource::StitchedLevel { .. } => Err(WsiError::Unsupported {
                reason: "JPEG passthrough is not available for stitched levels".into(),
            }),
            TileSource::Stripped { .. } | TileSource::ExternalJpeg { .. } => Err(WsiError::Unsupported {
                reason: "JPEG passthrough is only available for tiled image levels".into(),
            }),
        }
    }

    fn use_display_tile_cache(&self, req: &TileViewRequest) -> bool {
        let tile_req = TileRequest {
            scene: req.scene,
            series: req.series,
            level: req.level,
            plane: req.plane,
            col: req.col,
            row: req.row,
        };
        match self.tile_source_for(&tile_req) {
            Ok(
                TileSource::NdpiJpeg { .. }
                | TileSource::NdpiFullDecode { .. }
                | TileSource::SyntheticDownsample { .. },
            ) => false,
            Ok(TileSource::TiledIfd { .. }) => true,
            Ok(_) => true,
            Err(_) => true,
        }
    }

    fn read_region_fastpath(
        &self,
        ctx: &mut crate::core::registry::SlideReadContext<'_>,
        req: &RegionRequest,
    ) -> Option<Result<CpuTile, WsiError>> {
        let cache = ctx.tile_cache();
        let series = self
            .layout
            .dataset
            .scenes
            .get(req.scene.0)
            .and_then(|scene| scene.series.get(req.series.0))?;
        let level = series.levels.get(req.level.0 as usize)?;
        if !matches!(level.tile_layout, TileLayout::WholeLevel { .. }) {
            return None;
        }
        let plane = req.plane.0;

        let source = self.layout.tile_sources.get(&TileSourceKey {
            scene: req.scene.0,
            series: req.series.0,
            level: req.level.0,
            z: plane.z,
            c: plane.c,
            t: plane.t,
        })?;
        match source {
            TileSource::SyntheticDownsample { base_level, factor } => {
                Some(self.read_full_synthetic_region_fastpath(cache, req, *base_level, *factor))
            }
            _ => None,
        }
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        let source = self.tile_source_for(req)?;
        match source {
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                restart_interval,
                strip_offset,
                strip_byte_count,
            } => self.read_ndpi_restart_tile(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *restart_interval,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::NdpiFullDecode {
                ifd_id,
                jpeg_header,
                strip_offset,
                strip_byte_count,
            } => self.read_ndpi_full_decode_tile(
                req,
                *ifd_id,
                jpeg_header,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => self.read_tiled_ifd_tile(
                req,
                *ifd_id,
                jpeg_tables.as_deref(),
                *compression,
                BackendRequest::Auto,
            ),
            TileSource::StitchedLevel {
                components,
                direct_tiles,
            } => self.read_stitched_level_tile(req, components, direct_tiles),
            TileSource::SyntheticDownsample { base_level, factor } => {
                if req.col != 0 || req.row != 0 {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "synthetic NDPI whole-level tiles only support tile (0,0)".into(),
                    });
                }
                Ok(self
                    .get_or_decode_synthetic_level(req, *base_level, *factor)?
                    .as_ref()
                    .clone())
            }
            TileSource::Stripped { .. } => Err(WsiError::UnsupportedFormat(
                "Stripped pixel access via read_tile not supported; use read_associated()".into(),
            )),
            TileSource::ExternalJpeg { .. } => Err(WsiError::UnsupportedFormat(
                "External JPEG associated images cannot be read via read_tile()".into(),
            )),
        }
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let backend = output.backend().to_signinum();
        let require_device = output.requires_device();
        #[cfg(feature = "metal")]
        let prefer_device = output.prefers_device();
        #[cfg(feature = "metal")]
        let compressed_device_decode_enabled = output.compressed_device_decode_enabled();
        #[cfg(feature = "metal")]
        let metal_sessions = output.metal_sessions();

        #[cfg(feature = "metal")]
        if prefer_device && !reqs.is_empty() {
            if self.ndpi_jpeg_batchable(reqs)? {
                if compressed_device_decode_enabled || jpeg_device_decode_enabled() {
                    match self.decode_ndpi_jpeg_pixels(
                        reqs,
                        backend,
                        require_device,
                        metal_sessions,
                    ) {
                        Ok(tiles) => return Ok(tiles),
                        Err(err) if require_device => return Err(err),
                        Err(err) => {
                            tracing::debug!(
                                error = %err,
                                fallback_to_cpu = true,
                                fallback_reason = "ndpi_jpeg_device_decode_failed",
                                "NDPI JPEG device tile path failed; retrying through CPU output"
                            );
                        }
                    }
                } else if require_device {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "NDPI JPEG device decode is disabled; set {JPEG_DEVICE_DECODE_ENV}=1 or request compressed device decode to opt in"
                        ),
                    });
                }
            }

            let device_result = match self.tiled_ifd_batch_compression(reqs)? {
                Some(Compression::Jpeg)
                    if compressed_device_decode_enabled || jpeg_device_decode_enabled() =>
                {
                    Some(self.decode_tiled_ifd_jpeg_pixels(
                        reqs,
                        backend,
                        require_device,
                        metal_sessions,
                    ))
                }
                Some(Compression::Jpeg) if require_device => {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "JPEG device decode is disabled; set {JPEG_DEVICE_DECODE_ENV}=1 or request compressed device decode to opt in"
                        ),
                    });
                }
                Some(Compression::Jpeg) => None,
                Some(compression @ (Compression::Jp2kRgb | Compression::Jp2kYcbcr))
                    if compressed_device_decode_enabled || jp2k_device_decode_enabled() =>
                {
                    Some(self.decode_tiled_ifd_jp2k_pixels(
                        reqs,
                        compression,
                        backend,
                        require_device,
                        metal_sessions,
                    ))
                }
                Some(Compression::Jp2kRgb | Compression::Jp2kYcbcr) if require_device => {
                    return Err(WsiError::Unsupported {
                        reason: format!(
                            "JP2K device decode is disabled; set {JP2K_DEVICE_DECODE_ENV}=1 or request compressed device decode to opt in"
                        ),
                    });
                }
                Some(Compression::Jp2kRgb | Compression::Jp2kYcbcr) => None,
                _ if require_device => {
                    return Err(WsiError::Unsupported {
                        reason: "device backend not available for tiff_family".into(),
                    });
                }
                _ => None,
            };
            if let Some(result) = device_result {
                match result {
                    Ok(tiles) => return Ok(tiles),
                    Err(err) if require_device => return Err(err),
                    Err(err) => {
                        tracing::debug!(
                            error = %err,
                            fallback_to_cpu = true,
                            fallback_reason = "signinum_auto_chose_cpu",
                            "device tile path failed; retrying through CPU output"
                        );
                    }
                }
            }
        }

        #[cfg(not(feature = "metal"))]
        if require_device {
            return Err(WsiError::Unsupported {
                reason: "device backend not available for tiff_family".into(),
            });
        }

        self.read_tiles_cpu_with_backend(reqs, backend)
            .map(|tiles| tiles.into_iter().map(TilePixels::Cpu).collect())
    }

    fn read_tiles_cpu(&self, reqs: &[TileRequest]) -> Result<Vec<CpuTile>, WsiError> {
        self.read_tiles_cpu_with_backend(reqs, BackendRequest::Auto)
    }

    fn read_display_tile(&self, req: &TileViewRequest) -> Result<CpuTile, WsiError> {
        let source = self.tile_source_for(&TileRequest {
            scene: req.scene,
            series: req.series,
            level: req.level,
            plane: req.plane,
            col: req.col,
            row: req.row,
        })?;
        match source {
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag,
                tiles_across,
                tiles_down,
                strip_offset,
                strip_byte_count,
                ..
            } => self.read_ndpi_display_tile(
                req,
                *ifd_id,
                jpeg_header,
                *mcu_starts_tag,
                *tiles_across,
                *tiles_down,
                *strip_offset,
                *strip_byte_count,
            ),
            TileSource::NdpiFullDecode {
                ifd_id,
                strip_offset,
                strip_byte_count,
                ..
            } => self.read_ndpi_full_display_tile(req, *ifd_id, *strip_offset, *strip_byte_count),
            TileSource::SyntheticDownsample { base_level, factor } => {
                self.read_synthetic_display_tile(req, *base_level, *factor)
            }
            _ => read_display_tile_from_source(self, None, req, TileOutputPreference::cpu()),
        }
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let source = self
            .layout
            .associated_sources
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;

        match source {
            TileSource::Stripped {
                ifd_id,
                jpeg_tables,
                compression,
                strip_offsets,
                strip_byte_counts,
            } => {
                let info = self
                    .layout
                    .dataset
                    .associated_images
                    .get(name)
                    .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;

                match compression {
                    Compression::Jpeg => self.read_stripped_jpeg_image(
                        name,
                        *ifd_id,
                        jpeg_tables.as_deref(),
                        info.dimensions,
                        strip_offsets,
                        strip_byte_counts,
                    ),
                    Compression::None => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        self.decode_uncompressed_tile(
                            *ifd_id,
                            &data,
                            info.dimensions.0,
                            info.dimensions.1,
                        )
                    }
                    Compression::Lzw | Compression::Deflate | Compression::Zstd => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        let expected_bytes = self.expected_uncompressed_tile_bytes(
                            *ifd_id,
                            info.dimensions.0,
                            info.dimensions.1,
                        )?;
                        let decoded = self.decompress_tiff_payload(
                            *ifd_id,
                            *compression,
                            &data,
                            expected_bytes,
                            info.dimensions.0,
                            info.dimensions.1,
                        )?;
                        self.decode_uncompressed_tile(
                            *ifd_id,
                            &decoded,
                            info.dimensions.0,
                            info.dimensions.1,
                        )
                    }
                    Compression::Jp2kRgb => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        decode_one_jp2k(Jp2kDecodeJob {
                            data: Cow::Borrowed(&data),
                            expected_width: info.dimensions.0,
                            expected_height: info.dimensions.1,
                            rgb_color_space: true,
                            backend: BackendRequest::Auto,
                        })
                    }
                    Compression::Jp2kYcbcr => {
                        let data =
                            self.read_stripped_data(name, strip_offsets, strip_byte_counts)?;
                        decode_one_jp2k(Jp2kDecodeJob {
                            data: Cow::Borrowed(&data),
                            expected_width: info.dimensions.0,
                            expected_height: info.dimensions.1,
                            rgb_color_space: false,
                            backend: BackendRequest::Auto,
                        })
                    }
                    other => Err(WsiError::UnsupportedFormat(format!(
                        "associated image '{}' has unsupported compression {:?}",
                        name, other,
                    ))),
                }
            }
            TileSource::NdpiFullDecode {
                ifd_id,
                strip_offset,
                strip_byte_count,
                ..
            } => {
                let data = self
                    .container
                    .pread(*strip_offset, *strip_byte_count)
                    .map_err(|e| e.into_wsi_error(self.container.path()))?;

                let info = self
                    .layout
                    .dataset
                    .associated_images
                    .get(name)
                    .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;

                let options = signinum_decode_options(
                    self.tiff_jpeg_decode_options_for_data(*ifd_id, false, &data, None)
                        .color_transform,
                );
                let decoder = SigninumJpegDecoder::new_with_options(&data, options)
                    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
                let (pixels, outcome) = decoder
                    .decode(SigninumPixelFormat::Rgb8)
                    .map_err(|err| WsiError::Jpeg(err.to_string()))?;
                let decoded =
                    cpu_tile_from_rgb_pixels(outcome.decoded.w, outcome.decoded.h, pixels)?;
                if decoded.width > info.dimensions.0 || decoded.height > info.dimensions.1 {
                    crop_rgb_interleaved_u8_buffer(
                        &decoded,
                        0,
                        0,
                        info.dimensions.0,
                        info.dimensions.1,
                    )
                } else {
                    Ok(decoded)
                }
            }
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => {
                let info = self
                    .layout
                    .dataset
                    .associated_images
                    .get(name)
                    .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
                self.read_tiled_associated_image(
                    name,
                    *ifd_id,
                    jpeg_tables.as_deref(),
                    *compression,
                    info.dimensions,
                )
            }
            TileSource::ExternalJpeg { path } => {
                let data = std::fs::read(path).map_err(|err| WsiError::InvalidSlide {
                    path: path.clone(),
                    message: format!(
                        "failed to read external JPEG associated image '{}': {err}",
                        path.display()
                    ),
                })?;
                decode_one_jpeg(JpegDecodeJob {
                    data: Cow::Borrowed(&data),
                    tables: None,
                    expected_width: 0,
                    expected_height: 0,
                    color_transform: SigninumColorTransform::Auto,
                    force_dimensions: false,
                    requested_size: None,
                })
            }
            _ => Err(WsiError::UnsupportedFormat(format!(
                "associated image '{}' has unsupported source type",
                name,
            ))),
        }
    }

    fn recommended_shared_cache_bytes(&self) -> Option<u64> {
        self.layout
            .tile_sources
            .values()
            .any(|source| {
                matches!(
                    source,
                    TileSource::TiledIfd {
                        compression: Compression::Jp2kRgb | Compression::Jp2kYcbcr,
                        ..
                    }
                )
            })
            .then_some(DEFAULT_JP2K_SHARED_TILE_CACHE_BYTES)
    }
}
