use super::*;

fn validate_region_request<'a>(
    dataset: &'a Dataset,
    req: &RegionRequest,
) -> Result<(&'a Scene, &'a Series, &'a Level), WsiError> {
    if req.scene.0 >= dataset.scenes.len() {
        return Err(WsiError::SceneOutOfRange {
            index: req.scene.0,
            count: dataset.scenes.len(),
        });
    }
    let scene = &dataset.scenes[req.scene.0];

    if req.series.0 >= scene.series.len() {
        return Err(WsiError::SeriesOutOfRange {
            index: req.series.0,
            count: scene.series.len(),
        });
    }
    let series = &scene.series[req.series.0];

    if req.level.0 as usize >= series.levels.len() {
        return Err(WsiError::LevelOutOfRange {
            level: req.level.0,
            count: series.levels.len() as u32,
        });
    }
    let level = &series.levels[req.level.0 as usize];

    if req.plane.0.z >= series.axes.z {
        return Err(WsiError::PlaneOutOfRange {
            axis: "z".into(),
            value: req.plane.0.z,
            max: series.axes.z,
        });
    }
    if req.plane.0.c >= series.axes.c {
        return Err(WsiError::PlaneOutOfRange {
            axis: "c".into(),
            value: req.plane.0.c,
            max: series.axes.c,
        });
    }
    if req.plane.0.t >= series.axes.t {
        return Err(WsiError::PlaneOutOfRange {
            axis: "t".into(),
            value: req.plane.0.t,
            max: series.axes.t,
        });
    }

    Ok((scene, series, level))
}

pub(crate) fn composite_region_from_source<T: SlideReader + ?Sized>(
    source: &T,
    cache: Option<&TileCache>,
    req: &RegionRequest,
) -> Result<CpuTile, WsiError> {
    let dataset = source.dataset();
    let (_, series, level) = validate_region_request(dataset, req)?;
    let (x, y) = req.origin_px;
    let (w, h) = req.size_px;
    let plane = req.plane.0;

    let cache_key_for = |col: i64, row: i64| CacheKey {
        dataset_id: dataset.id,
        scene: req.scene.0 as u32,
        series: req.series.0 as u32,
        level: req.level.0,
        z: plane.z,
        c: plane.c,
        t: plane.t,
        tile_col: col,
        tile_row: row,
    };

    let tile_req_for = |col: i64, row: i64| TileRequest {
        scene: req.scene.0,
        series: req.series.0,
        level: req.level.0,
        plane,
        col,
        row,
    };

    let read_tile_cached = |col: i64, row: i64| -> Result<Arc<CpuTile>, WsiError> {
        let key = cache_key_for(col, row);

        if let Some(cache) = cache {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached);
            }
        }

        let tile = source.read_tile_cpu(&tile_req_for(col, row))?;
        let arc_tile = Arc::new(tile);
        if let Some(cache) = cache {
            cache.put(key, arc_tile.clone());
        }
        Ok(arc_tile)
    };

    let read_hit_tiles_cached = |hits: &[TileHit]| -> Result<Vec<Arc<CpuTile>>, WsiError> {
        let mut tiles = vec![None; hits.len()];
        let mut missed_slots = Vec::new();
        let mut missed_keys = Vec::new();
        let mut missed_reqs = Vec::new();

        for (slot, hit) in hits.iter().enumerate() {
            let key = cache_key_for(hit.col, hit.row);
            if let Some(cache) = cache {
                if let Some(cached) = cache.get(&key) {
                    tiles[slot] = Some(cached);
                    continue;
                }
            }
            missed_slots.push(slot);
            missed_keys.push(key);
            missed_reqs.push(tile_req_for(hit.col, hit.row));
        }

        if !missed_reqs.is_empty() {
            let decoded = if missed_reqs.len() == 1 {
                vec![source.read_tile_cpu(&missed_reqs[0])?]
            } else {
                source
                    .read_tiles(&missed_reqs, TileOutputPreference::cpu())?
                    .into_iter()
                    .map(|tile| match tile {
                        TilePixels::Cpu(cpu) => Ok(cpu),
                        TilePixels::Device(_) => Err(WsiError::Unsupported {
                            reason: "region composition requires CPU tiles".into(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            if decoded.len() != missed_reqs.len() {
                return Err(WsiError::TileRead {
                    col: missed_reqs.first().map_or(0, |req| req.col),
                    row: missed_reqs.first().map_or(0, |req| req.row),
                    level: req.level.0,
                    reason: format!(
                        "batched tile read returned {} tiles for {} requests",
                        decoded.len(),
                        missed_reqs.len()
                    ),
                });
            }

            for ((slot, key), tile) in missed_slots.into_iter().zip(missed_keys).zip(decoded) {
                let arc_tile = Arc::new(tile);
                if let Some(cache) = cache {
                    cache.put(key, arc_tile.clone());
                }
                tiles[slot] = Some(arc_tile);
            }
        }

        tiles
            .into_iter()
            .zip(hits.iter())
            .map(|(tile, hit)| {
                tile.ok_or_else(|| WsiError::TileRead {
                    col: hit.col,
                    row: hit.row,
                    level: req.level.0,
                    reason: "batched tile read did not populate requested tile".into(),
                })
            })
            .collect()
    };

    let hits = level.tile_layout.tiles_for_region(x, y, w, h);

    if hits.is_empty() {
        if let Some((probe_col, probe_row)) = metadata_probe_coordinate(&level.tile_layout) {
            if let Ok(template) = read_tile_cached(probe_col, probe_row) {
                return Ok(zero_sample_buffer_from_template(w, h, template.as_ref()));
            }
        }

        return Ok(zero_sample_buffer_from_series(w, h, series));
    }

    let hit_tiles = read_hit_tiles_cached(&hits)?;
    let first_tile = hit_tiles[0].clone();

    if first_tile.layout == CpuTileLayout::Planar {
        return Err(WsiError::DisplayConversion(
            "planar compositing not supported".into(),
        ));
    }

    let out_channels = first_tile.channels;
    let out_color_space = first_tile.color_space.clone();
    let out_layout = first_tile.layout;
    let out_w = w as usize;
    let out_h = h as usize;
    let region_pixels = w as u64 * h as u64;
    if region_pixels > DEFAULT_MAX_REGION_PIXELS {
        return Err(WsiError::DisplayConversion(format!(
            "region {}x{} ({} pixels) exceeds maximum of {} pixels",
            w, h, region_pixels, DEFAULT_MAX_REGION_PIXELS
        )));
    }
    let total_samples = out_w * out_h * out_channels as usize;
    let mut out_data = match &first_tile.data {
        CpuTileData::U8(_) => CpuTileData::u8(vec![0u8; total_samples]),
        CpuTileData::U16(_) => CpuTileData::u16(vec![0u16; total_samples]),
        CpuTileData::F32(_) => CpuTileData::f32(vec![0.0f32; total_samples]),
    };

    macro_rules! blit_tile {
        ($out_vec:expr, $tile_vec:expr, $tile:expr, $hit:expr) => {{
            let tw = $tile.width as i64;
            let th = $tile.height as i64;
            let ch = out_channels as usize;

            let src_x = (0i64).max(-$hit.dest_x) as usize;
            let src_y = (0i64).max(-$hit.dest_y) as usize;
            let dx = (0i64).max($hit.dest_x) as usize;
            let dy = (0i64).max($hit.dest_y) as usize;
            let copy_w = ((tw - src_x as i64) as usize).min(out_w - dx);
            let copy_h = ((th - src_y as i64) as usize).min(out_h - dy);
            let tile_row_stride = $tile.width as usize * ch;
            let out_row_stride = out_w * ch;

            for row in 0..copy_h {
                let src_off = (src_y + row) * tile_row_stride + src_x * ch;
                let dst_off = (dy + row) * out_row_stride + dx * ch;
                let len = copy_w * ch;
                $out_vec[dst_off..dst_off + len]
                    .copy_from_slice(&$tile_vec[src_off..src_off + len]);
            }
        }};
    }

    let needs_fractional_blit = |hit: &TileHit| {
        (hit.dest_x_f64 - hit.dest_x as f64).abs() > 1e-6
            || (hit.dest_y_f64 - hit.dest_y as f64).abs() > 1e-6
    };

    let mut alpha_buffer = matches!(&out_data, CpuTileData::U8(_))
        .then(|| hits.iter().any(needs_fractional_blit))
        .filter(|needed| *needed)
        .map(|_| vec![0.0f32; out_w * out_h]);

    let mark_tile_opaque = |alpha: &mut [f32], tile: &CpuTile, hit: &TileHit| {
        let tw = tile.width as i64;
        let th = tile.height as i64;
        let src_x = (0i64).max(-hit.dest_x) as usize;
        let src_y = (0i64).max(-hit.dest_y) as usize;
        let dx = (0i64).max(hit.dest_x) as usize;
        let dy = (0i64).max(hit.dest_y) as usize;
        let copy_w = ((tw - src_x as i64) as usize).min(out_w - dx);
        let copy_h = ((th - src_y as i64) as usize).min(out_h - dy);

        for row in 0..copy_h {
            let dst_off = (dy + row) * out_w + dx;
            alpha[dst_off..dst_off + copy_w].fill(1.0);
        }
    };

    let blit_tile_fractional_u8 = |out_vec: &mut Vec<u8>,
                                   alpha_vec: &mut [f32],
                                   tile_vec: &[u8],
                                   tile: &CpuTile,
                                   hit: &TileHit| {
        let ch = out_channels as usize;
        let tile_w = tile.width as i64;
        let tile_h = tile.height as i64;
        let start_x = hit.dest_x_f64.floor().max(0.0) as usize;
        let start_y = hit.dest_y_f64.floor().max(0.0) as usize;
        let end_x = (hit.dest_x_f64 + tile_w as f64).ceil().min(out_w as f64) as usize;
        let end_y = (hit.dest_y_f64 + tile_h as f64).ceil().min(out_h as f64) as usize;
        let out_row_stride = out_w * ch;
        let tile_row_stride = tile_w as usize * ch;

        for out_y in start_y..end_y {
            let src_y = out_y as f64 - hit.dest_y_f64;
            let y0 = src_y.floor() as i64;
            let y1 = y0 + 1;
            let wy = src_y - y0 as f64;
            let wy0 = (1.0 - wy) as f32;
            let wy1 = wy as f32;

            for out_x in start_x..end_x {
                let src_x = out_x as f64 - hit.dest_x_f64;
                let x0 = src_x.floor() as i64;
                let x1 = x0 + 1;
                let wx = src_x - x0 as f64;
                let wx0 = (1.0 - wx) as f32;
                let wx1 = wx as f32;
                let dst_off = out_y * out_row_stride + out_x * ch;
                let alpha_off = out_y * out_w + out_x;

                let in_bounds = |sx: i64, sy: i64| sx >= 0 && sx < tile_w && sy >= 0 && sy < tile_h;
                let a00 = if in_bounds(x0, y0) { wx0 * wy0 } else { 0.0 };
                let a10 = if in_bounds(x1, y0) { wx1 * wy0 } else { 0.0 };
                let a01 = if in_bounds(x0, y1) { wx0 * wy1 } else { 0.0 };
                let a11 = if in_bounds(x1, y1) { wx1 * wy1 } else { 0.0 };
                let src_alpha = a00 + a10 + a01 + a11;
                if src_alpha <= 0.0 {
                    continue;
                }

                let p00 = if in_bounds(x0, y0) {
                    Some((y0 as usize * tile_row_stride) + x0 as usize * ch)
                } else {
                    None
                };
                let p10 = if in_bounds(x1, y0) {
                    Some((y0 as usize * tile_row_stride) + x1 as usize * ch)
                } else {
                    None
                };
                let p01 = if in_bounds(x0, y1) {
                    Some((y1 as usize * tile_row_stride) + x0 as usize * ch)
                } else {
                    None
                };
                let p11 = if in_bounds(x1, y1) {
                    Some((y1 as usize * tile_row_stride) + x1 as usize * ch)
                } else {
                    None
                };
                let dst_alpha = alpha_vec[alpha_off];
                let out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);

                for channel in 0..ch {
                    let src_premult = p00
                        .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a00)
                        .unwrap_or(0.0)
                        + p10
                            .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a10)
                            .unwrap_or(0.0)
                        + p01
                            .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a01)
                            .unwrap_or(0.0)
                        + p11
                            .map(|idx| tile_vec[idx + channel] as f32 / 255.0 * a11)
                            .unwrap_or(0.0);
                    let dst_premult = (out_vec[dst_off + channel] as f32 / 255.0) * dst_alpha;
                    let out_premult = src_premult + dst_premult * (1.0 - src_alpha);
                    let value = if out_alpha > 0.0 {
                        out_premult / out_alpha
                    } else {
                        0.0
                    };
                    out_vec[dst_off + channel] = (value * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                alpha_vec[alpha_off] = out_alpha;
            }
        }
    };

    let mut blit_one_tile = |hit: &TileHit, tile: &Arc<CpuTile>| -> Result<(), WsiError> {
        match (&mut out_data, &tile.data) {
            (CpuTileData::U8(out_vec), CpuTileData::U8(tile_vec)) => {
                let out_vec = Arc::make_mut(out_vec);
                if needs_fractional_blit(hit) {
                    let alpha_vec = alpha_buffer.as_mut().ok_or_else(|| {
                        WsiError::DisplayConversion(
                            "fractional compositing alpha buffer missing".into(),
                        )
                    })?;
                    blit_tile_fractional_u8(out_vec, alpha_vec, tile_vec.as_slice(), tile, hit);
                } else {
                    blit_tile!(out_vec, tile_vec.as_slice(), tile, hit);
                    if let Some(alpha_vec) = alpha_buffer.as_mut() {
                        mark_tile_opaque(alpha_vec, tile, hit);
                    }
                }
            }
            (CpuTileData::U16(out_vec), CpuTileData::U16(tile_vec)) => {
                blit_tile!(Arc::make_mut(out_vec), tile_vec.as_slice(), tile, hit);
            }
            (CpuTileData::F32(out_vec), CpuTileData::F32(tile_vec)) => {
                blit_tile!(Arc::make_mut(out_vec), tile_vec.as_slice(), tile, hit);
            }
            _ => {
                return Err(WsiError::DisplayConversion(
                    "tile sample type mismatch during compositing".into(),
                ));
            }
        }
        Ok(())
    };

    blit_one_tile(&hits[0], &first_tile)?;
    for (hit, tile) in hits.iter().zip(hit_tiles.iter()).skip(1) {
        blit_one_tile(hit, tile)?;
    }

    Ok(CpuTile {
        width: w,
        height: h,
        channels: out_channels,
        color_space: out_color_space,
        layout: out_layout,
        data: out_data,
    })
}

fn metadata_probe_coordinate(layout: &TileLayout) -> Option<(i64, i64)> {
    match layout {
        TileLayout::Regular {
            tiles_across,
            tiles_down,
            ..
        } => (*tiles_across > 0 && *tiles_down > 0).then_some((0, 0)),
        TileLayout::WholeLevel { width, height, .. } => {
            (*width > 0 && *height > 0).then_some((0, 0))
        }
        TileLayout::Irregular { tiles, .. } => tiles
            .keys()
            .min_by(|(col_a, row_a), (col_b, row_b)| row_a.cmp(row_b).then(col_a.cmp(col_b)))
            .copied(),
    }
}

fn zero_sample_data(total_samples: usize, sample_type: SampleType) -> CpuTileData {
    match sample_type {
        SampleType::Uint8 => CpuTileData::u8(vec![0u8; total_samples]),
        SampleType::Uint16 => CpuTileData::u16(vec![0u16; total_samples]),
        SampleType::Float32 => CpuTileData::f32(vec![0.0f32; total_samples]),
    }
}

fn zero_sample_buffer_from_template(width: u32, height: u32, template: &CpuTile) -> CpuTile {
    let total_samples = width as usize * height as usize * template.channels as usize;
    CpuTile {
        width,
        height,
        channels: template.channels,
        color_space: template.color_space.clone(),
        layout: template.layout,
        data: zero_sample_data(total_samples, template.data.sample_type()),
    }
}

fn zero_sample_buffer_from_series(width: u32, height: u32, series: &Series) -> CpuTile {
    let channels = if series.channels.is_empty() {
        1u16
    } else {
        series.channels.len() as u16
    };
    let color_space = match channels {
        1 => ColorSpace::Grayscale,
        3 => ColorSpace::Rgb,
        4 => ColorSpace::Rgba,
        _ => ColorSpace::Unknown,
    };
    let total_samples = width as usize * height as usize * channels as usize;
    CpuTile {
        width,
        height,
        channels,
        color_space,
        layout: CpuTileLayout::Interleaved,
        data: zero_sample_data(total_samples, series.sample_type),
    }
}

pub(crate) fn crop_rgb_interleaved_u8_buffer(
    src: &CpuTile,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<CpuTile, WsiError> {
    if src.layout != CpuTileLayout::Interleaved || src.channels != 3 {
        return Err(WsiError::DisplayConversion(
            "RGB crop expects 3-channel interleaved data".into(),
        ));
    }
    if x > src.width
        || y > src.height
        || x.saturating_add(width) > src.width
        || y.saturating_add(height) > src.height
    {
        return Err(WsiError::DisplayConversion(format!(
            "crop {}x{} at {},{} exceeds source {}x{}",
            width, height, x, y, src.width, src.height
        )));
    }

    let src_data = src
        .data
        .as_u8()
        .ok_or_else(|| WsiError::DisplayConversion("RGB crop expects U8 source data".into()))?;
    let mut out = vec![0u8; width as usize * height as usize * 3];
    let src_stride = src.width as usize * 3;
    let dst_stride = width as usize * 3;
    for row in 0..height as usize {
        let src_start = (y as usize + row) * src_stride + x as usize * 3;
        let src_end = src_start + dst_stride;
        let dst_start = row * dst_stride;
        out[dst_start..dst_start + dst_stride].copy_from_slice(&src_data[src_start..src_end]);
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

pub(crate) fn read_display_tile_from_source<T: SlideReader + ?Sized>(
    source: &T,
    cache: Option<&TileCache>,
    req: &TileViewRequest,
    output: TileOutputPreference,
) -> Result<CpuTile, WsiError> {
    if matches!(output, TileOutputPreference::RequireDevice { .. }) {
        return Err(WsiError::Unsupported {
            reason: "display tile composition returns CPU pixels in Phase 2".into(),
        });
    }

    let dataset = source.dataset();
    let read_tile_uncached = |col: i64, row: i64| -> Result<CpuTile, WsiError> {
        let tile = source.read_tile(
            &TileRequest {
                scene: req.scene,
                series: req.series,
                level: req.level,
                plane: req.plane,
                col,
                row,
            },
            output.clone(),
        )?;
        match tile {
            TilePixels::Cpu(cpu) => Ok(cpu),
            TilePixels::Device(_) => Err(WsiError::Unsupported {
                reason: "display tile read requires CPU pixels".into(),
            }),
        }
    };
    let read_tile_cached = |col: i64, row: i64| -> Result<Arc<CpuTile>, WsiError> {
        let key = CacheKey {
            dataset_id: dataset.id,
            scene: req.scene as u32,
            series: req.series as u32,
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
            tile_col: col,
            tile_row: row,
        };

        if let Some(cache) = cache {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached);
            }
        }

        let tile = Arc::new(read_tile_uncached(col, row)?);
        if let Some(cache) = cache {
            cache.put(key, tile.clone());
        }
        Ok(tile)
    };
    let region_req = RegionRequest {
        scene: SceneId(req.scene),
        series: SeriesId(req.series),
        level: LevelIdx(req.level),
        plane: PlaneIdx(req.plane),
        origin_px: (
            req.col.saturating_mul(i64::from(req.tile_width)),
            req.row.saturating_mul(i64::from(req.tile_height)),
        ),
        size_px: (req.tile_width, req.tile_height),
    };
    let (_, _, level) = validate_region_request(dataset, &region_req)?;

    if let TileLayout::Regular {
        tile_width,
        tile_height,
        tiles_across,
        tiles_down,
    } = &level.tile_layout
    {
        if *tile_width == req.tile_width
            && *tile_height == req.tile_height
            && req.col >= 0
            && req.row >= 0
            && req.col < *tiles_across as i64
            && req.row < *tiles_down as i64
        {
            if cache.is_none() {
                return read_tile_uncached(req.col, req.row);
            }
            return Ok(read_tile_cached(req.col, req.row)?.as_ref().clone());
        }
    }

    let level_w = level.dimensions.0 as i64;
    let level_h = level.dimensions.1 as i64;
    if region_req.origin_px.0 >= level_w || region_req.origin_px.1 >= level_h {
        return Err(WsiError::TileRead {
            col: req.col,
            row: req.row,
            level: req.level,
            reason: "display tile origin out of bounds".into(),
        });
    }

    let clipped = RegionRequest {
        size_px: (
            req.tile_width
                .min((level_w - region_req.origin_px.0) as u32),
            req.tile_height
                .min((level_h - region_req.origin_px.1) as u32),
        ),
        ..region_req
    };
    composite_region_from_source(source, cache, &clipped)
}
