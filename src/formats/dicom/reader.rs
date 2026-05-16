use super::*;

pub(super) struct DicomReader {
    pub(super) slide: Arc<DicomSlide>,
}

impl SlideReader for DicomReader {
    fn dataset(&self) -> &Dataset {
        &self.slide.dataset
    }

    fn tile_codec_kind(&self, req: &TileRequest) -> TileCodecKind {
        self.slide
            .levels
            .get(req.level as usize)
            .map(|level| level.tile_codec_kind(req))
            .unwrap_or(TileCodecKind::Other)
    }

    fn use_display_tile_cache(&self, _req: &TileViewRequest) -> bool {
        true
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let backend = output.backend().to_signinum();

        #[cfg(feature = "metal")]
        if output.prefers_device() {
            match self.read_tiles_jp2k_device_batch(reqs, &output, backend) {
                Ok(Some(tiles)) => return Ok(tiles),
                Ok(None) => {}
                Err(err) if output.requires_device() => return Err(err),
                Err(err) => {
                    tracing::debug!(
                        error = %err,
                        fallback_to_cpu = true,
                        fallback_reason = "dicom_jp2k_device_batch_failed",
                        "DICOM JP2K device batch failed; retrying through CPU output"
                    );
                }
            }
            match self.read_tiles_jpeg_device_batch(reqs, &output, backend) {
                Ok(Some(tiles)) => return Ok(tiles),
                Ok(None) => {}
                Err(err) if output.requires_device() => return Err(err),
                Err(err) => {
                    tracing::debug!(
                        error = %err,
                        fallback_to_cpu = true,
                        fallback_reason = "dicom_jpeg_device_batch_failed",
                        "DICOM JPEG device batch failed; retrying through CPU output"
                    );
                }
            }
        }

        if output.requires_device() {
            return Err(WsiError::Unsupported {
                reason: "RequireDevice not supported for DICOM CPU fallback".into(),
            });
        }

        reqs.iter()
            .map(|req| {
                self.read_tile_with_backend(req, backend)
                    .map(TilePixels::Cpu)
            })
            .collect()
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.read_tile_with_backend(req, BackendRequest::Auto)
    }

    fn read_raw_compressed_tile(&self, req: &TileRequest) -> Result<RawCompressedTile, WsiError> {
        let image = self
            .slide
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: self.slide.levels.len() as u32,
            })?;
        image.read_raw_compressed_tile(req.col, req.row, req.level)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let image = self
            .slide
            .associated
            .get(name)
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
        image.read_associated(name)
    }
}

#[cfg(feature = "metal")]
pub(super) struct DicomDeviceDecodeJob {
    pub(super) slot: usize,
    pub(super) image: Arc<DicomImage>,
    pub(super) frame_index: u32,
}

#[cfg(feature = "metal")]
impl DicomReader {
    pub(super) fn read_tiles_jp2k_device_batch(
        &self,
        reqs: &[TileRequest],
        output: &TileOutputPreference,
        backend: BackendRequest,
    ) -> Result<Option<Vec<TilePixels>>, WsiError> {
        if reqs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if !output.compressed_device_decode_enabled() && !dicom_jp2k_device_decode_enabled() {
            return Ok(None);
        }
        let Some(metal_sessions) = output.metal_sessions() else {
            return Ok(None);
        };

        let mut results: Vec<Option<TilePixels>> = Vec::with_capacity(reqs.len());
        results.resize_with(reqs.len(), || None);
        let mut jobs = Vec::new();
        let mut job_meta = Vec::new();
        let mut saw_device_candidate = false;

        for (slot, req) in reqs.iter().enumerate() {
            let level =
                self.slide
                    .levels
                    .get(req.level as usize)
                    .ok_or(WsiError::LevelOutOfRange {
                        level: req.level,
                        count: self.slide.levels.len() as u32,
                    })?;
            if req.col < 0
                || req.row < 0
                || req.col >= level.tiles_across as i64
                || req.row >= level.tiles_down as i64
            {
                return Err(WsiError::Unsupported {
                    reason: format!(
                        "tile ({},{}) out of range for DICOM device decode",
                        req.col, req.row
                    ),
                });
            }

            let col = req.col as u32;
            let row = req.row as u32;
            let Some(image) = level.image_for_tile(col, row) else {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason:
                            "DICOM device batch cannot return CPU black tile for sparse missing tile"
                                .into(),
                    });
                }
                let (width, height) = level.actual_tile_dimensions(col, row);
                results[slot] = Some(TilePixels::Cpu(black_sample_buffer(width, height)));
                continue;
            };
            if !dicom_jp2k_device_batch_allowed(image.transfer_syntax_uid.as_str(), output) {
                continue;
            }
            let Some(frame_index) = image.frame_index(col, row) else {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason:
                            "DICOM device batch cannot return CPU black tile for sparse missing tile"
                                .into(),
                    });
                }
                let (width, height) = level.actual_tile_dimensions(col, row);
                results[slot] = Some(TilePixels::Cpu(black_sample_buffer(width, height)));
                continue;
            };
            let (actual_width, actual_height) = level.actual_tile_dimensions(col, row);
            if actual_width != image.tile_width || actual_height != image.tile_height {
                continue;
            }
            if image.samples_per_pixel != 3 {
                continue;
            }

            saw_device_candidate = true;
            if !output.requires_device() {
                if let Some(cached) = image.cached_decoded_frame(frame_index) {
                    results[slot] = Some(TilePixels::Cpu(cached.as_ref().clone()));
                    continue;
                }
            }

            let bytes =
                image.extract_encapsulated_frame(frame_index, req.level, req.col, req.row, true)?;
            jobs.push(Jp2kDecodeJob {
                data: Cow::Owned(bytes.as_ref().clone()),
                expected_width: image.tile_width,
                expected_height: image.tile_height,
                rgb_color_space: !matches!(
                    image.photometric_interpretation.as_str(),
                    "YBR_ICT" | "YBR_RCT"
                ),
                backend,
            });
            job_meta.push(DicomDeviceDecodeJob {
                slot,
                image: image.clone(),
                frame_index,
            });
        }

        if jobs.is_empty() && !saw_device_candidate {
            return Ok(None);
        }
        if jobs.is_empty() {
            return results
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(Some)
                .ok_or_else(|| WsiError::Unsupported {
                    reason: "DICOM device batch had no decodable JP2K frames".into(),
                });
        }

        let decoded =
            decode_batch_jp2k_pixels(&jobs, output.requires_device(), Some(metal_sessions));
        if decoded.len() != job_meta.len() {
            return Err(WsiError::Jp2k(format!(
                "DICOM JP2K device batch returned {} tiles for {} jobs",
                decoded.len(),
                job_meta.len()
            )));
        }

        for (meta, decoded) in job_meta.into_iter().zip(decoded) {
            let tile = decoded?;
            if let TilePixels::Cpu(cpu) = &tile {
                meta.image
                    .cache_decoded_frame(meta.frame_index, Arc::new(cpu.clone()));
            }
            results[meta.slot] = Some(tile);
        }

        for (slot, result) in results.iter_mut().enumerate() {
            if result.is_none() {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason: "DICOM device batch contained a non-device-decodable tile".into(),
                    });
                }
                *result = Some(TilePixels::Cpu(
                    self.read_tile_with_backend(&reqs[slot], backend)?,
                ));
            }
        }

        Ok(Some(
            results
                .into_iter()
                .map(|tile| {
                    tile.ok_or_else(|| WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "DICOM device batch result was not populated".into(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    pub(super) fn read_tiles_jpeg_device_batch(
        &self,
        reqs: &[TileRequest],
        output: &TileOutputPreference,
        backend: BackendRequest,
    ) -> Result<Option<Vec<TilePixels>>, WsiError> {
        if reqs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if !output.compressed_device_decode_enabled() {
            return Ok(None);
        }
        let Some(metal_sessions) = output.metal_sessions() else {
            return Ok(None);
        };

        let mut results: Vec<Option<TilePixels>> = Vec::with_capacity(reqs.len());
        results.resize_with(reqs.len(), || None);
        let mut jobs = Vec::new();
        let mut job_meta = Vec::new();
        let mut saw_device_candidate = false;

        for (slot, req) in reqs.iter().enumerate() {
            let level =
                self.slide
                    .levels
                    .get(req.level as usize)
                    .ok_or(WsiError::LevelOutOfRange {
                        level: req.level,
                        count: self.slide.levels.len() as u32,
                    })?;
            if req.col < 0
                || req.row < 0
                || req.col >= level.tiles_across as i64
                || req.row >= level.tiles_down as i64
            {
                return Err(WsiError::Unsupported {
                    reason: format!(
                        "tile ({},{}) out of range for DICOM device decode",
                        req.col, req.row
                    ),
                });
            }

            let col = req.col as u32;
            let row = req.row as u32;
            let Some(image) = level.image_for_tile(col, row) else {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason:
                            "DICOM device batch cannot return CPU black tile for sparse missing tile"
                                .into(),
                    });
                }
                let (width, height) = level.actual_tile_dimensions(col, row);
                results[slot] = Some(TilePixels::Cpu(black_sample_buffer(width, height)));
                continue;
            };
            if image.transfer_syntax_uid != JPEG_TRANSFER_SYNTAX {
                continue;
            }
            let Some(frame_index) = image.frame_index(col, row) else {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason:
                            "DICOM device batch cannot return CPU black tile for sparse missing tile"
                                .into(),
                    });
                }
                let (width, height) = level.actual_tile_dimensions(col, row);
                results[slot] = Some(TilePixels::Cpu(black_sample_buffer(width, height)));
                continue;
            };
            let (actual_width, actual_height) = level.actual_tile_dimensions(col, row);
            if actual_width != image.tile_width || actual_height != image.tile_height {
                continue;
            }
            if image.samples_per_pixel != 3 {
                continue;
            }

            saw_device_candidate = true;
            if !output.requires_device() {
                if let Some(cached) = image.cached_decoded_frame(frame_index) {
                    results[slot] = Some(TilePixels::Cpu(cached.as_ref().clone()));
                    continue;
                }
            }

            let bytes =
                image.extract_encapsulated_frame(frame_index, req.level, req.col, req.row, true)?;
            jobs.push(JpegDecodeJob {
                data: Cow::Owned(bytes.as_ref().clone()),
                tables: None,
                expected_width: image.tile_width,
                expected_height: image.tile_height,
                color_transform: signinum_jpeg::ColorTransform::Auto,
                force_dimensions: false,
                requested_size: None,
            });
            job_meta.push(DicomDeviceDecodeJob {
                slot,
                image: image.clone(),
                frame_index,
            });
        }

        if jobs.is_empty() && !saw_device_candidate {
            return Ok(None);
        }
        if jobs.is_empty() {
            return results
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(Some)
                .ok_or_else(|| WsiError::Unsupported {
                    reason: "DICOM device batch had no decodable JPEG frames".into(),
                });
        }

        let decoded = decode_batch_jpeg_pixels(
            &jobs,
            backend,
            output.requires_device(),
            Some(metal_sessions),
        );
        if decoded.len() != job_meta.len() {
            return Err(WsiError::Jpeg(format!(
                "DICOM JPEG device batch returned {} tiles for {} jobs",
                decoded.len(),
                job_meta.len()
            )));
        }

        for (meta, decoded) in job_meta.into_iter().zip(decoded) {
            let tile = decoded?;
            if let TilePixels::Cpu(cpu) = &tile {
                meta.image
                    .cache_decoded_frame(meta.frame_index, Arc::new(cpu.clone()));
            }
            results[meta.slot] = Some(tile);
        }

        for (slot, result) in results.iter_mut().enumerate() {
            if result.is_none() {
                if output.requires_device() {
                    return Err(WsiError::Unsupported {
                        reason: "DICOM device batch contained a non-device-decodable tile".into(),
                    });
                }
                *result = Some(TilePixels::Cpu(
                    self.read_tile_with_backend(&reqs[slot], backend)?,
                ));
            }
        }

        Ok(Some(
            results
                .into_iter()
                .map(|tile| {
                    tile.ok_or_else(|| WsiError::TileRead {
                        col: 0,
                        row: 0,
                        level: 0,
                        reason: "DICOM device batch result was not populated".into(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

pub(super) fn dicom_tile_codec_kind(transfer_syntax_uid: &str) -> TileCodecKind {
    if transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
        TileCodecKind::Jpeg
    } else if matches!(
        transfer_syntax_uid,
        HTJ2K_LOSSLESS_TRANSFER_SYNTAX | HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX
    ) {
        TileCodecKind::Htj2k
    } else if JP2K_TRANSFER_SYNTAXES.contains(&transfer_syntax_uid) {
        TileCodecKind::Jp2k
    } else {
        TileCodecKind::Other
    }
}

impl DicomReader {
    pub(super) fn read_tile_with_backend(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let image = self
            .slide
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: self.slide.levels.len() as u32,
            })?;
        image.read_tile(req.col, req.row, req.level, backend)
    }
}
