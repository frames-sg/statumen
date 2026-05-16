use super::*;

#[derive(Debug)]
pub(super) struct DicomImage {
    pub(super) path: PathBuf,
    pub(super) sop_instance_uid: String,
    pub(super) transfer_syntax_uid: String,
    pub(super) photometric_interpretation: String,
    pub(super) samples_per_pixel: u16,
    pub(super) planar_configuration: Option<u16>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) tiles_across: u32,
    pub(super) tiles_down: u32,
    pub(super) number_of_frames: u32,
    pub(super) grid: DicomGrid,
    pub(super) pixel_spacing: Option<(f64, f64)>,
    pub(super) objective_lens_power: Option<f64>,
    pub(super) encapsulated_frames: Mutex<Option<Arc<DicomEncapsulatedFrames>>>,
    pub(super) encapsulated_frame_cache: Mutex<LruCache<u32, Arc<Vec<u8>>>>,
    pub(super) decoded_frame_cache: Mutex<LruCache<u32, Arc<CpuTile>>>,
    pub(super) file: Mutex<Option<File>>,
}

#[derive(Debug)]
pub(super) enum DicomGrid {
    Full,
    Sparse(HashMap<(u32, u32), u32>),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DicomFragmentRef {
    pub(super) payload_offset: u64,
    pub(super) item_offset: u64,
    pub(super) len: u32,
}

#[derive(Debug)]
pub(super) struct DicomEncapsulatedFrames {
    pub(super) fragments: Vec<DicomFragmentRef>,
    pub(super) frame_ranges: Vec<std::ops::Range<usize>>,
}

impl DicomImage {
    pub(super) fn from_metadata(meta: ParsedDicomMetadata) -> Result<Self, WsiError> {
        let width = meta.total_pixel_matrix_columns.unwrap_or(meta.columns);
        let height = meta.total_pixel_matrix_rows.unwrap_or(meta.rows);
        let tile_width = meta.columns;
        let tile_height = meta.rows;
        let tiles_across = width.div_ceil(tile_width);
        let tiles_down = height.div_ceil(tile_height);
        let grid = if meta.dimension_organization_type.as_deref() == Some("TILED_SPARSE") {
            DicomGrid::Sparse(parse_sparse_tile_map(&meta.obj, tile_width, tile_height)?)
        } else {
            DicomGrid::Full
        };
        let frame_cache_entries =
            if JP2K_TRANSFER_SYNTAXES.contains(&meta.transfer_syntax_uid.as_str()) {
                2
            } else {
                1
            };
        Ok(Self {
            path: meta.path,
            sop_instance_uid: meta.sop_instance_uid,
            transfer_syntax_uid: meta.transfer_syntax_uid,
            photometric_interpretation: meta.photometric_interpretation,
            samples_per_pixel: meta.samples_per_pixel,
            planar_configuration: meta.planar_configuration,
            width,
            height,
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
            number_of_frames: meta.number_of_frames,
            grid,
            pixel_spacing: meta.pixel_spacing,
            objective_lens_power: meta.objective_lens_power,
            encapsulated_frames: Mutex::new(None),
            encapsulated_frame_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(frame_cache_entries).unwrap(),
            )),
            decoded_frame_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(frame_cache_entries).unwrap(),
            )),
            file: Mutex::new(None),
        })
    }

    pub(super) fn read_tile(
        &self,
        col: i64,
        row: i64,
        level: u32,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let span = tracing::info_span!(
            "dicom_read_tile",
            reader = "statumen",
            transfer_syntax = %self.transfer_syntax_uid,
        );
        let _guard = span.enter();
        if col < 0 || row < 0 || col >= self.tiles_across as i64 || row >= self.tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    self.tiles_across, self.tiles_down
                ),
            });
        }

        let col_u32 = col as u32;
        let row_u32 = row as u32;
        let Some(frame_index) = self.frame_index(col_u32, row_u32) else {
            let (width, height) = self.actual_tile_dimensions(col_u32, row_u32);
            return Ok(black_sample_buffer(width, height));
        };

        let (actual_width, actual_height) = self.actual_tile_dimensions(col_u32, row_u32);
        let buffer = self.decode_frame_sample_buffer(frame_index, level, col, row, backend)?;
        Ok(crop_sample_buffer_rgb(buffer, actual_width, actual_height))
    }

    pub(super) fn read_raw_compressed_tile(
        &self,
        col: i64,
        row: i64,
        level: u32,
    ) -> Result<RawCompressedTile, WsiError> {
        if col < 0 || row < 0 || col >= self.tiles_across as i64 || row >= self.tiles_down as i64 {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "tile ({col},{row}) out of range ({}x{})",
                    self.tiles_across, self.tiles_down
                ),
            });
        }

        let col_u32 = col as u32;
        let row_u32 = row as u32;
        let Some(frame_index) = self.frame_index(col_u32, row_u32) else {
            return Err(WsiError::Unsupported {
                reason: format!(
                    "raw compressed tile access is not available for sparse missing DICOM tile ({col}, {row}) at level {level}"
                ),
            });
        };
        let compression = raw_compression_for_transfer_syntax(
            &self.transfer_syntax_uid,
            &self.photometric_interpretation,
        )?;
        let photometric_interpretation = raw_photometric_interpretation(
            self.samples_per_pixel,
            &self.photometric_interpretation,
        )?;
        let bytes = self.extract_encapsulated_frame(frame_index, level, col, row, true)?;
        let mut data = bytes.as_ref().clone();
        trim_encapsulated_frame_padding(&mut data);

        Ok(RawCompressedTile {
            compression,
            width: self.tile_width,
            height: self.tile_height,
            bits_allocated: 8,
            samples_per_pixel: self.samples_per_pixel,
            photometric_interpretation,
            data,
        })
    }

    pub(super) fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let buffer = self
            .decode_frame_sample_buffer(0, 0, 0, 0, BackendRequest::Auto)
            .map_err(|err| match err {
                WsiError::TileRead { reason, .. } => {
                    WsiError::AssociatedImageNotFound(format!("{name}: {reason}"))
                }
                other => other,
            })?;
        Ok(crop_sample_buffer_rgb(buffer, self.width, self.height))
    }

    pub(super) fn frame_index(&self, col: u32, row: u32) -> Option<u32> {
        match &self.grid {
            DicomGrid::Full => Some(row * self.tiles_across + col),
            DicomGrid::Sparse(map) => map.get(&(col, row)).copied(),
        }
    }

    pub(super) fn is_full_grid(&self) -> bool {
        matches!(self.grid, DicomGrid::Full)
    }

    pub(super) fn actual_tile_dimensions(&self, col: u32, row: u32) -> (u32, u32) {
        let tile_x = col * self.tile_width;
        let tile_y = row * self.tile_height;
        let width = self.width.saturating_sub(tile_x).min(self.tile_width);
        let height = self.height.saturating_sub(tile_y).min(self.tile_height);
        (width, height)
    }

    pub(super) fn cached_decoded_frame(&self, frame_index: u32) -> Option<Arc<CpuTile>> {
        self.decoded_frame_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&frame_index)
            .cloned()
    }

    pub(super) fn cache_decoded_frame(&self, frame_index: u32, tile: Arc<CpuTile>) {
        self.decoded_frame_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(frame_index, tile);
    }

    pub(super) fn decode_uncompressed_frame_sample_buffer(
        &self,
        frame_index: u32,
        level: u32,
        col: i64,
        row: i64,
    ) -> Result<CpuTile, WsiError> {
        let obj = reopen_dicom_object(&self.path)?;
        let pixel_data = obj
            .element(tags::PIXEL_DATA)
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!("missing pixel data: {err}"),
            })?
            .to_bytes()
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!("failed to read DICOM pixel data: {err}"),
            })?;
        let frame_len = (self.tile_width as usize)
            .checked_mul(self.tile_height as usize)
            .and_then(|pixels| pixels.checked_mul(self.samples_per_pixel as usize))
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: "DICOM frame size overflow".into(),
            })?;
        let start = (frame_index as usize)
            .checked_mul(frame_len)
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: "DICOM frame offset overflow".into(),
            })?;
        let end = start
            .checked_add(frame_len)
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: "DICOM frame byte range overflow".into(),
            })?;
        if end > pixel_data.len() {
            return Err(WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "DICOM frame {frame_index} byte range {}..{} exceeds pixel data length {}",
                    start,
                    end,
                    pixel_data.len()
                ),
            });
        }
        frame_bytes_to_rgb_tile(
            &pixel_data[start..end],
            self.tile_width,
            self.tile_height,
            self.samples_per_pixel,
            self.planar_configuration.unwrap_or(0),
            &self.photometric_interpretation,
        )
        .map_err(|err| WsiError::TileRead {
            col,
            row,
            level,
            reason: err.to_string(),
        })
    }

    pub(super) fn decode_frame_sample_buffer(
        &self,
        frame_index: u32,
        level: u32,
        col: i64,
        row: i64,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let use_decoded_cache = is_encapsulated_transfer_syntax(&self.transfer_syntax_uid);
        if use_decoded_cache {
            if let Some(cached) = self.cached_decoded_frame(frame_index) {
                return Ok(cached.as_ref().clone());
            }
        }

        let buffer = if self.transfer_syntax_uid == JPEG_TRANSFER_SYNTAX {
            let bytes =
                self.extract_encapsulated_frame(frame_index, level, col, row, !use_decoded_cache)?;
            decode_batch_jpeg(&[JpegDecodeJob {
                data: Cow::Borrowed(bytes.as_slice()),
                tables: None,
                expected_width: self.tile_width,
                expected_height: self.tile_height,
                color_transform: signinum_jpeg::ColorTransform::Auto,
                force_dimensions: false,
                requested_size: None,
            }])
            .into_iter()
            .next()
            .expect("1-element JPEG facade batch")
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
        } else if JP2K_TRANSFER_SYNTAXES.contains(&self.transfer_syntax_uid.as_str()) {
            let bytes =
                self.extract_encapsulated_frame(frame_index, level, col, row, !use_decoded_cache)?;
            decode_batch_jp2k(&[Jp2kDecodeJob {
                data: Cow::Borrowed(bytes.as_slice()),
                expected_width: self.tile_width,
                expected_height: self.tile_height,
                rgb_color_space: !matches!(
                    self.photometric_interpretation.as_str(),
                    "YBR_ICT" | "YBR_RCT"
                ),
                backend,
            }])
            .into_iter()
            .next()
            .expect("1-element JP2K facade batch")
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
        } else if self.transfer_syntax_uid == RLE_TRANSFER_SYNTAX {
            let bytes =
                self.extract_encapsulated_frame(frame_index, level, col, row, !use_decoded_cache)?;
            decode_rle_lossless_frame(
                bytes.as_slice(),
                self.tile_width,
                self.tile_height,
                self.samples_per_pixel,
                &self.photometric_interpretation,
            )
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
        } else {
            self.decode_uncompressed_frame_sample_buffer(frame_index, level, col, row)?
        };

        let arc = Arc::new(buffer);
        if use_decoded_cache {
            self.cache_decoded_frame(frame_index, arc.clone());
        }
        Ok(arc.as_ref().clone())
    }

    pub(super) fn extract_encapsulated_frame(
        &self,
        frame_index: u32,
        level: u32,
        col: i64,
        row: i64,
        cache_result: bool,
    ) -> Result<Arc<Vec<u8>>, WsiError> {
        if is_encapsulated_transfer_syntax(&self.transfer_syntax_uid) {
            if cache_result {
                if let Some(bytes) = self
                    .encapsulated_frame_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&frame_index)
                    .cloned()
                {
                    return Ok(bytes);
                }
            }
            let encapsulated_frames = self.ensure_encapsulated_frames()?;
            let frame_range = encapsulated_frames
                .frame_ranges
                .get(frame_index as usize)
                .ok_or_else(|| WsiError::TileRead {
                    col,
                    row,
                    level,
                    reason: format!(
                        "encapsulated frame {frame_index} out of range for {} frames",
                        encapsulated_frames.frame_ranges.len()
                    ),
                })?;
            let bytes = Arc::new(self.read_encapsulated_fragments(
                &encapsulated_frames.fragments[frame_range.start..frame_range.end],
            )?);
            if cache_result {
                self.encapsulated_frame_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .put(frame_index, bytes.clone());
            }
            return Ok(bytes);
        }

        let obj = reopen_dicom_object(&self.path)?;
        let pixel_data = obj
            .element(tags::PIXEL_DATA)
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!("missing pixel data: {err}"),
            })?;
        let fragments = pixel_data.fragments().ok_or_else(|| WsiError::TileRead {
            col,
            row,
            level,
            reason: "pixel data is not encapsulated".into(),
        })?;
        let number_of_frames = optional_u32(&obj, tags::NUMBER_OF_FRAMES)
            .map_err(|err| WsiError::TileRead {
                col,
                row,
                level,
                reason: err.to_string(),
            })?
            .unwrap_or(1);

        if number_of_frames == 1 && fragments.len() > 1 {
            let total_len = fragments.iter().map(Vec::len).sum();
            let mut data = Vec::with_capacity(total_len);
            for fragment in fragments {
                data.extend_from_slice(fragment);
            }
            return Ok(Arc::new(data));
        }

        fragments
            .get(frame_index as usize)
            .map(|fragment| Arc::new(fragment.as_slice().to_vec()))
            .ok_or_else(|| WsiError::TileRead {
                col,
                row,
                level,
                reason: format!(
                    "encapsulated frame {frame_index} out of range for {} fragments",
                    fragments.len()
                ),
            })
    }

    pub(super) fn ensure_encapsulated_frames(
        &self,
    ) -> Result<Arc<DicomEncapsulatedFrames>, WsiError> {
        let mut guard = self
            .encapsulated_frames
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(frames) = &*guard {
            return Ok(frames.clone());
        }
        let frames = Arc::new(scan_encapsulated_frames(
            &self.path,
            &self.transfer_syntax_uid,
            self.number_of_frames,
        )?);
        *guard = Some(frames.clone());
        Ok(frames)
    }

    pub(super) fn read_encapsulated_fragments(
        &self,
        fragments: &[DicomFragmentRef],
    ) -> Result<Vec<u8>, WsiError> {
        let total_len: usize = fragments.iter().map(|fragment| fragment.len as usize).sum();
        let mut data = Vec::with_capacity(total_len);
        self.with_open_file(|file| {
            for fragment in fragments {
                file.seek(SeekFrom::Start(fragment.payload_offset))
                    .map_err(|source| WsiError::IoWithPath {
                        source: Arc::new(source),
                        path: self.path.clone(),
                    })?;
                let start = data.len();
                data.resize(start + fragment.len as usize, 0);
                file.read_exact(&mut data[start..])
                    .map_err(|source| WsiError::IoWithPath {
                        source: Arc::new(source),
                        path: self.path.clone(),
                    })?;
            }
            Ok(())
        })?;
        Ok(data)
    }

    pub(super) fn with_open_file<T>(
        &self,
        f: impl FnOnce(&mut File) -> Result<T, WsiError>,
    ) -> Result<T, WsiError> {
        let mut guard = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let file = File::open(&self.path).map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: self.path.clone(),
            })?;
            *guard = Some(file);
        }
        let file = guard.as_mut().expect("file must be initialized");
        f(file)
    }
}

pub(super) fn reopen_dicom_object(path: &Path) -> Result<DefaultDicomObject, WsiError> {
    dicom_object::open_file(path).map_err(|source| WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: format!("failed to reopen DICOM object: {source}"),
    })
}

pub(super) fn scan_encapsulated_frames(
    path: &Path,
    transfer_syntax_uid: &str,
    number_of_frames: u32,
) -> Result<DicomEncapsulatedFrames, WsiError> {
    let transfer_syntax = TransferSyntaxRegistry
        .get(transfer_syntax_uid)
        .or_else(|| {
            JP2K_TRANSFER_SYNTAXES
                .contains(&transfer_syntax_uid)
                .then(|| TransferSyntaxRegistry.get(uids::EXPLICIT_VR_LITTLE_ENDIAN))
                .flatten()
        })
        .ok_or_else(|| {
            invalid_slide(
                path,
                format!("unknown transfer syntax {transfer_syntax_uid}"),
            )
        })?;
    let mut reader = BufReader::new(File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?);
    position_reader_for_dicom_magic(&mut reader, path)?;
    let _meta = FileMetaTable::from_reader(&mut reader)
        .map_err(|source| invalid_slide(path, format!("cannot parse DICOM file meta: {source}")))?;
    let mut tokens = LazyDataSetReader::new_with_ts(reader, transfer_syntax)
        .map_err(|source| invalid_slide(path, format!("cannot stream DICOM dataset: {source}")))?;

    let mut in_pixel_sequence = false;
    let mut awaiting_offset_table = false;
    let mut offset_table = Vec::new();
    let mut fragments = Vec::new();

    while let Some(token) = tokens.advance() {
        let token = token
            .map_err(|source| invalid_slide(path, format!("cannot read DICOM token: {source}")))?;
        match token {
            LazyDataToken::PixelSequenceStart => {
                in_pixel_sequence = true;
                awaiting_offset_table = true;
            }
            LazyDataToken::ItemStart { len }
                if in_pixel_sequence && awaiting_offset_table && len.0 == 0 =>
            {
                awaiting_offset_table = false;
            }
            LazyDataToken::LazyItemValue { len, decoder }
                if in_pixel_sequence && awaiting_offset_table =>
            {
                decoder
                    .read_u32_to_vec(len, &mut offset_table)
                    .map_err(|source| {
                        invalid_slide(
                            path,
                            format!("cannot read DICOM basic offset table: {source}"),
                        )
                    })?;
                awaiting_offset_table = false;
            }
            LazyDataToken::LazyItemValue { len, decoder } if in_pixel_sequence => {
                let payload_offset = decoder.position();
                let item_offset = payload_offset.saturating_sub(8);
                decoder.skip_bytes(len).map_err(|source| {
                    invalid_slide(path, format!("cannot skip DICOM fragment: {source}"))
                })?;
                fragments.push(DicomFragmentRef {
                    payload_offset,
                    item_offset,
                    len,
                });
            }
            LazyDataToken::ItemStart { len } if in_pixel_sequence && len.0 == 0 => {
                return Err(invalid_slide(
                    path,
                    "zero-length DICOM pixel fragment is not supported",
                ));
            }
            LazyDataToken::SequenceEnd if in_pixel_sequence => break,
            other => {
                other.skip().map_err(|source| {
                    invalid_slide(path, format!("cannot skip DICOM token: {source}"))
                })?;
            }
        }
    }

    if fragments.is_empty() {
        if let Some(frames) = scan_encapsulated_frames_raw_little_endian(path, number_of_frames)? {
            return Ok(frames);
        }
    }

    build_encapsulated_frame_index(path, fragments, offset_table, number_of_frames)
}

pub(super) const PIXEL_DATA_TAG_LE: [u8; 4] = [0xE0, 0x7F, 0x10, 0x00];
pub(super) const DICOM_ITEM_TAG_LE: [u8; 4] = [0xFE, 0xFF, 0x00, 0xE0];
pub(super) const DICOM_SEQUENCE_DELIMITER_TAG_LE: [u8; 4] = [0xFE, 0xFF, 0xDD, 0xE0];
pub(super) const UNDEFINED_LENGTH_LE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
pub(super) const EXPLICIT_VR_LONG_HEADER_LEN: usize = 12;

pub(super) fn scan_encapsulated_frames_raw_little_endian(
    path: &Path,
    number_of_frames: u32,
) -> Result<Option<DicomEncapsulatedFrames>, WsiError> {
    let mut file = File::open(path).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })?;
    let Some(pixel_data_offset) = find_encapsulated_pixel_data_offset_le(&mut file, path)? else {
        return Ok(None);
    };

    let (fragments, offset_table) =
        scan_raw_encapsulated_pixel_sequence(&mut file, path, pixel_data_offset)?;
    build_encapsulated_frame_index(path, fragments, offset_table, number_of_frames).map(Some)
}

pub(super) fn find_encapsulated_pixel_data_offset_le(
    file: &mut File,
    path: &Path,
) -> Result<Option<u64>, WsiError> {
    const CHUNK_LEN: usize = 64 * 1024;
    let mut chunk = [0u8; CHUNK_LEN];
    let mut overlap = Vec::new();
    let mut chunk_offset = 0u64;

    file.seek(SeekFrom::Start(0))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;

    loop {
        let read_len = file
            .read(&mut chunk)
            .map_err(|source| WsiError::IoWithPath {
                source: Arc::new(source),
                path: path.to_path_buf(),
            })?;
        if read_len == 0 {
            return Ok(None);
        }

        let window_offset = chunk_offset.saturating_sub(overlap.len() as u64);
        let mut window = Vec::with_capacity(overlap.len() + read_len);
        window.extend_from_slice(&overlap);
        window.extend_from_slice(&chunk[..read_len]);

        for index in 0..=window.len().saturating_sub(EXPLICIT_VR_LONG_HEADER_LEN) {
            let header = &window[index..index + EXPLICIT_VR_LONG_HEADER_LEN];
            if is_encapsulated_pixel_data_header_le(header) {
                return Ok(Some(window_offset + index as u64));
            }
        }

        let keep = window.len().min(EXPLICIT_VR_LONG_HEADER_LEN - 1);
        overlap.clear();
        overlap.extend_from_slice(&window[window.len() - keep..]);
        chunk_offset = chunk_offset
            .checked_add(read_len as u64)
            .ok_or_else(|| invalid_slide(path, "DICOM raw Pixel Data scan offset overflow"))?;
    }
}

pub(super) fn is_encapsulated_pixel_data_header_le(header: &[u8]) -> bool {
    header.len() >= EXPLICIT_VR_LONG_HEADER_LEN
        && header[0..4] == PIXEL_DATA_TAG_LE
        && matches!(&header[4..6], b"OB" | b"OW" | b"UN")
        && header[6..8] == [0, 0]
        && header[8..12] == UNDEFINED_LENGTH_LE
}

pub(super) fn scan_raw_encapsulated_pixel_sequence(
    file: &mut File,
    path: &Path,
    pixel_data_offset: u64,
) -> Result<(Vec<DicomFragmentRef>, Vec<u32>), WsiError> {
    let mut cursor = pixel_data_offset
        .checked_add(EXPLICIT_VR_LONG_HEADER_LEN as u64)
        .ok_or_else(|| invalid_slide(path, "DICOM raw Pixel Data offset overflow"))?;
    let mut offset_table = None;
    let mut fragments = Vec::new();

    loop {
        let mut item_header = [0u8; 8];
        read_exact_at(file, path, cursor, &mut item_header)?;
        let tag = &item_header[0..4];
        let len = u32::from_le_bytes(
            item_header[4..8]
                .try_into()
                .expect("DICOM item length header is 4 bytes"),
        );
        cursor = cursor
            .checked_add(item_header.len() as u64)
            .ok_or_else(|| invalid_slide(path, "DICOM raw item offset overflow"))?;

        if tag == DICOM_SEQUENCE_DELIMITER_TAG_LE {
            return Ok((fragments, offset_table.unwrap_or_default()));
        }
        if tag != DICOM_ITEM_TAG_LE {
            return Err(invalid_slide(
                path,
                format!(
                    "unexpected DICOM pixel sequence tag {:02x?} at byte {}",
                    tag,
                    cursor - item_header.len() as u64
                ),
            ));
        }

        if offset_table.is_none() {
            offset_table = Some(read_basic_offset_table_at(file, path, cursor, len)?);
        } else {
            if len == 0 {
                return Err(invalid_slide(
                    path,
                    "zero-length DICOM pixel fragment is not supported",
                ));
            }
            fragments.push(DicomFragmentRef {
                payload_offset: cursor,
                item_offset: cursor - item_header.len() as u64,
                len,
            });
        }

        cursor = cursor
            .checked_add(len as u64)
            .ok_or_else(|| invalid_slide(path, "DICOM raw item payload offset overflow"))?;
    }
}

pub(super) fn read_basic_offset_table_at(
    file: &mut File,
    path: &Path,
    offset: u64,
    len: u32,
) -> Result<Vec<u32>, WsiError> {
    if !len.is_multiple_of(4) {
        return Err(invalid_slide(
            path,
            format!("DICOM basic offset table has non-u32 length {len}"),
        ));
    }
    let len = usize::try_from(len)
        .map_err(|_| invalid_slide(path, "DICOM basic offset table length overflow"))?;
    let mut bytes = vec![0u8; len];
    read_exact_at(file, path, offset, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            u32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("DICOM basic offset table chunk is 4 bytes"),
            )
        })
        .collect())
}

pub(super) fn read_exact_at(
    file: &mut File,
    path: &Path,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), WsiError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    file.read_exact(buf).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: path.to_path_buf(),
    })
}

pub(super) fn build_encapsulated_frame_index(
    path: &Path,
    fragments: Vec<DicomFragmentRef>,
    offset_table: Vec<u32>,
    number_of_frames: u32,
) -> Result<DicomEncapsulatedFrames, WsiError> {
    if number_of_frames == 0 {
        return Err(invalid_slide(path, "DICOM reported zero frames"));
    }
    if fragments.is_empty() {
        return Err(invalid_slide(
            path,
            "DICOM encapsulated pixel data has no fragments",
        ));
    }

    let frame_ranges = if number_of_frames == 1 {
        std::iter::once(0..fragments.len()).collect()
    } else if !offset_table.is_empty() {
        let base_item_offset = fragments[0].item_offset;
        let fragment_indices_by_offset: HashMap<u64, usize> = fragments
            .iter()
            .enumerate()
            .map(|(index, fragment)| (fragment.item_offset, index))
            .collect();
        let mut start_indices = Vec::with_capacity(offset_table.len());
        for offset in &offset_table {
            let target = base_item_offset + *offset as u64;
            let index = fragment_indices_by_offset
                .get(&target)
                .copied()
                .ok_or_else(|| {
                    invalid_slide(
                        path,
                        format!(
                            "DICOM basic offset table points to missing fragment offset {offset}"
                        ),
                    )
                })?;
            start_indices.push(index);
        }
        if start_indices.len() != number_of_frames as usize {
            return Err(invalid_slide(
                path,
                format!(
                    "DICOM basic offset table length {} does not match number_of_frames {}",
                    start_indices.len(),
                    number_of_frames
                ),
            ));
        }
        let mut ranges = Vec::with_capacity(start_indices.len());
        for (frame, start) in start_indices.iter().copied().enumerate() {
            let end = start_indices
                .get(frame + 1)
                .copied()
                .unwrap_or(fragments.len());
            ranges.push(start..end);
        }
        ranges
    } else if fragments.len() == number_of_frames as usize {
        (0..fragments.len()).map(|index| index..index + 1).collect()
    } else {
        return Err(invalid_slide(
            path,
            format!(
                "cannot map {} DICOM fragments to {} frames without a basic offset table",
                fragments.len(),
                number_of_frames
            ),
        ));
    };

    Ok(DicomEncapsulatedFrames {
        fragments,
        frame_ranges,
    })
}

pub(super) fn position_reader_for_dicom_magic<R: Read + Seek>(
    reader: &mut R,
    path: &Path,
) -> Result<(), WsiError> {
    let mut preamble = [0u8; 132];
    reader
        .read_exact(&mut preamble)
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    let start = if &preamble[128..] == b"DICM" { 128 } else { 0 };
    reader
        .seek(SeekFrom::Start(start))
        .map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
    Ok(())
}
