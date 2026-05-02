use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cfb::CompoundFile;
use flate2::read::ZlibDecoder;
use image::ImageFormat;
use lru::LruCache;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::core::types::*;
use crate::decode::jpeg::{decode_batch_jpeg, JpegDecodeJob};
use crate::error::WsiError;
use crate::properties::Properties;

const CFB_MAGIC: &[u8; 8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1";
const DEFAULT_TILE_PX: u32 = 256;
const ZVI_HEADER_PROBE_BYTES: usize = 4096;
const POSITION_DEDUP_TOLERANCE_PX: i64 = 128;

pub(crate) struct ZeissZviBackend {
    probe_cache: Mutex<LruCache<PathBuf, Arc<ZviSlide>>>,
}

impl ZeissZviBackend {
    pub(crate) fn new() -> Self {
        Self {
            probe_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(8).unwrap())),
        }
    }

    fn cache_key(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    fn parse(&self, path: &Path) -> Result<Arc<ZviSlide>, WsiError> {
        Ok(Arc::new(ZviSlide::parse(path)?))
    }
}

impl Default for ZeissZviBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatProbe for ZeissZviBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        let key = Self::cache_key(path);
        if self
            .probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .is_some()
        {
            return Ok(ProbeResult {
                detected: true,
                vendor: "zeiss".into(),
                confidence: ProbeConfidence::Definite,
            });
        }

        let mut magic = [0u8; 8];
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(_) => {
                return Ok(ProbeResult {
                    detected: false,
                    vendor: String::new(),
                    confidence: ProbeConfidence::Likely,
                });
            }
        };
        if file.read_exact(&mut magic).is_err() || magic != *CFB_MAGIC {
            return Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            });
        }

        let mut compound = match cfb::open(path) {
            Ok(compound) => compound,
            Err(_) => {
                return Ok(ProbeResult {
                    detected: false,
                    vendor: String::new(),
                    confidence: ProbeConfidence::Likely,
                });
            }
        };
        if !looks_like_zvi(&mut compound) {
            return Ok(ProbeResult {
                detected: false,
                vendor: String::new(),
                confidence: ProbeConfidence::Likely,
            });
        }

        Ok(ProbeResult {
            detected: true,
            vendor: "zeiss".into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for ZeissZviBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let key = Self::cache_key(path);
        let cached = self
            .probe_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned();
        let slide = match cached {
            Some(slide) => slide,
            None => {
                let slide = self.parse(path)?;
                self.probe_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .put(key, slide.clone());
                slide
            }
        };
        Ok(Box::new(ZviReader { slide }))
    }
}

struct ZviReader {
    slide: Arc<ZviSlide>,
}

impl SlideReader for ZviReader {
    fn dataset(&self) -> &Dataset {
        &self.slide.dataset
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.slide.read_tile(req)
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        self.slide
            .associated
            .get(name)
            .cloned()
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))
    }
}

struct ZviSlide {
    dataset: Dataset,
    compound: Mutex<CompoundFile<File>>,
    planes: Vec<ZviPlane>,
    plane_by_whole: HashMap<(u32, u32, u32), usize>,
    plane_by_tile: HashMap<(u32, u32, u32, i64, i64), usize>,
    associated: HashMap<String, CpuTile>,
}

#[derive(Clone)]
struct ZviPlane {
    stream_path: String,
    width: u32,
    height: u32,
    bytes_per_sample: u32,
    payload_offset: u64,
    compression: ZviCompression,
    z: u32,
    c: u32,
    t: u32,
    tile_index: i32,
    stage_position: Option<(f64, f64)>,
    pixel_offset: (i64, i64),
    grid_key: Option<(i64, i64)>,
    channel_name: Option<String>,
    channel_color: Option<[u8; 3]>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ZviCompression {
    Raw,
    Zlib,
    Jpeg,
}

impl ZviSlide {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let mut compound = cfb::open(path).map_err(|source| invalid_slide(path, source))?;
        let stream_paths = compound_stream_paths(&compound);
        let global_tags = read_tags_if_present(&mut compound, "/Image/Tags/Contents")?;
        let global_width = tag_u32(&global_tags, 515);
        let global_height = tag_u32(&global_tags, 516);
        let mpp_x = tag_f64(&global_tags, 769);
        let mpp_y = tag_f64(&global_tags, 772);

        let mut item_streams = stream_paths
            .iter()
            .filter_map(|stream_path| {
                item_contents_index(stream_path).map(|idx| (idx, stream_path.clone()))
            })
            .collect::<Vec<_>>();
        item_streams.sort_by_key(|(idx, _)| *idx);
        if item_streams.is_empty() {
            return Err(invalid_slide(path, "ZVI has no image item streams"));
        }

        let mut planes = Vec::with_capacity(item_streams.len());
        for (item_index, stream_path) in item_streams {
            let header = read_zvi_header(&mut compound, &stream_path)?;
            let tag_path = format!("/Image/Item({item_index})/Tags/Contents");
            let tags = read_tags_if_present(&mut compound, &tag_path)?;
            let stage_position = tag_f64(&tags, 2073).zip(tag_f64(&tags, 2074));
            planes.push(ZviPlane {
                stream_path,
                width: header.width,
                height: header.height,
                bytes_per_sample: header.bytes_per_sample,
                payload_offset: header.payload_offset,
                compression: header.compression,
                z: header.z,
                c: header.c,
                t: header.t,
                tile_index: header.tile_index,
                stage_position,
                pixel_offset: (0, 0),
                grid_key: None,
                channel_name: tag_string(&tags, 1284),
                channel_color: tag_color(&tags, 1282),
            });
        }

        if planes.is_empty() {
            return Err(invalid_slide(
                path,
                "ZVI image item streams were not readable",
            ));
        }

        let sample_type = if planes.iter().all(|plane| plane.bytes_per_sample == 2) {
            SampleType::Uint16
        } else if planes.iter().all(|plane| plane.bytes_per_sample == 1) {
            SampleType::Uint8
        } else {
            return Err(invalid_slide(
                path,
                "mixed ZVI sample byte depths are not supported",
            ));
        };
        let max_z = planes.iter().map(|plane| plane.z).max().unwrap_or(0);
        let max_c = planes.iter().map(|plane| plane.c).max().unwrap_or(0);
        let max_t = planes.iter().map(|plane| plane.t).max().unwrap_or(0);
        let size_z = max_z + 1;
        let size_c = max_c + 1;
        let size_t = max_t + 1;
        let plane_width = planes.iter().map(|plane| plane.width).max().unwrap_or(0);
        let plane_height = planes.iter().map(|plane| plane.height).max().unwrap_or(0);
        let mosaic = planes.iter().any(|plane| plane.tile_index != 0)
            || global_width.is_some_and(|width| width > u64::from(plane_width))
            || global_height.is_some_and(|height| height > u64::from(plane_height));

        let mut plane_by_whole = HashMap::new();
        let mut plane_by_tile = HashMap::new();
        let level_dimensions;
        let tile_layout;
        if mosaic {
            let mpp = mpp_x.zip(mpp_y).ok_or_else(|| {
                invalid_slide(path, "ZVI mosaic is missing global pixel scaling tags")
            })?;
            apply_mosaic_positions(&mut planes, mpp);
            let grid = build_mosaic_grid(&mut planes, plane_width, plane_height);
            for (idx, plane) in planes.iter().enumerate() {
                if let Some((col, row)) = plane.grid_key {
                    plane_by_tile.insert((plane.z, plane.c, plane.t, col, row), idx);
                }
            }
            level_dimensions = (
                global_width.unwrap_or_else(|| grid.width.max(plane_width as u64)),
                global_height.unwrap_or_else(|| grid.height.max(plane_height as u64)),
            );
            tile_layout = TileLayout::Irregular {
                tile_advance: (grid.advance_x, grid.advance_y),
                extra_tiles: (2, 2, 2, 2),
                tiles: grid.entries,
            };
        } else {
            for (idx, plane) in planes.iter().enumerate() {
                plane_by_whole.insert((plane.z, plane.c, plane.t), idx);
            }
            level_dimensions = (
                global_width.unwrap_or(plane_width as u64),
                global_height.unwrap_or(plane_height as u64),
            );
            tile_layout = TileLayout::WholeLevel {
                width: level_dimensions.0,
                height: level_dimensions.1,
                virtual_tile_width: DEFAULT_TILE_PX,
                virtual_tile_height: DEFAULT_TILE_PX,
            };
        }

        let quickhash = quickhash_for_zvi(path, &planes, level_dimensions)?;
        let dataset_id = dataset_id_from_quickhash(path, &quickhash)?;
        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "zeiss");
        properties.insert("openslide.quickhash-1", quickhash);
        properties.insert("zeiss.format", "zvi");
        properties.insert("zeiss.image.size_x", level_dimensions.0.to_string());
        properties.insert("zeiss.image.size_y", level_dimensions.1.to_string());
        properties.insert("zeiss.image.size_z", size_z.to_string());
        properties.insert("zeiss.image.size_c", size_c.to_string());
        properties.insert("zeiss.image.size_t", size_t.to_string());
        if let Some(mpp_x) = mpp_x {
            properties.insert("openslide.mpp-x", format!("{mpp_x:.6}"));
        }
        if let Some(mpp_y) = mpp_y {
            properties.insert("openslide.mpp-y", format!("{mpp_y:.6}"));
        }
        if let Some(objective) = tag_string(&global_tags, 2049) {
            properties.insert("zeiss.objective.name", objective);
        }
        if let Some(power) = tag_string(&global_tags, 2076) {
            properties.insert("openslide.objective-power", power);
        }

        let channels = build_zvi_channels(&planes, size_c);
        let associated = associated_images(&mut compound)?;
        let associated_metadata = associated
            .iter()
            .map(|(name, tile)| {
                (
                    name.clone(),
                    AssociatedImage {
                        dimensions: (tile.width, tile.height),
                        sample_type: tile.data.sample_type(),
                        channels: tile.channels,
                    },
                )
            })
            .collect();

        let dataset = Dataset {
            id: dataset_id,
            scenes: vec![Scene {
                id: "scene_0".to_string(),
                name: Some("Image".to_string()),
                series: vec![Series {
                    id: "series_0".to_string(),
                    axes: AxesShape {
                        z: size_z,
                        c: size_c,
                        t: size_t,
                    },
                    levels: vec![Level {
                        dimensions: level_dimensions,
                        downsample: 1.0,
                        tile_layout,
                    }],
                    sample_type,
                    channels,
                }],
            }],
            associated_images: associated_metadata,
            properties,
            icc_profiles: HashMap::new(),
        };

        Ok(Self {
            dataset,
            compound: Mutex::new(compound),
            planes,
            plane_by_whole,
            plane_by_tile,
            associated,
        })
    }

    fn read_tile(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        if req.scene != 0 || req.series != 0 || req.level != 0 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "ZVI exposes one scene, series, and level".into(),
            });
        }
        if req.plane.z >= self.dataset.scenes[0].series[0].axes.z
            || req.plane.c >= self.dataset.scenes[0].series[0].axes.c
            || req.plane.t >= self.dataset.scenes[0].series[0].axes.t
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "ZVI plane out of range".into(),
            });
        }

        let level = &self.dataset.scenes[0].series[0].levels[0];
        match &level.tile_layout {
            TileLayout::WholeLevel {
                width,
                height,
                virtual_tile_width,
                virtual_tile_height,
            } => {
                let plane_index = self
                    .plane_by_whole
                    .get(&(req.plane.z, req.plane.c, req.plane.t))
                    .copied()
                    .ok_or_else(|| WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "ZVI plane has no image payload".into(),
                    })?;
                let x = req.col.saturating_mul(i64::from(*virtual_tile_width));
                let y = req.row.saturating_mul(i64::from(*virtual_tile_height));
                if x < 0 || y < 0 || x >= *width as i64 || y >= *height as i64 {
                    return Err(WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "ZVI tile out of bounds".into(),
                    });
                }
                let w = (*virtual_tile_width).min((*width as i64 - x) as u32);
                let h = (*virtual_tile_height).min((*height as i64 - y) as u32);
                self.read_plane_window(plane_index, x as u32, y as u32, w, h)
            }
            TileLayout::Irregular { .. } => {
                let plane_index = self
                    .plane_by_tile
                    .get(&(req.plane.z, req.plane.c, req.plane.t, req.col, req.row))
                    .copied()
                    .ok_or_else(|| WsiError::TileRead {
                        col: req.col,
                        row: req.row,
                        level: req.level,
                        reason: "ZVI mosaic tile not found".into(),
                    })?;
                let plane = &self.planes[plane_index];
                self.read_plane_window(plane_index, 0, 0, plane.width, plane.height)
            }
            TileLayout::Regular { .. } => Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "ZVI does not use regular native tiles".into(),
            }),
        }
    }

    fn read_plane_window(
        &self,
        plane_index: usize,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<CpuTile, WsiError> {
        let plane = &self.planes[plane_index];
        if x > plane.width
            || y > plane.height
            || x.saturating_add(w) > plane.width
            || y.saturating_add(h) > plane.height
        {
            return Err(WsiError::TileRead {
                col: 0,
                row: 0,
                level: 0,
                reason: "ZVI plane window out of bounds".into(),
            });
        }

        match plane.compression {
            ZviCompression::Raw => self.read_raw_plane_window(plane, x, y, w, h),
            ZviCompression::Zlib => self.read_zlib_plane_window(plane, x, y, w, h),
            ZviCompression::Jpeg => self.read_jpeg_plane_window(plane, x, y, w, h),
        }
    }

    fn read_raw_plane_window(
        &self,
        plane: &ZviPlane,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<CpuTile, WsiError> {
        match plane.bytes_per_sample {
            1 => {
                let mut samples = vec![0u8; w as usize * h as usize];
                self.read_raw_rows(
                    plane,
                    RawReadWindow {
                        x,
                        y,
                        width: w,
                        height: h,
                        bytes_per_sample: 1,
                    },
                    &mut samples,
                )?;
                CpuTile::new(
                    w,
                    h,
                    1,
                    ColorSpace::Grayscale,
                    CpuTileLayout::Interleaved,
                    CpuTileData::u8(samples),
                )
            }
            2 => {
                let mut row_bytes = vec![0u8; w as usize * 2];
                let mut samples = vec![0u16; w as usize * h as usize];
                let mut compound = self.compound.lock().unwrap_or_else(|e| e.into_inner());
                let mut stream = compound.open_stream(&plane.stream_path)?;
                for row in 0..h {
                    let src_offset = plane
                        .payload_offset
                        .checked_add(
                            (u64::from(y + row) * u64::from(plane.width) + u64::from(x)) * 2,
                        )
                        .ok_or_else(|| {
                            WsiError::DisplayConversion("ZVI raw row offset overflow".into())
                        })?;
                    stream.seek(SeekFrom::Start(src_offset))?;
                    stream.read_exact(&mut row_bytes)?;
                    let dst = row as usize * w as usize;
                    for (slot, bytes) in samples[dst..dst + w as usize]
                        .iter_mut()
                        .zip(row_bytes.chunks_exact(2))
                    {
                        *slot = u16::from_le_bytes([bytes[0], bytes[1]]);
                    }
                }
                CpuTile::new(
                    w,
                    h,
                    1,
                    ColorSpace::Grayscale,
                    CpuTileLayout::Interleaved,
                    CpuTileData::u16(samples),
                )
            }
            other => Err(WsiError::Unsupported {
                reason: format!("unsupported ZVI raw sample byte depth {other}"),
            }),
        }
    }

    fn read_raw_rows(
        &self,
        plane: &ZviPlane,
        window: RawReadWindow,
        destination: &mut [u8],
    ) -> Result<(), WsiError> {
        let mut compound = self.compound.lock().unwrap_or_else(|e| e.into_inner());
        let mut stream = compound.open_stream(&plane.stream_path)?;
        let row_bytes = window.width as usize * window.bytes_per_sample as usize;
        for row in 0..window.height {
            let src_offset = plane
                .payload_offset
                .checked_add(
                    (u64::from(window.y + row) * u64::from(plane.width) + u64::from(window.x))
                        * window.bytes_per_sample,
                )
                .ok_or_else(|| WsiError::DisplayConversion("ZVI raw row offset overflow".into()))?;
            let dst = row as usize * row_bytes;
            stream.seek(SeekFrom::Start(src_offset))?;
            stream.read_exact(&mut destination[dst..dst + row_bytes])?;
        }
        Ok(())
    }

    fn read_zlib_plane_window(
        &self,
        plane: &ZviPlane,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<CpuTile, WsiError> {
        let compressed = self.read_plane_payload_to_end(plane)?;
        let mut decoder = ZlibDecoder::new(compressed.as_slice());
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        crop_decoded_zvi_plane(plane, &decompressed, x, y, w, h)
    }

    fn read_jpeg_plane_window(
        &self,
        plane: &ZviPlane,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<CpuTile, WsiError> {
        let jpeg = self.read_plane_payload_to_end(plane)?;
        let decoded = decode_batch_jpeg(&[JpegDecodeJob {
            data: std::borrow::Cow::Borrowed(jpeg.as_slice()),
            tables: None,
            expected_width: plane.width,
            expected_height: plane.height,
            color_transform: signinum_jpeg::ColorTransform::Auto,
            force_dimensions: false,
            requested_size: None,
        }])
        .into_iter()
        .next()
        .ok_or_else(|| WsiError::Jpeg("empty ZVI JPEG decode result".into()))??;
        crop_interleaved_tile(&decoded, x, y, w, h)
    }

    fn read_plane_payload_to_end(&self, plane: &ZviPlane) -> Result<Vec<u8>, WsiError> {
        let mut compound = self.compound.lock().unwrap_or_else(|e| e.into_inner());
        let mut stream = compound.open_stream(&plane.stream_path)?;
        stream.seek(SeekFrom::Start(plane.payload_offset))?;
        let mut payload = Vec::new();
        stream.read_to_end(&mut payload)?;
        Ok(payload)
    }
}

struct ZviImageHeader {
    width: u32,
    height: u32,
    bytes_per_sample: u32,
    payload_offset: u64,
    compression: ZviCompression,
    z: u32,
    c: u32,
    t: u32,
    tile_index: i32,
}

struct MosaicGrid {
    advance_x: f64,
    advance_y: f64,
    width: u64,
    height: u64,
    entries: HashMap<(i64, i64), TileEntry>,
}

#[derive(Clone, Copy)]
struct RawReadWindow {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    bytes_per_sample: u64,
}

fn looks_like_zvi(compound: &mut CompoundFile<File>) -> bool {
    compound.is_stream("/Image/Tags/Contents")
        && compound
            .walk()
            .any(|entry| entry.is_stream() && item_contents_index(&entry_path(&entry)).is_some())
}

fn compound_stream_paths(compound: &CompoundFile<File>) -> Vec<String> {
    let mut paths = compound
        .walk()
        .filter(|entry| entry.is_stream())
        .map(|entry| entry_path(&entry))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn entry_path(entry: &cfb::Entry) -> String {
    entry.path().to_string_lossy().replace('\\', "/")
}

fn item_contents_index(path: &str) -> Option<i32> {
    let rest = path.strip_prefix("/Image/Item(")?;
    let (index, suffix) = rest.split_once(')')?;
    (suffix == "/Contents")
        .then(|| index.parse::<i32>().ok())
        .flatten()
}

fn read_stream_prefix(
    compound: &mut CompoundFile<File>,
    path: &str,
    limit: usize,
) -> Result<Vec<u8>, WsiError> {
    let mut stream = compound.open_stream(path)?;
    let mut data = vec![0u8; limit];
    let count = stream.read(&mut data)?;
    data.truncate(count);
    Ok(data)
}

fn read_stream_to_end(compound: &mut CompoundFile<File>, path: &str) -> Result<Vec<u8>, WsiError> {
    let mut stream = compound.open_stream(path)?;
    let mut data = Vec::new();
    stream.read_to_end(&mut data)?;
    Ok(data)
}

fn read_zvi_header(
    compound: &mut CompoundFile<File>,
    stream_path: &str,
) -> Result<ZviImageHeader, WsiError> {
    let data = read_stream_prefix(compound, stream_path, ZVI_HEADER_PROBE_BYTES)?;
    parse_zvi_header(&data)
}

fn parse_zvi_header(data: &[u8]) -> Result<ZviImageHeader, WsiError> {
    let mut reader = ByteReader::new(data);
    for _ in 0..11 {
        let _ = reader.read_variant()?;
    }
    reader.skip(2)?;
    let coord_len = reader.read_i32()?.saturating_sub(20);
    reader.skip(8)?;
    let z = checked_axis(reader.read_i32()?)?;
    let c = checked_axis(reader.read_i32()?)?;
    let t = checked_axis(reader.read_i32()?)?;
    reader.skip(4)?;
    let tile_index = reader.read_i32()?;
    if coord_len < 8 {
        return Err(WsiError::DisplayConversion(
            "ZVI coordinate block is too short".into(),
        ));
    }
    reader.skip((coord_len - 8) as usize)?;
    for _ in 0..5 {
        let _ = reader.read_variant()?;
    }
    reader.skip(4)?;
    let width = checked_dimension(reader.read_i32()?)?;
    let height = checked_dimension(reader.read_i32()?)?;
    reader.skip(4)?;
    let bytes_per_sample = checked_dimension(reader.read_i32()?)?;
    reader.skip(4)?;
    let valid = reader.read_i32()?;
    let payload_offset = reader.position() as u64;
    let check = reader.read_bytes(4)?;
    let compression = if matches!(valid, 0 | 1) {
        if check == b"WZL\0" || &check[..3] == b"WZL" {
            ZviCompression::Zlib
        } else {
            ZviCompression::Jpeg
        }
    } else {
        ZviCompression::Raw
    };
    let payload_offset = if compression == ZviCompression::Zlib {
        payload_offset + 8
    } else {
        payload_offset
    };

    Ok(ZviImageHeader {
        width,
        height,
        bytes_per_sample,
        payload_offset,
        compression,
        z,
        c,
        t,
        tile_index,
    })
}

fn read_tags_if_present(
    compound: &mut CompoundFile<File>,
    path: &str,
) -> Result<HashMap<i32, String>, WsiError> {
    if !compound.is_stream(path) {
        return Ok(HashMap::new());
    }
    let data = read_stream_to_end(compound, path)?;
    parse_zvi_tags(&data)
}

fn parse_zvi_tags(data: &[u8]) -> Result<HashMap<i32, String>, WsiError> {
    let mut reader = ByteReader::new(data);
    reader.skip(8)?;
    let count = reader.read_i32()?.max(0) as usize;
    let mut tags = HashMap::new();
    for _ in 0..count {
        if reader.remaining() < 2 {
            break;
        }
        let value = reader
            .read_variant()?
            .trim_matches(char::from(0))
            .trim()
            .to_string();
        reader.skip(2)?;
        if reader.remaining() < 10 {
            break;
        }
        let tag_id = reader.read_i32()?;
        reader.skip(6)?;
        if tag_id != 1047 {
            tags.insert(tag_id, value);
        }
    }
    Ok(tags)
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn skip(&mut self, count: usize) -> Result<(), WsiError> {
        self.require(count)?;
        self.pos += count;
        Ok(())
    }

    fn read_bytes(&mut self, count: usize) -> Result<&'a [u8], WsiError> {
        self.require(count)?;
        let start = self.pos;
        self.pos += count;
        Ok(&self.data[start..self.pos])
    }

    fn read_u16(&mut self) -> Result<u16, WsiError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, WsiError> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i32(&mut self) -> Result<i32, WsiError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u32(&mut self) -> Result<u32, WsiError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, WsiError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, WsiError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, WsiError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_variant(&mut self) -> Result<String, WsiError> {
        let value_type = self.read_u16()?;
        match value_type {
            0 | 1 => Ok(String::new()),
            2 => self.read_i16().map(|value| value.to_string()),
            3 | 22 => self.read_i32().map(|value| value.to_string()),
            19 | 23 => self.read_u32().map(|value| value.to_string()),
            4 => self.read_f32().map(|value| value.to_string()),
            5 | 7 => self.read_f64().map(|value| value.to_string()),
            8 | 69 => {
                let byte_len = self.read_u32()? as usize;
                let raw = self.read_bytes(byte_len)?;
                Ok(decode_utf16le_lossy(raw))
            }
            9 | 13 => {
                self.skip(16)?;
                Ok(String::new())
            }
            11 => self.read_i16().map(|value| (value != 0).to_string()),
            20 => self.read_i64().map(|value| value.to_string()),
            21 => {
                let raw = self.read_bytes(8)?;
                Ok(u64::from_be_bytes([
                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                ])
                .to_string())
            }
            63 | 65 => {
                let byte_len = self.read_u32()? as usize;
                self.skip(byte_len)?;
                Ok(String::new())
            }
            66 => {
                let byte_len = self.read_u16()? as usize;
                let raw = self.read_bytes(byte_len)?;
                Ok(String::from_utf8_lossy(raw).into_owned())
            }
            _ => self.read_unknown_variant_lossy(),
        }
    }

    fn read_unknown_variant_lossy(&mut self) -> Result<String, WsiError> {
        let start = self.pos.saturating_sub(2);
        let mut scan = self.pos;
        while scan + 2 <= self.data.len() {
            if u16::from_le_bytes([self.data[scan], self.data[scan + 1]]) == 3 {
                break;
            }
            scan += 2;
        }
        self.pos = scan;
        Ok(decode_utf16le_lossy(&self.data[start..scan]))
    }

    fn require(&self, count: usize) -> Result<(), WsiError> {
        if self.remaining() < count {
            return Err(WsiError::DisplayConversion(
                "unexpected end of ZVI metadata stream".into(),
            ));
        }
        Ok(())
    }
}

fn checked_axis(value: i32) -> Result<u32, WsiError> {
    u32::try_from(value)
        .map_err(|_| WsiError::DisplayConversion("negative ZVI axis coordinate".into()))
}

fn checked_dimension(value: i32) -> Result<u32, WsiError> {
    let value = u32::try_from(value)
        .map_err(|_| WsiError::DisplayConversion("negative ZVI dimension".into()))?;
    if value == 0 {
        return Err(WsiError::DisplayConversion("zero ZVI dimension".into()));
    }
    Ok(value)
}

fn decode_utf16le_lossy(raw: &[u8]) -> String {
    let words = raw
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .take_while(|value| *value != 0)
        .collect::<Vec<_>>();
    String::from_utf16_lossy(&words)
}

fn tag_string(tags: &HashMap<i32, String>, tag_id: i32) -> Option<String> {
    tags.get(&tag_id)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn tag_u32(tags: &HashMap<i32, String>, tag_id: i32) -> Option<u64> {
    tag_string(tags, tag_id).and_then(|value| {
        value
            .parse::<u64>()
            .ok()
            .or_else(|| value.parse::<f64>().ok().map(|v| v.round() as u64))
    })
}

fn tag_f64(tags: &HashMap<i32, String>, tag_id: i32) -> Option<f64> {
    tag_string(tags, tag_id).and_then(|value| value.parse::<f64>().ok())
}

fn tag_color(tags: &HashMap<i32, String>, tag_id: i32) -> Option<[u8; 3]> {
    let value = tag_string(tags, tag_id)?.parse::<u32>().ok()?;
    Some([
        ((value >> 16) & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        (value & 0xff) as u8,
    ])
}

fn apply_mosaic_positions(planes: &mut [ZviPlane], mpp: (f64, f64)) {
    let min_x = planes
        .iter()
        .filter_map(|plane| plane.stage_position.map(|(x, _)| x))
        .fold(f64::INFINITY, f64::min);
    let min_y = planes
        .iter()
        .filter_map(|plane| plane.stage_position.map(|(_, y)| y))
        .fold(f64::INFINITY, f64::min);
    if !(min_x.is_finite() && min_y.is_finite() && mpp.0 > 0.0 && mpp.1 > 0.0) {
        return;
    }
    for plane in planes {
        if let Some((stage_x, stage_y)) = plane.stage_position {
            plane.pixel_offset = (
                ((stage_x - min_x) / mpp.0).round() as i64,
                ((stage_y - min_y) / mpp.1).round() as i64,
            );
        }
    }
}

fn build_mosaic_grid(planes: &mut [ZviPlane], tile_width: u32, tile_height: u32) -> MosaicGrid {
    let mut tile_offsets = BTreeMap::<i32, (i64, i64)>::new();
    for plane in planes.iter() {
        tile_offsets
            .entry(plane.tile_index)
            .or_insert(plane.pixel_offset);
    }

    let row_positions = dedup_positions(tile_offsets.values().map(|(_, y)| *y).collect::<Vec<_>>());
    let advance_y = median_step(&row_positions).unwrap_or(tile_height as f64);
    let mut row_columns: HashMap<i64, Vec<i64>> = HashMap::new();
    for (x, y) in tile_offsets.values() {
        let row = nearest_position_index(&row_positions, *y) as i64;
        row_columns.entry(row).or_default().push(*x);
    }
    for columns in row_columns.values_mut() {
        *columns = dedup_positions(std::mem::take(columns));
    }
    let advance_x = row_columns
        .values()
        .filter_map(|columns| median_step(columns))
        .next()
        .unwrap_or(tile_width as f64);

    let mut tile_key_by_index = HashMap::<i32, (i64, i64)>::new();
    let mut entries = HashMap::new();
    let mut width = 0u64;
    let mut height = 0u64;
    for (tile_index, (x, y)) in &tile_offsets {
        let row = nearest_position_index(&row_positions, *y) as i64;
        let columns = row_columns.get(&row).cloned().unwrap_or_default();
        let col = nearest_position_index(&columns, *x) as i64;
        tile_key_by_index.insert(*tile_index, (col, row));
        width = width.max((*x).max(0) as u64 + u64::from(tile_width));
        height = height.max((*y).max(0) as u64 + u64::from(tile_height));
        entries.insert(
            (col, row),
            TileEntry {
                offset: (
                    *x as f64 - col as f64 * advance_x,
                    *y as f64 - row as f64 * advance_y,
                ),
                dimensions: (tile_width, tile_height),
                tiff_tile_index: None,
            },
        );
    }

    for plane in planes {
        plane.grid_key = tile_key_by_index.get(&plane.tile_index).copied();
    }

    MosaicGrid {
        advance_x,
        advance_y,
        width,
        height,
        entries,
    }
}

fn dedup_positions(mut values: Vec<i64>) -> Vec<i64> {
    values.sort_unstable();
    let mut out: Vec<i64> = Vec::new();
    for value in values {
        if out
            .last()
            .is_none_or(|last| (value - *last).abs() > POSITION_DEDUP_TOLERANCE_PX)
        {
            out.push(value);
        }
    }
    out
}

fn nearest_position_index(values: &[i64], target: i64) -> usize {
    values
        .iter()
        .enumerate()
        .min_by_key(|(_, value)| (target - **value).abs())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn median_step(values: &[i64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mut steps = values
        .windows(2)
        .filter_map(|pair| {
            let step = pair[1] - pair[0];
            (step > POSITION_DEDUP_TOLERANCE_PX).then_some(step as f64)
        })
        .collect::<Vec<_>>();
    if steps.is_empty() {
        return None;
    }
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(steps[steps.len() / 2])
}

fn build_zvi_channels(planes: &[ZviPlane], size_c: u32) -> Vec<ChannelInfo> {
    (0..size_c)
        .map(|c| {
            let plane = planes.iter().find(|plane| plane.c == c);
            ChannelInfo {
                name: plane.and_then(|plane| plane.channel_name.clone()),
                color: plane.and_then(|plane| plane.channel_color),
                excitation_nm: None,
                emission_nm: None,
            }
        })
        .collect()
}

fn associated_images(
    compound: &mut CompoundFile<File>,
) -> Result<HashMap<String, CpuTile>, WsiError> {
    if !compound.is_stream("/Thumbnail") {
        return Ok(HashMap::new());
    }
    let data = read_stream_to_end(compound, "/Thumbnail")?;
    let Some(bmp_start) = data.windows(2).position(|bytes| bytes == b"BM") else {
        return Ok(HashMap::new());
    };
    let image = image::load_from_memory_with_format(&data[bmp_start..], ImageFormat::Bmp)
        .map_err(|source| WsiError::DisplayConversion(source.to_string()))?
        .to_rgb8();
    let tile = CpuTile::from_u8_interleaved(
        image.width(),
        image.height(),
        3,
        ColorSpace::Rgb,
        image.into_raw(),
    )?;
    Ok(HashMap::from([("thumbnail".to_string(), tile)]))
}

fn crop_decoded_zvi_plane(
    plane: &ZviPlane,
    data: &[u8],
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<CpuTile, WsiError> {
    match plane.bytes_per_sample {
        1 => {
            let mut samples = vec![0u8; w as usize * h as usize];
            for row in 0..h as usize {
                let src = ((y as usize + row) * plane.width as usize + x as usize)
                    .checked_mul(plane.bytes_per_sample as usize)
                    .ok_or_else(|| {
                        WsiError::DisplayConversion("ZVI decoded offset overflow".into())
                    })?;
                let dst = row * w as usize;
                samples[dst..dst + w as usize].copy_from_slice(&data[src..src + w as usize]);
            }
            CpuTile::new(
                w,
                h,
                1,
                ColorSpace::Grayscale,
                CpuTileLayout::Interleaved,
                CpuTileData::u8(samples),
            )
        }
        2 => {
            let mut samples = vec![0u16; w as usize * h as usize];
            for row in 0..h as usize {
                let src = ((y as usize + row) * plane.width as usize + x as usize)
                    .checked_mul(2)
                    .ok_or_else(|| {
                        WsiError::DisplayConversion("ZVI decoded offset overflow".into())
                    })?;
                let dst = row * w as usize;
                for (slot, bytes) in samples[dst..dst + w as usize]
                    .iter_mut()
                    .zip(data[src..src + w as usize * 2].chunks_exact(2))
                {
                    *slot = u16::from_le_bytes([bytes[0], bytes[1]]);
                }
            }
            CpuTile::new(
                w,
                h,
                1,
                ColorSpace::Grayscale,
                CpuTileLayout::Interleaved,
                CpuTileData::u16(samples),
            )
        }
        other => Err(WsiError::Unsupported {
            reason: format!("unsupported ZVI decoded sample byte depth {other}"),
        }),
    }
}

fn crop_interleaved_tile(
    src: &CpuTile,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<CpuTile, WsiError> {
    if src.layout != CpuTileLayout::Interleaved {
        return Err(WsiError::DisplayConversion(
            "cannot crop planar ZVI JPEG tile".into(),
        ));
    }
    let channels = src.channels as usize;
    let source = src
        .data
        .as_u8()
        .ok_or_else(|| WsiError::DisplayConversion("ZVI JPEG decoded to non-u8 samples".into()))?;
    let mut out = vec![0u8; width as usize * height as usize * channels];
    for row in 0..height as usize {
        let src_offset = ((y as usize + row) * src.width as usize + x as usize) * channels;
        let dst_offset = row * width as usize * channels;
        let len = width as usize * channels;
        out[dst_offset..dst_offset + len].copy_from_slice(&source[src_offset..src_offset + len]);
    }
    CpuTile::new(
        width,
        height,
        src.channels,
        src.color_space.clone(),
        CpuTileLayout::Interleaved,
        CpuTileData::u8(out),
    )
}

fn quickhash_for_zvi(
    path: &Path,
    planes: &[ZviPlane],
    dimensions: (u64, u64),
) -> Result<String, WsiError> {
    let mut quickhash = Quickhash1::new();
    quickhash.hash_string("zeiss-zvi");
    quickhash.hash_string(&path.display().to_string());
    quickhash.update(&dimensions.0.to_le_bytes());
    quickhash.update(&dimensions.1.to_le_bytes());
    for plane in planes.iter().take(64) {
        quickhash.hash_string(&plane.stream_path);
        quickhash.update(&plane.width.to_le_bytes());
        quickhash.update(&plane.height.to_le_bytes());
        quickhash.update(&plane.payload_offset.to_le_bytes());
    }
    quickhash
        .finish()
        .ok_or_else(|| WsiError::DisplayConversion("failed to compute ZVI quickhash".into()))
}

fn dataset_id_from_quickhash(path: &Path, quickhash: &str) -> Result<DatasetId, WsiError> {
    if quickhash.len() < 32 {
        return Err(invalid_slide(path, "ZVI quickhash too short"));
    }
    let value = u128::from_str_radix(&quickhash[..32], 16)
        .map_err(|_| invalid_slide(path, "ZVI quickhash is not valid hex"))?;
    Ok(DatasetId(value))
}

fn invalid_slide(path: &Path, message: impl ToString) -> WsiError {
    WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: message.to_string(),
    }
}
