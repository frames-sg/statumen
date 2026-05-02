use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use signinum_core::BackendRequest;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::core::types::*;
use crate::decode::jp2k::{decode_batch_jp2k, Jp2kDecodeJob};
use crate::error::WsiError;
use crate::properties::Properties;

const OLYMPUS_JPEG_2000: u32 = 3;

pub(crate) struct OlympusVsiBackend;

impl OlympusVsiBackend {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl FormatProbe for OlympusVsiBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        let detected = is_vsi_path(path) && companion_dir(path).is_some_and(|dir| dir.is_dir());
        Ok(ProbeResult {
            detected,
            vendor: if detected { "olympus" } else { "" }.into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for OlympusVsiBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        Ok(Box::new(OlympusVsiReader {
            slide: Arc::new(OlympusVsiSlide::parse(path)?),
        }))
    }
}

struct OlympusVsiReader {
    slide: Arc<OlympusVsiSlide>,
}

impl SlideReader for OlympusVsiReader {
    fn dataset(&self) -> &Dataset {
        &self.slide.dataset
    }

    fn use_display_tile_cache(&self, _req: &TileViewRequest) -> bool {
        true
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        let backend = match output {
            TileOutputPreference::Cpu { backend }
            | TileOutputPreference::PreferDevice { backend, .. } => backend.to_signinum(),
            TileOutputPreference::RequireDevice { .. } => {
                return Err(WsiError::Unsupported {
                    reason: "RequireDevice not supported for Olympus VSI".into(),
                });
            }
        };
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

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        Err(WsiError::AssociatedImageNotFound(name.into()))
    }
}

impl OlympusVsiReader {
    fn read_tile_with_backend(
        &self,
        req: &TileRequest,
        backend: BackendRequest,
    ) -> Result<CpuTile, WsiError> {
        let scene = self
            .slide
            .scenes
            .get(req.scene)
            .ok_or(WsiError::SceneOutOfRange {
                index: req.scene,
                count: self.slide.scenes.len(),
            })?;
        if req.series != 0 {
            return Err(WsiError::SeriesOutOfRange {
                index: req.series,
                count: 1,
            });
        }
        let level = scene
            .levels
            .get(req.level as usize)
            .ok_or(WsiError::LevelOutOfRange {
                level: req.level,
                count: scene.levels.len() as u32,
            })?;
        validate_plane(req.plane, scene.axes)?;
        if req.col < 0
            || req.row < 0
            || req.col >= level.tiles_across as i64
            || req.row >= level.tiles_down as i64
        {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: format!(
                    "tile ({},{}) out of range ({}x{})",
                    req.col, req.row, level.tiles_across, level.tiles_down
                ),
            });
        }

        let key = EtsTileKey {
            level: req.level,
            z: req.plane.z,
            c: req.plane.c,
            t: req.plane.t,
            col: req.col as u32,
            row: req.row as u32,
        };
        let Some(tile) = scene.tiles.get(&key) else {
            return Ok(scene.background_tile(level.tile_width, level.tile_height));
        };
        scene
            .decode_tile(tile, backend)
            .map_err(|err| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: err.to_string(),
            })
    }
}

struct OlympusVsiSlide {
    dataset: Dataset,
    scenes: Vec<EtsScene>,
}

impl OlympusVsiSlide {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let dir =
            companion_dir(path).ok_or_else(|| invalid_slide(path, "missing companion dir"))?;
        let mut ets_paths = find_ets_files(&dir)?;
        if ets_paths.is_empty() {
            return Err(invalid_slide(path, "no ETS frame files found"));
        }

        let mut scenes = ets_paths
            .drain(..)
            .map(|ets_path| EtsScene::parse(&ets_path))
            .collect::<Result<Vec<_>, _>>()?;
        scenes.sort_by_key(|scene| Reverse(scene.level0_area()));

        let mut quickhash = Quickhash1::new();
        quickhash.hash_string(&path.display().to_string());
        for scene in &scenes {
            quickhash.hash_string(&scene.path.display().to_string());
            quickhash.update(&scene.path.metadata()?.len().to_le_bytes());
        }
        let quickhash = quickhash
            .finish()
            .ok_or_else(|| invalid_slide(path, "failed to compute Olympus quickhash"))?;
        let dataset_id = dataset_id_from_quickhash(path, &quickhash)?;

        let public_scenes = scenes
            .iter()
            .enumerate()
            .map(|(scene_index, scene)| Scene {
                id: format!("s{scene_index}"),
                name: scene.name.clone(),
                series: vec![Series {
                    id: "ser0".into(),
                    axes: scene.axes,
                    levels: scene
                        .levels
                        .iter()
                        .map(|level| Level {
                            dimensions: (level.width as u64, level.height as u64),
                            downsample: scene.levels[0].width as f64 / level.width as f64,
                            tile_layout: TileLayout::Regular {
                                tile_width: level.tile_width,
                                tile_height: level.tile_height,
                                tiles_across: level.tiles_across as u64,
                                tiles_down: level.tiles_down as u64,
                            },
                        })
                        .collect(),
                    sample_type: scene.sample_type,
                    channels: scene.channels.clone(),
                }],
            })
            .collect();

        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "olympus");
        properties.insert("openslide.quickhash-1", quickhash);

        Ok(Self {
            dataset: Dataset {
                id: dataset_id,
                scenes: public_scenes,
                associated_images: HashMap::new(),
                properties,
                icc_profiles: HashMap::new(),
            },
            scenes,
        })
    }
}

struct EtsScene {
    path: PathBuf,
    name: Option<String>,
    levels: Vec<EtsLevel>,
    tiles: HashMap<EtsTileKey, EtsTile>,
    axes: AxesShape,
    sample_type: SampleType,
    samples_per_pixel: u32,
    background: Vec<u8>,
    channels: Vec<ChannelInfo>,
}

impl EtsScene {
    fn parse(path: &Path) -> Result<Self, WsiError> {
        let mut file = File::open(path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: path.to_path_buf(),
        })?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if !fourcc_matches(&magic, b"SIS") {
            return Err(invalid_slide(path, "invalid ETS SIS magic"));
        }
        let _header_size = file.read_u32::<LittleEndian>()?;
        let _version = file.read_u32::<LittleEndian>()?;
        let n_dimensions = file.read_u32::<LittleEndian>()?;
        let additional_header_offset = file.read_u64::<LittleEndian>()?;
        let _additional_header_size = file.read_u32::<LittleEndian>()?;
        file.seek(SeekFrom::Current(4))?;
        let used_chunk_offset = file.read_u64::<LittleEndian>()?;
        let n_used_chunks = file.read_u32::<LittleEndian>()?;
        file.seek(SeekFrom::Current(4))?;

        file.seek(SeekFrom::Start(additional_header_offset))?;
        file.read_exact(&mut magic)?;
        if !fourcc_matches(&magic, b"ETS") {
            return Err(invalid_slide(path, "invalid ETS header magic"));
        }
        file.seek(SeekFrom::Current(4))?;
        let pixel_type = file.read_u32::<LittleEndian>()?;
        let samples_per_pixel = file.read_u32::<LittleEndian>()?;
        let _colorspace = file.read_u32::<LittleEndian>()?;
        let compression = file.read_u32::<LittleEndian>()?;
        let _compression_quality = file.read_u32::<LittleEndian>()?;
        let tile_width = file.read_u32::<LittleEndian>()?;
        let tile_height = file.read_u32::<LittleEndian>()?;
        let _tile_z = file.read_u32::<LittleEndian>()?;
        file.seek(SeekFrom::Current(4 * 17))?;

        let sample_type = sample_type_from_ets(pixel_type)?;
        let background_len = samples_per_pixel as usize * sample_type.byte_size();
        let mut background = vec![0; background_len];
        file.read_exact(&mut background)?;
        let remaining_background = 40usize.saturating_sub(background_len);
        file.seek(SeekFrom::Current(remaining_background as i64))?;
        let _component_order = file.read_u32::<LittleEndian>()?;
        let use_pyramid = file.read_u32::<LittleEndian>()? != 0;

        if compression != OLYMPUS_JPEG_2000 {
            return Err(invalid_slide(
                path,
                format!("unsupported ETS compression {compression}"),
            ));
        }
        if n_dimensions < 3 {
            return Err(invalid_slide(
                path,
                "ETS coordinate dimensionality is too small",
            ));
        }

        file.seek(SeekFrom::Start(used_chunk_offset))?;
        let mut raw_tiles = Vec::with_capacity(n_used_chunks as usize);
        let mut max_level = 0u32;
        let mut max_z = 0u32;
        let mut max_c = 0u32;
        let mut max_t = 0u32;
        for _ in 0..n_used_chunks {
            file.seek(SeekFrom::Current(4))?;
            let mut coords = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                coords.push(file.read_i32::<LittleEndian>()?);
            }
            let offset = file.read_u64::<LittleEndian>()?;
            let byte_count = file.read_u32::<LittleEndian>()?;
            file.seek(SeekFrom::Current(4))?;

            let key = key_from_coords(&coords, use_pyramid)?;
            max_level = max_level.max(key.level);
            max_z = max_z.max(key.z);
            max_c = max_c.max(key.c);
            max_t = max_t.max(key.t);
            raw_tiles.push((key, offset, byte_count));
        }

        let mut max_col_by_level = vec![0u32; max_level as usize + 1];
        let mut max_row_by_level = vec![0u32; max_level as usize + 1];
        let mut tiles = HashMap::with_capacity(raw_tiles.len());
        for (key, offset, byte_count) in raw_tiles {
            let idx = key.level as usize;
            max_col_by_level[idx] = max_col_by_level[idx].max(key.col);
            max_row_by_level[idx] = max_row_by_level[idx].max(key.row);
            tiles.insert(key, EtsTile { offset, byte_count });
        }

        let levels = max_col_by_level
            .into_iter()
            .zip(max_row_by_level)
            .map(|(max_col, max_row)| EtsLevel {
                width: (max_col + 1) * tile_width,
                height: (max_row + 1) * tile_height,
                tile_width,
                tile_height,
                tiles_across: max_col + 1,
                tiles_down: max_row + 1,
            })
            .collect::<Vec<_>>();

        let channels = if samples_per_pixel == 3 {
            Vec::new()
        } else {
            (0..=max_c)
                .map(|c| ChannelInfo {
                    name: Some(format!("Channel {c}")),
                    color: None,
                    excitation_nm: None,
                    emission_nm: None,
                })
                .collect()
        };

        Ok(Self {
            path: path.to_path_buf(),
            name: path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .map(ToOwned::to_owned),
            levels,
            tiles,
            axes: AxesShape {
                z: max_z + 1,
                c: max_c + 1,
                t: max_t + 1,
            },
            sample_type,
            samples_per_pixel,
            background,
            channels,
        })
    }

    fn level0_area(&self) -> u64 {
        self.levels
            .first()
            .map(|level| level.width as u64 * level.height as u64)
            .unwrap_or(0)
    }

    fn decode_tile(&self, tile: &EtsTile, backend: BackendRequest) -> Result<CpuTile, WsiError> {
        let mut file = File::open(&self.path).map_err(|source| WsiError::IoWithPath {
            source: Arc::new(source),
            path: self.path.clone(),
        })?;
        file.seek(SeekFrom::Start(tile.offset))?;
        let mut bytes = vec![0; tile.byte_count as usize];
        file.read_exact(&mut bytes)?;
        decode_batch_jp2k(&[Jp2kDecodeJob {
            data: Cow::Owned(bytes),
            expected_width: self.levels[0].tile_width,
            expected_height: self.levels[0].tile_height,
            rgb_color_space: true,
            backend,
        }])
        .into_iter()
        .next()
        .expect("single JP2K decode job")
    }

    fn background_tile(&self, width: u32, height: u32) -> CpuTile {
        let mut bytes = Vec::with_capacity(width as usize * height as usize * 3);
        let rgb = if self.samples_per_pixel >= 3 && self.background.len() >= 3 {
            [self.background[0], self.background[1], self.background[2]]
        } else {
            let gray = self.background.first().copied().unwrap_or(0);
            [gray, gray, gray]
        };
        for _ in 0..(width as usize * height as usize) {
            bytes.extend_from_slice(&rgb);
        }
        CpuTile::new(
            width,
            height,
            3,
            ColorSpace::Rgb,
            CpuTileLayout::Interleaved,
            CpuTileData::u8(bytes),
        )
        .expect("background tile dimensions are valid")
    }
}

#[derive(Clone, Copy, Debug)]
struct EtsLevel {
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct EtsTileKey {
    level: u32,
    z: u32,
    c: u32,
    t: u32,
    col: u32,
    row: u32,
}

#[derive(Clone, Copy, Debug)]
struct EtsTile {
    offset: u64,
    byte_count: u32,
}

fn key_from_coords(coords: &[i32], use_pyramid: bool) -> Result<EtsTileKey, WsiError> {
    let upper = if use_pyramid {
        coords.len().saturating_sub(1)
    } else {
        coords.len()
    };
    let level = if use_pyramid {
        checked_coord(coords[coords.len() - 1], "resolution")?
    } else {
        0
    };
    let extra = &coords[2..upper];
    let z = extra
        .first()
        .copied()
        .map(|value| checked_coord(value, "z"))
        .transpose()?
        .unwrap_or(0);
    let c = extra
        .get(1)
        .copied()
        .map(|value| checked_coord(value, "c"))
        .transpose()?
        .unwrap_or(0);
    let t = extra
        .get(2)
        .copied()
        .map(|value| checked_coord(value, "t"))
        .transpose()?
        .unwrap_or(0);
    Ok(EtsTileKey {
        level,
        z,
        c,
        t,
        col: checked_coord(coords[0], "x")?,
        row: checked_coord(coords[1], "y")?,
    })
}

fn checked_coord(value: i32, name: &str) -> Result<u32, WsiError> {
    u32::try_from(value).map_err(|_| WsiError::InvalidSlide {
        path: PathBuf::new(),
        message: format!("negative ETS {name} coordinate {value}"),
    })
}

fn validate_plane(plane: PlaneSelection, axes: AxesShape) -> Result<(), WsiError> {
    if plane.z >= axes.z {
        return Err(WsiError::PlaneOutOfRange {
            axis: "z".into(),
            value: plane.z,
            max: axes.z.saturating_sub(1),
        });
    }
    if plane.c >= axes.c {
        return Err(WsiError::PlaneOutOfRange {
            axis: "c".into(),
            value: plane.c,
            max: axes.c.saturating_sub(1),
        });
    }
    if plane.t >= axes.t {
        return Err(WsiError::PlaneOutOfRange {
            axis: "t".into(),
            value: plane.t,
            max: axes.t.saturating_sub(1),
        });
    }
    Ok(())
}

fn sample_type_from_ets(pixel_type: u32) -> Result<SampleType, WsiError> {
    match pixel_type {
        1 | 2 => Ok(SampleType::Uint8),
        3 | 4 => Ok(SampleType::Uint16),
        9 => Ok(SampleType::Float32),
        other => Err(WsiError::UnsupportedFormat(format!(
            "unsupported ETS pixel type {other}"
        ))),
    }
}

fn find_ets_files(dir: &Path) -> Result<Vec<PathBuf>, WsiError> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(|source| WsiError::IoWithPath {
        source: Arc::new(source),
        path: dir.to_path_buf(),
    })? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let frame = path.join("frame_t.ets");
        if frame.is_file() {
            paths.push(frame);
        }
    }
    paths.sort();
    Ok(paths)
}

fn companion_dir(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?;
    let stem = path.file_stem()?.to_str()?;
    Some(parent.join(format!("_{stem}_")))
}

fn is_vsi_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(ext) if ext.eq_ignore_ascii_case("vsi")
    )
}

fn dataset_id_from_quickhash(path: &Path, quickhash: &str) -> Result<DatasetId, WsiError> {
    if quickhash.len() < 32 {
        return Err(invalid_slide(path, "quickhash too short"));
    }
    let value = u128::from_str_radix(&quickhash[..32], 16)
        .map_err(|_| invalid_slide(path, "quickhash is not valid hex"))?;
    Ok(DatasetId(value))
}

fn invalid_slide(path: &Path, message: impl Into<String>) -> WsiError {
    WsiError::InvalidSlide {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn fourcc_matches(bytes: &[u8; 4], tag: &[u8; 3]) -> bool {
    &bytes[..3] == tag && (bytes[3] == 0 || bytes[3] == b' ')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::registry::Slide;

    #[test]
    fn opens_olympus_vsi_when_corpus_is_available() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let path = workspace_root
            .join("downloads/openslide-testdata-extracted/full/Olympus/OS-1/OS-1.vsi");
        if !path.exists() {
            return;
        }

        let slide = Slide::open(&path).expect("open Olympus VSI");
        let dataset = slide.dataset();
        assert!(!dataset.scenes.is_empty());
        assert!(dataset.scenes[0].series[0].levels.len() >= 2);
        let tile = slide
            .read_tile(
                &TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: 0,
                    row: 0,
                },
                TileOutputPreference::cpu(),
            )
            .expect("read Olympus VSI tile");
        assert!(matches!(tile, TilePixels::Cpu(_)));
    }
}
