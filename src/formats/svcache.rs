use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::registry::{
    DatasetReader, FormatProbe, FormatRegistry, ProbeConfidence, ProbeResult, Slide, SlideReader,
};
use crate::core::types::{
    AssociatedImage, AxesShape, ChannelInfo, ColorSpace, CpuTile, CpuTileLayout, Dataset,
    DatasetId, Level, PlaneSelection, SampleType, Scene, Series, TileLayout, TileOutputPreference,
    TilePixels, TileRequest, TileViewRequest,
};
use crate::error::WsiError;
use crate::properties::Properties;

const MAGIC: &[u8; 8] = b"SVCACHE1";
const SCHEMA_VERSION: u32 = 2;
const DEFAULT_TILE_SIZE: u32 = 256;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SvcachePolicy {
    #[default]
    Off,
    PreferFresh,
    RequireFresh,
}

impl SvcachePolicy {
    pub fn from_env_value(value: Option<&str>) -> Self {
        match value.unwrap_or("off").trim().to_ascii_lowercase().as_str() {
            "prefer" | "on" | "true" | "1" => Self::PreferFresh,
            "required" | "require" => Self::RequireFresh,
            _ => Self::Off,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SvcacheMetadata {
    schema_version: u32,
    #[serde(default = "default_cache_complete")]
    complete: bool,
    source: SourceFingerprint,
    properties: Vec<(String, String)>,
    scenes: Vec<SceneMeta>,
    associated: Vec<AssociatedMeta>,
}

fn default_cache_complete() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SourceFingerprint {
    path: String,
    len: u64,
    modified_unix_nanos: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SceneMeta {
    id: String,
    name: Option<String>,
    series: Vec<SeriesMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeriesMeta {
    id: String,
    axes: AxesMeta,
    sample_type: SampleTypeMeta,
    channels: Vec<ChannelMeta>,
    levels: Vec<LevelMeta>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct AxesMeta {
    z: u32,
    c: u32,
    t: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum SampleTypeMeta {
    Uint8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChannelMeta {
    name: Option<String>,
    color: Option<[u8; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LevelMeta {
    dimensions: (u64, u64),
    downsample: f64,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u64,
    tiles_down: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tiles: Vec<Option<TileMeta>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sparse_tiles: Vec<SparseTileMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseTileMeta {
    index: u64,
    tile: TileMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AssociatedMeta {
    name: String,
    dimensions: (u32, u32),
    tile: TileMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TileMeta {
    payload_offset: u64,
    payload_len: u64,
    decoded_len: usize,
    width: u32,
    height: u32,
    channels: u16,
    color_space: ColorSpaceMeta,
    codec: PayloadCodec,
    sha256: String,
}

impl LevelMeta {
    fn tile_meta_for_index(&self, index: u64) -> Option<&TileMeta> {
        if !self.tiles.is_empty() {
            return usize::try_from(index)
                .ok()
                .and_then(|idx| self.tiles.get(idx))
                .and_then(Option::as_ref);
        }
        self.sparse_tiles
            .binary_search_by_key(&index, |entry| entry.index)
            .ok()
            .map(|idx| &self.sparse_tiles[idx].tile)
    }

    fn insert_tile_for_index(&mut self, index: u64, tile: TileMeta) {
        if !self.tiles.is_empty() {
            if let Ok(idx) = usize::try_from(index) {
                if let Some(slot) = self.tiles.get_mut(idx) {
                    *slot = Some(tile);
                }
            }
            return;
        }

        match self
            .sparse_tiles
            .binary_search_by_key(&index, |entry| entry.index)
        {
            Ok(idx) => self.sparse_tiles[idx].tile = tile,
            Err(idx) => self
                .sparse_tiles
                .insert(idx, SparseTileMeta { index, tile }),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum PayloadCodec {
    Zstd,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum ColorSpaceMeta {
    Rgb,
    Rgba,
    Grayscale,
}

pub struct SvcacheBackend;

pub struct SvcacheReader {
    file: Mutex<File>,
    payload_start: u64,
    metadata: SvcacheMetadata,
    dataset: Dataset,
    associated_index: HashMap<String, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SvcacheTileSelection {
    pub scene: usize,
    pub series: usize,
    pub level: u32,
    pub plane: PlaneSelection,
    pub col: i64,
    pub row: i64,
}

pub fn default_svcache_path(source_path: &Path) -> PathBuf {
    let name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("slide");
    source_path.with_file_name(format!("{name}.svcache"))
}

pub fn cache_dir_svcache_path(source_path: &Path) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(source_path.to_string_lossy().as_bytes());
    let hash = hex_encode(&hasher.finalize());
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache")
        .join("slideviewer")
        .join("svcache")
        .join(format!("{hash}.svcache"))
}

pub fn svcache_candidate_paths(source_path: &Path) -> [PathBuf; 2] {
    [
        default_svcache_path(source_path),
        cache_dir_svcache_path(source_path),
    ]
}

pub(crate) fn resolve_open_path_with_policy(
    path: &Path,
    policy: SvcachePolicy,
) -> Result<PathBuf, WsiError> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("svcache"))
    {
        return Ok(path.to_path_buf());
    }
    if matches!(policy, SvcachePolicy::Off) {
        return Ok(path.to_path_buf());
    }

    for candidate in svcache_candidate_paths(path) {
        if candidate.is_file() && is_fresh_svcache(&candidate, path).unwrap_or(false) {
            return Ok(candidate);
        }
    }

    if matches!(policy, SvcachePolicy::RequireFresh) {
        return Err(WsiError::UnsupportedFormat(format!(
            "fresh .svcache required for {}",
            path.display()
        )));
    }
    Ok(path.to_path_buf())
}

pub fn build_svcache(source_path: &Path, out_path: &Path) -> Result<(), WsiError> {
    let registry = FormatRegistry::builtin_native();
    let source = registry.open_exact(source_path)?;
    let slide = Slide::from_source_with_cache_bytes(source, 256 * 1024 * 1024);
    let source_fingerprint = fingerprint_source(source_path)?;

    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut payload = tempfile::tempfile()?;
    let mut scenes = Vec::new();

    for (scene_idx, scene) in slide.dataset().scenes.iter().enumerate() {
        let mut series_meta = Vec::new();
        for (series_idx, series) in scene.series.iter().enumerate() {
            let mut levels_meta = Vec::new();
            for (level_idx, level) in series.levels.iter().enumerate() {
                let (tile_width, tile_height, tiles_across, tiles_down) =
                    cache_grid_for_level(level);
                let mut tiles = Vec::with_capacity(
                    usize::try_from(tiles_across.saturating_mul(tiles_down)).unwrap_or(0),
                );
                for row in 0..tiles_down {
                    for col in 0..tiles_across {
                        let request = TileViewRequest {
                            scene: scene_idx,
                            series: series_idx,
                            level: level_idx as u32,
                            plane: PlaneSelection::default(),
                            col: i64::try_from(col).unwrap_or(i64::MAX),
                            row: i64::try_from(row).unwrap_or(i64::MAX),
                            tile_width,
                            tile_height,
                        };
                        let tile = slide.read_display_tile(&request)?;
                        tiles.push(Some(write_tile_payload(&mut payload, &tile)?));
                    }
                }
                levels_meta.push(LevelMeta {
                    dimensions: level.dimensions,
                    downsample: level.downsample,
                    tile_width,
                    tile_height,
                    tiles_across,
                    tiles_down,
                    tiles,
                    sparse_tiles: Vec::new(),
                });
            }
            series_meta.push(SeriesMeta {
                id: series.id.clone(),
                axes: AxesMeta {
                    z: series.axes.z,
                    c: series.axes.c,
                    t: series.axes.t,
                },
                sample_type: SampleTypeMeta::Uint8,
                channels: series
                    .channels
                    .iter()
                    .map(|channel| ChannelMeta {
                        name: channel.name.clone(),
                        color: channel.color,
                    })
                    .collect(),
                levels: levels_meta,
            });
        }
        scenes.push(SceneMeta {
            id: scene.id.clone(),
            name: scene.name.clone(),
            series: series_meta,
        });
    }

    let associated = build_associated_payloads(&slide, &mut payload)?;
    let metadata = SvcacheMetadata {
        schema_version: SCHEMA_VERSION,
        complete: true,
        source: source_fingerprint,
        properties: slide
            .dataset()
            .properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        scenes,
        associated,
    };
    write_svcache_file(out_path, &metadata, payload)
}

pub fn build_svcache_tiles(
    source_path: &Path,
    out_path: &Path,
    selections: &[SvcacheTileSelection],
) -> Result<usize, WsiError> {
    build_svcache_tiles_with_existing_policy(
        source_path,
        out_path,
        selections,
        ExistingTilePolicy::Preserve,
    )
}

pub fn build_svcache_tiles_replace(
    source_path: &Path,
    out_path: &Path,
    selections: &[SvcacheTileSelection],
) -> Result<usize, WsiError> {
    build_svcache_tiles_with_existing_policy(
        source_path,
        out_path,
        selections,
        ExistingTilePolicy::Replace,
    )
}

pub fn build_svcache_tile_payloads_replace(
    source_path: &Path,
    out_path: &Path,
    tiles: &[(SvcacheTileSelection, CpuTile)],
) -> Result<usize, WsiError> {
    build_svcache_tile_payloads_with_existing_policy(
        source_path,
        out_path,
        tiles,
        ExistingTilePolicy::Replace,
    )
}

pub fn build_svcache_tile_payloads_merge(
    source_path: &Path,
    out_path: &Path,
    tiles: &[(SvcacheTileSelection, CpuTile)],
) -> Result<usize, WsiError> {
    build_svcache_tile_payloads_with_existing_policy(
        source_path,
        out_path,
        tiles,
        ExistingTilePolicy::Preserve,
    )
}

fn build_svcache_tile_payloads_with_existing_policy(
    source_path: &Path,
    out_path: &Path,
    tiles: &[(SvcacheTileSelection, CpuTile)],
    existing_tile_policy: ExistingTilePolicy,
) -> Result<usize, WsiError> {
    let registry = FormatRegistry::builtin_native();
    let source = registry.open_exact(source_path)?;
    let slide = Slide::from_source_with_cache_bytes(source, 256 * 1024 * 1024);
    let source_fingerprint = fingerprint_source(source_path)?;

    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut payload = tempfile::tempfile()?;
    let mut scenes = metadata_shell(slide.dataset())?;
    let _copied = copy_existing_svcache_tiles_with_policy(
        out_path,
        source_path,
        &mut scenes,
        &mut payload,
        existing_tile_policy,
    )?;

    let mut unique = tiles
        .iter()
        .map(|(selection, tile)| (*selection, tile))
        .collect::<Vec<_>>();
    unique.sort_by_key(|(selection, _)| {
        (
            selection.scene,
            selection.series,
            selection.level,
            selection.plane.z,
            selection.plane.c,
            selection.plane.t,
            selection.row,
            selection.col,
        )
    });
    unique.dedup_by_key(|(selection, _)| *selection);

    let mut written = 0usize;
    for (selection, tile) in unique {
        let (_, _, tiles_across, tiles_down) =
            level_grid_for_selection(slide.dataset(), selection)?;
        if selection.col < 0 || selection.row < 0 {
            return Err(WsiError::TileRead {
                col: selection.col,
                row: selection.row,
                level: selection.level,
                reason: ".svcache selection has negative tile coordinate".into(),
            });
        }
        let col = selection.col as u64;
        let row = selection.row as u64;
        if col >= tiles_across || row >= tiles_down {
            return Err(WsiError::TileRead {
                col: selection.col,
                row: selection.row,
                level: selection.level,
                reason: ".svcache selection tile coordinate out of range".into(),
            });
        }
        let idx = row
            .checked_mul(tiles_across)
            .and_then(|base| base.checked_add(col))
            .ok_or_else(|| WsiError::TileRead {
                col: selection.col,
                row: selection.row,
                level: selection.level,
                reason: ".svcache selection tile index overflow".into(),
            })?;
        if scenes[selection.scene].series[selection.series].levels[selection.level as usize]
            .tile_meta_for_index(idx)
            .is_some()
        {
            continue;
        }
        let tile_meta = write_tile_payload(&mut payload, tile)?;
        scenes[selection.scene].series[selection.series].levels[selection.level as usize]
            .insert_tile_for_index(idx, tile_meta);
        written += 1;
    }

    let metadata = SvcacheMetadata {
        schema_version: SCHEMA_VERSION,
        complete: false,
        source: source_fingerprint,
        properties: slide
            .dataset()
            .properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        scenes,
        associated: Vec::new(),
    };
    write_svcache_file(out_path, &metadata, payload)?;
    Ok(written)
}

fn build_svcache_tiles_with_existing_policy(
    source_path: &Path,
    out_path: &Path,
    selections: &[SvcacheTileSelection],
    existing_tile_policy: ExistingTilePolicy,
) -> Result<usize, WsiError> {
    let registry = FormatRegistry::builtin_native();
    let source = registry.open_exact(source_path)?;
    let slide = Slide::from_source_with_cache_bytes(source, 256 * 1024 * 1024);
    let source_fingerprint = fingerprint_source(source_path)?;

    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut payload = tempfile::tempfile()?;
    let mut scenes = metadata_shell(slide.dataset())?;
    let _copied = copy_existing_svcache_tiles_with_policy(
        out_path,
        source_path,
        &mut scenes,
        &mut payload,
        existing_tile_policy,
    )?;
    let mut seen = HashSet::new();
    let mut unique = Vec::with_capacity(selections.len());
    for &selection in selections {
        if seen.insert(selection) {
            unique.push(selection);
        }
    }
    unique.sort_by_key(|selection| {
        (
            selection.scene,
            selection.series,
            selection.level,
            selection.plane.z,
            selection.plane.c,
            selection.plane.t,
            selection.row,
            selection.col,
        )
    });

    let mut written = 0usize;
    for selection in unique {
        let (tile_width, tile_height, tiles_across, tiles_down) =
            level_grid_for_selection(slide.dataset(), selection)?;
        if selection.col < 0 || selection.row < 0 {
            return Err(WsiError::TileRead {
                col: selection.col,
                row: selection.row,
                level: selection.level,
                reason: ".svcache selection has negative tile coordinate".into(),
            });
        }
        let col = selection.col as u64;
        let row = selection.row as u64;
        if col >= tiles_across || row >= tiles_down {
            return Err(WsiError::TileRead {
                col: selection.col,
                row: selection.row,
                level: selection.level,
                reason: ".svcache selection tile coordinate out of range".into(),
            });
        }
        let idx = row
            .checked_mul(tiles_across)
            .and_then(|base| base.checked_add(col))
            .ok_or_else(|| WsiError::TileRead {
                col: selection.col,
                row: selection.row,
                level: selection.level,
                reason: ".svcache selection tile index overflow".into(),
            })?;
        if scenes[selection.scene].series[selection.series].levels[selection.level as usize]
            .tile_meta_for_index(idx)
            .is_some()
        {
            continue;
        }
        let request = TileViewRequest {
            scene: selection.scene,
            series: selection.series,
            level: selection.level,
            plane: selection.plane,
            col: selection.col,
            row: selection.row,
            tile_width,
            tile_height,
        };
        let tile = slide.read_display_tile(&request)?;
        let tile_meta = write_tile_payload(&mut payload, &tile)?;
        scenes[selection.scene].series[selection.series].levels[selection.level as usize]
            .insert_tile_for_index(idx, tile_meta);
        written += 1;
    }

    let metadata = SvcacheMetadata {
        schema_version: SCHEMA_VERSION,
        complete: false,
        source: source_fingerprint,
        properties: slide
            .dataset()
            .properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        scenes,
        associated: Vec::new(),
    };
    write_svcache_file(out_path, &metadata, payload)?;
    Ok(written)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExistingTilePolicy {
    Preserve,
    Replace,
}

fn copy_existing_svcache_tiles_with_policy(
    out_path: &Path,
    source_path: &Path,
    scenes: &mut [SceneMeta],
    payload: &mut File,
    policy: ExistingTilePolicy,
) -> Result<usize, WsiError> {
    match policy {
        ExistingTilePolicy::Preserve => {
            copy_existing_svcache_tiles(out_path, source_path, scenes, payload)
        }
        ExistingTilePolicy::Replace => Ok(0),
    }
}

fn copy_existing_svcache_tiles(
    out_path: &Path,
    source_path: &Path,
    scenes: &mut [SceneMeta],
    payload: &mut File,
) -> Result<usize, WsiError> {
    if !out_path.is_file() || !svcache_matches_source(out_path, source_path).unwrap_or(false) {
        return Ok(0);
    }
    let (mut existing_file, payload_start, existing_metadata) = read_svcache(out_path)?;
    let mut copied = 0usize;

    for (scene_idx, scene) in scenes.iter_mut().enumerate() {
        let Some(existing_scene) = existing_metadata.scenes.get(scene_idx) else {
            continue;
        };
        for (series_idx, series) in scene.series.iter_mut().enumerate() {
            let Some(existing_series) = existing_scene.series.get(series_idx) else {
                continue;
            };
            for (level_idx, level) in series.levels.iter_mut().enumerate() {
                let Some(existing_level) = existing_series.levels.get(level_idx) else {
                    continue;
                };
                let tile_limit = level
                    .tiles_across
                    .checked_mul(level.tiles_down)
                    .ok_or_else(|| WsiError::InvalidSlide {
                        path: out_path.into(),
                        message: "svcache level tile count overflow".into(),
                    })?;
                for (idx, existing_slot) in existing_level.tiles.iter().enumerate() {
                    let idx = u64::try_from(idx).map_err(|_| WsiError::InvalidSlide {
                        path: out_path.into(),
                        message: "svcache tile index overflow".into(),
                    })?;
                    if idx >= tile_limit {
                        break;
                    }
                    if level.tile_meta_for_index(idx).is_some() {
                        continue;
                    }
                    let Some(existing_tile) = existing_slot else {
                        continue;
                    };
                    let copied_tile = copy_tile_payload(
                        &mut existing_file,
                        payload_start,
                        existing_tile,
                        payload,
                    )?;
                    level.insert_tile_for_index(idx, copied_tile);
                    copied += 1;
                }
                for existing_entry in &existing_level.sparse_tiles {
                    if existing_entry.index >= tile_limit {
                        return Err(WsiError::InvalidSlide {
                            path: out_path.into(),
                            message: "svcache sparse tile index out of range".into(),
                        });
                    }
                    if level.tile_meta_for_index(existing_entry.index).is_some() {
                        continue;
                    }
                    let copied_tile = copy_tile_payload(
                        &mut existing_file,
                        payload_start,
                        &existing_entry.tile,
                        payload,
                    )?;
                    level.insert_tile_for_index(existing_entry.index, copied_tile);
                    copied += 1;
                }
            }
        }
    }

    Ok(copied)
}

fn copy_tile_payload(
    existing_file: &mut File,
    payload_start: u64,
    existing_tile: &TileMeta,
    payload: &mut File,
) -> Result<TileMeta, WsiError> {
    let source_offset = payload_start
        .checked_add(existing_tile.payload_offset)
        .ok_or_else(|| WsiError::InvalidSlide {
            path: PathBuf::from(".svcache"),
            message: "svcache payload offset overflow".into(),
        })?;
    existing_file.seek(SeekFrom::Start(source_offset))?;
    let payload_offset = payload.seek(SeekFrom::End(0))?;
    let mut limited = existing_file.take(existing_tile.payload_len);
    std::io::copy(&mut limited, payload)?;
    let mut copied = existing_tile.clone();
    copied.payload_offset = payload_offset;
    Ok(copied)
}

impl SvcacheBackend {
    pub fn new() -> Self {
        Self
    }
}

impl FormatProbe for SvcacheBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        let mut file = File::open(path)?;
        let mut magic = [0_u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return Ok(ProbeResult {
                detected: false,
                vendor: "svcache".into(),
                confidence: ProbeConfidence::Likely,
            });
        }
        Ok(ProbeResult {
            detected: &magic == MAGIC,
            vendor: "svcache".into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for SvcacheBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let (file, payload_start, metadata) = read_svcache(path)?;
        let dataset = dataset_from_metadata(path, &metadata);
        let associated_index = metadata
            .associated
            .iter()
            .enumerate()
            .map(|(idx, assoc)| (assoc.name.clone(), idx))
            .collect();
        Ok(Box::new(SvcacheReader {
            file: Mutex::new(file),
            payload_start,
            metadata,
            dataset,
            associated_index,
        }))
    }
}

impl SlideReader for SvcacheReader {
    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        let tile = self.tile_meta(req)?;
        self.read_tile_meta(tile)
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        if matches!(output, TileOutputPreference::RequireDevice { .. }) {
            return Err(WsiError::Unsupported {
                reason: ".svcache device output is not implemented".into(),
            });
        }
        reqs.iter()
            .map(|req| self.read_tile_cpu(req).map(TilePixels::Cpu))
            .collect()
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        let idx = self
            .associated_index
            .get(name)
            .copied()
            .ok_or_else(|| WsiError::AssociatedImageNotFound(name.into()))?;
        self.read_tile_meta(&self.metadata.associated[idx].tile)
    }
}

impl SvcacheReader {
    fn tile_meta(&self, req: &TileRequest) -> Result<&TileMeta, WsiError> {
        let level = self
            .metadata
            .scenes
            .get(req.scene)
            .and_then(|scene| scene.series.get(req.series))
            .and_then(|series| series.levels.get(req.level as usize))
            .ok_or_else(|| WsiError::LevelOutOfRange {
                level: req.level,
                count: 0,
            })?;
        if req.col < 0 || req.row < 0 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "negative .svcache tile coordinate".into(),
            });
        }
        let col = req.col as u64;
        let row = req.row as u64;
        if col >= level.tiles_across || row >= level.tiles_down {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: ".svcache tile coordinate out of range".into(),
            });
        }
        let idx = row
            .checked_mul(level.tiles_across)
            .and_then(|base| base.checked_add(col))
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: ".svcache tile index overflow".into(),
            })?;
        level
            .tile_meta_for_index(idx)
            .ok_or_else(|| WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: ".svcache tile not populated".into(),
            })
    }

    fn read_tile_meta(&self, tile: &TileMeta) -> Result<CpuTile, WsiError> {
        let mut encoded = vec![0_u8; tile.payload_len as usize];
        {
            let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
            file.seek(SeekFrom::Start(self.payload_start + tile.payload_offset))?;
            file.read_exact(&mut encoded)?;
        }
        let actual_hash = hex_encode(&Sha256::digest(&encoded));
        if actual_hash != tile.sha256 {
            return Err(WsiError::InvalidSlide {
                path: PathBuf::from(&self.metadata.source.path),
                message: "svcache tile checksum mismatch".into(),
            });
        }
        let decoded = match tile.codec {
            PayloadCodec::Zstd => {
                zstd::bulk::decompress(&encoded, tile.decoded_len).map_err(|err| {
                    WsiError::Codec {
                        codec: "svcache-zstd",
                        source: Box::new(err),
                    }
                })?
            }
        };
        CpuTile::from_u8_interleaved(
            tile.width,
            tile.height,
            tile.channels,
            tile.color_space.into(),
            decoded,
        )
    }
}

fn metadata_shell(dataset: &Dataset) -> Result<Vec<SceneMeta>, WsiError> {
    let mut scenes = Vec::with_capacity(dataset.scenes.len());
    for scene in &dataset.scenes {
        let mut series_meta = Vec::with_capacity(scene.series.len());
        for series in &scene.series {
            let mut levels_meta = Vec::with_capacity(series.levels.len());
            for level in &series.levels {
                let (tile_width, tile_height, tiles_across, tiles_down) =
                    cache_grid_for_level(level);
                levels_meta.push(LevelMeta {
                    dimensions: level.dimensions,
                    downsample: level.downsample,
                    tile_width,
                    tile_height,
                    tiles_across,
                    tiles_down,
                    tiles: Vec::new(),
                    sparse_tiles: Vec::new(),
                });
            }
            series_meta.push(SeriesMeta {
                id: series.id.clone(),
                axes: AxesMeta {
                    z: series.axes.z,
                    c: series.axes.c,
                    t: series.axes.t,
                },
                sample_type: SampleTypeMeta::Uint8,
                channels: series
                    .channels
                    .iter()
                    .map(|channel| ChannelMeta {
                        name: channel.name.clone(),
                        color: channel.color,
                    })
                    .collect(),
                levels: levels_meta,
            });
        }
        scenes.push(SceneMeta {
            id: scene.id.clone(),
            name: scene.name.clone(),
            series: series_meta,
        });
    }
    Ok(scenes)
}

fn level_grid_for_selection(
    dataset: &Dataset,
    selection: SvcacheTileSelection,
) -> Result<(u32, u32, u64, u64), WsiError> {
    let level = dataset
        .scenes
        .get(selection.scene)
        .and_then(|scene| scene.series.get(selection.series))
        .and_then(|series| series.levels.get(selection.level as usize))
        .ok_or_else(|| WsiError::LevelOutOfRange {
            level: selection.level,
            count: 0,
        })?;
    Ok(cache_grid_for_level(level))
}

fn cache_grid_for_level(level: &Level) -> (u32, u32, u64, u64) {
    match &level.tile_layout {
        TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } => (*tile_width, *tile_height, *tiles_across, *tiles_down),
        TileLayout::WholeLevel { width, height, .. } => (
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
            width.div_ceil(u64::from(DEFAULT_TILE_SIZE)),
            height.div_ceil(u64::from(DEFAULT_TILE_SIZE)),
        ),
        TileLayout::Irregular { .. } => {
            let width = level.dimensions.0;
            let height = level.dimensions.1;
            (
                DEFAULT_TILE_SIZE,
                DEFAULT_TILE_SIZE,
                width.div_ceil(u64::from(DEFAULT_TILE_SIZE)),
                height.div_ceil(u64::from(DEFAULT_TILE_SIZE)),
            )
        }
    }
}

fn write_tile_payload(file: &mut File, tile: &CpuTile) -> Result<TileMeta, WsiError> {
    if tile.layout != CpuTileLayout::Interleaved || tile.data.sample_type() != SampleType::Uint8 {
        return Err(WsiError::UnsupportedFormat(
            ".svcache builder only supports interleaved uint8 display tiles".into(),
        ));
    }
    let raw = tile.data.as_u8().ok_or_else(|| {
        WsiError::UnsupportedFormat(".svcache builder expected uint8 tile data".into())
    })?;
    let color_space = ColorSpaceMeta::try_from(&tile.color_space)?;
    let encoded = zstd::bulk::compress(raw, 1).map_err(|err| WsiError::Codec {
        codec: "svcache-zstd",
        source: Box::new(err),
    })?;
    let payload_offset = file.stream_position()?;
    file.write_all(&encoded)?;
    Ok(TileMeta {
        payload_offset,
        payload_len: encoded.len() as u64,
        decoded_len: raw.len(),
        width: tile.width,
        height: tile.height,
        channels: tile.channels,
        color_space,
        codec: PayloadCodec::Zstd,
        sha256: hex_encode(&Sha256::digest(&encoded)),
    })
}

fn build_associated_payloads(
    slide: &Slide,
    payload: &mut File,
) -> Result<Vec<AssociatedMeta>, WsiError> {
    let mut associated = Vec::new();
    let mut names = slide
        .dataset()
        .associated_images
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    if names.is_empty() {
        names.extend(
            ["thumbnail", "macro", "label"]
                .into_iter()
                .map(str::to_string),
        );
    }
    names.sort();
    names.dedup();
    for name in names {
        match slide.read_associated(&name) {
            Ok(tile) => associated.push(AssociatedMeta {
                name,
                dimensions: (tile.width, tile.height),
                tile: write_tile_payload(payload, &tile)?,
            }),
            Err(WsiError::AssociatedImageNotFound(_)) => {}
            Err(err) => return Err(err),
        }
    }
    Ok(associated)
}

fn write_svcache_file(
    out_path: &Path,
    metadata: &SvcacheMetadata,
    mut payload: File,
) -> Result<(), WsiError> {
    let metadata_json = serde_json::to_vec(metadata).map_err(|err| WsiError::InvalidSlide {
        path: out_path.into(),
        message: format!("serialize svcache metadata: {err}"),
    })?;
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = tempfile::NamedTempFile::new_in(parent)?;
    out.write_all(MAGIC)?;
    out.write_all(&(metadata_json.len() as u64).to_le_bytes())?;
    out.write_all(&metadata_json)?;
    payload.seek(SeekFrom::Start(0))?;
    std::io::copy(&mut payload, &mut out)?;
    out.flush()?;
    out.persist(out_path)
        .map_err(|err| WsiError::Io(err.error))?;
    Ok(())
}

fn read_svcache(path: &Path) -> Result<(File, u64, SvcacheMetadata), WsiError> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(WsiError::UnsupportedFormat(format!(
            "{} is not an .svcache file",
            path.display()
        )));
    }
    let mut len = [0_u8; 8];
    file.read_exact(&mut len)?;
    let metadata_len = u64::from_le_bytes(len);
    if metadata_len > 128 * 1024 * 1024 {
        return Err(WsiError::InvalidSlide {
            path: path.into(),
            message: "svcache metadata is too large".into(),
        });
    }
    let mut metadata_bytes = vec![0_u8; metadata_len as usize];
    file.read_exact(&mut metadata_bytes)?;
    let metadata: SvcacheMetadata =
        serde_json::from_slice(&metadata_bytes).map_err(|err| WsiError::InvalidSlide {
            path: path.into(),
            message: format!("parse svcache metadata: {err}"),
        })?;
    if metadata.schema_version != SCHEMA_VERSION {
        return Err(WsiError::UnsupportedFormat(format!(
            "unsupported svcache schema {}",
            metadata.schema_version
        )));
    }
    let payload_start = 16 + metadata_len;
    Ok((file, payload_start, metadata))
}

fn is_fresh_svcache(cache_path: &Path, source_path: &Path) -> Result<bool, WsiError> {
    let (_, _, metadata) = read_svcache(cache_path)?;
    Ok(metadata.complete && metadata.source == fingerprint_source(source_path)?)
}

pub fn svcache_matches_source(cache_path: &Path, source_path: &Path) -> Result<bool, WsiError> {
    let (_, _, metadata) = read_svcache(cache_path)?;
    Ok(metadata.source == fingerprint_source(source_path)?)
}

fn fingerprint_source(path: &Path) -> Result<SourceFingerprint, WsiError> {
    let meta = std::fs::metadata(path)?;
    let modified_unix_nanos = meta
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos());
    Ok(SourceFingerprint {
        path: path.to_string_lossy().to_string(),
        len: meta.len(),
        modified_unix_nanos,
    })
}

fn dataset_from_metadata(path: &Path, metadata: &SvcacheMetadata) -> Dataset {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(metadata.source.path.as_bytes());
    let digest = hasher.finalize();
    let mut id_bytes = [0_u8; 16];
    id_bytes.copy_from_slice(&digest[..16]);
    let mut properties = Properties::new();
    for (key, value) in &metadata.properties {
        properties.insert(key.clone(), value.clone());
    }
    properties.insert("openslide.vendor", "svcache");

    Dataset {
        id: DatasetId(u128::from_le_bytes(id_bytes)),
        scenes: metadata
            .scenes
            .iter()
            .map(|scene| Scene {
                id: scene.id.clone(),
                name: scene.name.clone(),
                series: scene
                    .series
                    .iter()
                    .map(|series| Series {
                        id: series.id.clone(),
                        axes: AxesShape {
                            z: series.axes.z,
                            c: series.axes.c,
                            t: series.axes.t,
                        },
                        levels: series
                            .levels
                            .iter()
                            .map(|level| Level {
                                dimensions: level.dimensions,
                                downsample: level.downsample,
                                tile_layout: TileLayout::Regular {
                                    tile_width: level.tile_width,
                                    tile_height: level.tile_height,
                                    tiles_across: level.tiles_across,
                                    tiles_down: level.tiles_down,
                                },
                            })
                            .collect(),
                        sample_type: SampleType::Uint8,
                        channels: series
                            .channels
                            .iter()
                            .map(|channel| ChannelInfo {
                                name: channel.name.clone(),
                                color: channel.color,
                                excitation_nm: None,
                                emission_nm: None,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
        associated_images: metadata
            .associated
            .iter()
            .map(|assoc| {
                (
                    assoc.name.clone(),
                    AssociatedImage {
                        dimensions: assoc.dimensions,
                        sample_type: SampleType::Uint8,
                        channels: assoc.tile.channels,
                    },
                )
            })
            .collect(),
        properties,
        icc_profiles: HashMap::new(),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

impl TryFrom<&ColorSpace> for ColorSpaceMeta {
    type Error = WsiError;

    fn try_from(value: &ColorSpace) -> Result<Self, Self::Error> {
        match value {
            ColorSpace::Rgb => Ok(Self::Rgb),
            ColorSpace::Rgba => Ok(Self::Rgba),
            ColorSpace::Grayscale => Ok(Self::Grayscale),
            other => Err(WsiError::UnsupportedFormat(format!(
                ".svcache builder does not support {:?} display tiles",
                other
            ))),
        }
    }
}

impl From<ColorSpaceMeta> for ColorSpace {
    fn from(value: ColorSpaceMeta) -> Self {
        match value {
            ColorSpaceMeta::Rgb => ColorSpace::Rgb,
            ColorSpaceMeta::Rgba => ColorSpace::Rgba,
            ColorSpaceMeta::Grayscale => ColorSpace::Grayscale,
        }
    }
}

impl PartialEq for SourceFingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.modified_unix_nanos == other.modified_unix_nanos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::CpuTile;

    #[test]
    fn svcache_round_trips_single_tile() {
        let mut payload = tempfile::tempfile().unwrap();
        let tile =
            CpuTile::from_u8_interleaved(1, 1, 3, ColorSpace::Rgb, vec![10_u8, 20_u8, 30_u8])
                .unwrap();
        let tile_meta = write_tile_payload(&mut payload, &tile).unwrap();
        let source = tempfile::NamedTempFile::new().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("roundtrip.svcache");
        let metadata = SvcacheMetadata {
            schema_version: SCHEMA_VERSION,
            complete: true,
            source: fingerprint_source(source.path()).unwrap(),
            properties: Vec::new(),
            scenes: vec![SceneMeta {
                id: "scene-0".into(),
                name: None,
                series: vec![SeriesMeta {
                    id: "series-0".into(),
                    axes: AxesMeta { z: 1, c: 1, t: 1 },
                    sample_type: SampleTypeMeta::Uint8,
                    channels: Vec::new(),
                    levels: vec![LevelMeta {
                        dimensions: (1, 1),
                        downsample: 1.0,
                        tile_width: 1,
                        tile_height: 1,
                        tiles_across: 1,
                        tiles_down: 1,
                        tiles: vec![Some(tile_meta)],
                        sparse_tiles: Vec::new(),
                    }],
                }],
            }],
            associated: Vec::new(),
        };
        write_svcache_file(&out_path, &metadata, payload).unwrap();

        let backend = SvcacheBackend::new();
        let reader = backend.open(&out_path).unwrap();
        let decoded = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();

        assert_eq!(decoded.data.as_u8().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn svcache_sparse_level_reports_missing_tile() {
        let payload = tempfile::tempfile().unwrap();
        let source = tempfile::NamedTempFile::new().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("sparse.svcache");
        let metadata = SvcacheMetadata {
            schema_version: SCHEMA_VERSION,
            complete: false,
            source: fingerprint_source(source.path()).unwrap(),
            properties: Vec::new(),
            scenes: vec![SceneMeta {
                id: "scene-0".into(),
                name: None,
                series: vec![SeriesMeta {
                    id: "series-0".into(),
                    axes: AxesMeta { z: 1, c: 1, t: 1 },
                    sample_type: SampleTypeMeta::Uint8,
                    channels: Vec::new(),
                    levels: vec![LevelMeta {
                        dimensions: (2, 1),
                        downsample: 1.0,
                        tile_width: 1,
                        tile_height: 1,
                        tiles_across: 2,
                        tiles_down: 1,
                        tiles: vec![None, None],
                        sparse_tiles: Vec::new(),
                    }],
                }],
            }],
            associated: Vec::new(),
        };
        write_svcache_file(&out_path, &metadata, payload).unwrap();

        let backend = SvcacheBackend::new();
        let reader = backend.open(&out_path).unwrap();
        let err = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 0,
            })
            .unwrap_err();

        assert!(
            err.to_string().contains(".svcache tile not populated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn whole_level_cache_grid_uses_display_tiles() {
        let level = Level {
            dimensions: (3_596, 2_912),
            downsample: 32.0,
            tile_layout: TileLayout::WholeLevel {
                width: 3_596,
                height: 2_912,
                virtual_tile_width: 3_596,
                virtual_tile_height: 2_912,
            },
        };

        assert_eq!(cache_grid_for_level(&level), (256, 256, 15, 12));
    }

    #[test]
    fn partial_svcache_metadata_shell_starts_sparse() {
        let dataset = Dataset {
            id: DatasetId(42),
            scenes: vec![Scene {
                id: "scene-0".into(),
                name: None,
                series: vec![Series {
                    id: "series-0".into(),
                    axes: AxesShape::default(),
                    levels: vec![Level {
                        dimensions: (65_536, 65_536),
                        downsample: 1.0,
                        tile_layout: TileLayout::Regular {
                            tile_width: 256,
                            tile_height: 256,
                            tiles_across: 256,
                            tiles_down: 256,
                        },
                    }],
                    sample_type: SampleType::Uint8,
                    channels: Vec::new(),
                }],
            }],
            associated_images: std::collections::HashMap::new(),
            properties: Properties::new(),
            icc_profiles: std::collections::HashMap::new(),
        };

        let scenes = metadata_shell(&dataset).unwrap();
        let level = &scenes[0].series[0].levels[0];

        assert!(level.tiles.is_empty());
        assert!(level.sparse_tiles.is_empty());
        assert_eq!(level.tiles_across, 256);
        assert_eq!(level.tiles_down, 256);
    }

    #[test]
    fn sparse_svcache_is_not_fresh_for_auto_resolution() {
        let payload = tempfile::tempfile().unwrap();
        let source = tempfile::NamedTempFile::new().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("sparse.svcache");
        let metadata = SvcacheMetadata {
            schema_version: SCHEMA_VERSION,
            complete: false,
            source: fingerprint_source(source.path()).unwrap(),
            properties: Vec::new(),
            scenes: vec![SceneMeta {
                id: "scene-0".into(),
                name: None,
                series: Vec::new(),
            }],
            associated: Vec::new(),
        };
        write_svcache_file(&out_path, &metadata, payload).unwrap();

        assert!(!is_fresh_svcache(&out_path, source.path()).unwrap());
    }

    #[test]
    fn sparse_svcache_can_match_source_for_read_through_overlay() {
        let payload = tempfile::tempfile().unwrap();
        let source = tempfile::NamedTempFile::new().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("sparse-overlay.svcache");
        let metadata = SvcacheMetadata {
            schema_version: SCHEMA_VERSION,
            complete: false,
            source: fingerprint_source(source.path()).unwrap(),
            properties: Vec::new(),
            scenes: Vec::new(),
            associated: Vec::new(),
        };
        write_svcache_file(&out_path, &metadata, payload).unwrap();

        assert!(svcache_matches_source(&out_path, source.path()).unwrap());
    }

    #[test]
    fn sparse_svcache_merge_preserves_existing_tiles() {
        let mut existing_payload = tempfile::tempfile().unwrap();
        let tile =
            CpuTile::from_u8_interleaved(1, 1, 3, ColorSpace::Rgb, vec![1_u8, 2_u8, 3_u8]).unwrap();
        let existing_tile = write_tile_payload(&mut existing_payload, &tile).unwrap();
        let source = tempfile::NamedTempFile::new().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("merge.svcache");
        let metadata = SvcacheMetadata {
            schema_version: SCHEMA_VERSION,
            complete: false,
            source: fingerprint_source(source.path()).unwrap(),
            properties: Vec::new(),
            scenes: vec![SceneMeta {
                id: "scene-0".into(),
                name: None,
                series: vec![SeriesMeta {
                    id: "series-0".into(),
                    axes: AxesMeta { z: 1, c: 1, t: 1 },
                    sample_type: SampleTypeMeta::Uint8,
                    channels: Vec::new(),
                    levels: vec![LevelMeta {
                        dimensions: (2, 1),
                        downsample: 1.0,
                        tile_width: 1,
                        tile_height: 1,
                        tiles_across: 2,
                        tiles_down: 1,
                        tiles: vec![Some(existing_tile), None],
                        sparse_tiles: Vec::new(),
                    }],
                }],
            }],
            associated: Vec::new(),
        };
        write_svcache_file(&out_path, &metadata, existing_payload).unwrap();

        let mut merged_payload = tempfile::tempfile().unwrap();
        let mut scenes = metadata.scenes.clone();
        scenes[0].series[0].levels[0].tiles = vec![None, None];

        let copied =
            copy_existing_svcache_tiles(&out_path, source.path(), &mut scenes, &mut merged_payload)
                .unwrap();

        assert_eq!(copied, 1);
        assert!(scenes[0].series[0].levels[0].tiles[0].is_some());
        assert!(scenes[0].series[0].levels[0].tiles[1].is_none());
    }

    #[test]
    fn sparse_svcache_replace_does_not_copy_existing_tiles() {
        let mut existing_payload = tempfile::tempfile().unwrap();
        let tile =
            CpuTile::from_u8_interleaved(1, 1, 3, ColorSpace::Rgb, vec![1_u8, 2_u8, 3_u8]).unwrap();
        let existing_tile = write_tile_payload(&mut existing_payload, &tile).unwrap();
        let source = tempfile::NamedTempFile::new().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("replace.svcache");
        let metadata = SvcacheMetadata {
            schema_version: SCHEMA_VERSION,
            complete: false,
            source: fingerprint_source(source.path()).unwrap(),
            properties: Vec::new(),
            scenes: vec![SceneMeta {
                id: "scene-0".into(),
                name: None,
                series: vec![SeriesMeta {
                    id: "series-0".into(),
                    axes: AxesMeta { z: 1, c: 1, t: 1 },
                    sample_type: SampleTypeMeta::Uint8,
                    channels: Vec::new(),
                    levels: vec![LevelMeta {
                        dimensions: (2, 1),
                        downsample: 1.0,
                        tile_width: 1,
                        tile_height: 1,
                        tiles_across: 2,
                        tiles_down: 1,
                        tiles: vec![Some(existing_tile), None],
                        sparse_tiles: Vec::new(),
                    }],
                }],
            }],
            associated: Vec::new(),
        };
        write_svcache_file(&out_path, &metadata, existing_payload).unwrap();

        let mut replacement_payload = tempfile::tempfile().unwrap();
        let mut scenes = metadata.scenes.clone();
        scenes[0].series[0].levels[0].tiles = vec![None, None];

        let copied = copy_existing_svcache_tiles_with_policy(
            &out_path,
            source.path(),
            &mut scenes,
            &mut replacement_payload,
            ExistingTilePolicy::Replace,
        )
        .unwrap();

        assert_eq!(copied, 0);
        assert!(scenes[0].series[0].levels[0].tiles[0].is_none());
        assert!(scenes[0].series[0].levels[0].tiles[1].is_none());
    }

    #[test]
    fn build_svcache_tiles_replace_rewrites_selected_tiles_when_cache_exists() {
        let mut source = tempfile::Builder::new().suffix(".j2c").tempfile().unwrap();
        source
            .write_all(include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k"))
            .unwrap();
        source.flush().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("viewport.svcache");
        let selections = [SvcacheTileSelection {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        }];

        let first_written = build_svcache_tiles(source.path(), &out_path, &selections).unwrap();
        let merged_written = build_svcache_tiles(source.path(), &out_path, &selections).unwrap();
        let replaced_written =
            build_svcache_tiles_replace(source.path(), &out_path, &selections).unwrap();

        assert_eq!(first_written, 1);
        assert_eq!(merged_written, 0);
        assert_eq!(
            replaced_written, 1,
            "replace mode must not treat copied existing tiles as already populated"
        );

        let (_, _, metadata) = read_svcache(&out_path).unwrap();
        let level = &metadata.scenes[0].series[0].levels[0];
        assert!(
            level.tiles.is_empty(),
            "viewport caches must not serialize dense empty tile slots"
        );
        assert_eq!(level.sparse_tiles.len(), 1);
        assert_eq!(level.sparse_tiles[0].index, 0);

        let backend = SvcacheBackend::new();
        let reader = backend.open(&out_path).unwrap();
        reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();
    }

    #[test]
    fn build_svcache_tile_payloads_replace_writes_sparse_decoded_tiles() {
        let mut source = tempfile::Builder::new().suffix(".j2c").tempfile().unwrap();
        source
            .write_all(include_bytes!("../../tests/fixtures/jp2k/rgb_nomct.j2k"))
            .unwrap();
        source.flush().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("decoded-tiles.svcache");
        let selection = SvcacheTileSelection {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        };
        let tile = CpuTile::from_u8_interleaved(
            1,
            1,
            4,
            ColorSpace::Rgba,
            vec![11_u8, 22_u8, 33_u8, 44_u8],
        )
        .unwrap();

        let written =
            build_svcache_tile_payloads_replace(source.path(), &out_path, &[(selection, tile)])
                .unwrap();

        assert_eq!(written, 1);
        let (_, _, metadata) = read_svcache(&out_path).unwrap();
        let level = &metadata.scenes[0].series[0].levels[0];
        assert!(level.tiles.is_empty());
        assert_eq!(level.sparse_tiles.len(), 1);

        let backend = SvcacheBackend::new();
        let reader = backend.open(&out_path).unwrap();
        let decoded = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();
        assert_eq!(decoded.data.as_u8().unwrap(), &[11, 22, 33, 44]);
    }
}
