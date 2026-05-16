//! Signinum, reference, and OpenSlide oracle helpers.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jpeg_decoder::{Decoder as ReferenceJpegDecoder, PixelFormat as ReferenceJpegPixelFormat};
use statumen::{
    CpuTile, FormatRegistry, LevelIdx, PlaneIdx, PlaneSelection, RegionRequest, SceneId, SeriesId,
    Slide, TileLayout, TileRequest,
};

#[derive(Debug, Clone)]
pub struct TileBuffer {
    pub pixels_rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[allow(clippy::too_many_arguments)]
fn region_request(
    scene: usize,
    series: usize,
    level: u32,
    plane: PlaneSelection,
    x: i64,
    y: i64,
    w: u32,
    h: u32,
) -> RegionRequest {
    RegionRequest {
        scene: SceneId(scene),
        series: SeriesId(series),
        level: LevelIdx(level),
        plane: PlaneIdx(plane),
        origin_px: (x, y),
        size_px: (w, h),
    }
}

pub trait Oracle {
    fn name(&self) -> &'static str;
    fn open(&self, slide_path: &Path) -> Result<OpenedSlide, String>;
}

pub struct OpenedSlide {
    pub path: PathBuf,
    pub oracle_name: &'static str,
    pub level_count: u32,
    pub level_dimensions: Vec<(u64, u64)>,
    pub tile_sizes: Vec<Option<(u32, u32)>>,
    pub reader: OracleReader,
    pub region_reader: OracleReader,
}

pub type OracleReader =
    Box<dyn Fn(u32, i64, i64, u32, u32) -> Result<TileBuffer, String> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    Tile,
    Region,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeRequest {
    pub level: u32,
    pub x: i64,
    pub y: i64,
    pub width: u32,
    pub height: u32,
    pub kind: ProbeKind,
}

pub fn top_left_probe(slide: &OpenedSlide, level: u32) -> Option<ProbeRequest> {
    if let Some(Some((width, height))) = slide.tile_sizes.get(level as usize) {
        return Some(ProbeRequest {
            level,
            x: 0,
            y: 0,
            width: *width,
            height: *height,
            kind: ProbeKind::Tile,
        });
    }

    let (level_width, level_height) = *slide.level_dimensions.get(level as usize)?;
    let width = u32::try_from(level_width.min(256)).ok()?;
    let height = u32::try_from(level_height.min(256)).ok()?;
    if width == 0 || height == 0 {
        return None;
    }

    Some(ProbeRequest {
        level,
        x: 0,
        y: 0,
        width,
        height,
        kind: ProbeKind::Region,
    })
}

pub fn read_probe(slide: &OpenedSlide, probe: ProbeRequest) -> Result<TileBuffer, String> {
    match probe.kind {
        ProbeKind::Tile => (slide.reader)(probe.level, probe.x, probe.y, probe.width, probe.height),
        ProbeKind::Region => {
            (slide.region_reader)(probe.level, probe.x, probe.y, probe.width, probe.height)
        }
    }
}

pub fn is_reference_oracle_unsupported(err: &str) -> bool {
    err.starts_with("reference oracle unsupported")
}

pub struct SigninumOracle;

impl Oracle for SigninumOracle {
    fn name(&self) -> &'static str {
        "signinum"
    }

    fn open(&self, slide_path: &Path) -> Result<OpenedSlide, String> {
        open_via_statumen(slide_path, "signinum", false)
    }
}

pub struct ReferenceOracle;

impl Oracle for ReferenceOracle {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn open(&self, slide_path: &Path) -> Result<OpenedSlide, String> {
        open_via_statumen(slide_path, "reference", true)
    }
}

fn open_via_statumen(
    slide_path: &Path,
    name: &'static str,
    use_reference_jpeg: bool,
) -> Result<OpenedSlide, String> {
    let registry = FormatRegistry::builtin();
    let handle = Slide::open_with_cache_bytes(slide_path, &registry, 64 * 1024 * 1024)
        .map_err(|e| format!("statumen::open_with({}): {e}", slide_path.display()))?;
    let levels = &handle.dataset().scenes[0].series[0].levels;
    let level_count = levels.len() as u32;
    let level_dimensions: Vec<(u64, u64)> = levels.iter().map(|level| level.dimensions).collect();
    let tile_sizes = levels
        .iter()
        .map(|level| match &level.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                ..
            } => Some((*tile_width, *tile_height)),
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => Some((*virtual_tile_width, *virtual_tile_height)),
            TileLayout::Irregular { .. } => None,
        })
        .collect();
    let path_owned = slide_path.to_path_buf();
    let reader_path = path_owned.clone();
    let region_path = path_owned.clone();
    let reader_level_dimensions = level_dimensions.clone();
    let handle = Arc::new(handle);

    let reader_handle = Arc::clone(&handle);
    let reader: OracleReader = Box::new(move |level, x, y, width, height| {
        let level_meta = reader_handle
            .dataset()
            .scenes
            .first()
            .and_then(|scene| scene.series.first())
            .and_then(|series| series.levels.get(level as usize))
            .ok_or_else(|| format!("oracle: level {level} out of range"))?;
        let (tile_width, tile_height) = match &level_meta.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                ..
            } => (*tile_width, *tile_height),
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                ..
            } => (*virtual_tile_width, *virtual_tile_height),
            TileLayout::Irregular { .. } => {
                return Err(format!(
                    "oracle: irregular tile layout at level {level} not supported"
                ));
            }
        };
        if width != tile_width || height != tile_height {
            return Err(format!(
                "oracle: read size ({width}x{height}) must match tile size ({tile_width}x{tile_height})"
            ));
        }
        if x % i64::from(tile_width) != 0 || y % i64::from(tile_height) != 0 {
            return Err(format!(
                "oracle: read origin ({x},{y}) must be tile-aligned"
            ));
        }
        if use_reference_jpeg {
            return require_reference_tile(
                read_reference_tiff_jpeg_tile(
                    &reader_path,
                    &reader_level_dimensions,
                    level,
                    x,
                    y,
                    width,
                    height,
                ),
                format!(
                    "{} level={level} tile origin=({x},{y}) size={width}x{height}",
                    reader_path.display()
                ),
            );
        }
        let req = TileRequest {
            scene: 0,
            series: 0,
            level,
            plane: PlaneSelection::default(),
            col: x / i64::from(tile_width),
            row: y / i64::from(tile_height),
        };
        let buf = reader_handle
            .source()
            .read_tile_cpu(&req)
            .map_err(|e| format!("read_tile: {e}"))?;
        sample_buffer_to_rgba(buf)
    });

    let region_handle = Arc::clone(&handle);
    let region_reader: OracleReader = Box::new(move |level, x, y, width, height| {
        if use_reference_jpeg {
            return Err(format!(
                "reference oracle unsupported for {} level={level} region origin=({x},{y}) size={width}x{height}: independent region reference decode is not implemented",
                region_path.display()
            ));
        }
        let req = region_request(0, 0, level, PlaneSelection::default(), x, y, width, height);
        let buf = region_handle
            .read_region(&req)
            .map_err(|e| format!("read_region: {e}"))?;
        sample_buffer_to_rgba(buf)
    });

    Ok(OpenedSlide {
        path: path_owned,
        oracle_name: name,
        level_count,
        level_dimensions,
        tile_sizes,
        reader,
        region_reader,
    })
}

pub(crate) fn sample_buffer_to_rgba(buf: CpuTile) -> Result<TileBuffer, String> {
    let width = buf.width;
    let height = buf.height;
    let rgba = buf
        .into_rgba()
        .map_err(|e| format!("oracle: convert tile to RGBA: {e}"))?;
    Ok(TileBuffer {
        pixels_rgba: rgba.into_raw(),
        width,
        height,
    })
}

#[derive(Debug)]
pub(crate) enum ReferenceTileError {
    Unsupported(String),
    Fatal(String),
}

impl ReferenceTileError {
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    fn fatal(message: impl Into<String>) -> Self {
        Self::Fatal(message.into())
    }
}

pub(crate) fn require_reference_tile(
    result: Result<TileBuffer, ReferenceTileError>,
    context: impl AsRef<str>,
) -> Result<TileBuffer, String> {
    match result {
        Ok(tile) => Ok(tile),
        Err(ReferenceTileError::Unsupported(reason)) => Err(format!(
            "reference oracle unsupported for {}: {reason}",
            context.as_ref()
        )),
        Err(ReferenceTileError::Fatal(reason)) => Err(reason),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TiffEndian {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TiffKind {
    Classic,
    Big,
}

#[derive(Clone, Debug)]
struct TiffEntry {
    typ: u16,
    count: u64,
    value_or_offset: Vec<u8>,
    inline: bool,
}

#[derive(Clone, Debug)]
struct TiffIfd {
    offset: u64,
    tags: HashMap<u16, TiffEntry>,
}

struct TiffFile {
    bytes: Vec<u8>,
    endian: TiffEndian,
    kind: TiffKind,
    first_ifd: u64,
}

const TIFF_IMAGE_WIDTH: u16 = 256;
const TIFF_IMAGE_LENGTH: u16 = 257;
const TIFF_COMPRESSION: u16 = 259;
const TIFF_SUB_IFDS: u16 = 330;
const TIFF_TILE_WIDTH: u16 = 322;
const TIFF_TILE_LENGTH: u16 = 323;
const TIFF_TILE_OFFSETS: u16 = 324;
const TIFF_TILE_BYTE_COUNTS: u16 = 325;
const TIFF_JPEG_TABLES: u16 = 347;

fn read_reference_tiff_jpeg_tile(
    slide_path: &Path,
    level_dimensions: &[(u64, u64)],
    level: u32,
    x: i64,
    y: i64,
    width: u32,
    height: u32,
) -> Result<TileBuffer, ReferenceTileError> {
    let target_dimensions = level_dimensions
        .get(level as usize)
        .copied()
        .ok_or_else(|| ReferenceTileError::fatal(format!("reference: level {level} missing")))?;
    let tiff = TiffFile::open(slide_path)?;
    let ifds = tiff.collect_ifds()?;
    let ifd = ifds
        .iter()
        .find(|ifd| {
            ifd.get_u64(&tiff, TIFF_IMAGE_WIDTH) == Some(target_dimensions.0)
                && ifd.get_u64(&tiff, TIFF_IMAGE_LENGTH) == Some(target_dimensions.1)
        })
        .ok_or_else(|| {
            ReferenceTileError::unsupported(format!(
                "reference: no TIFF IFD for level {level} dimensions {target_dimensions:?}"
            ))
        })?;
    let compression = ifd.get_u64(&tiff, TIFF_COMPRESSION).unwrap_or(1);
    if compression != 7 {
        return Err(ReferenceTileError::unsupported(format!(
            "reference: TIFF IFD at {} compression {compression} is not JPEG",
            ifd.offset
        )));
    }

    let tile_width = ifd.get_u64(&tiff, TIFF_TILE_WIDTH).ok_or_else(|| {
        ReferenceTileError::unsupported(format!(
            "reference: TIFF IFD at {} is not tiled",
            ifd.offset
        ))
    })?;
    let tile_height = ifd.get_u64(&tiff, TIFF_TILE_LENGTH).ok_or_else(|| {
        ReferenceTileError::unsupported(format!(
            "reference: TIFF IFD at {} is not tiled",
            ifd.offset
        ))
    })?;
    if u64::from(width) != tile_width || u64::from(height) != tile_height {
        return Err(ReferenceTileError::unsupported(format!(
            "reference: requested {width}x{height}, TIFF tile is {tile_width}x{tile_height}"
        )));
    }
    if x < 0
        || y < 0
        || !(x as u64).is_multiple_of(tile_width)
        || !(y as u64).is_multiple_of(tile_height)
    {
        return Err(ReferenceTileError::unsupported(format!(
            "reference: request origin ({x},{y}) is not aligned to TIFF tile {tile_width}x{tile_height}"
        )));
    }

    let tiles_across = target_dimensions.0.div_ceil(tile_width);
    let col = x as u64 / tile_width;
    let row = y as u64 / tile_height;
    let tile_index = row
        .checked_mul(tiles_across)
        .and_then(|value| value.checked_add(col))
        .ok_or_else(|| ReferenceTileError::fatal("reference: TIFF tile index overflow"))?;

    let offsets = ifd.get_u64_array(&tiff, TIFF_TILE_OFFSETS)?;
    let byte_counts = ifd.get_u64_array(&tiff, TIFF_TILE_BYTE_COUNTS)?;
    let offset = *offsets.get(tile_index as usize).ok_or_else(|| {
        ReferenceTileError::fatal(format!(
            "reference: TIFF tile index {tile_index} missing offset in IFD {}",
            ifd.offset
        ))
    })?;
    let byte_count = *byte_counts.get(tile_index as usize).ok_or_else(|| {
        ReferenceTileError::fatal(format!(
            "reference: TIFF tile index {tile_index} missing byte count in IFD {}",
            ifd.offset
        ))
    })?;
    let data = tiff.read_range(offset, byte_count)?;
    let tables = match ifd.get_bytes(&tiff, TIFF_JPEG_TABLES) {
        Ok(bytes) => Some(bytes),
        Err(ReferenceTileError::Unsupported(_)) => None,
        Err(err) => return Err(err),
    };
    decode_reference_jpeg(&data, tables.as_deref(), width, height)
}

fn decode_reference_jpeg(
    data: &[u8],
    tables: Option<&[u8]>,
    width: u32,
    height: u32,
) -> Result<TileBuffer, ReferenceTileError> {
    let input = assemble_jpeg_tables(data, tables);
    let mut decoder = ReferenceJpegDecoder::new(Cursor::new(input.as_ref()));
    let pixels = decoder
        .decode()
        .map_err(|err| ReferenceTileError::fatal(format!("reference jpeg decode: {err}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| ReferenceTileError::fatal("reference jpeg decode returned no metadata"))?;
    let decoded_width = u32::from(info.width);
    let decoded_height = u32::from(info.height);
    if decoded_width < width || decoded_height < height {
        return Err(ReferenceTileError::fatal(format!(
            "reference jpeg decoded {decoded_width}x{decoded_height}, expected at least {width}x{height}"
        )));
    }
    let rgba = match info.pixel_format {
        ReferenceJpegPixelFormat::RGB24 => crop_rgb_to_rgba(&pixels, decoded_width, width, height)?,
        ReferenceJpegPixelFormat::L8 => {
            crop_luma_to_rgba(&pixels, decoded_width, width, height, 1)?
        }
        ReferenceJpegPixelFormat::L16 => {
            crop_luma_to_rgba(&pixels, decoded_width, width, height, 2)?
        }
        other => {
            return Err(ReferenceTileError::fatal(format!(
                "reference jpeg unsupported pixel format {other:?}"
            )));
        }
    };
    Ok(TileBuffer {
        pixels_rgba: rgba,
        width,
        height,
    })
}

fn assemble_jpeg_tables<'a>(data: &'a [u8], tables: Option<&[u8]>) -> std::borrow::Cow<'a, [u8]> {
    let Some(tables) = tables else {
        return std::borrow::Cow::Borrowed(data);
    };
    let tables_end = if tables.len() >= 2 && tables[tables.len() - 2..] == [0xff, 0xd9] {
        tables.len() - 2
    } else {
        tables.len()
    };
    let data_start = if data.len() >= 2 && data[..2] == [0xff, 0xd8] {
        2
    } else {
        0
    };
    std::borrow::Cow::Owned([&tables[..tables_end], &data[data_start..]].concat())
}

fn crop_rgb_to_rgba(
    pixels: &[u8],
    decoded_width: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ReferenceTileError> {
    let decoded_stride = decoded_width as usize * 3;
    let required = decoded_stride
        .checked_mul(height as usize)
        .ok_or_else(|| ReferenceTileError::fatal("reference jpeg RGB size overflow"))?;
    if pixels.len() < required {
        return Err(ReferenceTileError::fatal(format!(
            "reference jpeg RGB data too short: {} < {required}",
            pixels.len()
        )));
    }
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for row in 0..height as usize {
        let start = row * decoded_stride;
        for rgb in pixels[start..start + width as usize * 3].chunks_exact(3) {
            rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    Ok(rgba)
}

fn crop_luma_to_rgba(
    pixels: &[u8],
    decoded_width: u32,
    width: u32,
    height: u32,
    bytes_per_sample: usize,
) -> Result<Vec<u8>, ReferenceTileError> {
    let decoded_stride = decoded_width as usize * bytes_per_sample;
    let required = decoded_stride
        .checked_mul(height as usize)
        .ok_or_else(|| ReferenceTileError::fatal("reference jpeg luma size overflow"))?;
    if pixels.len() < required {
        return Err(ReferenceTileError::fatal(format!(
            "reference jpeg luma data too short: {} < {required}",
            pixels.len()
        )));
    }
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for row in 0..height as usize {
        let start = row * decoded_stride;
        for sample in
            pixels[start..start + width as usize * bytes_per_sample].chunks_exact(bytes_per_sample)
        {
            let value = sample[0];
            rgba.extend_from_slice(&[value, value, value, 255]);
        }
    }
    Ok(rgba)
}

impl TiffFile {
    fn open(path: &Path) -> Result<Self, ReferenceTileError> {
        let bytes = std::fs::read(path).map_err(|err| {
            ReferenceTileError::unsupported(format!(
                "reference: cannot read TIFF candidate {}: {err}",
                path.display()
            ))
        })?;
        if bytes.len() < 8 {
            return Err(ReferenceTileError::unsupported(
                "reference: file too small for TIFF header",
            ));
        }
        let endian = match &bytes[..2] {
            b"II" => TiffEndian::Little,
            b"MM" => TiffEndian::Big,
            _ => {
                return Err(ReferenceTileError::unsupported(
                    "reference: file is not TIFF",
                ));
            }
        };
        let magic = read_u16(endian, &bytes[2..4]);
        match magic {
            42 => Ok(Self {
                first_ifd: read_u32(endian, &bytes[4..8]) as u64,
                bytes,
                endian,
                kind: TiffKind::Classic,
            }),
            43 => {
                if bytes.len() < 16 {
                    return Err(ReferenceTileError::fatal(
                        "reference: BigTIFF header is truncated",
                    ));
                }
                if read_u16(endian, &bytes[4..6]) != 8 || read_u16(endian, &bytes[6..8]) != 0 {
                    return Err(ReferenceTileError::unsupported(
                        "reference: unsupported BigTIFF offset size",
                    ));
                }
                Ok(Self {
                    first_ifd: read_u64(endian, &bytes[8..16]),
                    bytes,
                    endian,
                    kind: TiffKind::Big,
                })
            }
            _ => Err(ReferenceTileError::unsupported(
                "reference: file is not classic TIFF or BigTIFF",
            )),
        }
    }

    fn collect_ifds(&self) -> Result<Vec<TiffIfd>, ReferenceTileError> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        self.collect_ifd_chain(self.first_ifd, &mut seen, &mut out)?;
        Ok(out)
    }

    fn collect_ifd_chain(
        &self,
        mut offset: u64,
        seen: &mut HashSet<u64>,
        out: &mut Vec<TiffIfd>,
    ) -> Result<(), ReferenceTileError> {
        while offset != 0 {
            if !seen.insert(offset) {
                break;
            }
            let ifd = self.read_ifd(offset)?;
            if let Some(entry) = ifd.tags.get(&TIFF_SUB_IFDS) {
                for sub_ifd_offset in entry.u64_array(self)? {
                    self.collect_ifd_chain(sub_ifd_offset, seen, out)?;
                }
            }
            offset = self.next_ifd_offset(offset)?;
            out.push(ifd);
        }
        Ok(())
    }

    fn read_ifd(&self, offset: u64) -> Result<TiffIfd, ReferenceTileError> {
        let count = match self.kind {
            TiffKind::Classic => self.read_u16_at(offset)? as u64,
            TiffKind::Big => self.read_u64_at(offset)?,
        };
        let entry_size = match self.kind {
            TiffKind::Classic => 12u64,
            TiffKind::Big => 20u64,
        };
        let entries_start = match self.kind {
            TiffKind::Classic => offset + 2,
            TiffKind::Big => offset + 8,
        };
        let mut tags = HashMap::new();
        for index in 0..count {
            let entry_offset = entries_start + index * entry_size;
            let raw = self.read_range(entry_offset, entry_size)?;
            let tag = read_u16(self.endian, &raw[0..2]);
            let typ = read_u16(self.endian, &raw[2..4]);
            let (entry_count, value_field) = match self.kind {
                TiffKind::Classic => (
                    read_u32(self.endian, &raw[4..8]) as u64,
                    raw[8..12].to_vec(),
                ),
                TiffKind::Big => (read_u64(self.endian, &raw[4..12]), raw[12..20].to_vec()),
            };
            let value_bytes = tiff_type_size(typ)
                .and_then(|size| size.checked_mul(entry_count))
                .ok_or_else(|| {
                    ReferenceTileError::fatal("reference: TIFF tag byte size overflow")
                })?;
            let inline = value_bytes <= value_field.len() as u64;
            tags.insert(
                tag,
                TiffEntry {
                    typ,
                    count: entry_count,
                    value_or_offset: value_field,
                    inline,
                },
            );
        }
        Ok(TiffIfd { offset, tags })
    }

    fn next_ifd_offset(&self, offset: u64) -> Result<u64, ReferenceTileError> {
        let count = match self.kind {
            TiffKind::Classic => self.read_u16_at(offset)? as u64,
            TiffKind::Big => self.read_u64_at(offset)?,
        };
        let next_offset_pos = match self.kind {
            TiffKind::Classic => offset + 2 + count * 12,
            TiffKind::Big => offset + 8 + count * 20,
        };
        match self.kind {
            TiffKind::Classic => Ok(self.read_u32_at(next_offset_pos)? as u64),
            TiffKind::Big => self.read_u64_at(next_offset_pos),
        }
    }

    fn read_range(&self, offset: u64, byte_len: u64) -> Result<Vec<u8>, ReferenceTileError> {
        let start = usize::try_from(offset)
            .map_err(|_| ReferenceTileError::fatal("reference: TIFF offset exceeds usize"))?;
        let len = usize::try_from(byte_len)
            .map_err(|_| ReferenceTileError::fatal("reference: TIFF byte count exceeds usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| ReferenceTileError::fatal("reference: TIFF byte range overflow"))?;
        let slice = self.bytes.get(start..end).ok_or_else(|| {
            ReferenceTileError::fatal(format!(
                "reference: TIFF byte range {offset}..{} is outside file",
                offset + byte_len
            ))
        })?;
        Ok(slice.to_vec())
    }

    fn read_u16_at(&self, offset: u64) -> Result<u16, ReferenceTileError> {
        let bytes = self.read_range(offset, 2)?;
        Ok(read_u16(self.endian, &bytes))
    }

    fn read_u32_at(&self, offset: u64) -> Result<u32, ReferenceTileError> {
        let bytes = self.read_range(offset, 4)?;
        Ok(read_u32(self.endian, &bytes))
    }

    fn read_u64_at(&self, offset: u64) -> Result<u64, ReferenceTileError> {
        let bytes = self.read_range(offset, 8)?;
        Ok(read_u64(self.endian, &bytes))
    }
}

impl TiffIfd {
    fn get_u64(&self, tiff: &TiffFile, tag: u16) -> Option<u64> {
        self.tags
            .get(&tag)
            .and_then(|entry| entry.u64_array(tiff).ok())
            .and_then(|values| values.into_iter().next())
    }

    fn get_u64_array(&self, tiff: &TiffFile, tag: u16) -> Result<Vec<u64>, ReferenceTileError> {
        self.tags
            .get(&tag)
            .ok_or_else(|| ReferenceTileError::fatal(format!("reference: TIFF tag {tag} missing")))?
            .u64_array(tiff)
    }

    fn get_bytes(&self, tiff: &TiffFile, tag: u16) -> Result<Vec<u8>, ReferenceTileError> {
        self.tags
            .get(&tag)
            .ok_or_else(|| {
                ReferenceTileError::unsupported(format!("reference: TIFF tag {tag} missing"))
            })?
            .bytes(tiff)
    }
}

impl TiffEntry {
    fn bytes(&self, tiff: &TiffFile) -> Result<Vec<u8>, ReferenceTileError> {
        let byte_len = tiff_type_size(self.typ)
            .and_then(|size| size.checked_mul(self.count))
            .ok_or_else(|| ReferenceTileError::fatal("reference: TIFF tag byte size overflow"))?;
        if self.inline {
            let len = usize::try_from(byte_len)
                .map_err(|_| ReferenceTileError::fatal("reference: TIFF inline tag too large"))?;
            return Ok(self.value_or_offset[..len].to_vec());
        }
        let offset = match tiff.kind {
            TiffKind::Classic => read_u32(tiff.endian, &self.value_or_offset[..4]) as u64,
            TiffKind::Big => read_u64(tiff.endian, &self.value_or_offset[..8]),
        };
        tiff.read_range(offset, byte_len)
    }

    fn u64_array(&self, tiff: &TiffFile) -> Result<Vec<u64>, ReferenceTileError> {
        let bytes = self.bytes(tiff)?;
        let size = tiff_type_size(self.typ).ok_or_else(|| {
            ReferenceTileError::fatal(format!("reference: unsupported TIFF type {}", self.typ))
        })? as usize;
        let mut values = Vec::with_capacity(self.count as usize);
        for chunk in bytes.chunks_exact(size) {
            let value = match self.typ {
                3 => read_u16(tiff.endian, chunk) as u64,
                4 | 13 => read_u32(tiff.endian, chunk) as u64,
                16 | 18 => read_u64(tiff.endian, chunk),
                other => {
                    return Err(ReferenceTileError::fatal(format!(
                        "reference: TIFF type {other} cannot be read as integer"
                    )));
                }
            };
            values.push(value);
        }
        Ok(values)
    }
}

fn tiff_type_size(typ: u16) -> Option<u64> {
    match typ {
        1 | 2 | 6 | 7 => Some(1),
        3 | 8 => Some(2),
        4 | 9 | 11 | 13 => Some(4),
        5 | 10 | 12 | 16 | 17 | 18 => Some(8),
        _ => None,
    }
}

fn read_u16(endian: TiffEndian, bytes: &[u8]) -> u16 {
    match endian {
        TiffEndian::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        TiffEndian::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    }
}

fn read_u32(endian: TiffEndian, bytes: &[u8]) -> u32 {
    match endian {
        TiffEndian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        TiffEndian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

fn read_u64(endian: TiffEndian, bytes: &[u8]) -> u64 {
    match endian {
        TiffEndian::Little => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        TiffEndian::Big => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

#[cfg(feature = "parity-openslide")]
pub struct OpenSlideOracle {
    pub lib: super::openslide_shim::LoadedOpenSlide,
}

#[cfg(feature = "parity-openslide")]
impl Oracle for OpenSlideOracle {
    fn name(&self) -> &'static str {
        "openslide"
    }

    fn open(&self, slide_path: &Path) -> Result<OpenedSlide, String> {
        let osr = self.lib.open(slide_path)?;
        let level_count = osr.level_count();
        let level_dimensions = (0..level_count)
            .map(|level| osr.level_dimensions(level))
            .collect::<Vec<_>>();
        let osr = Arc::new(osr);
        let reader_osr = Arc::clone(&osr);
        let reader: OracleReader = Box::new(move |level, x, y, width, height| {
            let pixels_rgba = reader_osr.read_region(x, y, level, width, height)?;
            Ok(TileBuffer {
                pixels_rgba,
                width,
                height,
            })
        });
        let region_osr = Arc::clone(&osr);
        let region_reader: OracleReader = Box::new(move |level, x, y, width, height| {
            let pixels_rgba = region_osr.read_region(x, y, level, width, height)?;
            Ok(TileBuffer {
                pixels_rgba,
                width,
                height,
            })
        });
        Ok(OpenedSlide {
            path: slide_path.to_path_buf(),
            oracle_name: "openslide",
            level_count,
            level_dimensions,
            tile_sizes: vec![None; level_count as usize],
            reader,
            region_reader,
        })
    }
}
