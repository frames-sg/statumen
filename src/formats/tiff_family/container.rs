use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use tracing::debug;

use super::error::{IfdId, TiffParseError};

/// Maximum number of entries allowed in a single IFD. Prevents DoS from crafted
/// BigTIFF files with huge entry counts.
const MAX_IFD_ENTRIES: u64 = 1_000_000;

/// Maximum byte size for a single tag payload read. Prevents OOM from crafted
/// tags with enormous count × type_size products.
const MAX_TAG_PAYLOAD: u64 = 256 * 1024 * 1024;

// ── Endianness ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Endian {
    Little,
    Big,
}

// ── TIFF tag constants ─────────────────────────────────────────────

pub(crate) mod tags {
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const IMAGE_DESCRIPTION: u16 = 270;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    #[cfg(test)]
    pub const SUB_IFDS: u16 = 330;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const X_RESOLUTION: u16 = 282;
    pub const Y_RESOLUTION: u16 = 283;
    pub const RESOLUTION_UNIT: u16 = 296;
    pub const PREDICTOR: u16 = 317;
    pub const JPEG_TABLES: u16 = 347;
    pub const XMP: u16 = 700;
    /// Hamamatsu NDPI marker tag.
    pub const NDPI_MARKER: u16 = 65420;
}

// ── TiffType ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub(crate) enum TiffType {
    Byte,      // 1: u8
    Ascii,     // 2: null-terminated string
    Short,     // 3: u16
    Long,      // 4: u32
    Rational,  // 5: u32/u32
    SByte,     // 6: i8
    Undefined, // 7: arbitrary bytes
    SShort,    // 8: i16
    SLong,     // 9: i32
    SRational, // 10: i32/i32
    Float,     // 11: f32
    Double,    // 12: f64
    Ifd,       // 13: u32 IFD offset
    Long8,     // 16: u64 (BigTIFF)
    SLong8,    // 17: i64 (BigTIFF)
    Ifd8,      // 18: u64 IFD offset (BigTIFF)
}

impl TiffType {
    pub fn from_u16(id: u16) -> Option<TiffType> {
        match id {
            1 => Some(TiffType::Byte),
            2 => Some(TiffType::Ascii),
            3 => Some(TiffType::Short),
            4 => Some(TiffType::Long),
            5 => Some(TiffType::Rational),
            6 => Some(TiffType::SByte),
            7 => Some(TiffType::Undefined),
            8 => Some(TiffType::SShort),
            9 => Some(TiffType::SLong),
            10 => Some(TiffType::SRational),
            11 => Some(TiffType::Float),
            12 => Some(TiffType::Double),
            13 => Some(TiffType::Ifd),
            16 => Some(TiffType::Long8),
            17 => Some(TiffType::SLong8),
            18 => Some(TiffType::Ifd8),
            _ => None,
        }
    }

    pub fn byte_size(&self) -> u64 {
        match self {
            TiffType::Byte | TiffType::Ascii | TiffType::SByte | TiffType::Undefined => 1,
            TiffType::Short | TiffType::SShort => 2,
            TiffType::Long | TiffType::SLong | TiffType::Float | TiffType::Ifd => 4,
            TiffType::Rational
            | TiffType::SRational
            | TiffType::Double
            | TiffType::Long8
            | TiffType::SLong8
            | TiffType::Ifd8 => 8,
        }
    }
}

// ── InlineValue ────────────────────────────────────────────────────

/// Stores inline tag data. Invariant: len <= 12.
/// Capacity of 12 covers BigTIFF's 8-byte value slot with headroom.
#[derive(Clone, Debug)]
pub(crate) struct InlineValue {
    len: u8,
    bytes: [u8; 12],
}

impl InlineValue {
    pub fn new(data: &[u8]) -> Self {
        assert!(data.len() <= 12, "InlineValue data exceeds 12 bytes");
        let mut bytes = [0u8; 12];
        bytes[..data.len()].copy_from_slice(data);
        InlineValue {
            len: data.len() as u8,
            bytes,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

// ── TagValue ───────────────────────────────────────────────────────

pub(crate) enum TagValue {
    Inline(InlineValue),
    Lazy {
        offset: u64,
        byte_len: u64,
        resolved: OnceLock<Result<Vec<u8>, TiffParseError>>,
    },
}

impl std::fmt::Debug for TagValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TagValue::Inline(v) => f.debug_tuple("Inline").field(v).finish(),
            TagValue::Lazy {
                offset, byte_len, ..
            } => f
                .debug_struct("Lazy")
                .field("offset", offset)
                .field("byte_len", byte_len)
                .finish(),
        }
    }
}

// ── TagEntry ───────────────────────────────────────────────────────

pub(crate) struct TagEntry {
    pub tiff_type: TiffType,
    pub count: u64,
    pub value: TagValue,
    pub(super) decoded_u64: OnceLock<Vec<u64>>,
}

impl std::fmt::Debug for TagEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TagEntry")
            .field("tiff_type", &self.tiff_type)
            .field("count", &self.count)
            .field("value", &self.value)
            .finish()
    }
}

impl TagEntry {
    pub fn new_inline(tiff_type: TiffType, count: u64, data: &[u8]) -> Self {
        TagEntry {
            tiff_type,
            count,
            value: TagValue::Inline(InlineValue::new(data)),
            decoded_u64: OnceLock::new(),
        }
    }

    pub fn new_lazy(tiff_type: TiffType, count: u64, offset: u64, byte_len: u64) -> Self {
        TagEntry {
            tiff_type,
            count,
            value: TagValue::Lazy {
                offset,
                byte_len,
                resolved: OnceLock::new(),
            },
            decoded_u64: OnceLock::new(),
        }
    }
}

// ── Ifd ────────────────────────────────────────────────────────────

pub(crate) struct Ifd {
    pub id: IfdId,
    pub offset: u64,
    pub tags: HashMap<u16, TagEntry>,
    /// References to SubIFDs by stable ID.
    pub sub_ifds: Vec<IfdId>,
}

impl std::fmt::Debug for Ifd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ifd")
            .field("id", &self.id)
            .field("offset", &self.offset)
            .field("tag_count", &self.tags.len())
            .field("sub_ifds", &self.sub_ifds)
            .finish()
    }
}

// ── ParseReader (used only during open()) ──────────────────────────

/// Sequential reader used during TiffContainer::open().
/// Wraps a BufReader and provides endian-aware reading.
/// Dropped when open() returns — not stored on TiffContainer.
struct ParseReader {
    reader: std::io::BufReader<std::fs::File>,
    endian: Endian,
    bigtiff: bool,
}

impl ParseReader {
    fn new(file: std::fs::File, endian: Endian, bigtiff: bool) -> Self {
        ParseReader {
            reader: std::io::BufReader::new(file),
            endian,
            bigtiff,
        }
    }

    fn read_u16(&mut self) -> Result<u16, TiffParseError> {
        let val = match self.endian {
            Endian::Little => self.reader.read_u16::<LittleEndian>()?,
            Endian::Big => self.reader.read_u16::<BigEndian>()?,
        };
        Ok(val)
    }

    fn read_u32(&mut self) -> Result<u32, TiffParseError> {
        let val = match self.endian {
            Endian::Little => self.reader.read_u32::<LittleEndian>()?,
            Endian::Big => self.reader.read_u32::<BigEndian>()?,
        };
        Ok(val)
    }

    fn read_u64(&mut self) -> Result<u64, TiffParseError> {
        let val = match self.endian {
            Endian::Little => self.reader.read_u64::<LittleEndian>()?,
            Endian::Big => self.reader.read_u64::<BigEndian>()?,
        };
        Ok(val)
    }

    fn read_bytes(&mut self, len: usize) -> Result<Vec<u8>, TiffParseError> {
        use std::io::Read;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn seek(&mut self, offset: u64) -> Result<(), TiffParseError> {
        use std::io::{Seek, SeekFrom};
        self.reader.seek(SeekFrom::Start(offset))?;
        Ok(())
    }
}

// ── TiffContainer ──────────────────────────────────────────────────

pub(crate) struct TiffContainer {
    path: Arc<PathBuf>,
    file_len: u64,
    /// Persistent file handle for pread — avoids re-opening per call.
    /// pread(2) on Unix is atomic and doesn't use the file position, so a
    /// single File can be shared across threads safely.
    file: std::fs::File,
    /// On Windows, `seek_read` is not atomic (it modifies the shared file
    /// position), so concurrent pread calls must be serialized.
    #[cfg(windows)]
    pread_lock: std::sync::Mutex<()>,
    endian: Endian,
    bigtiff: bool,
    ndpi: bool,
    /// Top-level IFD order as encountered in the main chain.
    top_ifds: Vec<IfdId>,
    /// Flat arena: every parsed IFD lives here exactly once, keyed by byte offset.
    ifds: HashMap<IfdId, Ifd>,
}

impl std::fmt::Debug for TiffContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiffContainer")
            .field("path", &self.path)
            .field("endian", &self.endian)
            .field("bigtiff", &self.bigtiff)
            .field("ndpi", &self.ndpi)
            .field("ifd_count", &self.ifds.len())
            .finish()
    }
}

impl TiffContainer {
    /// Open and parse a TIFF or BigTIFF file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TiffParseError> {
        let started = Instant::now();
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();

        // We need at least 8 bytes for a classic TIFF header
        if file_len < 8 {
            return Err(TiffParseError::Structure(format!(
                "file too small for TIFF header: {} bytes",
                file_len
            )));
        }

        // Parse header with a sequential reader
        let mut reader = ParseReader::new(file, Endian::Little, false); // endian TBD

        // Read byte order mark
        let bom = reader.read_bytes(2)?;
        let endian = match (bom[0], bom[1]) {
            (b'I', b'I') => Endian::Little,
            (b'M', b'M') => Endian::Big,
            _ => {
                return Err(TiffParseError::Structure(format!(
                    "invalid TIFF byte order marker: [{:#04x}, {:#04x}]",
                    bom[0], bom[1]
                )));
            }
        };
        reader.endian = endian;

        // Read magic number
        let magic = reader.read_u16()?;
        let bigtiff = match magic {
            42 => false,
            43 => true,
            _ => {
                return Err(TiffParseError::Structure(format!(
                    "invalid TIFF magic number: {} (expected 42 or 43)",
                    magic
                )));
            }
        };
        reader.bigtiff = bigtiff;

        // BigTIFF: validate offset size and reserved field
        if bigtiff {
            let offset_size = reader.read_u16()?;
            if offset_size != 8 {
                return Err(TiffParseError::Structure(format!(
                    "invalid BigTIFF offset size: {} (expected 8)",
                    offset_size
                )));
            }
            let reserved = reader.read_u16()?;
            if reserved != 0 {
                return Err(TiffParseError::Structure(format!(
                    "invalid BigTIFF reserved field: {} (expected 0)",
                    reserved
                )));
            }
        }

        let initial_ndpi = is_ndpi_extension(path);

        // Read first IFD offset
        let raw_first_ifd_offset = if bigtiff {
            reader.read_u64()?
        } else {
            reader.read_u32()? as u64
        };
        let first_ifd_offset = if initial_ndpi && !bigtiff {
            repair_ndpi_first_ifd_offset(&mut reader, raw_first_ifd_offset, file_len)?
        } else {
            raw_first_ifd_offset
        };

        let path_arc = Arc::new(path.to_path_buf());

        // Re-open a dedicated handle for pread (the parse reader consumes
        // its File and we don't want to share seek state).
        let pread_file = std::fs::File::open(path)?;

        let mut container = TiffContainer {
            path: path_arc,
            file_len,
            file: pread_file,
            #[cfg(windows)]
            pread_lock: std::sync::Mutex::new(()),
            endian,
            bigtiff,
            ndpi: initial_ndpi,
            top_ifds: Vec::new(),
            ifds: HashMap::new(),
        };

        // Walk the IFD chain
        container.walk_ifd_chain(&mut reader, first_ifd_offset)?;

        debug!(
            path = %path.display(),
            file_len = container.file_len,
            top_ifd_count = container.top_ifds.len(),
            ifd_count = container.ifds.len(),
            bigtiff = container.bigtiff,
            ndpi = container.ndpi,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "parsed TIFF container"
        );

        Ok(container)
    }

    #[cfg(test)]
    fn open_parse_reader(&self) -> Result<ParseReader, TiffParseError> {
        let file = std::fs::File::open(self.path.as_ref())?;
        Ok(ParseReader::new(file, self.endian, self.bigtiff))
    }

    // ── IFD chain walking ──────────────────────────────────────

    fn walk_ifd_chain(
        &mut self,
        reader: &mut ParseReader,
        first_ifd_offset: u64,
    ) -> Result<(), TiffParseError> {
        let mut ifd_offset = first_ifd_offset;
        let mut is_first = true;

        while ifd_offset != 0 {
            // Global safety limit
            if self.ifds.len() >= 10_000 {
                return Err(TiffParseError::Structure(
                    "too many IFDs (>10000), possible corrupt file".into(),
                ));
            }

            // Main-chain loop detection
            let ifd_id = IfdId(ifd_offset);
            if self.ifds.contains_key(&ifd_id) {
                return Err(TiffParseError::Structure(format!(
                    "IFD chain loop: offset {} already visited",
                    ifd_offset
                )));
            }

            let (ifd, next_offset) = self.parse_ifd(reader, ifd_offset)?;

            // NDPI detection: tag 65420 in first IFD
            if is_first && ifd.tags.contains_key(&tags::NDPI_MARKER) {
                self.ndpi = true;
            }
            is_first = false;

            self.top_ifds.push(ifd_id);
            self.ifds.insert(ifd_id, ifd);

            ifd_offset = next_offset;
        }

        Ok(())
    }

    /// Parse a single IFD at the given offset. Returns the Ifd and the next-IFD offset.
    fn parse_ifd(
        &mut self,
        reader: &mut ParseReader,
        offset: u64,
    ) -> Result<(Ifd, u64), TiffParseError> {
        reader.seek(offset)?;

        // Read entry count
        let entry_count: u64 = if self.bigtiff {
            reader.read_u64()?
        } else {
            reader.read_u16()? as u64
        };

        if entry_count > MAX_IFD_ENTRIES {
            return Err(TiffParseError::Structure(format!(
                "IFD entry count {} exceeds maximum {}",
                entry_count, MAX_IFD_ENTRIES
            )));
        }

        let slot_size: u64 = if self.bigtiff { 8 } else { 4 };
        let mut tags_map = HashMap::new();

        for _ in 0..entry_count {
            let tag_id = reader.read_u16()?;
            let type_id = reader.read_u16()?;

            let count: u64 = if self.bigtiff {
                reader.read_u64()?
            } else {
                reader.read_u32()? as u64
            };

            // Read the value/offset slot as raw bytes
            let slot_bytes = reader.read_bytes(slot_size as usize)?;

            // Try to interpret the type
            let tiff_type = match TiffType::from_u16(type_id) {
                Some(t) => t,
                None => {
                    // Unknown type — skip this entry
                    continue;
                }
            };

            let total_bytes = count.checked_mul(tiff_type.byte_size()).ok_or_else(|| {
                TiffParseError::InvalidTag {
                    ifd_offset: offset,
                    tag: tag_id,
                    message: format!(
                        "payload too large: count={}, type_size={}",
                        count,
                        tiff_type.byte_size()
                    ),
                }
            })?;

            if total_bytes > MAX_TAG_PAYLOAD {
                return Err(TiffParseError::InvalidTag {
                    ifd_offset: offset,
                    tag: tag_id,
                    message: format!(
                        "tag payload {} bytes exceeds {} byte limit",
                        total_bytes, MAX_TAG_PAYLOAD
                    ),
                });
            }

            let entry = if total_bytes <= slot_size {
                // Inline
                TagEntry::new_inline(tiff_type, count, &slot_bytes[..total_bytes as usize])
            } else {
                // Out-of-line: interpret slot as offset
                let data_offset = if self.bigtiff {
                    let mut c = std::io::Cursor::new(&slot_bytes);
                    match self.endian {
                        Endian::Little => c.read_u64::<LittleEndian>()?,
                        Endian::Big => c.read_u64::<BigEndian>()?,
                    }
                } else {
                    let mut c = std::io::Cursor::new(&slot_bytes);
                    let raw = match self.endian {
                        Endian::Little => c.read_u32::<LittleEndian>()?,
                        Endian::Big => c.read_u32::<BigEndian>()?,
                    };
                    let off = raw as u64;
                    // Apply NDPI fixup for out-of-line classic TIFF offsets
                    if self.ndpi {
                        fix_offset_ndpi(offset, off)
                    } else {
                        off
                    }
                };

                TagEntry::new_lazy(tiff_type, count, data_offset, total_bytes)
            };

            tags_map.insert(tag_id, entry);
        }

        // Read next IFD offset
        let next_offset = if self.ndpi && !self.bigtiff {
            // NDPI: read 8-byte next-IFD pointers even in classic TIFF
            reader.read_u64()?
        } else if self.bigtiff {
            reader.read_u64()?
        } else {
            reader.read_u32()? as u64
        };

        let ifd = Ifd {
            id: IfdId(offset),
            offset,
            tags: tags_map,
            sub_ifds: Vec::new(),
        };

        Ok((ifd, next_offset))
    }

    // ── SubIFD parsing ────────────────────────────────────────

    /// Parse SubIFDs referenced by tag 330 in an IFD.
    /// Adds newly discovered IFDs to the global arena. Deduplicates by offset.
    #[cfg(test)]
    pub fn materialize_sub_ifds(
        &mut self,
        parent_ifd_id: IfdId,
        max_depth: u32,
    ) -> Result<(), TiffParseError> {
        let mut reader = self.open_parse_reader()?;
        let mut ancestry = vec![parent_ifd_id];
        self.parse_sub_ifds(&mut reader, parent_ifd_id, 0, max_depth, &mut ancestry)
    }

    #[cfg(test)]
    pub fn materialize_all_sub_ifds(&mut self, max_depth: u32) -> Result<(), TiffParseError> {
        let root_ids = self.top_ifds.clone();
        let mut reader = self.open_parse_reader()?;
        for root_id in root_ids {
            let mut ancestry = vec![root_id];
            self.parse_sub_ifds(&mut reader, root_id, 0, max_depth, &mut ancestry)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn parse_sub_ifds(
        &mut self,
        reader: &mut ParseReader,
        parent_ifd_id: IfdId,
        depth: u32,
        max_depth: u32,
        ancestry: &mut Vec<IfdId>,
    ) -> Result<(), TiffParseError> {
        if depth > max_depth {
            return Err(TiffParseError::Structure(format!(
                "SubIFD depth limit exceeded (max {})",
                max_depth
            )));
        }

        // Check if parent has a SUB_IFDS tag
        let sub_ifd_offsets = {
            let parent = self
                .ifds
                .get(&parent_ifd_id)
                .ok_or(TiffParseError::IfdNotFound(parent_ifd_id))?;

            let entry = match parent.tags.get(&tags::SUB_IFDS) {
                Some(e) => e,
                None => return Ok(()), // No SubIFDs
            };

            // Extract offsets from inline or lazy data
            let bytes = match &entry.value {
                TagValue::Inline(v) => v.as_bytes().to_vec(),
                TagValue::Lazy {
                    offset, byte_len, ..
                } => self.pread(*offset, *byte_len)?,
            };

            // Decode as array of IFD offsets
            let elem_size = entry.tiff_type.byte_size() as usize;
            if elem_size == 0 {
                return Ok(());
            }
            let mut offsets = Vec::new();
            for chunk in bytes.chunks_exact(elem_size) {
                let off = match (entry.tiff_type, self.endian) {
                    (TiffType::Long | TiffType::Ifd, Endian::Little) => {
                        u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64
                    }
                    (TiffType::Long | TiffType::Ifd, Endian::Big) => {
                        u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64
                    }
                    (TiffType::Long8 | TiffType::Ifd8, Endian::Little) => u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]),
                    (TiffType::Long8 | TiffType::Ifd8, Endian::Big) => u64::from_be_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]),
                    _ => continue,
                };
                if off != 0 {
                    offsets.push(off);
                }
            }
            offsets
        };

        // Parse each SubIFD
        let mut child_ids = Vec::new();
        for sub_offset in &sub_ifd_offsets {
            let sub_id = IfdId(*sub_offset);

            if ancestry.contains(&sub_id) {
                return Err(TiffParseError::Structure(format!(
                    "SubIFD loop detected: offset {} already in ancestry",
                    sub_offset
                )));
            }

            if !self.ifds.contains_key(&sub_id) {
                // Safety limit
                if self.ifds.len() >= 10_000 {
                    return Err(TiffParseError::Structure(
                        "too many IFDs (>10000), possible corrupt file".into(),
                    ));
                }

                let (ifd, _next_offset) = self.parse_ifd(reader, *sub_offset)?;
                self.ifds.insert(sub_id, ifd);
            }
            child_ids.push(sub_id);

            ancestry.push(sub_id);
            self.parse_sub_ifds(reader, sub_id, depth + 1, max_depth, ancestry)?;
            ancestry.pop();
        }

        // Store child IDs on the parent
        if let Some(parent) = self.ifds.get_mut(&parent_ifd_id) {
            parent.sub_ifds = child_ids;
        }

        Ok(())
    }

    // ── pread (stateless positional read) ──────────────────────

    /// Perform a positional read using the persistent file handle.
    /// Uses pread(2) on Unix for lock-free concurrent reads (no seek state).
    pub fn pread(&self, offset: u64, len: u64) -> Result<Vec<u8>, TiffParseError> {
        // Bounds check with checked arithmetic
        let end = offset
            .checked_add(len)
            .ok_or(TiffParseError::Bounds { offset, len })?;
        if end > self.file_len {
            return Err(TiffParseError::Bounds { offset, len });
        }

        let len_usize = len as usize;
        let mut buf = vec![0u8; len_usize];

        #[cfg(unix)]
        {
            self.file.read_exact_at(&mut buf, offset)?;
        }

        #[cfg(windows)]
        {
            // seek_read on Windows is not atomic (it modifies the shared file
            // position), so we must serialize concurrent pread calls.
            let _guard = self.pread_lock.lock().unwrap_or_else(|e| e.into_inner());
            let n = self.file.seek_read(&mut buf, offset)?;
            if n != len_usize {
                return Err(TiffParseError::Io {
                    kind: std::io::ErrorKind::UnexpectedEof,
                    source: Arc::new(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!("seek_read: expected {} bytes, got {}", len_usize, n),
                    )),
                    path: Some(self.path.clone()),
                });
            }
        }

        #[cfg(not(any(unix, windows)))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(self.path.as_ref())?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut buf)?;
        }

        Ok(buf)
    }

    // ── Tag resolution ─────────────────────────────────────────

    /// Resolve a tag's raw bytes. For Inline, returns the bytes directly.
    /// For Lazy, performs a pread on first access and caches the result
    /// (including errors) permanently via OnceLock.
    pub fn resolve_tag(&self, ifd_id: IfdId, tag: u16) -> Result<&[u8], TiffParseError> {
        let ifd = self.ifd_by_id(ifd_id)?;
        let entry = ifd.tags.get(&tag).ok_or(TiffParseError::TagNotFound {
            ifd_offset: ifd.offset,
            tag,
        })?;

        match &entry.value {
            TagValue::Inline(v) => Ok(v.as_bytes()),
            TagValue::Lazy {
                offset,
                byte_len,
                resolved,
            } => {
                let offset = *offset;
                let byte_len = *byte_len;
                let result = resolved.get_or_init(|| self.pread(offset, byte_len));
                result.as_ref().map(|v| v.as_slice()).map_err(|e| e.clone())
            }
        }
    }

    /// Alias for resolve_tag — returns raw bytes without type interpretation.
    pub fn get_bytes(&self, ifd_id: IfdId, tag: u16) -> Result<&[u8], TiffParseError> {
        self.resolve_tag(ifd_id, tag)
    }

    // ── Typed scalar accessors ─────────────────────────────────

    /// Read a single u32 value. Accepts BYTE, SHORT, LONG types.
    pub fn get_u32(&self, ifd_id: IfdId, tag: u16) -> Result<u32, TiffParseError> {
        let ifd = self.ifd_by_id(ifd_id)?;
        let entry = ifd.tags.get(&tag).ok_or(TiffParseError::TagNotFound {
            ifd_offset: ifd.offset,
            tag,
        })?;
        if entry.count != 1 {
            return Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!("expected count=1, got {}", entry.count),
            });
        }
        let bytes = self.resolve_tag(ifd_id, tag)?;
        match entry.tiff_type {
            TiffType::Byte => Ok(bytes[0] as u32),
            TiffType::Short => {
                let val = match self.endian {
                    Endian::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
                    Endian::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
                };
                Ok(val as u32)
            }
            TiffType::Long => {
                let val = match self.endian {
                    Endian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    Endian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                };
                Ok(val)
            }
            _ => Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!("get_u32 expects BYTE/SHORT/LONG, got {:?}", entry.tiff_type),
            }),
        }
    }

    /// Read a single u64 value. Accepts BYTE, SHORT, LONG, LONG8 types.
    pub fn get_u64(&self, ifd_id: IfdId, tag: u16) -> Result<u64, TiffParseError> {
        let ifd = self.ifd_by_id(ifd_id)?;
        let entry = ifd.tags.get(&tag).ok_or(TiffParseError::TagNotFound {
            ifd_offset: ifd.offset,
            tag,
        })?;
        if entry.count != 1 {
            return Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!("expected count=1, got {}", entry.count),
            });
        }
        let bytes = self.resolve_tag(ifd_id, tag)?;
        match entry.tiff_type {
            TiffType::Byte => Ok(bytes[0] as u64),
            TiffType::Short => {
                let val = match self.endian {
                    Endian::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
                    Endian::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
                };
                Ok(val as u64)
            }
            TiffType::Long => {
                let val = match self.endian {
                    Endian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    Endian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                };
                let val = val as u64;
                if self.ndpi && is_ndpi_data_offset_tag(tag) {
                    Ok(self.fix_ndpi_data_offset(ifd.offset, val))
                } else {
                    Ok(val)
                }
            }
            TiffType::Long8 => {
                let val = match self.endian {
                    Endian::Little => u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]),
                    Endian::Big => u64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]),
                };
                Ok(val)
            }
            _ => Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!(
                    "get_u64 expects BYTE/SHORT/LONG/LONG8, got {:?}",
                    entry.tiff_type
                ),
            }),
        }
    }

    /// Read a single f64 value. Accepts FLOAT, DOUBLE, RATIONAL, SRATIONAL types.
    pub fn get_f64(&self, ifd_id: IfdId, tag: u16) -> Result<f64, TiffParseError> {
        let ifd = self.ifd_by_id(ifd_id)?;
        let entry = ifd.tags.get(&tag).ok_or(TiffParseError::TagNotFound {
            ifd_offset: ifd.offset,
            tag,
        })?;
        if entry.count != 1 {
            return Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!("expected count=1, got {}", entry.count),
            });
        }
        let bytes = self.resolve_tag(ifd_id, tag)?;
        match entry.tiff_type {
            TiffType::Float => {
                let val = match self.endian {
                    Endian::Little => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    Endian::Big => f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                };
                Ok(val as f64)
            }
            TiffType::Double => {
                let val = match self.endian {
                    Endian::Little => f64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]),
                    Endian::Big => f64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]),
                };
                Ok(val)
            }
            TiffType::Rational => {
                let (num, den) = self.decode_rational(bytes)?;
                if den == 0 {
                    Ok(0.0)
                } else {
                    Ok(num as f64 / den as f64)
                }
            }
            TiffType::SRational => {
                let (num, den) = self.decode_srational(bytes)?;
                if den == 0 {
                    Ok(0.0)
                } else {
                    Ok(num as f64 / den as f64)
                }
            }
            _ => Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!(
                    "get_f64 expects FLOAT/DOUBLE/RATIONAL/SRATIONAL, got {:?}",
                    entry.tiff_type
                ),
            }),
        }
    }

    /// Read an ASCII string value.
    pub fn get_string(&self, ifd_id: IfdId, tag: u16) -> Result<&str, TiffParseError> {
        let ifd = self.ifd_by_id(ifd_id)?;
        let entry = ifd.tags.get(&tag).ok_or(TiffParseError::TagNotFound {
            ifd_offset: ifd.offset,
            tag,
        })?;
        if entry.tiff_type != TiffType::Ascii {
            return Err(TiffParseError::InvalidTag {
                ifd_offset: ifd.offset,
                tag,
                message: format!("get_string expects ASCII, got {:?}", entry.tiff_type),
            });
        }
        let bytes = self.resolve_tag(ifd_id, tag)?;
        // Strip null terminator(s)
        let trimmed = match bytes.iter().position(|&b| b == 0) {
            Some(pos) => &bytes[..pos],
            None => bytes,
        };
        std::str::from_utf8(trimmed).map_err(|e| TiffParseError::InvalidTag {
            ifd_offset: ifd.offset,
            tag,
            message: format!("invalid UTF-8 in ASCII tag: {}", e),
        })
    }

    // ── Internal decode helpers ────────────────────────────────

    fn decode_rational(&self, bytes: &[u8]) -> Result<(u32, u32), TiffParseError> {
        if bytes.len() < 8 {
            return Err(TiffParseError::Structure(
                "RATIONAL requires 8 bytes".into(),
            ));
        }
        let num = match self.endian {
            Endian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Endian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        };
        let den = match self.endian {
            Endian::Little => u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            Endian::Big => u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        };
        Ok((num, den))
    }

    fn decode_srational(&self, bytes: &[u8]) -> Result<(i32, i32), TiffParseError> {
        if bytes.len() < 8 {
            return Err(TiffParseError::Structure(
                "SRATIONAL requires 8 bytes".into(),
            ));
        }
        let num = match self.endian {
            Endian::Little => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Endian::Big => i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        };
        let den = match self.endian {
            Endian::Little => i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            Endian::Big => i32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        };
        Ok((num, den))
    }

    // ── Typed array accessors (cached decode) ──────────────────

    /// Read a u64 array. Accepts SHORT, LONG, LONG8 types. Cached via OnceLock.
    pub fn get_u64_array(&self, ifd_id: IfdId, tag: u16) -> Result<&[u64], TiffParseError> {
        let ifd = self.ifd_by_id(ifd_id)?;
        let entry = ifd.tags.get(&tag).ok_or(TiffParseError::TagNotFound {
            ifd_offset: ifd.offset,
            tag,
        })?;

        if let Some(cached) = entry.decoded_u64.get() {
            return Ok(cached.as_slice());
        }

        let bytes = self.resolve_tag(ifd_id, tag)?;
        let count = entry.count as usize;

        let decoded: Vec<u64> = match entry.tiff_type {
            TiffType::Short => bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| match self.endian {
                    Endian::Little => u16::from_le_bytes([c[0], c[1]]) as u64,
                    Endian::Big => u16::from_be_bytes([c[0], c[1]]) as u64,
                })
                .collect(),
            TiffType::Long => bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| match self.endian {
                    Endian::Little => u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64,
                    Endian::Big => u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as u64,
                })
                .map(|value| {
                    if self.ndpi && is_ndpi_data_offset_tag(tag) {
                        self.fix_ndpi_data_offset(ifd.offset, value)
                    } else {
                        value
                    }
                })
                .collect(),
            TiffType::Long8 => bytes
                .chunks_exact(8)
                .take(count)
                .map(|c| match self.endian {
                    Endian::Little => {
                        u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    }
                    Endian::Big => {
                        u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    }
                })
                .collect(),
            _ => {
                return Err(TiffParseError::InvalidTag {
                    ifd_offset: ifd.offset,
                    tag,
                    message: format!(
                        "get_u64_array expects SHORT/LONG/LONG8, got {:?}",
                        entry.tiff_type
                    ),
                });
            }
        };

        Ok(entry.decoded_u64.get_or_init(|| decoded).as_slice())
    }

    // ── Metadata accessors ─────────────────────────────────────

    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    pub fn endian(&self) -> Endian {
        self.endian
    }

    pub fn is_bigtiff(&self) -> bool {
        self.bigtiff
    }

    pub fn is_ndpi(&self) -> bool {
        self.ndpi
    }

    pub fn ifd_count(&self) -> usize {
        self.ifds.len()
    }

    pub fn top_ifds(&self) -> &[IfdId] {
        &self.top_ifds
    }

    pub fn ifd_by_id(&self, id: IfdId) -> Result<&Ifd, TiffParseError> {
        self.ifds.get(&id).ok_or(TiffParseError::IfdNotFound(id))
    }

    fn fix_ndpi_data_offset(&self, ifd_offset: u64, raw_offset: u64) -> u64 {
        let fixed = fix_offset_ndpi(ifd_offset, raw_offset);
        for candidate in [
            raw_offset,
            raw_offset.saturating_sub(1),
            fixed,
            fixed.saturating_sub(1),
        ] {
            if candidate < self.file_len && self.looks_like_jpeg_soi(candidate) {
                return candidate;
            }
        }
        fixed
    }

    fn looks_like_jpeg_soi(&self, offset: u64) -> bool {
        self.pread(offset, 2)
            .is_ok_and(|bytes| bytes.as_slice() == [0xFF, 0xD8])
    }
}

// ── NDPI offset fixup ──────────────────────────────────────────────

/// NDPI stores >4GB offsets using only the low 32 bits in classic TIFF fields.
/// Heuristic: reconstruct high bits from the IFD's own offset.
/// Ported from the established tifflike offset-reconstruction behavior.
fn fix_offset_ndpi(diroff: u64, offset: u64) -> u64 {
    let mut result = (diroff & !u64::from(u32::MAX)) | (offset & u64::from(u32::MAX));
    if result >= diroff {
        result = result.saturating_sub(u64::from(u32::MAX) + 1).min(result);
    }
    result
}

fn is_ndpi_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(ext) if ext.eq_ignore_ascii_case("ndpi")
    )
}

fn is_ndpi_data_offset_tag(tag: u16) -> bool {
    matches!(tag, tags::STRIP_OFFSETS | tags::TILE_OFFSETS)
}

fn repair_ndpi_first_ifd_offset(
    reader: &mut ParseReader,
    raw_offset: u64,
    file_len: u64,
) -> Result<u64, TiffParseError> {
    let four_gb = u64::from(u32::MAX) + 1;
    let mut candidate = raw_offset;
    while candidate < file_len {
        if is_plausible_ifd_offset(reader, candidate, file_len)? {
            return Ok(candidate);
        }
        candidate = match candidate.checked_add(four_gb) {
            Some(next) => next,
            None => break,
        };
    }
    if raw_offset >= file_len {
        return Err(TiffParseError::Structure(format!(
            "NDPI first IFD offset {raw_offset} is outside file length {file_len}; file may be truncated"
        )));
    }
    Ok(raw_offset)
}

fn is_plausible_ifd_offset(
    reader: &mut ParseReader,
    offset: u64,
    file_len: u64,
) -> Result<bool, TiffParseError> {
    if offset + 2 > file_len {
        return Ok(false);
    }
    reader.seek(offset)?;
    let entry_count = reader.read_u16()? as u64;
    if entry_count == 0 || entry_count > 4096 {
        return Ok(false);
    }
    let ifd_bytes = 2u64
        .checked_add(entry_count.saturating_mul(12))
        .and_then(|value| value.checked_add(8))
        .unwrap_or(u64::MAX);
    Ok(offset.saturating_add(ifd_bytes) <= file_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TiffType ───────────────────────────────────────────────

    #[test]
    fn tiff_type_from_u16_known_types() {
        assert_eq!(TiffType::from_u16(1), Some(TiffType::Byte));
        assert_eq!(TiffType::from_u16(2), Some(TiffType::Ascii));
        assert_eq!(TiffType::from_u16(3), Some(TiffType::Short));
        assert_eq!(TiffType::from_u16(4), Some(TiffType::Long));
        assert_eq!(TiffType::from_u16(5), Some(TiffType::Rational));
        assert_eq!(TiffType::from_u16(6), Some(TiffType::SByte));
        assert_eq!(TiffType::from_u16(7), Some(TiffType::Undefined));
        assert_eq!(TiffType::from_u16(8), Some(TiffType::SShort));
        assert_eq!(TiffType::from_u16(9), Some(TiffType::SLong));
        assert_eq!(TiffType::from_u16(10), Some(TiffType::SRational));
        assert_eq!(TiffType::from_u16(11), Some(TiffType::Float));
        assert_eq!(TiffType::from_u16(12), Some(TiffType::Double));
        assert_eq!(TiffType::from_u16(13), Some(TiffType::Ifd));
        assert_eq!(TiffType::from_u16(16), Some(TiffType::Long8));
        assert_eq!(TiffType::from_u16(17), Some(TiffType::SLong8));
        assert_eq!(TiffType::from_u16(18), Some(TiffType::Ifd8));
    }

    #[test]
    fn tiff_type_from_u16_unknown() {
        assert_eq!(TiffType::from_u16(0), None);
        assert_eq!(TiffType::from_u16(14), None);
        assert_eq!(TiffType::from_u16(15), None);
        assert_eq!(TiffType::from_u16(19), None);
        assert_eq!(TiffType::from_u16(255), None);
    }

    #[test]
    fn tiff_type_byte_sizes() {
        assert_eq!(TiffType::Byte.byte_size(), 1);
        assert_eq!(TiffType::Ascii.byte_size(), 1);
        assert_eq!(TiffType::Short.byte_size(), 2);
        assert_eq!(TiffType::Long.byte_size(), 4);
        assert_eq!(TiffType::Rational.byte_size(), 8);
        assert_eq!(TiffType::Float.byte_size(), 4);
        assert_eq!(TiffType::Double.byte_size(), 8);
        assert_eq!(TiffType::Long8.byte_size(), 8);
        assert_eq!(TiffType::Ifd.byte_size(), 4);
        assert_eq!(TiffType::Ifd8.byte_size(), 8);
    }

    #[test]
    fn tiff_type_round_trip() {
        for id in 1..=18 {
            if let Some(tt) = TiffType::from_u16(id) {
                assert!(tt.byte_size() > 0, "type {:?} has zero byte_size", tt);
            }
        }
    }

    // ── InlineValue ────────────────────────────────────────────

    #[test]
    fn inline_value_construction_and_access() {
        let val = InlineValue::new(&[1, 2, 3, 4]);
        assert_eq!(val.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn inline_value_empty() {
        let val = InlineValue::new(&[]);
        assert_eq!(val.as_bytes().len(), 0);
    }

    #[test]
    fn inline_value_max_capacity() {
        let data = [0xFFu8; 12];
        let val = InlineValue::new(&data);
        assert_eq!(val.as_bytes().len(), 12);
        assert_eq!(val.as_bytes(), &data);
    }

    #[test]
    #[should_panic(expected = "exceeds 12 bytes")]
    fn inline_value_rejects_oversized() {
        let _ = InlineValue::new(&[0u8; 13]);
    }

    // ── Endian ─────────────────────────────────────────────────

    #[test]
    fn endian_equality() {
        assert_eq!(Endian::Little, Endian::Little);
        assert_eq!(Endian::Big, Endian::Big);
        assert_ne!(Endian::Little, Endian::Big);
    }

    // ── TagEntry / TagValue ────────────────────────────────────

    #[test]
    fn tag_entry_inline_construction() {
        let entry = TagEntry::new_inline(TiffType::Short, 1, &[0x00, 0x01]);
        assert_eq!(entry.tiff_type, TiffType::Short);
        assert_eq!(entry.count, 1);
        match &entry.value {
            TagValue::Inline(v) => assert_eq!(v.as_bytes(), &[0x00, 0x01]),
            TagValue::Lazy { .. } => panic!("expected Inline"),
        }
    }

    #[test]
    fn tag_entry_lazy_construction() {
        let entry = TagEntry::new_lazy(TiffType::Long, 100, 4096, 400);
        assert_eq!(entry.tiff_type, TiffType::Long);
        assert_eq!(entry.count, 100);
        match &entry.value {
            TagValue::Lazy {
                offset, byte_len, ..
            } => {
                assert_eq!(*offset, 4096);
                assert_eq!(*byte_len, 400);
            }
            TagValue::Inline(_) => panic!("expected Lazy"),
        }
    }

    #[test]
    fn tag_entry_decoded_oncelocks_start_empty() {
        let entry = TagEntry::new_inline(TiffType::Long, 1, &[0, 0, 0, 1]);
        assert!(entry.decoded_u64.get().is_none());
    }

    #[test]
    fn ifd_construction() {
        let mut tags = HashMap::new();
        tags.insert(256, TagEntry::new_inline(TiffType::Long, 1, &[0, 0, 4, 0]));
        let ifd = Ifd {
            id: IfdId(1024),
            offset: 1024,
            tags,
            sub_ifds: vec![],
        };
        assert_eq!(ifd.id, IfdId(1024));
        assert_eq!(ifd.tags.len(), 1);
        assert!(ifd.tags.contains_key(&256));
    }

    // ── Test helpers for synthetic TIFFs ───────────────────────

    use std::io::Write;

    /// Write a u16 in the given endianness.
    fn write_u16(buf: &mut Vec<u8>, endian: Endian, val: u16) {
        match endian {
            Endian::Little => buf.extend_from_slice(&val.to_le_bytes()),
            Endian::Big => buf.extend_from_slice(&val.to_be_bytes()),
        }
    }

    /// Write a u32 in the given endianness.
    fn write_u32(buf: &mut Vec<u8>, endian: Endian, val: u32) {
        match endian {
            Endian::Little => buf.extend_from_slice(&val.to_le_bytes()),
            Endian::Big => buf.extend_from_slice(&val.to_be_bytes()),
        }
    }

    /// Write a u64 in the given endianness.
    fn write_u64(buf: &mut Vec<u8>, endian: Endian, val: u64) {
        match endian {
            Endian::Little => buf.extend_from_slice(&val.to_le_bytes()),
            Endian::Big => buf.extend_from_slice(&val.to_be_bytes()),
        }
    }

    /// A synthetic IFD entry for test data construction.
    struct SyntheticEntry {
        tag: u16,
        tiff_type: u16,
        count: u64,
        /// Inline data (used when total_bytes <= slot_size).
        /// If None and data is out-of-line, `out_of_line_data` is used.
        inline_data: Option<Vec<u8>>,
        /// Out-of-line data to be written at a computed offset.
        out_of_line_data: Option<Vec<u8>>,
    }

    /// Build a minimal classic TIFF with one IFD.
    /// Returns (bytes, ifd_offset).
    fn make_classic_tiff_single(endian: Endian, entries: &[SyntheticEntry]) -> Vec<u8> {
        let mut buf = Vec::new();
        // Byte order
        match endian {
            Endian::Little => buf.extend_from_slice(b"II"),
            Endian::Big => buf.extend_from_slice(b"MM"),
        }
        // Magic
        write_u16(&mut buf, endian, 42);
        // First IFD offset (immediately after the header, at offset 8)
        let ifd_offset = 8u32;
        write_u32(&mut buf, endian, ifd_offset);

        // IFD entry count
        let entry_count = entries.len() as u16;
        write_u16(&mut buf, endian, entry_count);

        // Compute where out-of-line data goes:
        // After header(8) + entry_count(2) + entries(12 each) + next_ifd_offset(4)
        let mut ool_offset = 8u64 + 2 + (entries.len() as u64 * 12) + 4;

        // Collect out-of-line data with their offsets
        let mut ool_chunks: Vec<(u64, Vec<u8>)> = Vec::new();

        for entry in entries {
            write_u16(&mut buf, endian, entry.tag);
            write_u16(&mut buf, endian, entry.tiff_type);
            write_u32(&mut buf, endian, entry.count as u32);

            let type_size = TiffType::from_u16(entry.tiff_type)
                .map(|t| t.byte_size())
                .unwrap_or(1);
            let total_bytes = entry.count * type_size;

            if total_bytes <= 4 {
                // Inline: write up to 4 bytes, pad with zeros
                let data = entry.inline_data.as_deref().unwrap_or(&[]);
                let mut slot = [0u8; 4];
                let copy_len = data.len().min(4);
                slot[..copy_len].copy_from_slice(&data[..copy_len]);
                buf.extend_from_slice(&slot);
            } else {
                // Out-of-line: write offset
                let data = entry
                    .out_of_line_data
                    .as_ref()
                    .expect("out-of-line entry must have out_of_line_data");
                write_u32(&mut buf, endian, ool_offset as u32);
                ool_chunks.push((ool_offset, data.clone()));
                ool_offset += data.len() as u64;
            }
        }

        // Next IFD offset = 0 (no more IFDs)
        write_u32(&mut buf, endian, 0);

        // Write out-of-line data
        for (_offset, data) in &ool_chunks {
            buf.extend_from_slice(data);
        }

        buf
    }

    /// Build a minimal BigTIFF with one IFD.
    fn make_bigtiff_single(endian: Endian, entries: &[SyntheticEntry]) -> Vec<u8> {
        let mut buf = Vec::new();
        // Byte order
        match endian {
            Endian::Little => buf.extend_from_slice(b"II"),
            Endian::Big => buf.extend_from_slice(b"MM"),
        }
        // Magic 43
        write_u16(&mut buf, endian, 43);
        // Offset size = 8
        write_u16(&mut buf, endian, 8);
        // Reserved = 0
        write_u16(&mut buf, endian, 0);
        // First IFD offset (immediately after header, at offset 16)
        let ifd_offset = 16u64;
        write_u64(&mut buf, endian, ifd_offset);

        // IFD entry count (8 bytes for BigTIFF)
        let entry_count = entries.len() as u64;
        write_u64(&mut buf, endian, entry_count);

        // Compute where out-of-line data goes:
        // After header(16) + entry_count(8) + entries(20 each) + next_ifd_offset(8)
        let mut ool_offset = 16u64 + 8 + (entries.len() as u64 * 20) + 8;

        let mut ool_chunks: Vec<(u64, Vec<u8>)> = Vec::new();

        for entry in entries {
            write_u16(&mut buf, endian, entry.tag);
            write_u16(&mut buf, endian, entry.tiff_type);
            write_u64(&mut buf, endian, entry.count);

            let type_size = TiffType::from_u16(entry.tiff_type)
                .map(|t| t.byte_size())
                .unwrap_or(1);
            let total_bytes = entry.count * type_size;

            if total_bytes <= 8 {
                // Inline: write up to 8 bytes, pad with zeros
                let data = entry.inline_data.as_deref().unwrap_or(&[]);
                let mut slot = [0u8; 8];
                let copy_len = data.len().min(8);
                slot[..copy_len].copy_from_slice(&data[..copy_len]);
                buf.extend_from_slice(&slot);
            } else {
                // Out-of-line: write offset
                let data = entry
                    .out_of_line_data
                    .as_ref()
                    .expect("out-of-line entry must have out_of_line_data");
                write_u64(&mut buf, endian, ool_offset);
                ool_chunks.push((ool_offset, data.clone()));
                ool_offset += data.len() as u64;
            }
        }

        // Next IFD offset = 0
        write_u64(&mut buf, endian, 0);

        // Write out-of-line data
        for (_offset, data) in &ool_chunks {
            buf.extend_from_slice(data);
        }

        buf
    }

    /// Write synthetic TIFF bytes to a tempfile, return the path.
    fn write_tiff_tempfile(data: &[u8]) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(data).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    // ── Header parsing tests ──────────────────────────────────

    #[test]
    fn parse_classic_le_header() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4, // LONG
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0]), // 1024 LE
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.endian(), Endian::Little);
        assert!(!container.is_bigtiff());
        assert!(!container.is_ndpi());
    }

    #[test]
    fn parse_classic_be_header() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 0, 4, 0]), // 1024 BE
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Big, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.endian(), Endian::Big);
        assert!(!container.is_bigtiff());
    }

    #[test]
    fn parse_bigtiff_le_header() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 16, // LONG8
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0, 0, 0, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_bigtiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.endian(), Endian::Little);
        assert!(container.is_bigtiff());
    }

    #[test]
    fn parse_bigtiff_be_header() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 16,
            count: 1,
            inline_data: Some(vec![0, 0, 0, 0, 0, 0, 4, 0]),
            out_of_line_data: None,
        }];
        let data = make_bigtiff_single(Endian::Big, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.endian(), Endian::Big);
        assert!(container.is_bigtiff());
    }

    #[test]
    fn reject_invalid_magic() {
        let mut data = vec![b'I', b'I'];
        data.extend_from_slice(&99u16.to_le_bytes()); // bad magic
        data.extend_from_slice(&8u32.to_le_bytes()); // dummy offset
        let tmp = write_tiff_tempfile(&data);
        let result = TiffContainer::open(tmp.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, TiffParseError::Structure(_)),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn reject_bad_bigtiff_offset_size() {
        let mut data = Vec::new();
        data.extend_from_slice(b"II");
        data.extend_from_slice(&43u16.to_le_bytes()); // BigTIFF magic
        data.extend_from_slice(&4u16.to_le_bytes()); // wrong offset size (should be 8)
        data.extend_from_slice(&0u16.to_le_bytes()); // reserved
        data.extend_from_slice(&16u64.to_le_bytes()); // first IFD offset
        let tmp = write_tiff_tempfile(&data);
        let result = TiffContainer::open(tmp.path());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TiffParseError::Structure(_)));
    }

    #[test]
    fn pread_bounds_check_rejects_out_of_bounds() {
        // Create a minimal valid TIFF so TiffContainer::open() succeeds
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 1, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let file_len = data.len() as u64;
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        // Try to read beyond file end
        let result = container.pread(file_len - 2, 10);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TiffParseError::Bounds { .. }));
    }

    #[test]
    fn pread_offset_overflow_rejected() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 1, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        // Offset that would overflow u64
        let result = container.pread(u64::MAX, 10);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TiffParseError::Bounds { .. }));
    }

    // ── Multi-IFD test helper ─────────────────────────────────

    /// Build a classic TIFF with two chained IFDs, each with the given entries.
    fn make_classic_tiff_two_ifds(
        endian: Endian,
        entries1: &[SyntheticEntry],
        entries2: &[SyntheticEntry],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // Byte order
        match endian {
            Endian::Little => buf.extend_from_slice(b"II"),
            Endian::Big => buf.extend_from_slice(b"MM"),
        }
        // Magic
        write_u16(&mut buf, endian, 42);
        // First IFD offset = 8
        write_u32(&mut buf, endian, 8);

        // === IFD 1 ===
        let ifd1_offset = 8u64;
        write_u16(&mut buf, endian, entries1.len() as u16);

        // Compute IFD1 total size: 2 + entries*12 + 4
        let ifd1_size = 2 + (entries1.len() as u64 * 12) + 4;
        // IFD2 starts after IFD1 + any OOL data from IFD1
        // For simplicity, assume no OOL data in entries1
        let ifd2_offset = ifd1_offset + ifd1_size;

        for entry in entries1 {
            write_u16(&mut buf, endian, entry.tag);
            write_u16(&mut buf, endian, entry.tiff_type);
            write_u32(&mut buf, endian, entry.count as u32);
            let data = entry.inline_data.as_deref().unwrap_or(&[0, 0, 0, 0]);
            let mut slot = [0u8; 4];
            let copy_len = data.len().min(4);
            slot[..copy_len].copy_from_slice(&data[..copy_len]);
            buf.extend_from_slice(&slot);
        }

        // Next IFD offset -> IFD2
        write_u32(&mut buf, endian, ifd2_offset as u32);

        // === IFD 2 ===
        write_u16(&mut buf, endian, entries2.len() as u16);

        for entry in entries2 {
            write_u16(&mut buf, endian, entry.tag);
            write_u16(&mut buf, endian, entry.tiff_type);
            write_u32(&mut buf, endian, entry.count as u32);
            let data = entry.inline_data.as_deref().unwrap_or(&[0, 0, 0, 0]);
            let mut slot = [0u8; 4];
            let copy_len = data.len().min(4);
            slot[..copy_len].copy_from_slice(&data[..copy_len]);
            buf.extend_from_slice(&slot);
        }

        // Next IFD offset = 0 (end)
        write_u32(&mut buf, endian, 0);

        buf
    }

    // ── IFD chain walking tests ───────────────────────────────

    #[test]
    fn single_ifd_parsed() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4, // LONG
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.ifd_count(), 1);
        assert_eq!(container.top_ifds().len(), 1);
        let ifd = container.ifd_by_id(container.top_ifds()[0]).unwrap();
        assert!(ifd.tags.contains_key(&tags::IMAGE_WIDTH));
    }

    #[test]
    fn two_chained_ifds_parsed() {
        let e1 = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0]),
            out_of_line_data: None,
        }];
        let e2 = vec![SyntheticEntry {
            tag: tags::IMAGE_LENGTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 3, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_two_ifds(Endian::Little, &e1, &e2);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.ifd_count(), 2);
        assert_eq!(container.top_ifds().len(), 2);
        // IFDs should have different IDs (offsets)
        assert_ne!(container.top_ifds()[0], container.top_ifds()[1]);
    }

    #[test]
    fn ifd_chain_loop_detected() {
        let mut data = Vec::new();
        data.extend_from_slice(b"II");
        write_u16(&mut data, Endian::Little, 42);
        // First IFD at offset 8
        write_u32(&mut data, Endian::Little, 8);
        // IFD with 0 entries
        write_u16(&mut data, Endian::Little, 0);
        // Next IFD offset points back to 8 (loop!)
        write_u32(&mut data, Endian::Little, 8);

        let tmp = write_tiff_tempfile(&data);
        let result = TiffContainer::open(tmp.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            TiffParseError::Structure(msg) => assert!(msg.contains("loop"), "got: {}", msg),
            other => panic!("expected Structure, got: {:?}", other),
        }
    }

    #[test]
    fn empty_ifd_accepted() {
        let mut data = Vec::new();
        data.extend_from_slice(b"II");
        write_u16(&mut data, Endian::Little, 42);
        write_u32(&mut data, Endian::Little, 8); // IFD at offset 8
        write_u16(&mut data, Endian::Little, 0); // 0 entries
        write_u32(&mut data, Endian::Little, 0); // no next IFD

        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert_eq!(container.ifd_count(), 1);
        let ifd = container.ifd_by_id(container.top_ifds()[0]).unwrap();
        assert_eq!(ifd.tags.len(), 0);
    }

    #[test]
    fn unknown_type_id_skipped() {
        // Create an entry with type ID 99 (unknown) — should be skipped
        let entries = vec![
            SyntheticEntry {
                tag: 999,
                tiff_type: 99, // unknown
                count: 1,
                inline_data: Some(vec![0, 0, 0, 0]),
                out_of_line_data: None,
            },
            SyntheticEntry {
                tag: tags::IMAGE_WIDTH,
                tiff_type: 4, // LONG
                count: 1,
                inline_data: Some(vec![0, 4, 0, 0]),
                out_of_line_data: None,
            },
        ];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd = container.ifd_by_id(container.top_ifds()[0]).unwrap();
        // Unknown type entry skipped, only IMAGE_WIDTH present
        assert!(!ifd.tags.contains_key(&999));
        assert!(ifd.tags.contains_key(&tags::IMAGE_WIDTH));
    }

    #[test]
    fn ndpi_detection_via_tag_65420() {
        // NDPI uses 8-byte next-IFD pointers, so we manually construct the data.
        let mut ndpi_data = Vec::new();
        ndpi_data.extend_from_slice(b"II");
        write_u16(&mut ndpi_data, Endian::Little, 42);
        write_u32(&mut ndpi_data, Endian::Little, 8); // first IFD at 8

        // IFD: 2 entries
        write_u16(&mut ndpi_data, Endian::Little, 2);
        // Entry 1: NDPI marker tag
        write_u16(&mut ndpi_data, Endian::Little, tags::NDPI_MARKER);
        write_u16(&mut ndpi_data, Endian::Little, 4); // LONG
        write_u32(&mut ndpi_data, Endian::Little, 1);
        ndpi_data.extend_from_slice(&1u32.to_le_bytes());
        // Entry 2: IMAGE_WIDTH
        write_u16(&mut ndpi_data, Endian::Little, tags::IMAGE_WIDTH);
        write_u16(&mut ndpi_data, Endian::Little, 4); // LONG
        write_u32(&mut ndpi_data, Endian::Little, 1);
        ndpi_data.extend_from_slice(&1024u32.to_le_bytes());
        // Next IFD offset: 8 bytes for NDPI (value = 0)
        ndpi_data.extend_from_slice(&0u64.to_le_bytes());

        let tmp = write_tiff_tempfile(&ndpi_data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        assert!(container.is_ndpi());
    }

    #[test]
    fn ifd_by_id_not_found() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 1, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let result = container.ifd_by_id(IfdId(999999));
        assert!(matches!(
            result.unwrap_err(),
            TiffParseError::IfdNotFound(_)
        ));
    }

    // ── NDPI offset fixup tests ───────────────────────────────

    #[test]
    fn ndpi_fixup_low_offset_unchanged() {
        // When both diroff and offset are in the low 4GB, result should be offset
        let result = fix_offset_ndpi(1000, 500);
        assert_eq!(result, 500);
    }

    #[test]
    fn ndpi_fixup_high_diroff_reconstructs() {
        // diroff at 5GB, offset stored as low 32 bits of 4.5GB
        let diroff: u64 = 5 * 1024 * 1024 * 1024; // 5 GB
        let real_offset: u64 = 4 * 1024 * 1024 * 1024 + 500_000_000; // 4.5 GB
        let stored_offset = real_offset & u64::from(u32::MAX); // low 32 bits
        let result = fix_offset_ndpi(diroff, stored_offset);
        assert_eq!(result, real_offset);
    }

    #[test]
    fn ndpi_fixup_result_below_diroff() {
        // The fixup should always produce a result <= diroff
        // (data referenced by an IFD should precede it)
        let diroff: u64 = 6 * 1024 * 1024 * 1024;
        let stored_offset: u64 = 100;
        let result = fix_offset_ndpi(diroff, stored_offset);
        assert!(result <= diroff, "result {} > diroff {}", result, diroff);
    }

    #[test]
    fn ndpi_fixup_zero_diroff() {
        // When diroff is 0, the heuristic clamps: result >= diroff triggers
        // saturating_sub(4GB) which floors to 0.
        let result = fix_offset_ndpi(0, 12345);
        assert_eq!(result, 0);
    }

    #[test]
    fn opens_wrapped_first_ifd_ndpi_when_corpus_is_available() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let path = workspace_root.join("downloads/openslide-testdata/Hamamatsu/Hamamatsu-1.ndpi");
        if !path.exists() {
            return;
        }

        let container = TiffContainer::open(&path).expect("open wrapped-offset NDPI");
        assert!(container.is_ndpi());
        assert!(!container.top_ifds().is_empty());
    }

    // ── SubIFD test helpers ───────────────────────────────────

    /// Build a classic TIFF with one main IFD that has a SUB_IFDS tag
    /// pointing to one SubIFD.
    fn make_classic_tiff_with_subifd(endian: Endian) -> Vec<u8> {
        let mut buf = Vec::new();
        match endian {
            Endian::Little => buf.extend_from_slice(b"II"),
            Endian::Big => buf.extend_from_slice(b"MM"),
        }
        write_u16(&mut buf, endian, 42);
        write_u32(&mut buf, endian, 8); // first IFD at 8

        // Main IFD: 2 entries (IMAGE_WIDTH + SUB_IFDS)
        write_u16(&mut buf, endian, 2);

        // Entry 1: IMAGE_WIDTH = 1024
        write_u16(&mut buf, endian, tags::IMAGE_WIDTH);
        write_u16(&mut buf, endian, 4); // LONG
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, 1024);

        // Entry 2: SUB_IFDS — inline, count=1, pointing to SubIFD
        // Main IFD: header(8) + count(2) + 2*entries(24) + next(4) = 38
        // SubIFD will be at offset 38
        let sub_ifd_offset = 38u32;
        write_u16(&mut buf, endian, tags::SUB_IFDS);
        write_u16(&mut buf, endian, 4); // LONG (IFD offsets as LONG)
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, sub_ifd_offset);

        // Next IFD offset = 0
        write_u32(&mut buf, endian, 0);

        // === SubIFD at offset 38 ===
        assert_eq!(buf.len(), sub_ifd_offset as usize);
        write_u16(&mut buf, endian, 1); // 1 entry

        // Entry: IMAGE_LENGTH = 768
        write_u16(&mut buf, endian, tags::IMAGE_LENGTH);
        write_u16(&mut buf, endian, 4); // LONG
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, 768);

        // Next IFD offset = 0
        write_u32(&mut buf, endian, 0);

        buf
    }

    // ── SubIFD tests ──────────────────────────────────────────

    #[test]
    fn open_does_not_materialize_sub_ifds_eagerly() {
        let data = make_classic_tiff_with_subifd(Endian::Little);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();

        assert_eq!(container.top_ifds().len(), 1);
        assert_eq!(container.ifd_count(), 1);

        let main_ifd = container.ifd_by_id(container.top_ifds()[0]).unwrap();
        assert!(main_ifd.sub_ifds.is_empty());
    }

    #[test]
    fn sub_ifd_parsed() {
        let data = make_classic_tiff_with_subifd(Endian::Little);
        let tmp = write_tiff_tempfile(&data);
        let mut container = TiffContainer::open(tmp.path()).unwrap();

        let main_id = container.top_ifds()[0];
        container
            .materialize_sub_ifds(main_id, 4)
            .expect("materialize subifds");

        assert_eq!(container.top_ifds().len(), 1);
        assert_eq!(container.ifd_count(), 2); // main + sub

        let main_ifd = container.ifd_by_id(main_id).unwrap();
        assert_eq!(main_ifd.sub_ifds.len(), 1);

        let sub_ifd = container.ifd_by_id(main_ifd.sub_ifds[0]).unwrap();
        assert!(sub_ifd.tags.contains_key(&tags::IMAGE_LENGTH));
    }

    #[test]
    fn sub_ifd_nested_depth_2() {
        // Build a TIFF with main IFD -> SubIFD -> sub-SubIFD
        let endian = Endian::Little;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        write_u16(&mut buf, endian, 42);
        write_u32(&mut buf, endian, 8); // main IFD at 8

        // Main IFD at 8: 1 entry (SUB_IFDS)
        // Size: 2 + 12 + 4 = 18, so SubIFD1 at 8+18 = 26
        write_u16(&mut buf, endian, 1);
        write_u16(&mut buf, endian, tags::SUB_IFDS);
        write_u16(&mut buf, endian, 4); // LONG
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, 26); // SubIFD1 at 26
        write_u32(&mut buf, endian, 0); // next IFD = 0

        // SubIFD1 at 26: 1 entry (SUB_IFDS) pointing to SubIFD2
        // Size: 2 + 12 + 4 = 18, so SubIFD2 at 26+18 = 44
        assert_eq!(buf.len(), 26);
        write_u16(&mut buf, endian, 1);
        write_u16(&mut buf, endian, tags::SUB_IFDS);
        write_u16(&mut buf, endian, 4); // LONG
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, 44); // SubIFD2 at 44
        write_u32(&mut buf, endian, 0); // next IFD = 0

        // SubIFD2 at 44: 1 entry (IMAGE_WIDTH)
        assert_eq!(buf.len(), 44);
        write_u16(&mut buf, endian, 1);
        write_u16(&mut buf, endian, tags::IMAGE_WIDTH);
        write_u16(&mut buf, endian, 4); // LONG
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, 512);
        write_u32(&mut buf, endian, 0); // next IFD = 0

        let tmp = write_tiff_tempfile(&buf);
        let mut container = TiffContainer::open(tmp.path()).unwrap();
        let main_id = container.top_ifds()[0];
        container
            .materialize_sub_ifds(main_id, 4)
            .expect("materialize nested subifds");

        assert_eq!(container.ifd_count(), 3); // main + sub1 + sub2
        let main_ifd = container.ifd_by_id(main_id).unwrap();
        assert_eq!(main_ifd.sub_ifds.len(), 1);
        let sub1 = container.ifd_by_id(main_ifd.sub_ifds[0]).unwrap();
        assert_eq!(sub1.sub_ifds.len(), 1);
        let sub2 = container.ifd_by_id(sub1.sub_ifds[0]).unwrap();
        assert!(sub2.tags.contains_key(&tags::IMAGE_WIDTH));
    }

    #[test]
    fn sub_ifd_duplicate_offset_dedup() {
        // Two entries in SUB_IFDS tag pointing to the same offset
        let endian = Endian::Little;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        write_u16(&mut buf, endian, 42);
        write_u32(&mut buf, endian, 8);

        // Main IFD at 8: 1 entry (SUB_IFDS with count=2, inline since 2*4=8 > 4 -> OOL)
        // Actually 2 LONGs = 8 bytes > 4 byte slot -> out-of-line
        // Main IFD size: 2 + 12 + 4 = 18, OOL data at 8+18 = 26
        // SubIFD at 26+8 = 34
        write_u16(&mut buf, endian, 1);
        write_u16(&mut buf, endian, tags::SUB_IFDS);
        write_u16(&mut buf, endian, 4); // LONG
        write_u32(&mut buf, endian, 2); // count=2
        write_u32(&mut buf, endian, 26); // OOL data offset
        write_u32(&mut buf, endian, 0); // next IFD = 0

        // OOL data at 26: two offsets both pointing to 34
        assert_eq!(buf.len(), 26);
        write_u32(&mut buf, endian, 34);
        write_u32(&mut buf, endian, 34); // duplicate!

        // SubIFD at 34: 1 entry
        assert_eq!(buf.len(), 34);
        write_u16(&mut buf, endian, 1);
        write_u16(&mut buf, endian, tags::IMAGE_WIDTH);
        write_u16(&mut buf, endian, 4);
        write_u32(&mut buf, endian, 1);
        write_u32(&mut buf, endian, 256);
        write_u32(&mut buf, endian, 0);

        let tmp = write_tiff_tempfile(&buf);
        let mut container = TiffContainer::open(tmp.path()).unwrap();
        let main_id = container.top_ifds()[0];
        container
            .materialize_sub_ifds(main_id, 4)
            .expect("materialize duplicate subifds");

        // Only 2 unique IFDs (main + one SubIFD), despite two references
        assert_eq!(container.ifd_count(), 2);

        let main_ifd = container.ifd_by_id(main_id).unwrap();
        // Both references stored (preserves topology)
        assert_eq!(main_ifd.sub_ifds.len(), 2);
        assert_eq!(main_ifd.sub_ifds[0], main_ifd.sub_ifds[1]);
    }

    #[test]
    fn sub_ifd_depth_limit() {
        // Build 5 levels of nested SubIFDs (limit is 4)
        let endian = Endian::Little;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        write_u16(&mut buf, endian, 42);
        write_u32(&mut buf, endian, 8);

        // Each IFD: 2 (count) + 12 (1 entry) + 4 (next) = 18 bytes
        let ifd_size = 18u32;
        let mut current_offset = 8u32;

        for i in 0..6 {
            let is_last = i == 5;
            assert_eq!(buf.len(), current_offset as usize);
            write_u16(&mut buf, endian, 1); // 1 entry

            if is_last {
                // Last IFD has IMAGE_WIDTH instead of SUB_IFDS
                write_u16(&mut buf, endian, tags::IMAGE_WIDTH);
                write_u16(&mut buf, endian, 4);
                write_u32(&mut buf, endian, 1);
                write_u32(&mut buf, endian, 100);
            } else {
                let next_sub = current_offset + ifd_size;
                write_u16(&mut buf, endian, tags::SUB_IFDS);
                write_u16(&mut buf, endian, 4); // LONG
                write_u32(&mut buf, endian, 1);
                write_u32(&mut buf, endian, next_sub);
            }
            write_u32(&mut buf, endian, 0); // no next IFD in chain
            current_offset += ifd_size;
        }

        let tmp = write_tiff_tempfile(&buf);
        let mut container = TiffContainer::open(tmp.path()).unwrap();
        let main_id = container.top_ifds()[0];
        let result = container.materialize_sub_ifds(main_id, 4);
        // Should fail because depth exceeds 4
        assert!(result.is_err());
        match result.unwrap_err() {
            TiffParseError::Structure(msg) => {
                assert!(msg.contains("depth"), "got: {}", msg);
            }
            other => panic!("expected Structure, got: {:?}", other),
        }
    }

    #[test]
    fn sub_ifd_cross_reference_preserved() {
        let data = make_classic_tiff_with_subifd(Endian::Little);
        let tmp = write_tiff_tempfile(&data);
        let mut container = TiffContainer::open(tmp.path()).unwrap();
        container
            .materialize_all_sub_ifds(4)
            .expect("materialize all subifds");

        let main_ifd = container.ifd_by_id(container.top_ifds()[0]).unwrap();
        let sub_id = main_ifd.sub_ifds[0];

        // SubIFD accessible via flat arena lookup (O(1))
        let sub_ifd = container.ifd_by_id(sub_id).unwrap();
        assert_eq!(sub_ifd.id, sub_id);
        assert_eq!(sub_ifd.offset, sub_id.0);
    }

    // ── Lazy resolution tests ─────────────────────────────────

    #[test]
    fn resolve_inline_tag_returns_bytes() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4, // LONG
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0]), // 1024 LE
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();

        let ifd_id = container.top_ifds()[0];
        let bytes = container.resolve_tag(ifd_id, tags::IMAGE_WIDTH).unwrap();
        assert_eq!(bytes, &[0, 4, 0, 0]);
    }

    #[test]
    fn resolve_lazy_tag_triggers_io() {
        // Create a tag with out-of-line data (>4 bytes for classic)
        let ool_data: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0]; // two LONGs
        let entries = vec![SyntheticEntry {
            tag: tags::TILE_OFFSETS,
            tiff_type: 4, // LONG
            count: 2,
            inline_data: None,
            out_of_line_data: Some(ool_data.clone()),
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();

        let ifd_id = container.top_ifds()[0];
        let bytes = container.resolve_tag(ifd_id, tags::TILE_OFFSETS).unwrap();
        assert_eq!(bytes, &ool_data);
    }

    #[test]
    fn resolve_lazy_tag_cached() {
        let ool_data: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0];
        let entries = vec![SyntheticEntry {
            tag: tags::TILE_OFFSETS,
            tiff_type: 4,
            count: 2,
            inline_data: None,
            out_of_line_data: Some(ool_data.clone()),
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();

        let ifd_id = container.top_ifds()[0];
        let bytes1 = container.resolve_tag(ifd_id, tags::TILE_OFFSETS).unwrap();
        let bytes2 = container.resolve_tag(ifd_id, tags::TILE_OFFSETS).unwrap();
        // Same slice returned (same OnceLock)
        assert_eq!(bytes1.as_ptr(), bytes2.as_ptr());
    }

    #[test]
    fn resolve_tag_not_found() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 1, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();

        let ifd_id = container.top_ifds()[0];
        let result = container.resolve_tag(ifd_id, tags::TILE_OFFSETS);
        assert!(matches!(
            result.unwrap_err(),
            TiffParseError::TagNotFound { .. }
        ));
    }

    #[test]
    fn inline_classification_correct() {
        // 1 LONG = 4 bytes -> inline in classic TIFF (slot_size=4)
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4,
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0]),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let ifd = container.ifd_by_id(ifd_id).unwrap();
        let entry = ifd.tags.get(&tags::IMAGE_WIDTH).unwrap();
        assert!(matches!(entry.value, TagValue::Inline(_)));
    }

    // ── Typed scalar accessor tests ───────────────────────────

    #[test]
    fn get_u32_from_short() {
        let entries = vec![SyntheticEntry {
            tag: tags::BITS_PER_SAMPLE,
            tiff_type: 3, // SHORT
            count: 1,
            inline_data: Some(vec![8, 0, 0, 0]), // 8 LE
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        assert_eq!(container.get_u32(ifd_id, tags::BITS_PER_SAMPLE).unwrap(), 8);
    }

    #[test]
    fn get_u32_from_long() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 4, // LONG
            count: 1,
            inline_data: Some(vec![0, 4, 0, 0]), // 1024 LE
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        assert_eq!(container.get_u32(ifd_id, tags::IMAGE_WIDTH).unwrap(), 1024);
    }

    #[test]
    fn get_u64_from_long8() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_WIDTH,
            tiff_type: 16, // LONG8
            count: 1,
            inline_data: Some(vec![0, 0, 0, 1, 0, 0, 0, 0]), // 16777216 LE
            out_of_line_data: None,
        }];
        let data = make_bigtiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        assert_eq!(
            container.get_u64(ifd_id, tags::IMAGE_WIDTH).unwrap(),
            16777216
        );
    }

    #[test]
    fn get_f64_from_rational() {
        let mut rational_bytes = Vec::new();
        rational_bytes.extend_from_slice(&72u32.to_le_bytes());
        rational_bytes.extend_from_slice(&1u32.to_le_bytes());

        let entries = vec![SyntheticEntry {
            tag: 282,
            tiff_type: 5, // RATIONAL
            count: 1,
            inline_data: None,
            out_of_line_data: Some(rational_bytes),
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let val = container.get_f64(ifd_id, 282).unwrap();
        assert!((val - 72.0).abs() < f64::EPSILON);
    }

    #[test]
    fn get_f64_from_float() {
        let float_val: f32 = std::f32::consts::PI;
        let entries = vec![SyntheticEntry {
            tag: 500,      // arbitrary tag
            tiff_type: 11, // FLOAT
            count: 1,
            inline_data: Some(float_val.to_le_bytes().to_vec()),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let val = container.get_f64(ifd_id, 500).unwrap();
        assert!(
            (val - f64::from(std::f32::consts::PI)).abs() < 0.001,
            "got: {}",
            val
        );
    }

    #[test]
    fn get_string_ascii() {
        let text = b"Hello\0";
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_DESCRIPTION,
            tiff_type: 2, // ASCII
            count: text.len() as u64,
            inline_data: None,
            out_of_line_data: Some(text.to_vec()),
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let s = container
            .get_string(ifd_id, tags::IMAGE_DESCRIPTION)
            .unwrap();
        assert_eq!(s, "Hello");
    }

    #[test]
    fn get_u32_type_mismatch() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_DESCRIPTION,
            tiff_type: 2, // ASCII
            count: 4,
            inline_data: Some(b"foo\0".to_vec()),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let result = container.get_u32(ifd_id, tags::IMAGE_DESCRIPTION);
        assert!(matches!(
            result.unwrap_err(),
            TiffParseError::InvalidTag { .. }
        ));
    }

    // ── Typed array accessor tests ────────────────────────────

    #[test]
    fn get_u64_array_from_long() {
        // Two LONGs = 8 bytes -> out-of-line for classic
        let mut ool = Vec::new();
        ool.extend_from_slice(&100u32.to_le_bytes());
        ool.extend_from_slice(&200u32.to_le_bytes());

        let entries = vec![SyntheticEntry {
            tag: tags::TILE_OFFSETS,
            tiff_type: 4, // LONG
            count: 2,
            inline_data: None,
            out_of_line_data: Some(ool),
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let arr = container.get_u64_array(ifd_id, tags::TILE_OFFSETS).unwrap();
        assert_eq!(arr, &[100, 200]);
    }

    #[test]
    fn get_u64_array_from_long8() {
        let mut ool = Vec::new();
        ool.extend_from_slice(&5_000_000_000u64.to_le_bytes());
        ool.extend_from_slice(&6_000_000_000u64.to_le_bytes());

        let entries = vec![SyntheticEntry {
            tag: tags::TILE_OFFSETS,
            tiff_type: 16, // LONG8
            count: 2,
            inline_data: None,
            out_of_line_data: Some(ool),
        }];
        let data = make_bigtiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let arr = container.get_u64_array(ifd_id, tags::TILE_OFFSETS).unwrap();
        assert_eq!(arr, &[5_000_000_000, 6_000_000_000]);
    }

    #[test]
    fn get_u64_array_cached_pointer_equality() {
        let mut ool = Vec::new();
        ool.extend_from_slice(&100u32.to_le_bytes());
        ool.extend_from_slice(&200u32.to_le_bytes());

        let entries = vec![SyntheticEntry {
            tag: tags::TILE_OFFSETS,
            tiff_type: 4,
            count: 2,
            inline_data: None,
            out_of_line_data: Some(ool),
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];

        let arr1 = container.get_u64_array(ifd_id, tags::TILE_OFFSETS).unwrap();
        let arr2 = container.get_u64_array(ifd_id, tags::TILE_OFFSETS).unwrap();
        // Same pointer — cached, not re-decoded
        assert_eq!(arr1.as_ptr(), arr2.as_ptr());
    }

    #[test]
    fn get_u64_array_type_mismatch() {
        let entries = vec![SyntheticEntry {
            tag: tags::IMAGE_DESCRIPTION,
            tiff_type: 2, // ASCII
            count: 4,
            inline_data: Some(b"foo\0".to_vec()),
            out_of_line_data: None,
        }];
        let data = make_classic_tiff_single(Endian::Little, &entries);
        let tmp = write_tiff_tempfile(&data);
        let container = TiffContainer::open(tmp.path()).unwrap();
        let ifd_id = container.top_ifds()[0];
        let result = container.get_u64_array(ifd_id, tags::IMAGE_DESCRIPTION);
        assert!(matches!(
            result.unwrap_err(),
            TiffParseError::InvalidTag { .. }
        ));
    }
}
