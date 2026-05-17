use super::*;

// ── FullDecodeCache ───────────────────────────────────────────────

/// Default maximum cache size: 128 MB.
pub(super) const DEFAULT_FULL_DECODE_CACHE_BYTES: u64 = 128 * 1024 * 1024;
pub(super) const FULL_DECODE_CACHE_BYTES_ENV: &str = "STATUMEN_FULL_DECODE_CACHE_BYTES";
/// Default maximum cache size for decoded NDPI strips: 8 MB.
///
/// Large NDPI display traces are effectively one-way walks through the strip
/// space; retaining a much larger working set inflated RSS without improving
/// the measured tail. Keep the budget tight and allow local override for
/// targeted tuning.
pub(super) const DEFAULT_NDPI_STRIP_CACHE_BYTES: u64 = 1024 * 1024;
pub(super) const NDPI_STRIP_CACHE_BYTES_ENV: &str = "STATUMEN_NDPI_STRIP_CACHE_BYTES";
/// Default maximum cache size for synthetic NDPI tail levels: 16 MB.
pub(super) const DEFAULT_SYNTHETIC_LEVEL_CACHE_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const SYNTHETIC_LEVEL_CACHE_BYTES_ENV: &str = "STATUMEN_SYNTHETIC_LEVEL_CACHE_BYTES";
pub(super) const DEFAULT_JP2K_SHARED_TILE_CACHE_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const DEFAULT_STITCHED_COMPONENT_TILE_CACHE_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const NDPI_DISPLAY_WIDE_STRIP_BATCH: usize = 4;
pub(super) const NDPI_DISPLAY_NARROW_STRIP_BATCH: usize = 8;
#[cfg(feature = "metal")]
pub(super) const JPEG_DEVICE_DECODE_ENV: &str = "STATUMEN_JPEG_DEVICE_DECODE";
#[cfg(feature = "metal")]
pub(super) const JP2K_DEVICE_DECODE_ENV: &str = "STATUMEN_JP2K_DEVICE_DECODE";

pub(super) type NdpiMcuStartsCache = HashMap<(IfdId, u16), Arc<Vec<u64>>>;
pub(super) type SyntheticDeepestKey = (usize, usize, u32, u32, u32);
pub(super) type SyntheticDeepestValue = (u32, u32, u32);
pub(super) const NDPI_DISPLAY_WIDE_STRIP_WIDTH: u32 = 1024;

pub(super) struct NdpiJpegTilePayload {
    pub(super) jpeg: Vec<u8>,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[cfg(feature = "metal")]
pub(super) fn jpeg_device_decode_enabled() -> bool {
    std::env::var(JPEG_DEVICE_DECODE_ENV).is_ok_and(|value| {
        value.eq_ignore_ascii_case("1")
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("yes")
            || value.eq_ignore_ascii_case("on")
    })
}

#[cfg(feature = "metal")]
pub(super) fn jp2k_device_decode_enabled() -> bool {
    std::env::var(JP2K_DEVICE_DECODE_ENV).is_ok_and(|value| {
        value.eq_ignore_ascii_case("1")
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("yes")
            || value.eq_ignore_ascii_case("on")
    })
}

/// Byte-budgeted LRU cache for fully decoded NDPI levels.
///
/// NDPI levels without restart markers require decoding the entire JPEG
/// image to extract a single tile. This cache stores the decoded image
/// so subsequent tile requests from the same level are satisfied from
/// memory instead of re-decoding.
///
/// Same pattern as TileCache from Plan 1: byte budget drives eviction,
/// oversize entries are rejected (not cached but still returned).
pub(super) struct FullDecodeCache {
    pub(super) entries: LruCache<IfdId, Arc<CpuTile>>,
    pub(super) current_bytes: u64,
    pub(super) max_bytes: u64,
}

impl FullDecodeCache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    /// Get a cached decoded image by IFD ID.
    pub fn get(&mut self, key: &IfdId) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    /// Insert a decoded image. Evicts LRU entries to make room.
    /// Rejects entries larger than max_bytes (returns without storing).
    pub fn put(&mut self, key: IfdId, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;

        if byte_size > self.max_bytes {
            return; // Oversize — don't cache
        }

        // Remove existing entry if present
        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        // Evict LRU entries until there's room
        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for FullDecodeCache {
    fn default() -> Self {
        Self::new(DEFAULT_FULL_DECODE_CACHE_BYTES)
    }
}

pub(super) struct NdpiStripCache {
    pub(super) entries: LruCache<NdpiStripKey, Arc<CpuTile>>,
    pub(super) current_bytes: u64,
    pub(super) max_bytes: u64,
}

impl NdpiStripCache {
    pub(super) fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    pub(super) fn get(&mut self, key: &NdpiStripKey) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    pub(super) fn put(&mut self, key: NdpiStripKey, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;

        if byte_size > self.max_bytes {
            return;
        }

        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for NdpiStripCache {
    fn default() -> Self {
        Self::new(DEFAULT_NDPI_STRIP_CACHE_BYTES)
    }
}

pub(super) struct SyntheticLevelCache {
    pub(super) entries: LruCache<SyntheticLevelKey, Arc<CpuTile>>,
    pub(super) current_bytes: u64,
    pub(super) max_bytes: u64,
}

impl SyntheticLevelCache {
    pub(super) fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    pub(super) fn get(&mut self, key: &SyntheticLevelKey) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    pub(super) fn put(&mut self, key: SyntheticLevelKey, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;

        if byte_size > self.max_bytes {
            return;
        }

        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for SyntheticLevelCache {
    fn default() -> Self {
        Self::new(DEFAULT_SYNTHETIC_LEVEL_CACHE_BYTES)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct StitchedComponentTileKey {
    pub(super) ifd_id: IfdId,
    pub(super) tile_idx: usize,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) struct StitchedComponentTileCache {
    pub(super) entries: LruCache<StitchedComponentTileKey, Arc<CpuTile>>,
    pub(super) current_bytes: u64,
    pub(super) max_bytes: u64,
}

impl StitchedComponentTileCache {
    pub(super) fn new(max_bytes: u64) -> Self {
        Self {
            entries: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    pub(super) fn get(&mut self, key: &StitchedComponentTileKey) -> Option<Arc<CpuTile>> {
        self.entries.get(key).cloned()
    }

    pub(super) fn put(&mut self, key: StitchedComponentTileKey, data: Arc<CpuTile>) {
        let byte_size = data.data.byte_size() as u64;
        if byte_size > self.max_bytes {
            return;
        }

        if let Some((_, existing)) = self.entries.pop_entry(&key) {
            self.current_bytes -= existing.data.byte_size() as u64;
        }

        while self.current_bytes + byte_size > self.max_bytes {
            if let Some((_, evicted)) = self.entries.pop_lru() {
                self.current_bytes -= evicted.data.byte_size() as u64;
            } else {
                break;
            }
        }

        self.entries.put(key, data);
        self.current_bytes += byte_size;
    }
}

impl Default for StitchedComponentTileCache {
    fn default() -> Self {
        Self::new(DEFAULT_STITCHED_COMPONENT_TILE_CACHE_BYTES)
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct FullDecodeFlight {
    pub(super) waiters: usize,
    pub(super) result: Option<Result<Arc<CpuTile>, String>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct NdpiStripKey {
    pub(super) ifd_id: IfdId,
    pub(super) col: u32,
    pub(super) native_row: u32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct NdpiStripFlight {
    pub(super) waiters: usize,
    pub(super) result: Option<Result<Arc<CpuTile>, String>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SyntheticLevelKey {
    pub(super) scene: usize,
    pub(super) series: usize,
    pub(super) base_level: u32,
    pub(super) target_level: u32,
    pub(super) z: u32,
    pub(super) c: u32,
    pub(super) t: u32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct SyntheticLevelFlight {
    pub(super) waiters: usize,
    pub(super) result: Option<Result<Arc<CpuTile>, String>>,
}
