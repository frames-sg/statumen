//! Layer 2 types for TIFF-family layout interpretation.
//!
//! A `TiffLayoutInterpreter` maps raw IFDs from a `TiffContainer` into the
//! normalized `Dataset` model plus a `DatasetLayout` that records how to
//! access each level's pixel data at decode time.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::core::hash::Quickhash1;
use crate::core::types::{Compression, Dataset, DatasetId};
use crate::error::WsiError;
use crate::formats::tiff_family::container::{tags, TiffContainer};
use crate::formats::tiff_family::error::{IfdId, TiffParseError};

pub(crate) mod aperio;
pub(crate) mod generic;
pub(crate) mod leica;
pub(crate) mod ndpi;
pub(crate) mod philips;
pub(crate) mod trestle;
pub(crate) mod ventana;

const QUICKHASH_MAX_LEVEL_BYTES: u64 = 5 << 20;
const TAG_DOCUMENT_NAME: u16 = 269;
const TAG_MAKE: u16 = 271;
const TAG_MODEL: u16 = 272;
const TAG_SOFTWARE: u16 = 305;
const TAG_DATETIME: u16 = 306;
const TAG_ARTIST: u16 = 315;
const TAG_HOST_COMPUTER: u16 = 316;
const TAG_COPYRIGHT: u16 = 33432;

pub(crate) struct DatasetIdentity {
    pub dataset_id: DatasetId,
    pub quickhash1: Option<String>,
}

// ── Interpreter trait ────────────────────────────────────────────────────────

/// Trait for vendor-specific TIFF layout interpretation.
/// Implementations map raw IFDs into the normalized Dataset model.
pub(crate) trait TiffLayoutInterpreter: Send + Sync {
    /// Returns true if this interpreter can handle the given container.
    /// Must be cheap — only inspect tags already loaded into `container`.
    fn detect(&self, container: &TiffContainer) -> bool;

    /// Interpret the container, producing a `DatasetLayout`.
    /// Returns `TiffParseError`; callers convert to `WsiError` at the
    /// `TiffFamilyBackend` boundary.
    fn interpret(&self, container: &TiffContainer) -> Result<DatasetLayout, TiffParseError>;

    /// Short vendor identifier for probe results (e.g., "aperio", "leica").
    fn vendor_name(&self) -> &'static str;
}

// ── Output types ─────────────────────────────────────────────────────────────

/// Output of a layout interpreter. Bundles the normalized metadata `Dataset`
/// with the two maps the pixel reader needs for dispatch.
#[derive(Debug)]
pub(crate) struct DatasetLayout {
    /// Normalized metadata tree for this file.
    pub dataset: Dataset,

    /// Maps each (scene, series, level, z, c, t) plane to its pixel source.
    /// The pixel reader uses this to dispatch `TileRequest`s to the correct
    /// IFD or NDPI JPEG path.
    pub tile_sources: HashMap<TileSourceKey, TileSource>,

    /// Maps associated image names ("macro", "label", "thumbnail") to their
    /// pixel sources. `Dataset::associated_images` stores only metadata
    /// (dimensions, sample_type); this map provides the IFD/strip reference
    /// needed to decode.
    pub associated_sources: HashMap<String, TileSource>,
}

/// Composite key for tile source lookup. Identifies a plane, not a tile.
/// The pixel reader computes tile addressing within the plane.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub(crate) struct TileSourceKey {
    pub scene: usize,
    pub series: usize,
    pub level: u32,
    pub z: u32,
    pub c: u32,
    pub t: u32,
}

/// Describes how to access a level's or associated image's pixel data.
#[derive(Debug, Clone)]
pub(crate) enum TileSource {
    /// Standard tiled TIFF IFD (Aperio SVS, OME-TIFF, etc.).
    TiledIfd {
        ifd_id: IfdId,
        /// JPEG tables from JPEGTables tag (tag 347), if present.
        jpeg_tables: Option<Vec<u8>>,
        compression: Compression,
    },

    /// NDPI giant JPEG with MCU-boundary extraction (fast path).
    ///
    /// `tiles_across` and `tiles_down` are derived from
    /// `ceil(image_width / virtual_tile_width)` and
    /// `ceil(image_height / virtual_tile_height)` at layout time.
    /// The pixel reader uses row-major indexing:
    ///   `idx = row * tiles_across + col`
    ///
    /// `strip_offset` and `strip_byte_count` are needed to compute the
    /// end boundary for the last tile:
    ///   `end = strip_offset + strip_byte_count` when
    ///   `idx + 1 == mcu_starts.len()`
    NdpiJpeg {
        ifd_id: IfdId,
        /// JPEG header bytes (SOI through end of SOS segment).
        jpeg_header: Vec<u8>,
        /// Tag number of the MCU-starts array (NDPI tag 65426).
        /// Resolved lazily by the pixel reader on first tile access.
        mcu_starts_tag: u16,
        /// Number of virtual tiles per row.
        tiles_across: u32,
        /// Number of virtual tile rows.
        tiles_down: u32,
        /// Restart interval in MCUs (from DRI marker).
        restart_interval: u16,
        /// Strip byte offset — used to compute last-tile end boundary.
        strip_offset: u64,
        /// Strip byte count — used to compute last-tile end boundary.
        strip_byte_count: u64,
    },

    /// NDPI level without restart markers — full decode required.
    ///
    /// The pixel reader decodes the entire JPEG once, caches it in
    /// `FullDecodeCache`, and extracts virtual tile regions on demand.
    NdpiFullDecode {
        ifd_id: IfdId,
        /// JPEG header bytes (SOI through end of SOS segment).
        jpeg_header: Vec<u8>,
        strip_offset: u64,
        strip_byte_count: u64,
    },

    /// Synthetic NDPI power-of-two level derived from a higher-resolution level.
    ///
    /// The compatibility model exposes a complete power-of-two NDPI pyramid even when the
    /// underlying file only stores sparse physical resolutions. `base_level`
    /// points at the nearest higher-resolution level already present in the
    /// public pyramid, and `factor` is the power-of-two reduction relative to
    /// that level.
    SyntheticDownsample { base_level: u32, factor: u32 },

    /// A public level composed from multiple tiled TIFF IFD-backed images.
    ///
    /// Leica SCN exposes one stitched public slide assembled from multiple
    /// brightfield images positioned within the collection coordinate space.
    /// Each component contributes a rectangular region at this public level.
    #[allow(dead_code)]
    StitchedLevel {
        components: Vec<StitchedLevelComponent>,
        direct_tiles: HashMap<(i64, i64), usize>,
    },

    /// Stripped TIFF (associated images, older formats).
    Stripped {
        ifd_id: IfdId,
        /// JPEG tables from JPEGTables tag (tag 347), if present.
        jpeg_tables: Option<Vec<u8>>,
        compression: Compression,
        strip_offsets: Vec<u64>,
        strip_byte_counts: Vec<u64>,
    },

    /// Associated image stored as an external JPEG sidecar file.
    ExternalJpeg { path: PathBuf },
}

#[derive(Debug, Clone)]
pub(crate) struct StitchedLevelComponent {
    pub ifd_id: IfdId,
    pub jpeg_tables: Option<Vec<u8>>,
    pub compression: Compression,
    pub origin_x: i64,
    pub origin_y: i64,
    pub width: u64,
    pub height: u64,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_across: u64,
    pub tiles_down: u64,
}

pub(crate) fn compute_tiff_dataset_identity(
    container: &TiffContainer,
    lowest_resolution_ifd: IfdId,
    property_ifd: IfdId,
) -> Result<DatasetIdentity, TiffParseError> {
    let quickhash1 = compute_tiff_quickhash(container, lowest_resolution_ifd, property_ifd)?;
    let dataset_id = match quickhash1.as_deref() {
        Some(hash) => dataset_id_from_hex(hash)?,
        None => fallback_dataset_id(container, lowest_resolution_ifd, property_ifd)?,
    };
    Ok(DatasetIdentity {
        dataset_id,
        quickhash1,
    })
}

fn compute_tiff_quickhash(
    container: &TiffContainer,
    lowest_resolution_ifd: IfdId,
    property_ifd: IfdId,
) -> Result<Option<String>, TiffParseError> {
    let mut hash = Quickhash1::new();

    if !hash_tiff_level(&mut hash, container, lowest_resolution_ifd)? {
        return Ok(None);
    }

    hash_tiff_string_properties(&mut hash, container, property_ifd);
    Ok(hash.finish())
}

fn hash_tiff_level(
    hash: &mut Quickhash1,
    container: &TiffContainer,
    ifd_id: IfdId,
) -> Result<bool, TiffParseError> {
    let ranges = tiff_data_ranges(container, ifd_id)?;
    let total_bytes: u64 = ranges.iter().map(|(_, len)| *len).sum();
    if total_bytes > QUICKHASH_MAX_LEVEL_BYTES {
        hash.disable();
        return Ok(false);
    }

    for (offset, len) in ranges {
        if len == 0 {
            continue;
        }
        hash.hash_file_part(container.path(), offset, Some(len))
            .map_err(wsi_to_tiff_error)?;
    }

    Ok(true)
}

fn tiff_data_ranges(
    container: &TiffContainer,
    ifd_id: IfdId,
) -> Result<Vec<(u64, u64)>, TiffParseError> {
    let ifd = container.ifd_by_id(ifd_id)?;
    let (offset_tag, length_tag) = if ifd.tags.contains_key(&tags::TILE_WIDTH) {
        (tags::TILE_OFFSETS, tags::TILE_BYTE_COUNTS)
    } else {
        (tags::STRIP_OFFSETS, tags::STRIP_BYTE_COUNTS)
    };
    let offsets = match container.get_u64_array(ifd_id, offset_tag) {
        Ok(values) => values,
        Err(TiffParseError::TagNotFound { .. }) => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let lengths = match container.get_u64_array(ifd_id, length_tag) {
        Ok(values) => values,
        Err(TiffParseError::TagNotFound { .. }) => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };

    if offsets.len() != lengths.len() {
        return Err(TiffParseError::Structure(format!(
            "IFD {} has {} offsets but {} byte counts",
            ifd_id,
            offsets.len(),
            lengths.len()
        )));
    }

    Ok(offsets
        .iter()
        .copied()
        .zip(lengths.iter().copied())
        .collect())
}

fn hash_tiff_string_properties(hash: &mut Quickhash1, container: &TiffContainer, ifd_id: IfdId) {
    const STRING_PROPS: [(&str, u16); 8] = [
        ("tiff.ImageDescription", tags::IMAGE_DESCRIPTION),
        ("tiff.Make", TAG_MAKE),
        ("tiff.Model", TAG_MODEL),
        ("tiff.Software", TAG_SOFTWARE),
        ("tiff.DateTime", TAG_DATETIME),
        ("tiff.Artist", TAG_ARTIST),
        ("tiff.HostComputer", TAG_HOST_COMPUTER),
        ("tiff.Copyright", TAG_COPYRIGHT),
    ];

    for (name, tag) in STRING_PROPS {
        hash_named_tiff_string_property(hash, container, ifd_id, name, tag);
    }
    hash_named_tiff_string_property(
        hash,
        container,
        ifd_id,
        "tiff.DocumentName",
        TAG_DOCUMENT_NAME,
    );
}

fn hash_named_tiff_string_property(
    hash: &mut Quickhash1,
    container: &TiffContainer,
    ifd_id: IfdId,
    name: &str,
    tag: u16,
) {
    hash.hash_string(name);
    hash.hash_string(container.get_string(ifd_id, tag).unwrap_or(""));
}

fn dataset_id_from_hex(hex: &str) -> Result<DatasetId, TiffParseError> {
    if hex.len() < 32 {
        return Err(TiffParseError::Structure(format!(
            "quickhash too short: expected at least 32 hex chars, got {}",
            hex.len()
        )));
    }

    let prefix = &hex[..32];
    let value = u128::from_str_radix(prefix, 16).map_err(|err| {
        TiffParseError::Structure(format!("invalid quickhash hex prefix '{prefix}': {err}"))
    })?;
    Ok(DatasetId(value))
}

fn fallback_dataset_id(
    container: &TiffContainer,
    lowest_resolution_ifd: IfdId,
    property_ifd: IfdId,
) -> Result<DatasetId, TiffParseError> {
    let mut hash = Quickhash1::new();
    hash.update(&container.ifd_count().to_le_bytes());
    hash.update(&(container.top_ifds().len() as u64).to_le_bytes());
    hash.update(&[match container.endian() {
        crate::formats::tiff_family::container::Endian::Little => 0,
        crate::formats::tiff_family::container::Endian::Big => 1,
    }]);
    hash.update(&[u8::from(container.is_bigtiff())]);

    let ifd = container.ifd_by_id(lowest_resolution_ifd)?;
    hash.update(&ifd.offset.to_le_bytes());
    for (offset, len) in tiff_data_ranges(container, lowest_resolution_ifd)? {
        hash.update(&offset.to_le_bytes());
        hash.update(&len.to_le_bytes());
    }
    hash_tiff_string_properties(&mut hash, container, property_ifd);

    let hex = hash.finish().ok_or_else(|| {
        TiffParseError::Structure("fallback dataset hash unexpectedly disabled".into())
    })?;
    dataset_id_from_hex(&hex)
}

fn wsi_to_tiff_error(err: WsiError) -> TiffParseError {
    match err {
        WsiError::Io(source) => source.into(),
        WsiError::IoWithPath { source, path } => TiffParseError::Io {
            kind: source.kind(),
            source,
            path: Some(std::sync::Arc::new(path)),
        },
        other => TiffParseError::Structure(other.to_string()),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{AxesShape, DatasetId, Level, SampleType, Scene, Series, TileLayout};
    use std::collections::HashSet;

    // ── TileSourceKey ──────────────────────────────────────────────────────────

    #[test]
    fn tile_source_key_equality() {
        let a = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        let b = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn tile_source_key_inequality_on_each_field() {
        let base = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        let cases: &[TileSourceKey] = &[
            TileSourceKey {
                scene: 1,
                ..base.clone()
            },
            TileSourceKey {
                series: 1,
                ..base.clone()
            },
            TileSourceKey {
                level: 1,
                ..base.clone()
            },
            TileSourceKey {
                z: 1,
                ..base.clone()
            },
            TileSourceKey {
                c: 1,
                ..base.clone()
            },
            TileSourceKey {
                t: 1,
                ..base.clone()
            },
        ];
        for key in cases {
            assert_ne!(base, *key, "expected {:?} != {:?}", base, key);
        }
    }

    #[test]
    fn tile_source_key_hash_consistency() {
        let mut set = HashSet::new();
        let key = TileSourceKey {
            scene: 0,
            series: 0,
            level: 2,
            z: 0,
            c: 0,
            t: 0,
        };
        set.insert(key.clone());
        set.insert(key.clone());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn tile_source_key_distinct_keys_in_hashmap() {
        let mut map: HashMap<TileSourceKey, u32> = HashMap::new();
        for level in 0..4u32 {
            let key = TileSourceKey {
                scene: 0,
                series: 0,
                level,
                z: 0,
                c: 0,
                t: 0,
            };
            map.insert(key, level * 10);
        }
        assert_eq!(map.len(), 4);
        let k = TileSourceKey {
            scene: 0,
            series: 0,
            level: 2,
            z: 0,
            c: 0,
            t: 0,
        };
        assert_eq!(map[&k], 20);
    }

    // ── TileSource construction ────────────────────────────────────────────────

    #[test]
    fn tile_source_tiled_ifd_construction() {
        let src = TileSource::TiledIfd {
            ifd_id: IfdId(512),
            jpeg_tables: Some(vec![0xFF, 0xD8]),
            compression: Compression::Jpeg,
        };
        match src {
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables,
                compression,
            } => {
                assert_eq!(ifd_id, IfdId(512));
                assert!(jpeg_tables.is_some());
                assert_eq!(compression, Compression::Jpeg);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn tile_source_ndpi_restart_construction() {
        let src = TileSource::NdpiJpeg {
            ifd_id: IfdId(1024),
            jpeg_header: vec![0xFF, 0xD8, 0xFF, 0xC0],
            mcu_starts_tag: 65426,
            tiles_across: 8,
            tiles_down: 6,
            restart_interval: 16,
            strip_offset: 4096,
            strip_byte_count: 1_000_000,
        };
        match src {
            TileSource::NdpiJpeg {
                ifd_id,
                tiles_across,
                tiles_down,
                restart_interval,
                strip_offset,
                strip_byte_count,
                mcu_starts_tag,
                ..
            } => {
                assert_eq!(ifd_id, IfdId(1024));
                assert_eq!(tiles_across, 8);
                assert_eq!(tiles_down, 6);
                assert_eq!(restart_interval, 16);
                assert_eq!(strip_offset, 4096);
                assert_eq!(strip_byte_count, 1_000_000);
                assert_eq!(mcu_starts_tag, 65426);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn tile_source_ndpi_full_decode_construction() {
        let src = TileSource::NdpiFullDecode {
            ifd_id: IfdId(2048),
            jpeg_header: vec![0xFF, 0xD8],
            strip_offset: 8192,
            strip_byte_count: 500_000,
        };
        match src {
            TileSource::NdpiFullDecode {
                ifd_id,
                strip_offset,
                strip_byte_count,
                ..
            } => {
                assert_eq!(ifd_id, IfdId(2048));
                assert_eq!(strip_offset, 8192);
                assert_eq!(strip_byte_count, 500_000);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn tile_source_stripped_construction() {
        let src = TileSource::Stripped {
            ifd_id: IfdId(4096),
            jpeg_tables: None,
            compression: Compression::None,
            strip_offsets: vec![0],
            strip_byte_counts: vec![0],
        };
        match src {
            TileSource::Stripped {
                ifd_id,
                compression,
                ..
            } => {
                assert_eq!(ifd_id, IfdId(4096));
                assert_eq!(compression, Compression::None);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ── DatasetLayout construction ─────────────────────────────────────────────

    fn make_minimal_dataset() -> Dataset {
        Dataset {
            id: DatasetId(1),
            scenes: vec![Scene {
                id: "scene-0".into(),
                name: None,
                series: vec![Series {
                    id: "series-0".into(),
                    axes: AxesShape::default(),
                    levels: vec![Level {
                        dimensions: (1024, 768),
                        downsample: 1.0,
                        tile_layout: TileLayout::Regular {
                            tile_width: 256,
                            tile_height: 256,
                            tiles_across: 4,
                            tiles_down: 3,
                        },
                    }],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Default::default(),
            icc_profiles: HashMap::new(),
        }
    }

    #[test]
    fn dataset_layout_construction_with_tile_sources() {
        let key = TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        };
        let source = TileSource::TiledIfd {
            ifd_id: IfdId(8),
            jpeg_tables: None,
            compression: Compression::Jpeg,
        };

        let mut tile_sources = HashMap::new();
        tile_sources.insert(key.clone(), source);

        let layout = DatasetLayout {
            dataset: make_minimal_dataset(),
            tile_sources,
            associated_sources: HashMap::new(),
        };

        assert_eq!(layout.dataset.id, DatasetId(1));
        assert!(layout.tile_sources.contains_key(&key));
        assert!(layout.associated_sources.is_empty());
    }

    #[test]
    fn dataset_layout_construction_with_associated_sources() {
        let macro_src = TileSource::Stripped {
            ifd_id: IfdId(256),
            jpeg_tables: None,
            compression: Compression::Jpeg,
            strip_offsets: vec![0],
            strip_byte_counts: vec![0],
        };

        let mut associated_sources = HashMap::new();
        associated_sources.insert("macro".to_string(), macro_src);

        let layout = DatasetLayout {
            dataset: make_minimal_dataset(),
            tile_sources: HashMap::new(),
            associated_sources,
        };

        assert!(layout.associated_sources.contains_key("macro"));
        assert!(layout.tile_sources.is_empty());
    }

    #[test]
    fn dataset_layout_multiple_levels() {
        let mut tile_sources = HashMap::new();
        for level in 0..4u32 {
            let key = TileSourceKey {
                scene: 0,
                series: 0,
                level,
                z: 0,
                c: 0,
                t: 0,
            };
            let src = TileSource::TiledIfd {
                ifd_id: IfdId(level as u64 * 512),
                jpeg_tables: None,
                compression: Compression::Jpeg,
            };
            tile_sources.insert(key, src);
        }

        let layout = DatasetLayout {
            dataset: make_minimal_dataset(),
            tile_sources,
            associated_sources: HashMap::new(),
        };

        assert_eq!(layout.tile_sources.len(), 4);
        let k2 = TileSourceKey {
            scene: 0,
            series: 0,
            level: 2,
            z: 0,
            c: 0,
            t: 0,
        };
        assert!(layout.tile_sources.contains_key(&k2));
    }
}
