use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::core::types::{
    AssociatedImage, AxesShape, ChannelInfo, Compression, CpuTile, Dataset, DatasetId,
    EncodedTilePhotometricInterpretation, Level, PlaneSelection, RawCompressedTile, SampleType,
    Scene, Series, TileCodecKind, TileLayout, TileOutputPreference, TilePixels, TileRequest,
};
use crate::decode::jp2k::{decode_jp2k_to_sample_buffer, Jp2kColorSpace};
use crate::decode::jp2k_codestream::{parse_codestream_header, validate_narrow_subset};
use crate::error::WsiError;
use crate::properties::Properties;

const MARKER_SOC_BYTES: [u8; 2] = [0xFF, 0x4F];

pub(crate) struct RawJp2kBackend;

impl RawJp2kBackend {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl FormatProbe for RawJp2kBackend {
    fn probe(&self, path: &Path) -> Result<ProbeResult, WsiError> {
        let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
            return Ok(not_detected());
        };
        if !matches!(extension.to_ascii_lowercase().as_str(), "j2k" | "j2c") {
            return Ok(not_detected());
        }

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return Ok(not_detected()),
        };
        let mut magic = [0u8; 2];
        if file.read_exact(&mut magic).is_err() || magic != MARKER_SOC_BYTES {
            return Ok(not_detected());
        }

        Ok(ProbeResult {
            detected: true,
            vendor: "raw-jp2k".into(),
            confidence: ProbeConfidence::Definite,
        })
    }
}

impl DatasetReader for RawJp2kBackend {
    fn open(&self, path: &Path) -> Result<Box<dyn SlideReader>, WsiError> {
        let data = std::fs::read(path)?;
        let header = parse_codestream_header(&data)?;
        validate_narrow_subset(&header)?;
        Ok(Box::new(RawJp2kReader {
            dataset: dataset_for_codestream(path, &data, header.image_width, header.image_height)?,
            data,
            width: header.image_width,
            height: header.image_height,
        }))
    }
}

fn not_detected() -> ProbeResult {
    ProbeResult {
        detected: false,
        vendor: "raw-jp2k".into(),
        confidence: ProbeConfidence::Likely,
    }
}

fn dataset_for_codestream(
    path: &Path,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Dataset, WsiError> {
    Ok(Dataset {
        id: dataset_id_for_raw_codestream(path, data)?,
        scenes: vec![Scene {
            id: "raw-jp2k".into(),
            name: path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned()),
            series: vec![Series {
                id: "0".into(),
                axes: AxesShape::default(),
                levels: vec![Level {
                    dimensions: (u64::from(width), u64::from(height)),
                    downsample: 1.0,
                    tile_layout: TileLayout::Regular {
                        tile_width: width,
                        tile_height: height,
                        tiles_across: 1,
                        tiles_down: 1,
                    },
                }],
                sample_type: SampleType::Uint8,
                channels: rgb_channels(),
            }],
        }],
        associated_images: HashMap::<String, AssociatedImage>::new(),
        properties: Properties::new(),
        icc_profiles: HashMap::new(),
    })
}

fn dataset_id_for_raw_codestream(path: &Path, data: &[u8]) -> Result<DatasetId, WsiError> {
    let mut hasher = Quickhash1::new();
    hasher.hash_string("raw-jp2k");
    hasher.hash_string(&path.display().to_string());
    hasher.update(data);
    let hash = hasher
        .finish()
        .ok_or_else(|| WsiError::Jp2k("raw JP2K dataset hash disabled".into()))?;
    let value = u128::from_str_radix(&hash[..32], 16)
        .map_err(|_| WsiError::Jp2k("raw JP2K dataset hash is not valid hex".into()))?;
    Ok(DatasetId(value))
}

fn rgb_channels() -> Vec<ChannelInfo> {
    vec![
        ChannelInfo {
            name: Some("R".into()),
            color: Some([255, 0, 0]),
            excitation_nm: None,
            emission_nm: None,
        },
        ChannelInfo {
            name: Some("G".into()),
            color: Some([0, 255, 0]),
            excitation_nm: None,
            emission_nm: None,
        },
        ChannelInfo {
            name: Some("B".into()),
            color: Some([0, 0, 255]),
            excitation_nm: None,
            emission_nm: None,
        },
    ]
}

struct RawJp2kReader {
    dataset: Dataset,
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl RawJp2kReader {
    fn validate_request(&self, req: &TileRequest) -> Result<(), WsiError> {
        if req.scene != 0 || req.series != 0 || req.level != 0 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "raw JP2K source has one scene, one series, and one level".into(),
            });
        }
        if req.plane != PlaneSelection::default() {
            return Err(WsiError::Unsupported {
                reason: "raw JP2K source has only the default plane".into(),
            });
        }
        if req.col != 0 || req.row != 0 {
            return Err(WsiError::TileRead {
                col: req.col,
                row: req.row,
                level: req.level,
                reason: "raw JP2K source has exactly one tile at (0, 0)".into(),
            });
        }
        Ok(())
    }
}

impl SlideReader for RawJp2kReader {
    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn tile_codec_kind(&self, _req: &TileRequest) -> TileCodecKind {
        TileCodecKind::Jp2k
    }

    fn read_tiles(
        &self,
        reqs: &[TileRequest],
        output: TileOutputPreference,
    ) -> Result<Vec<TilePixels>, WsiError> {
        if matches!(output, TileOutputPreference::RequireDevice { .. }) {
            return Err(WsiError::Unsupported {
                reason: "raw JP2K backend does not provide resident device tiles".into(),
            });
        }
        reqs.iter()
            .map(|req| self.read_tile_cpu(req).map(TilePixels::Cpu))
            .collect()
    }

    fn read_tile_cpu(&self, req: &TileRequest) -> Result<CpuTile, WsiError> {
        self.validate_request(req)?;
        decode_jp2k_to_sample_buffer(&self.data, self.width, self.height, Jp2kColorSpace::Rgb)
    }

    fn read_raw_compressed_tile(&self, req: &TileRequest) -> Result<RawCompressedTile, WsiError> {
        self.validate_request(req)?;
        Ok(RawCompressedTile {
            compression: Compression::Jp2kRgb,
            width: self.width,
            height: self.height,
            bits_allocated: 8,
            samples_per_pixel: 3,
            photometric_interpretation: EncodedTilePhotometricInterpretation::Rgb,
            data: self.data.clone(),
        })
    }

    fn read_associated(&self, name: &str) -> Result<CpuTile, WsiError> {
        Err(WsiError::AssociatedImageNotFound(name.into()))
    }
}
