use crate::error::WsiError;

const MARKER_SOC: u16 = 0xFF4F;
const MARKER_SIZ: u16 = 0xFF51;
const MARKER_COD: u16 = 0xFF52;
const MARKER_QCD: u16 = 0xFF5C;
#[cfg(test)]
const MARKER_POC: u16 = 0xFF5F;
const MARKER_SOT: u16 = 0xFF90;
const MARKER_SOD: u16 = 0xFF93;
const MARKER_EOC: u16 = 0xFFD9;
const MARKER_EPH: u16 = 0xFF92;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2kProgressionOrder {
    Lrcp,
    Rlcp,
    Rpcl,
    Pcrl,
    Cprl,
    Unknown(u8),
}

impl From<u8> for Jp2kProgressionOrder {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Lrcp,
            1 => Self::Rlcp,
            2 => Self::Rpcl,
            3 => Self::Pcrl,
            4 => Self::Cprl,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2kWaveletTransform {
    Irreversible9x7,
    Reversible5x3,
    Unknown(u8),
}

impl From<u8> for Jp2kWaveletTransform {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Irreversible9x7,
            1 => Self::Reversible5x3,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2kQuantizationStyle {
    NoQuantization,
    ScalarDerived,
    ScalarExpounded,
    Reserved(u8),
}

impl From<u8> for Jp2kQuantizationStyle {
    fn from(value: u8) -> Self {
        match value & 0x1F {
            0 => Self::NoQuantization,
            1 => Self::ScalarDerived,
            2 => Self::ScalarExpounded,
            other => Self::Reserved(other),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kComponentInfo {
    pub precision_bits: u8,
    pub is_signed: bool,
    pub horizontal_sample_separation: u8,
    pub vertical_sample_separation: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kCodingStyleInfo {
    pub progression_order: Jp2kProgressionOrder,
    pub layers: u16,
    pub multiple_component_transform: bool,
    pub decomposition_levels: u8,
    pub code_block_width_exponent: u8,
    pub code_block_height_exponent: u8,
    pub code_block_style: u8,
    pub transform: Jp2kWaveletTransform,
    pub custom_precincts: bool,
    pub sop_markers: bool,
    pub eph_markers: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kQuantStep {
    pub exponent: u8,
    pub mantissa: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kQuantizationInfo {
    pub style: Jp2kQuantizationStyle,
    pub guard_bits: u8,
    pub steps: Vec<Jp2kQuantStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kTilePartHeader {
    pub tile_index: u16,
    pub tile_part_length: u32,
    pub tile_part_index: u8,
    pub tile_part_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kTilePartInfo {
    pub header: Jp2kTilePartHeader,
    pub data_offset: usize,
    pub data_length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2kCodestreamInfo {
    pub image_origin_x: u32,
    pub image_origin_y: u32,
    pub image_width: u32,
    pub image_height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_origin_x: u32,
    pub tile_origin_y: u32,
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub components: Vec<Jp2kComponentInfo>,
    pub coding_style: Jp2kCodingStyleInfo,
    pub quantization: Jp2kQuantizationInfo,
    pub main_header_length: usize,
    pub tile_parts: Vec<Jp2kTilePartInfo>,
    pub seen_markers: Vec<u16>,
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn set_offset(&mut self, offset: usize) -> Result<(), WsiError> {
        if offset > self.data.len() {
            return Err(WsiError::Jp2k("codestream offset out of range".into()));
        }
        self.offset = offset;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, WsiError> {
        if self.remaining() < 1 {
            return Err(WsiError::Jp2k("unexpected end of codestream".into()));
        }
        let value = self.data[self.offset];
        self.offset += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16, WsiError> {
        if self.remaining() < 2 {
            return Err(WsiError::Jp2k("unexpected end of codestream".into()));
        }
        let value = u16::from_be_bytes([self.data[self.offset], self.data[self.offset + 1]]);
        self.offset += 2;
        Ok(value)
    }

    fn read_u32(&mut self) -> Result<u32, WsiError> {
        if self.remaining() < 4 {
            return Err(WsiError::Jp2k("unexpected end of codestream".into()));
        }
        let value = u32::from_be_bytes([
            self.data[self.offset],
            self.data[self.offset + 1],
            self.data[self.offset + 2],
            self.data[self.offset + 3],
        ]);
        self.offset += 4;
        Ok(value)
    }

    fn read_segment_bytes(&mut self) -> Result<&'a [u8], WsiError> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            return Err(WsiError::Jp2k("invalid marker segment length".into()));
        }
        let payload_len = length - 2;
        if self.remaining() < payload_len {
            return Err(WsiError::Jp2k("truncated marker segment".into()));
        }
        let start = self.offset;
        self.offset += payload_len;
        Ok(&self.data[start..start + payload_len])
    }
}

impl Jp2kCodingStyleInfo {
    #[cfg(any(debug_assertions, test))]
    pub fn resolution_count(&self) -> u8 {
        self.decomposition_levels + 1
    }

    pub fn code_block_width(&self) -> u32 {
        1u32 << (self.code_block_width_exponent as u32 + 2)
    }

    pub fn code_block_height(&self) -> u32 {
        1u32 << (self.code_block_height_exponent as u32 + 2)
    }

    pub fn expected_expounded_quant_steps(&self) -> usize {
        3 * self.decomposition_levels as usize + 1
    }
}

struct SizInfo {
    image_origin_x: u32,
    image_origin_y: u32,
    image_width: u32,
    image_height: u32,
    tile_width: u32,
    tile_height: u32,
    tile_origin_x: u32,
    tile_origin_y: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    components: Vec<Jp2kComponentInfo>,
}

fn div_ceil_u32(value: u32, divisor: u32) -> Result<u32, WsiError> {
    if divisor == 0 {
        return Err(WsiError::Jp2k(
            "invalid zero tile divisor in SIZ marker".into(),
        ));
    }
    Ok(value.div_ceil(divisor))
}

fn parse_siz(segment: &[u8]) -> Result<SizInfo, WsiError> {
    let mut reader = Reader::new(segment);
    let _capabilities = reader.read_u16()?;
    let x_size = reader.read_u32()?;
    let y_size = reader.read_u32()?;
    let x_origin = reader.read_u32()?;
    let y_origin = reader.read_u32()?;
    let tile_width = reader.read_u32()?;
    let tile_height = reader.read_u32()?;
    let tile_origin_x = reader.read_u32()?;
    let tile_origin_y = reader.read_u32()?;
    let component_count = reader.read_u16()? as usize;

    let image_width = x_size
        .checked_sub(x_origin)
        .ok_or_else(|| WsiError::Jp2k("invalid SIZ image width".into()))?;
    let image_height = y_size
        .checked_sub(y_origin)
        .ok_or_else(|| WsiError::Jp2k("invalid SIZ image height".into()))?;
    let tile_span_x = x_size
        .checked_sub(tile_origin_x)
        .ok_or_else(|| WsiError::Jp2k("invalid SIZ tile origin x".into()))?;
    let tile_span_y = y_size
        .checked_sub(tile_origin_y)
        .ok_or_else(|| WsiError::Jp2k("invalid SIZ tile origin y".into()))?;
    let tile_count_x = div_ceil_u32(tile_span_x, tile_width)?;
    let tile_count_y = div_ceil_u32(tile_span_y, tile_height)?;

    let mut components = Vec::with_capacity(component_count);
    for _ in 0..component_count {
        let ssiz = reader.read_u8()?;
        let precision_bits = (ssiz & 0x7F) + 1;
        let is_signed = ssiz & 0x80 != 0;
        let horizontal_sample_separation = reader.read_u8()?;
        let vertical_sample_separation = reader.read_u8()?;
        components.push(Jp2kComponentInfo {
            precision_bits,
            is_signed,
            horizontal_sample_separation,
            vertical_sample_separation,
        });
    }

    Ok(SizInfo {
        image_origin_x: x_origin,
        image_origin_y: y_origin,
        image_width,
        image_height,
        tile_width,
        tile_height,
        tile_origin_x,
        tile_origin_y,
        tile_count_x,
        tile_count_y,
        components,
    })
}

fn parse_cod(segment: &[u8]) -> Result<Jp2kCodingStyleInfo, WsiError> {
    let mut reader = Reader::new(segment);
    let scod = reader.read_u8()?;
    let progression_order = Jp2kProgressionOrder::from(reader.read_u8()?);
    let layers = reader.read_u16()?;
    let multiple_component_transform = reader.read_u8()? != 0;
    let decomposition_levels = reader.read_u8()?;
    let code_block_width_exponent = reader.read_u8()?;
    let code_block_height_exponent = reader.read_u8()?;
    let code_block_style = reader.read_u8()?;
    let transform = Jp2kWaveletTransform::from(reader.read_u8()?);

    Ok(Jp2kCodingStyleInfo {
        progression_order,
        layers,
        multiple_component_transform,
        decomposition_levels,
        code_block_width_exponent,
        code_block_height_exponent,
        code_block_style,
        transform,
        custom_precincts: scod & 0x01 != 0,
        sop_markers: scod & 0x02 != 0,
        eph_markers: scod & 0x04 != 0,
    })
}

fn parse_qcd(segment: &[u8]) -> Result<Jp2kQuantizationInfo, WsiError> {
    let mut reader = Reader::new(segment);
    let sqcd = reader.read_u8()?;
    let style = Jp2kQuantizationStyle::from(sqcd);
    let steps = match style {
        Jp2kQuantizationStyle::NoQuantization => segment[1..]
            .iter()
            .map(|value| Jp2kQuantStep {
                exponent: value >> 3,
                mantissa: 0,
            })
            .collect(),
        Jp2kQuantizationStyle::ScalarDerived | Jp2kQuantizationStyle::ScalarExpounded => {
            if !(segment.len() - 1).is_multiple_of(2) {
                return Err(WsiError::Jp2k(
                    "invalid QCD marker payload length for scalar quantization".into(),
                ));
            }
            segment[1..]
                .chunks_exact(2)
                .map(|chunk| {
                    let packed = u16::from_be_bytes([chunk[0], chunk[1]]);
                    Jp2kQuantStep {
                        exponent: (packed >> 11) as u8,
                        mantissa: packed & 0x07FF,
                    }
                })
                .collect()
        }
        Jp2kQuantizationStyle::Reserved(value) => {
            return Err(WsiError::Jp2k(format!(
                "unsupported JP2K quantization style value: {}",
                value
            )));
        }
    };
    Ok(Jp2kQuantizationInfo {
        style,
        guard_bits: sqcd >> 5,
        steps,
    })
}

fn parse_sot(segment: &[u8]) -> Result<Jp2kTilePartHeader, WsiError> {
    let mut reader = Reader::new(segment);
    Ok(Jp2kTilePartHeader {
        tile_index: reader.read_u16()?,
        tile_part_length: reader.read_u32()?,
        tile_part_index: reader.read_u8()?,
        tile_part_count: reader.read_u8()?,
    })
}

pub(crate) fn parse_codestream_header(data: &[u8]) -> Result<Jp2kCodestreamInfo, WsiError> {
    let mut reader = Reader::new(data);
    let soc = reader.read_u16()?;
    if soc != MARKER_SOC {
        return Err(WsiError::Jp2k(
            "expected raw J2K codestream starting with SOC marker".into(),
        ));
    }

    let mut seen_markers = vec![MARKER_SOC];
    let mut siz = None;
    let mut cod = None;
    let mut qcd = None;
    let mut main_header_length = data.len();
    let mut tile_parts = Vec::new();

    while reader.remaining() >= 2 {
        let marker_offset = reader.offset();
        let marker = reader.read_u16()?;
        seen_markers.push(marker);
        match marker {
            MARKER_SOD | MARKER_EOC => {
                main_header_length = marker_offset;
                break;
            }
            MARKER_EPH => continue,
            MARKER_SIZ => siz = Some(parse_siz(reader.read_segment_bytes()?)?),
            MARKER_COD => cod = Some(parse_cod(reader.read_segment_bytes()?)?),
            MARKER_QCD => qcd = Some(parse_qcd(reader.read_segment_bytes()?)?),
            MARKER_SOT => {
                main_header_length = marker_offset;
                let tile_part_header = parse_sot(reader.read_segment_bytes()?)?;
                loop {
                    if reader.remaining() < 2 {
                        return Err(WsiError::Jp2k(
                            "tile-part header ended before SOD marker".into(),
                        ));
                    }
                    let inner_marker = reader.read_u16()?;
                    seen_markers.push(inner_marker);
                    match inner_marker {
                        MARKER_SOD => {
                            let data_offset = reader.offset();
                            let data_end = if tile_part_header.tile_part_length == 0 {
                                data.len()
                            } else {
                                marker_offset
                                    .checked_add(tile_part_header.tile_part_length as usize)
                                    .ok_or_else(|| {
                                        WsiError::Jp2k("tile-part length overflow".into())
                                    })?
                            };
                            if data_end < data_offset || data_end > data.len() {
                                return Err(WsiError::Jp2k(
                                    "invalid JP2K tile-part payload bounds".into(),
                                ));
                            }
                            tile_parts.push(Jp2kTilePartInfo {
                                header: tile_part_header,
                                data_offset,
                                data_length: data_end - data_offset,
                            });
                            reader.set_offset(data_end)?;
                            break;
                        }
                        MARKER_EPH => continue,
                        MARKER_EOC => {
                            return Err(WsiError::Jp2k(
                                "tile-part terminated before SOD marker".into(),
                            ));
                        }
                        _ => {
                            let _ = reader.read_segment_bytes()?;
                        }
                    }
                }
            }
            _ => {
                let _ = reader.read_segment_bytes()?;
            }
        }
    }

    let siz = siz.ok_or_else(|| WsiError::Jp2k("missing SIZ marker".into()))?;
    let coding_style = cod.ok_or_else(|| WsiError::Jp2k("missing COD marker".into()))?;
    let quantization = qcd.ok_or_else(|| WsiError::Jp2k("missing QCD marker".into()))?;

    Ok(Jp2kCodestreamInfo {
        image_origin_x: siz.image_origin_x,
        image_origin_y: siz.image_origin_y,
        image_width: siz.image_width,
        image_height: siz.image_height,
        tile_width: siz.tile_width,
        tile_height: siz.tile_height,
        tile_origin_x: siz.tile_origin_x,
        tile_origin_y: siz.tile_origin_y,
        tile_count_x: siz.tile_count_x,
        tile_count_y: siz.tile_count_y,
        components: siz.components,
        coding_style,
        quantization,
        main_header_length,
        tile_parts,
        seen_markers,
    })
}

pub(crate) fn validate_narrow_subset(info: &Jp2kCodestreamInfo) -> Result<(), WsiError> {
    if info.tile_count_x != 1 || info.tile_count_y != 1 {
        return Err(WsiError::Jp2k(format!(
            "unsupported JP2K tile grid: expected single tile, found {}x{}",
            info.tile_count_x, info.tile_count_y
        )));
    }
    if info.tile_parts.len() != 1 {
        return Err(WsiError::Jp2k(format!(
            "unsupported JP2K tile-part count: expected 1, found {}",
            info.tile_parts.len()
        )));
    }
    if info.tile_parts[0].header.tile_part_index != 0 {
        return Err(WsiError::Jp2k(
            "unsupported non-zero JP2K tile-part index".into(),
        ));
    }
    if info.tile_parts[0].header.tile_part_count > 1 {
        return Err(WsiError::Jp2k(format!(
            "unsupported JP2K tile-part count marker: {}",
            info.tile_parts[0].header.tile_part_count
        )));
    }
    if info.components.len() != 3 {
        return Err(WsiError::Jp2k(format!(
            "unsupported JP2K component count: expected 3, found {}",
            info.components.len()
        )));
    }

    for (index, component) in info.components.iter().enumerate() {
        if component.precision_bits != 8 || component.is_signed {
            return Err(WsiError::Jp2k(format!(
                "unsupported JP2K component {} precision/sign: expected unsigned 8-bit",
                index
            )));
        }
        if component.horizontal_sample_separation == 0 || component.vertical_sample_separation == 0
        {
            return Err(WsiError::Jp2k(format!(
                "invalid JP2K sampling factors for component {}",
                index
            )));
        }
    }

    let y = &info.components[0];
    let cb = &info.components[1];
    let cr = &info.components[2];
    if y.horizontal_sample_separation != 1 || y.vertical_sample_separation != 1 {
        return Err(WsiError::Jp2k(
            "unsupported JP2K luma sampling factors".into(),
        ));
    }
    if cb != cr {
        return Err(WsiError::Jp2k(
            "unsupported JP2K asymmetric chroma subsampling".into(),
        ));
    }
    if !matches!(cb.horizontal_sample_separation, 1 | 2)
        || !matches!(cb.vertical_sample_separation, 1 | 2)
    {
        return Err(WsiError::Jp2k(
            "unsupported JP2K chroma subsampling; expected 4:4:4, 4:2:2, or 4:2:0".into(),
        ));
    }
    if !matches!(
        info.coding_style.transform,
        Jp2kWaveletTransform::Irreversible9x7 | Jp2kWaveletTransform::Reversible5x3
    ) {
        return Err(WsiError::Jp2k(
            "unsupported JP2K wavelet transform; expected irreversible 9/7 or reversible 5/3"
                .into(),
        ));
    }
    if info.coding_style.code_block_width() > 64 || info.coding_style.code_block_height() > 64 {
        return Err(WsiError::Jp2k(
            "unsupported JP2K code-block size; expected at most 64x64".into(),
        ));
    }
    if info.coding_style.code_block_width_exponent as u16
        + info.coding_style.code_block_height_exponent as u16
        > 8
    {
        return Err(WsiError::Jp2k(
            "unsupported JP2K code-block exponent sum".into(),
        ));
    }
    if !matches!(info.coding_style.code_block_style, 0 | 0x40) {
        return Err(WsiError::Jp2k(
            "unsupported JP2K code-block style; expected default or HT block coding".into(),
        ));
    }
    if info.coding_style.layers == 0 {
        return Err(WsiError::Jp2k("invalid JP2K layer count".into()));
    }
    match info.quantization.style {
        Jp2kQuantizationStyle::NoQuantization => {}
        Jp2kQuantizationStyle::ScalarDerived => {
            if info.quantization.steps.len() != 1 {
                return Err(WsiError::Jp2k(
                    "invalid JP2K derived quantization step count".into(),
                ));
            }
        }
        Jp2kQuantizationStyle::ScalarExpounded => {
            let expected_steps = info.coding_style.expected_expounded_quant_steps();
            if info.quantization.steps.len() != expected_steps {
                return Err(WsiError::Jp2k(format!(
                    "invalid JP2K expounded quantization step count: expected {}, found {}",
                    expected_steps,
                    info.quantization.steps.len()
                )));
            }
        }
        _ => {
            return Err(WsiError::Jp2k(
                "unsupported JP2K quantization style; expected scalar deadzone".into(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(marker: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&marker.to_be_bytes());
        out.extend_from_slice(&((payload.len() as u16) + 2).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn build_siz(
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
        chroma_dx: u8,
        chroma_dy: u8,
        precision: u8,
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u16.to_be_bytes());
        payload.extend_from_slice(&width.to_be_bytes());
        payload.extend_from_slice(&height.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&tile_width.to_be_bytes());
        payload.extend_from_slice(&tile_height.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&3u16.to_be_bytes());
        payload.extend_from_slice(&[(precision - 1) & 0x7F, 1, 1]);
        payload.extend_from_slice(&[(precision - 1) & 0x7F, chroma_dx, chroma_dy]);
        payload.extend_from_slice(&[(precision - 1) & 0x7F, chroma_dx, chroma_dy]);
        segment(MARKER_SIZ, &payload)
    }

    fn build_cod(transform: u8, mct: bool) -> Vec<u8> {
        build_cod_with_code_block_style(transform, mct, 0)
    }

    fn build_cod_with_code_block_style(transform: u8, mct: bool, code_block_style: u8) -> Vec<u8> {
        let payload = [
            0,
            0,
            0,
            1,
            u8::from(mct),
            5,
            4,
            4,
            code_block_style,
            transform,
        ];
        segment(MARKER_COD, &payload)
    }

    fn build_qcd(style: u8) -> Vec<u8> {
        let mut payload = vec![0b0100_0000 | style];
        match style {
            1 => payload.extend_from_slice(&[0x08, 0x00]),
            2 => {
                for _ in 0..16 {
                    payload.extend_from_slice(&[0x08, 0x00]);
                }
            }
            _ => payload.push(0x40),
        }
        segment(MARKER_QCD, &payload)
    }

    fn build_sot(tile_part_length: u32, tile_part_index: u8, tile_part_count: u8) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u16.to_be_bytes());
        payload.extend_from_slice(&tile_part_length.to_be_bytes());
        payload.push(tile_part_index);
        payload.push(tile_part_count);
        segment(MARKER_SOT, &payload)
    }

    fn build_supported_codestream(
        chroma_dx: u8,
        chroma_dy: u8,
        mct: bool,
        tile_width: u32,
        tile_height: u32,
        tile_part_count: u8,
    ) -> Vec<u8> {
        let mut stream = Vec::new();
        let entropy_data = [0x00, 0x01, 0x02, 0x03];
        let tile_part_length = 2 + (2 + 8) + 2 + entropy_data.len() as u32;
        stream.extend_from_slice(&MARKER_SOC.to_be_bytes());
        stream.extend_from_slice(&build_siz(
            512,
            256,
            tile_width,
            tile_height,
            chroma_dx,
            chroma_dy,
            8,
        ));
        stream.extend_from_slice(&build_cod(0, mct));
        stream.extend_from_slice(&build_qcd(2));
        stream.extend_from_slice(&build_sot(tile_part_length, 0, tile_part_count));
        stream.extend_from_slice(&MARKER_SOD.to_be_bytes());
        stream.extend_from_slice(&entropy_data);
        stream.extend_from_slice(&MARKER_EOC.to_be_bytes());
        stream
    }

    #[test]
    fn parse_supported_codestream_header() {
        let stream = build_supported_codestream(2, 2, true, 512, 256, 1);
        let info = parse_codestream_header(&stream).unwrap();
        assert_eq!(info.image_width, 512);
        assert_eq!(info.image_height, 256);
        assert_eq!(info.tile_width, 512);
        assert_eq!(info.tile_height, 256);
        assert_eq!(info.tile_count_x, 1);
        assert_eq!(info.tile_count_y, 1);
        assert_eq!(info.components.len(), 3);
        assert_eq!(
            info.coding_style.transform,
            Jp2kWaveletTransform::Irreversible9x7
        );
        assert_eq!(
            info.quantization.style,
            Jp2kQuantizationStyle::ScalarExpounded
        );
        assert_eq!(info.quantization.steps.len(), 16);
        assert_eq!(info.quantization.steps[0].exponent, 1);
        assert_eq!(info.coding_style.code_block_width(), 64);
        assert_eq!(info.coding_style.code_block_height(), 64);
        assert_eq!(info.tile_parts.len(), 1);
        assert_eq!(info.tile_parts[0].header.tile_part_index, 0);
        assert_eq!(info.tile_parts[0].data_length, 4);
    }

    #[test]
    fn validate_supported_subset_accepts_420() {
        let stream = build_supported_codestream(2, 2, true, 512, 256, 1);
        let info = parse_codestream_header(&stream).unwrap();
        validate_narrow_subset(&info).unwrap();
    }

    #[test]
    fn validate_supported_subset_accepts_444() {
        let stream = build_supported_codestream(1, 1, false, 512, 256, 1);
        let info = parse_codestream_header(&stream).unwrap();
        validate_narrow_subset(&info).unwrap();
    }

    #[test]
    fn reject_missing_soc() {
        let result = parse_codestream_header(&[0x00, 0x00, 0xFF, 0x51]);
        assert!(result.is_err());
    }

    #[test]
    fn reject_non_8bit_subset() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&MARKER_SOC.to_be_bytes());
        stream.extend_from_slice(&build_siz(64, 64, 64, 64, 1, 1, 12));
        stream.extend_from_slice(&build_cod(0, false));
        stream.extend_from_slice(&build_qcd(1));
        stream.extend_from_slice(&build_sot(14, 0, 1));
        stream.extend_from_slice(&MARKER_SOD.to_be_bytes());
        let info = parse_codestream_header(&stream).unwrap();
        let err = validate_narrow_subset(&info).unwrap_err().to_string();
        assert!(err.contains("unsigned 8-bit"));
    }

    #[test]
    fn validate_supported_subset_accepts_reversible_lossless_transform() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&MARKER_SOC.to_be_bytes());
        stream.extend_from_slice(&build_siz(64, 64, 64, 64, 1, 1, 8));
        stream.extend_from_slice(&build_cod(1, false));
        stream.extend_from_slice(&build_qcd(0));
        stream.extend_from_slice(&build_sot(14, 0, 1));
        stream.extend_from_slice(&MARKER_SOD.to_be_bytes());
        let info = parse_codestream_header(&stream).unwrap();
        validate_narrow_subset(&info).unwrap();
    }

    #[test]
    fn validate_supported_subset_accepts_htj2k_lossless_profile() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&MARKER_SOC.to_be_bytes());
        stream.extend_from_slice(&build_siz(64, 64, 64, 64, 1, 1, 8));
        stream.extend_from_slice(&build_cod_with_code_block_style(1, true, 0x40));
        stream.extend_from_slice(&build_qcd(0));
        stream.extend_from_slice(&build_sot(14, 0, 1));
        stream.extend_from_slice(&MARKER_SOD.to_be_bytes());
        let info = parse_codestream_header(&stream).unwrap();
        validate_narrow_subset(&info).unwrap();
    }

    #[test]
    fn accept_decoder_supported_marker_segments() {
        let mut stream = build_supported_codestream(1, 1, false, 512, 256, 1);
        let insert_at = 2;
        stream.splice(insert_at..insert_at, segment(MARKER_POC, &[0, 0, 0]));
        let info = parse_codestream_header(&stream).unwrap();
        validate_narrow_subset(&info).unwrap();
    }

    #[test]
    fn reject_invalid_expounded_quant_step_count() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&MARKER_SOC.to_be_bytes());
        stream.extend_from_slice(&build_siz(64, 64, 64, 64, 1, 1, 8));
        stream.extend_from_slice(&build_cod(0, false));
        stream.extend_from_slice(&segment(MARKER_QCD, &[0b0100_0010, 0x08, 0x00]));
        stream.extend_from_slice(&build_sot(14, 0, 1));
        stream.extend_from_slice(&MARKER_SOD.to_be_bytes());
        let info = parse_codestream_header(&stream).unwrap();
        let err = validate_narrow_subset(&info).unwrap_err().to_string();
        assert!(err.contains("expounded quantization step count"));
    }

    #[test]
    fn reject_multi_tile_subset() {
        let stream = build_supported_codestream(1, 1, false, 256, 256, 1);
        let info = parse_codestream_header(&stream).unwrap();
        let err = validate_narrow_subset(&info).unwrap_err().to_string();
        assert!(err.contains("single tile"));
    }

    #[test]
    fn reject_multi_tile_part_subset() {
        let stream = build_supported_codestream(1, 1, false, 512, 256, 2);
        let info = parse_codestream_header(&stream).unwrap();
        let err = validate_narrow_subset(&info).unwrap_err().to_string();
        assert!(err.contains("tile-part count"));
    }
}
