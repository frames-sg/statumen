use crate::decode::jp2k::Jp2kColorSpace;
use crate::decode::jp2k_codestream::Jp2kCodestreamInfo;
#[cfg(test)]
use crate::error::WsiError;

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct DecodedComponent {
    pub width: usize,
    pub height: usize,
    pub samples: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct DecodedImage {
    pub width: usize,
    pub height: usize,
    pub components: [DecodedComponent; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedInterleavedImage {
    pub width: usize,
    pub height: usize,
    pub colorspace: Jp2kColorSpace,
    pub pixels: Vec<u8>,
}

pub(crate) fn effective_output_colorspace(
    header: &Jp2kCodestreamInfo,
    requested_colorspace: Jp2kColorSpace,
) -> Jp2kColorSpace {
    if header.coding_style.multiple_component_transform {
        Jp2kColorSpace::Rgb
    } else {
        requested_colorspace
    }
}

#[cfg(test)]
fn interleaved_to_components(
    data: &[u8],
    width: usize,
    height: usize,
    _colorspace: Jp2kColorSpace,
) -> Result<DecodedImage, WsiError> {
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| WsiError::Jp2k("decoded JP2K image size overflow".into()))?;
    let expected_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| WsiError::Jp2k("decoded JP2K buffer size overflow".into()))?;
    if data.len() != expected_len {
        return Err(WsiError::Jp2k(format!(
            "unexpected decoded JP2K buffer length: expected {}, found {}",
            expected_len,
            data.len()
        )));
    }

    let mut r = Vec::with_capacity(pixel_count);
    let mut g = Vec::with_capacity(pixel_count);
    let mut b = Vec::with_capacity(pixel_count);
    for pixel in data.chunks_exact(3) {
        r.push(pixel[0] as i32);
        g.push(pixel[1] as i32);
        b.push(pixel[2] as i32);
    }

    Ok(DecodedImage {
        width,
        height,
        components: [
            DecodedComponent {
                width,
                height,
                samples: r,
            },
            DecodedComponent {
                width,
                height,
                samples: g,
            },
            DecodedComponent {
                width,
                height,
                samples: b,
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::jp2k_codestream::{
        Jp2kCodingStyleInfo, Jp2kProgressionOrder, Jp2kQuantStep, Jp2kQuantizationInfo,
        Jp2kQuantizationStyle, Jp2kWaveletTransform,
    };

    fn test_header(multiple_component_transform: bool) -> Jp2kCodestreamInfo {
        Jp2kCodestreamInfo {
            image_origin_x: 0,
            image_origin_y: 0,
            image_width: 2,
            image_height: 1,
            tile_width: 2,
            tile_height: 1,
            tile_origin_x: 0,
            tile_origin_y: 0,
            tile_count_x: 1,
            tile_count_y: 1,
            components: vec![],
            coding_style: Jp2kCodingStyleInfo {
                progression_order: Jp2kProgressionOrder::Lrcp,
                layers: 1,
                multiple_component_transform,
                decomposition_levels: 0,
                code_block_width_exponent: 4,
                code_block_height_exponent: 4,
                code_block_style: 0,
                transform: Jp2kWaveletTransform::Irreversible9x7,
                custom_precincts: false,
                sop_markers: false,
                eph_markers: false,
            },
            quantization: Jp2kQuantizationInfo {
                style: Jp2kQuantizationStyle::ScalarExpounded,
                guard_bits: 2,
                steps: vec![Jp2kQuantStep {
                    exponent: 8,
                    mantissa: 0,
                }],
            },
            main_header_length: 0,
            tile_parts: vec![],
            seen_markers: vec![],
        }
    }

    #[test]
    fn multiple_component_transform_forces_rgb_output() {
        let header = test_header(true);
        assert_eq!(
            effective_output_colorspace(&header, Jp2kColorSpace::YCbCr),
            Jp2kColorSpace::Rgb
        );
    }

    #[test]
    fn raw_ycbcr_without_mct_preserves_requested_colorspace() {
        let header = test_header(false);
        assert_eq!(
            effective_output_colorspace(&header, Jp2kColorSpace::YCbCr),
            Jp2kColorSpace::YCbCr
        );
    }

    #[test]
    fn interleaved_rgb_bytes_split_into_components() {
        let decoded =
            interleaved_to_components(&[10, 20, 30, 40, 50, 60], 2, 1, Jp2kColorSpace::Rgb)
                .unwrap();
        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 1);
        assert_eq!(decoded.components[0].samples, vec![10, 40]);
        assert_eq!(decoded.components[1].samples, vec![20, 50]);
        assert_eq!(decoded.components[2].samples, vec![30, 60]);
    }
}
