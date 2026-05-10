use statumen::{CpuTile, WsiError};

pub(crate) fn tile_to_premultiplied_argb(tile: CpuTile) -> Result<Vec<u32>, WsiError> {
    let rgba = tile.into_rgba()?;
    let mut argb = Vec::with_capacity(rgba.len() / 4);
    for pixel in rgba.chunks_exact(4) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        let a = pixel[3];
        let premultiply = |channel: u8| -> u8 {
            ((u16::from(channel) * u16::from(a) + 127) / 255).min(255) as u8
        };
        let r = premultiply(r);
        let g = premultiply(g);
        let b = premultiply(b);
        argb.push((u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b));
    }
    Ok(argb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use statumen::{ColorSpace, CpuTile};

    #[test]
    fn rgba_converts_to_premultiplied_argb_words() {
        let tile = CpuTile::from_u8_interleaved(
            2,
            1,
            4,
            ColorSpace::Rgba,
            vec![100, 50, 200, 128, 10, 20, 30, 0],
        )
        .expect("valid tile");

        let argb = tile_to_premultiplied_argb(tile).expect("convert");

        assert_eq!(argb, vec![0x8032_1964, 0x0000_0000]);
    }
}
