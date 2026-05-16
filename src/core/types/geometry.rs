use super::*;

// ── Tile layout ────────────────────────────────────────────────────

/// How tiles are organized at a given level.
#[derive(Debug)]
pub enum TileLayout {
    /// Regular grid — fixed tile size, row-major.
    Regular {
        tile_width: u32,
        tile_height: u32,
        tiles_across: u64,
        tiles_down: u64,
    },
    /// Per-tile offsets (Ventana BIF, some DICOM).
    /// Geometry follows compatibility tilemap semantics and uses floating-point
    /// tile advances/offsets plus conservative extra-tile expansion.
    Irregular {
        tile_advance: (f64, f64),
        /// Extra tiles to consider around the nominal tilemap region, in the
        /// order `(top, bottom, left, right)`.
        extra_tiles: (u32, u32, u32, u32),
        tiles: HashMap<(i64, i64), TileEntry>,
    },
    /// Entire level is one contiguous image (NDPI giant JPEG).
    /// Backend exposes it as a virtual tile grid.
    WholeLevel {
        width: u64,
        height: u64,
        virtual_tile_width: u32,
        virtual_tile_height: u32,
    },
}

/// Result of tile intersection computation.
#[derive(Debug, Clone)]
pub struct TileHit {
    pub col: i64,
    pub row: i64,
    /// Pixel offset where this tile's top-left lands in the output buffer.
    pub dest_x: i64,
    pub dest_y: i64,
    /// Floating-point placement used by irregular tilemaps.
    pub dest_x_f64: f64,
    pub dest_y_f64: f64,
}

impl TileLayout {
    /// Compute which tiles intersect the given pixel region.
    pub fn tiles_for_region(&self, x: i64, y: i64, w: u32, h: u32) -> Vec<TileHit> {
        match self {
            TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across,
                tiles_down,
            } => {
                let tw = *tile_width as i64;
                let th = *tile_height as i64;
                let start_col = if x >= 0 { x / tw } else { (x - tw + 1) / tw };
                let start_row = if y >= 0 { y / th } else { (y - th + 1) / th };
                let end_col = (x + w as i64 + tw - 1) / tw;
                let end_row = (y + h as i64 + th - 1) / th;

                let mut hits = Vec::new();
                for row in start_row..end_row {
                    for col in start_col..end_col {
                        if col >= 0
                            && col < *tiles_across as i64
                            && row >= 0
                            && row < *tiles_down as i64
                        {
                            hits.push(TileHit {
                                col,
                                row,
                                dest_x: col * tw - x,
                                dest_y: row * th - y,
                                dest_x_f64: (col * tw - x) as f64,
                                dest_y_f64: (row * th - y) as f64,
                            });
                        }
                    }
                }
                hits
            }
            TileLayout::WholeLevel {
                width,
                height,
                virtual_tile_width,
                virtual_tile_height,
            } => {
                let vtw = *virtual_tile_width as i64;
                let vth = *virtual_tile_height as i64;
                let max_col = (*width as i64 + vtw - 1) / vtw;
                let max_row = (*height as i64 + vth - 1) / vth;

                let start_col = if x >= 0 { x / vtw } else { (x - vtw + 1) / vtw }.max(0);
                let start_row = if y >= 0 { y / vth } else { (y - vth + 1) / vth }.max(0);
                let end_col = ((x + w as i64 + vtw - 1) / vtw).min(max_col);
                let end_row = ((y + h as i64 + vth - 1) / vth).min(max_row);

                let mut hits = Vec::new();
                for row in start_row..end_row {
                    for col in start_col..end_col {
                        hits.push(TileHit {
                            col,
                            row,
                            dest_x: col * vtw - x,
                            dest_y: row * vth - y,
                            dest_x_f64: (col * vtw - x) as f64,
                            dest_y_f64: (row * vth - y) as f64,
                        });
                    }
                }
                hits
            }
            TileLayout::Irregular {
                tile_advance,
                extra_tiles,
                tiles,
            } => {
                let adv_x = tile_advance.0;
                let adv_y = tile_advance.1;
                if !(adv_x.is_finite() && adv_y.is_finite()) || adv_x <= 0.0 || adv_y <= 0.0 {
                    return Vec::new();
                }

                let (extra_top, extra_bottom, extra_left, extra_right) = *extra_tiles;
                let region_x = x as f64;
                let region_y = y as f64;
                let region_x2 = region_x + w as f64;
                let region_y2 = region_y + h as f64;
                let start_col = (region_x / adv_x) as i64 - i64::from(extra_left);
                let end_col = (region_x2 / adv_x).ceil() as i64 + i64::from(extra_right);
                let start_row = (region_y / adv_y) as i64 - i64::from(extra_top);
                let end_row = (region_y2 / adv_y).ceil() as i64 + i64::from(extra_bottom);
                let mut hits = Vec::new();
                for row in start_row..end_row {
                    for col in start_col..end_col {
                        if let Some(entry) = tiles.get(&(col, row)) {
                            let tile_x = col as f64 * adv_x + entry.offset.0;
                            let tile_y = row as f64 * adv_y + entry.offset.1;
                            let tile_x2 = tile_x + entry.dimensions.0 as f64;
                            let tile_y2 = tile_y + entry.dimensions.1 as f64;

                            if tile_x2 > region_x
                                && tile_x < region_x2
                                && tile_y2 > region_y
                                && tile_y < region_y2
                            {
                                hits.push(TileHit {
                                    col,
                                    row,
                                    dest_x: (tile_x - region_x).round() as i64,
                                    dest_y: (tile_y - region_y).round() as i64,
                                    dest_x_f64: tile_x - region_x,
                                    dest_y_f64: tile_y - region_y,
                                });
                            }
                        }
                    }
                }
                hits
            }
        }
    }
}

/// Per-tile position and size in an Irregular layout.
#[derive(Debug, Clone)]
pub struct TileEntry {
    pub offset: (f64, f64),
    pub dimensions: (u32, u32),
    /// For irregular TIFF tile grids (e.g. Ventana BIF), the exact TIFF tile
    /// index to use when reading from tile_offsets/tile_byte_counts arrays.
    /// `None` for regular row-major addressing.
    pub tiff_tile_index: Option<usize>,
}
