// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::PathBuf;

use statumen::{Compression, PlaneSelection, Slide, TileCodecKind, TileLayout, TileRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let slide = Slide::open(&args.slide_path)?;
    let dataset = slide.dataset();
    let scene = dataset
        .scenes
        .get(args.scene)
        .ok_or_else(|| format!("scene {} is out of range", args.scene))?;
    let series = scene
        .series
        .get(args.series)
        .ok_or_else(|| format!("series {} is out of range", args.series))?;
    let level = series
        .levels
        .get(args.level as usize)
        .ok_or_else(|| format!("level {} is out of range", args.level))?;

    let TileLayout::Regular {
        tile_width,
        tile_height,
        tiles_across,
        tiles_down,
    } = level.tile_layout
    else {
        return Err(format!("level {} is not a regular tiled level", args.level).into());
    };

    let mut written = 0usize;
    for (col, row) in centered_tile_coords(tiles_across, tiles_down, args.count) {
        let req = TileRequest {
            scene: args.scene,
            series: args.series,
            level: args.level,
            plane: PlaneSelection::default(),
            col,
            row,
        };

        if slide.tile_codec_kind(&req) != TileCodecKind::Jpeg {
            continue;
        }

        let raw = slide.source().read_raw_compressed_tile(&req)?;
        if raw.compression != Compression::Jpeg
            || raw.width != tile_width
            || raw.height != tile_height
            || raw.bits_allocated != 8
            || raw.samples_per_pixel != 3
        {
            continue;
        }

        let name = format!(
            "level{:02}_row{:05}_col{:05}_{}x{}.jpg",
            args.level, row, col, raw.width, raw.height
        );
        fs::write(args.out_dir.join(name), raw.data)?;
        written += 1;
        if written == args.count {
            break;
        }
    }

    println!(
        "wrote {written} JPEG tiles from {} level {} to {}",
        args.slide_path.display(),
        args.level,
        args.out_dir.display()
    );
    if written == 0 {
        return Err("no full-size JPEG tiles were extracted".into());
    }
    Ok(())
}

struct Args {
    slide_path: PathBuf,
    out_dir: PathBuf,
    count: usize,
    scene: usize,
    series: usize,
    level: u32,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = env::args_os().skip(1);
        let slide_path = args.next().map(PathBuf::from).ok_or_else(usage)?;
        let out_dir = args.next().map(PathBuf::from).ok_or_else(usage)?;
        let count = parse_optional(args.next(), 23, "count")?;
        let level = parse_optional(args.next(), 0, "level")?;
        let scene = parse_optional(args.next(), 0, "scene")?;
        let series = parse_optional(args.next(), 0, "series")?;

        if count == 0 {
            return Err("count must be greater than zero".to_string());
        }
        Ok(Self {
            slide_path,
            out_dir,
            count,
            scene,
            series,
            level,
        })
    }
}

fn parse_optional<T>(raw: Option<std::ffi::OsString>, default: T, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    match raw {
        Some(raw) => raw
            .to_str()
            .ok_or_else(|| format!("{name} must be valid UTF-8"))?
            .parse()
            .map_err(|_| format!("{name} has an invalid value")),
        None => Ok(default),
    }
}

fn usage() -> String {
    "usage: cargo run --release --example extract_jpeg_tiles -- <slide-path> <out-dir> [count] [level] [scene] [series]".to_string()
}

fn centered_tile_coords(tiles_across: u64, tiles_down: u64, count: usize) -> Vec<(i64, i64)> {
    let side = (count as f64).sqrt().ceil() as u64;
    let start_col = tiles_across.saturating_sub(side) / 2;
    let start_row = tiles_down.saturating_sub(side) / 2;
    let end_col = (start_col + side).min(tiles_across);
    let end_row = (start_row + side).min(tiles_down);

    let mut coords = Vec::with_capacity(count);
    for row in start_row..end_row {
        for col in start_col..end_col {
            coords.push((col as i64, row as i64));
            if coords.len() == count {
                return coords;
            }
        }
    }

    for row in 0..tiles_down {
        for col in 0..tiles_across {
            let coord = (col as i64, row as i64);
            if !coords.contains(&coord) {
                coords.push(coord);
                if coords.len() == count {
                    return coords;
                }
            }
        }
    }
    coords
}
