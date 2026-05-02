use std::path::PathBuf;

use statumen::{
    build_svcache, build_svcache_tiles, default_svcache_path, CacheConfig, PlaneSelection, Slide,
    SlideOpenOptions, SvcacheTileSelection, TileLayout,
};

fn usage() -> &'static str {
    "usage:
  svcache build <slide-path> [--out <cache-path>]
  svcache build-window <slide-path> --size <width>x<height> [--level <n>] [--z <n>] [--origin <x>,<y> | --center <x>,<y>] [--margin-tiles <n>] [--out <cache-path>]"
}

fn build(args: &[String]) -> Result<(), String> {
    let mut slide_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--out" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--out requires a cache path".to_string())?;
                out_path = Some(PathBuf::from(value));
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option {value}"));
            }
            value => {
                if slide_path.is_some() {
                    return Err(format!("unexpected argument {value}"));
                }
                slide_path = Some(PathBuf::from(value));
            }
        }
        idx += 1;
    }

    let slide_path = slide_path.ok_or_else(|| usage().to_string())?;
    let out_path = out_path.unwrap_or_else(|| default_svcache_path(&slide_path));
    build_svcache(&slide_path, &out_path).map_err(|err| err.to_string())?;
    println!("{}", out_path.display());
    Ok(())
}

#[derive(Debug, Clone)]
struct WindowArgs {
    slide_path: PathBuf,
    out_path: Option<PathBuf>,
    level: u32,
    z: u32,
    size: (u64, u64),
    origin: Option<(u64, u64)>,
    center: Option<(u64, u64)>,
    margin_tiles: u64,
}

fn parse_u32_option(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}

fn parse_u64_option(name: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}

fn parse_size(value: &str) -> Result<(u64, u64), String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| "size must use <width>x<height>".to_string())?;
    let width = parse_u64_option("width", width)?;
    let height = parse_u64_option("height", height)?;
    if width == 0 || height == 0 {
        return Err("size dimensions must be > 0".into());
    }
    Ok((width, height))
}

fn parse_pair(value: &str, name: &str) -> Result<(u64, u64), String> {
    let (x, y) = value
        .split_once(',')
        .ok_or_else(|| format!("{name} must use <x>,<y>"))?;
    Ok((parse_u64_option("x", x)?, parse_u64_option("y", y)?))
}

fn parse_build_window_args(args: &[String]) -> Result<WindowArgs, String> {
    let mut slide_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut level = 0;
    let mut z = 0;
    let mut size = None;
    let mut origin = None;
    let mut center = None;
    let mut margin_tiles = 1;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--out" => {
                idx += 1;
                out_path = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--out requires a cache path".to_string())?,
                ));
            }
            "--level" => {
                idx += 1;
                level = parse_u32_option(
                    "--level",
                    args.get(idx)
                        .ok_or_else(|| "--level requires a value".to_string())?,
                )?;
            }
            "--z" => {
                idx += 1;
                z = parse_u32_option(
                    "--z",
                    args.get(idx)
                        .ok_or_else(|| "--z requires a value".to_string())?,
                )?;
            }
            "--size" | "--viewport" => {
                idx += 1;
                size =
                    Some(parse_size(args.get(idx).ok_or_else(|| {
                        "--size requires <width>x<height>".to_string()
                    })?)?);
            }
            "--origin" => {
                idx += 1;
                origin = Some(parse_pair(
                    args.get(idx)
                        .ok_or_else(|| "--origin requires <x>,<y>".to_string())?,
                    "--origin",
                )?);
            }
            "--center" => {
                idx += 1;
                center = Some(parse_pair(
                    args.get(idx)
                        .ok_or_else(|| "--center requires <x>,<y>".to_string())?,
                    "--center",
                )?);
            }
            "--margin-tiles" => {
                idx += 1;
                margin_tiles = parse_u64_option(
                    "--margin-tiles",
                    args.get(idx)
                        .ok_or_else(|| "--margin-tiles requires a value".to_string())?,
                )?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option {value}"));
            }
            value => {
                if slide_path.is_some() {
                    return Err(format!("unexpected argument {value}"));
                }
                slide_path = Some(PathBuf::from(value));
            }
        }
        idx += 1;
    }

    if origin.is_some() && center.is_some() {
        return Err("--origin and --center are mutually exclusive".into());
    }

    Ok(WindowArgs {
        slide_path: slide_path.ok_or_else(|| usage().to_string())?,
        out_path,
        level,
        z,
        size: size.ok_or_else(|| "--size is required".to_string())?,
        origin,
        center,
        margin_tiles,
    })
}

fn cache_grid(level: &statumen::Level) -> (u32, u32, u64, u64) {
    match level.tile_layout {
        TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        } => (tile_width, tile_height, tiles_across, tiles_down),
        TileLayout::WholeLevel { width, height, .. } => {
            (256, 256, width.div_ceil(256), height.div_ceil(256))
        }
        TileLayout::Irregular { .. } => {
            let tile = 256_u32;
            (
                tile,
                tile,
                level.dimensions.0.div_ceil(u64::from(tile)),
                level.dimensions.1.div_ceil(u64::from(tile)),
            )
        }
    }
}

fn window_origin(
    level_dimensions: (u64, u64),
    size: (u64, u64),
    origin: Option<(u64, u64)>,
    center: Option<(u64, u64)>,
) -> (u64, u64) {
    if let Some(origin) = origin {
        return origin;
    }
    if let Some((cx, cy)) = center {
        return (cx.saturating_sub(size.0 / 2), cy.saturating_sub(size.1 / 2));
    }
    (
        level_dimensions.0.saturating_sub(size.0) / 2,
        level_dimensions.1.saturating_sub(size.1) / 2,
    )
}

fn selections_for_window(
    slide: &Slide,
    args: &WindowArgs,
) -> Result<Vec<SvcacheTileSelection>, String> {
    let scene = slide
        .dataset()
        .scenes
        .first()
        .ok_or_else(|| "slide has no scenes".to_string())?;
    let series = scene
        .series
        .first()
        .ok_or_else(|| "slide has no series".to_string())?;
    let level = series
        .levels
        .get(args.level as usize)
        .ok_or_else(|| format!("slide has no level {}", args.level))?;
    let (tile_width, tile_height, tiles_across, tiles_down) = cache_grid(level);
    if tiles_across == 0 || tiles_down == 0 {
        return Err(format!("level {} has an empty tile grid", args.level));
    }

    let (origin_x, origin_y) = window_origin(level.dimensions, args.size, args.origin, args.center);
    let max_x = level.dimensions.0.saturating_sub(1);
    let max_y = level.dimensions.1.saturating_sub(1);
    let end_x = origin_x
        .saturating_add(args.size.0.saturating_sub(1))
        .min(max_x);
    let end_y = origin_y
        .saturating_add(args.size.1.saturating_sub(1))
        .min(max_y);
    let start_col = (origin_x / u64::from(tile_width)).saturating_sub(args.margin_tiles);
    let start_row = (origin_y / u64::from(tile_height)).saturating_sub(args.margin_tiles);
    let end_col = ((end_x / u64::from(tile_width)) + args.margin_tiles).min(tiles_across - 1);
    let end_row = ((end_y / u64::from(tile_height)) + args.margin_tiles).min(tiles_down - 1);

    let mut selections =
        Vec::with_capacity(((end_col - start_col + 1) * (end_row - start_row + 1)) as usize);
    for row in start_row..=end_row {
        for col in start_col..=end_col {
            selections.push(SvcacheTileSelection {
                scene: 0,
                series: 0,
                level: args.level,
                plane: PlaneSelection {
                    z: args.z,
                    c: 0,
                    t: 0,
                },
                col: col as i64,
                row: row as i64,
            });
        }
    }
    Ok(selections)
}

fn build_window(args: &[String]) -> Result<(), String> {
    let args = parse_build_window_args(args)?;
    let slide = Slide::open_with_options(
        &args.slide_path,
        SlideOpenOptions::default().with_cache_config(
            CacheConfig::deterministic().with_shared_tile_bytes(64 * 1024 * 1024),
        ),
    )
    .map_err(|err| format!("open {}: {err}", args.slide_path.display()))?;
    let selections = selections_for_window(&slide, &args)?;
    let out_path = args
        .out_path
        .clone()
        .unwrap_or_else(|| default_svcache_path(&args.slide_path));
    let written = build_svcache_tiles(&args.slide_path, &out_path, &selections)
        .map_err(|err| err.to_string())?;
    println!("{} ({written} tiles)", out_path.display());
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(usage().to_string());
    }
    let command = args.remove(0);
    match command.as_str() {
        "build" => build(&args),
        "build-window" => build_window(&args),
        "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown command {other}\n{}", usage())),
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_build_window_requires_size() {
        let err = parse_build_window_args(&["slide.svs".into()]).unwrap_err();
        assert!(err.contains("--size"));
    }

    #[test]
    fn centered_window_defaults_to_middle_origin() {
        assert_eq!(
            window_origin((1000, 800), (200, 100), None, None),
            (400, 350)
        );
        assert_eq!(
            window_origin((1000, 800), (200, 100), None, Some((100, 50))),
            (0, 0)
        );
    }
}
