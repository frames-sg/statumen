use super::*;

// --- AxesShape ---

#[test]
fn axes_shape_default_is_2d() {
    let axes = AxesShape::default();
    assert_eq!(axes.z, 1);
    assert_eq!(axes.c, 1);
    assert_eq!(axes.t, 1);
}

// --- PlaneSelection ---

#[test]
fn plane_selection_default_is_origin() {
    let plane = PlaneSelection::default();
    assert_eq!(plane.z, 0);
    assert_eq!(plane.c, 0);
    assert_eq!(plane.t, 0);
}

// --- DatasetId ---

#[test]
fn dataset_id_equality() {
    let a = DatasetId(42);
    let b = DatasetId(42);
    let c = DatasetId(99);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn dataset_id_hash_consistent() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(DatasetId(1));
    set.insert(DatasetId(1));
    assert_eq!(set.len(), 1);
}

#[test]
fn tile_output_preference_constructors_map_correctly() {
    match TileOutputPreference::cpu() {
        TileOutputPreference::Cpu { backend } => {
            assert!(matches!(backend, OutputBackendRequest::Auto));
        }
        other => panic!("cpu() must produce Cpu/Auto, got {other:?}"),
    }
    match TileOutputPreference::cpu_only() {
        TileOutputPreference::Cpu { backend } => {
            assert!(matches!(backend, OutputBackendRequest::Cpu));
        }
        other => panic!("cpu_only() must produce Cpu/Cpu, got {other:?}"),
    }
    assert!(matches!(
        TileOutputPreference::prefer_device_auto(),
        TileOutputPreference::PreferDevice {
            backend: OutputBackendRequest::Auto,
            ..
        }
    ));
    assert!(matches!(
        TileOutputPreference::require_metal(),
        TileOutputPreference::RequireDevice {
            backend: OutputBackendRequest::Metal,
            ..
        }
    ));
}

#[test]
fn tile_output_preference_compressed_device_decode_is_explicit() {
    assert!(!TileOutputPreference::prefer_device_auto().compressed_device_decode_enabled());
    assert!(
        TileOutputPreference::prefer_device_auto_with_compressed_decode()
            .compressed_device_decode_enabled()
    );
    assert!(
        TileOutputPreference::require_device_auto_with_compressed_decode()
            .compressed_device_decode_enabled()
    );
    assert!(TileOutputPreference::require_device_auto_with_compressed_decode().requires_device());
}

#[test]
fn tile_output_preference_can_disable_adaptive_decode_route() {
    let preference = TileOutputPreference::prefer_device_auto_with_compressed_decode();
    assert!(preference.adaptive_decode_route_enabled());

    let preference = preference.without_adaptive_decode_route();
    assert!(!preference.adaptive_decode_route_enabled());
    assert!(preference.compressed_device_decode_enabled());
}

// --- TileLayout intersection ---

#[test]
fn regular_tiles_for_region_basic() {
    let layout = TileLayout::Regular {
        tile_width: 256,
        tile_height: 256,
        tiles_across: 4,
        tiles_down: 4,
    };
    // 300x300 at (100, 100) → cols 0-1, rows 0-1 → 4 tiles
    let tiles = layout.tiles_for_region(100, 100, 300, 300);
    assert_eq!(tiles.len(), 4);
    let coords: Vec<(i64, i64)> = tiles.iter().map(|t| (t.col, t.row)).collect();
    assert!(coords.contains(&(0, 0)));
    assert!(coords.contains(&(1, 0)));
    assert!(coords.contains(&(0, 1)));
    assert!(coords.contains(&(1, 1)));
}

#[test]
fn regular_tiles_single_tile() {
    let layout = TileLayout::Regular {
        tile_width: 256,
        tile_height: 256,
        tiles_across: 4,
        tiles_down: 4,
    };
    let tiles = layout.tiles_for_region(0, 0, 100, 100);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].col, 0);
    assert_eq!(tiles[0].row, 0);
}

#[test]
fn regular_tiles_clipped_at_bounds() {
    let layout = TileLayout::Regular {
        tile_width: 256,
        tile_height: 256,
        tiles_across: 2,
        tiles_down: 2,
    };
    // Region extends beyond grid
    let tiles = layout.tiles_for_region(256, 256, 512, 512);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].col, 1);
    assert_eq!(tiles[0].row, 1);
}

#[test]
fn regular_tiles_negative_coords() {
    let layout = TileLayout::Regular {
        tile_width: 256,
        tile_height: 256,
        tiles_across: 4,
        tiles_down: 4,
    };
    // Negative start — only in-bounds tiles returned
    let tiles = layout.tiles_for_region(-100, -100, 200, 200);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].col, 0);
    assert_eq!(tiles[0].row, 0);
}

#[test]
fn whole_level_tiles_for_region() {
    let layout = TileLayout::WholeLevel {
        width: 1024,
        height: 768,
        virtual_tile_width: 256,
        virtual_tile_height: 256,
    };
    // Region covering the entire image → ceil(1024/256) * ceil(768/256) = 4*3 = 12 tiles
    let tiles = layout.tiles_for_region(0, 0, 1024, 768);
    assert_eq!(tiles.len(), 12);
}

#[test]
fn whole_level_small_region() {
    let layout = TileLayout::WholeLevel {
        width: 4096,
        height: 4096,
        virtual_tile_width: 512,
        virtual_tile_height: 512,
    };
    // 100x100 at origin → 1 tile
    let tiles = layout.tiles_for_region(0, 0, 100, 100);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].col, 0);
    assert_eq!(tiles[0].row, 0);
}

#[test]
fn whole_level_negative_coords_clamp_to_first_tile() {
    let layout = TileLayout::WholeLevel {
        width: 1024,
        height: 1024,
        virtual_tile_width: 256,
        virtual_tile_height: 256,
    };

    let tiles = layout.tiles_for_region(-300, -300, 400, 400);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].col, 0);
    assert_eq!(tiles[0].row, 0);
    assert_eq!(tiles[0].dest_x, 300);
    assert_eq!(tiles[0].dest_y, 300);
}

#[test]
fn irregular_tiles_for_region_basic() {
    let mut tiles_map = std::collections::HashMap::new();
    tiles_map.insert(
        (0i64, 0i64),
        TileEntry {
            offset: (0.0, 0.0),
            dimensions: (256, 256),
            tiff_tile_index: None,
        },
    );
    tiles_map.insert(
        (1, 0),
        TileEntry {
            offset: (5.0, 0.0),
            dimensions: (256, 256),
            tiff_tile_index: None,
        },
    );
    tiles_map.insert(
        (0, 1),
        TileEntry {
            offset: (0.0, 3.0),
            dimensions: (256, 256),
            tiff_tile_index: None,
        },
    );

    let layout = TileLayout::Irregular {
        tile_advance: (256.0, 256.0),
        extra_tiles: (1, 0, 1, 0),
        tiles: tiles_map,
    };

    let result = layout.tiles_for_region(0, 0, 512, 512);
    assert_eq!(result.len(), 3);
}

#[test]
fn irregular_tiles_negative_offset() {
    let mut tiles_map = std::collections::HashMap::new();
    tiles_map.insert(
        (0i64, 0i64),
        TileEntry {
            offset: (-10.0, -5.0),
            dimensions: (256, 256),
            tiff_tile_index: None,
        },
    );

    let layout = TileLayout::Irregular {
        tile_advance: (256.0, 256.0),
        extra_tiles: (0, 1, 0, 1),
        tiles: tiles_map,
    };

    // Tile actual position is (-10, -5) to (246, 251)
    // Region (0, 0, 100, 100) should hit it
    let result = layout.tiles_for_region(0, 0, 100, 100);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dest_x, -10);
    assert_eq!(result[0].dest_y, -5);
}

#[test]
fn irregular_tiles_no_match() {
    let mut tiles_map = std::collections::HashMap::new();
    tiles_map.insert(
        (0i64, 0i64),
        TileEntry {
            offset: (0.0, 0.0),
            dimensions: (256, 256),
            tiff_tile_index: None,
        },
    );

    let layout = TileLayout::Irregular {
        tile_advance: (256.0, 256.0),
        extra_tiles: (0, 0, 0, 0),
        tiles: tiles_map,
    };

    let result = layout.tiles_for_region(10000, 10000, 100, 100);
    assert_eq!(result.len(), 0);
}

// --- Compression ---

#[test]
fn compression_equality() {
    assert_eq!(Compression::Jpeg, Compression::Jpeg);
    assert_ne!(Compression::Jpeg, Compression::Jp2kRgb);
    assert_eq!(Compression::Other(99), Compression::Other(99));
    assert_ne!(Compression::Other(99), Compression::Other(100));
}

// --- SampleType ---

#[test]
fn sample_type_byte_size() {
    assert_eq!(SampleType::Uint8.byte_size(), 1);
    assert_eq!(SampleType::Uint16.byte_size(), 2);
    assert_eq!(SampleType::Float32.byte_size(), 4);
}

// --- CpuTile display conversion ---

#[test]
fn to_rgba_from_rgb_u8() {
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![255, 0, 0, 0, 255, 0]),
    };
    let img = buf.to_rgba().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
    assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0, 255]);
}

#[test]
fn to_rgba_from_rgb_u8_planar() {
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Planar,
        data: CpuTileData::u8(vec![255, 0, 0, 255, 0, 0]),
    };
    let img = buf.to_rgba().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
    assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0, 255]);
}

#[test]
fn to_rgba_from_grayscale_u8() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![128]),
    };
    let img = buf.to_rgba().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [128, 128, 128, 255]);
}

#[test]
fn to_rgba_from_palette() {
    let lut = vec![[255, 0, 0], [0, 255, 0]];
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Palette(Arc::new(lut)),
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![0, 1]),
    };
    let img = buf.to_rgba().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
    assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0, 255]);
}

#[test]
fn to_rgba_rejects_non_u8() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u16(vec![1000]),
    };
    assert!(buf.to_rgba().is_err());
}

#[test]
fn to_rgba_windowed_u16() {
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u16(vec![0, 1000]),
    };
    let window = DisplayWindow {
        min: 0.0,
        max: 1000.0,
    };
    let img = buf.to_rgba_windowed(&window).unwrap();
    assert_eq!(img.get_pixel(0, 0).0[0], 0); // 0 maps to 0
    assert_eq!(img.get_pixel(1, 0).0[0], 255); // 1000 maps to 255
}

#[test]
fn to_rgba_windowed_u16_planar_rgb() {
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Planar,
        data: CpuTileData::u16(vec![0, 1000, 0, 1000, 0, 0]),
    };
    let window = DisplayWindow {
        min: 0.0,
        max: 1000.0,
    };
    let img = buf.to_rgba_windowed(&window).unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 255]);
    assert_eq!(img.get_pixel(1, 0).0, [255, 255, 0, 255]);
}

#[test]
fn to_rgba_windowed_zero_range_errors() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u16(vec![100]),
    };
    let window = DisplayWindow {
        min: 50.0,
        max: 50.0,
    };
    assert!(buf.to_rgba_windowed(&window).is_err());
}

#[test]
fn to_rgb_from_rgb_u8() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![100, 150, 200]),
    };
    let img = buf.to_rgb().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [100, 150, 200]);
}

#[test]
fn into_rgb_reuses_interleaved_rgb_storage() {
    let raw = vec![100, 150, 200];
    let ptr = raw.as_ptr();
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(raw),
    };
    let img = buf.into_rgb().unwrap();
    assert_eq!(img.as_raw().as_ptr(), ptr);
    assert_eq!(img.get_pixel(0, 0).0, [100, 150, 200]);
}

#[test]
fn into_rgba_reuses_interleaved_rgba_storage() {
    let raw = vec![100, 150, 200, 255];
    let ptr = raw.as_ptr();
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 4,
        color_space: ColorSpace::Rgba,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(raw),
    };
    let img = buf.into_rgba().unwrap();
    assert_eq!(img.as_raw().as_ptr(), ptr);
    assert_eq!(img.get_pixel(0, 0).0, [100, 150, 200, 255]);
}

// --- CpuTile::new() ---

#[test]
fn sample_buffer_new_valid() {
    let buf = CpuTile::new(
        2,
        1,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(vec![0; 6]),
    );
    assert!(buf.is_ok());
    assert_eq!(buf.unwrap().width, 2);
}

#[test]
fn sample_buffer_new_invalid_length() {
    let buf = CpuTile::new(
        2,
        1,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(vec![0; 5]),
    );
    assert!(buf.is_err());
}

#[test]
#[should_panic(expected = "CpuTile::new_for_test currently stores packed interleaved data")]
fn cpu_tile_new_for_test_rejects_padded_stride() {
    CpuTile::new_for_test(
        Arc::<[u8]>::from(vec![0u8; 32]),
        2,
        2,
        16,
        PixelFormat::Rgba8,
    );
}

#[test]
fn sample_buffer_new_overflow_dimensions() {
    let buf = CpuTile::new(
        u32::MAX,
        u32::MAX,
        3,
        ColorSpace::Rgb,
        CpuTileLayout::Interleaved,
        CpuTileData::u8(vec![]),
    );
    assert!(buf.is_err());
}

// --- Direct to_rgb() paths ---

#[test]
fn to_rgb_direct_path_rgb8() {
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![255, 0, 0, 0, 255, 0]),
    };
    let img = buf.to_rgb().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0]);
}

#[test]
fn to_rgb_direct_path_grayscale() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![128]),
    };
    let img = buf.to_rgb().unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [128, 128, 128]);
}

#[test]
fn to_rgb_rejects_non_u8() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u16(vec![1000]),
    };
    assert!(buf.to_rgb().is_err());
}

#[test]
fn to_rgb_windowed_u16_direct() {
    let buf = CpuTile {
        width: 2,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Grayscale,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u16(vec![0, 1000]),
    };
    let window = DisplayWindow {
        min: 0.0,
        max: 1000.0,
    };
    let img = buf.to_rgb_windowed(&window).unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0]);
    assert_eq!(img.get_pixel(1, 0).0, [255, 255, 255]);
}

#[test]
fn to_rgb_windowed_f32_3ch_direct() {
    let buf = CpuTile {
        width: 1,
        height: 1,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::f32(vec![0.0, 0.5, 1.0]),
    };
    let window = DisplayWindow { min: 0.0, max: 1.0 };
    let img = buf.to_rgb_windowed(&window).unwrap();
    assert_eq!(img.get_pixel(0, 0).0, [0, 128, 255]);
}

// --- Arc Palette ---

#[test]
fn palette_clone_is_cheap() {
    let lut = Arc::new(vec![[255, 0, 0]; 256]);
    let cs = ColorSpace::Palette(lut.clone());
    let cs2 = cs.clone();
    drop(cs2);
    assert_eq!(Arc::strong_count(&lut), 2); // original + cs
}
