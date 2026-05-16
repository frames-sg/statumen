use super::*;
use crate::formats::tiff_family::container::TiffContainer;
use crate::formats::tiff_family::layout::DatasetLayout;
use crate::properties::Properties;
use flate2::write::ZlibEncoder;
use flate2::Compression as DeflateCompression;
use image::{DynamicImage, ImageFormat};
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
use std::collections::HashMap;
use std::io::Cursor;
use std::io::Write;
use tempfile::NamedTempFile;

#[allow(clippy::too_many_arguments)]
fn region_request(
    scene: usize,
    series: usize,
    level: u32,
    plane: PlaneSelection,
    x: i64,
    y: i64,
    w: u32,
    h: u32,
) -> RegionRequest {
    RegionRequest {
        scene: SceneId(scene),
        series: SeriesId(series),
        level: LevelIdx(level),
        plane: PlaneIdx(plane),
        origin_px: (x, y),
        size_px: (w, h),
    }
}

fn make_sample_buffer(size: usize) -> CpuTile {
    CpuTile {
        width: 64,
        height: 64,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(vec![0u8; size]),
    }
}

fn jpeg_sof(ids: [u8; 3], sampling: [(u8, u8); 3]) -> Vec<u8> {
    let mut jpeg = vec![
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01, 0x00, 0x01, 0x03,
    ];
    for idx in 0..3 {
        jpeg.push(ids[idx]);
        jpeg.push((sampling[idx].0 << 4) | sampling[idx].1);
        jpeg.push(0);
    }
    jpeg
}

#[test]
fn jpeg_rgb_component_ids_zero_one_two_follow_tiff_photometric() {
    let jpeg = jpeg_sof([0, 1, 2], [(1, 1), (1, 1), (1, 1)]);

    assert_eq!(
        jpeg_bitstream_color_hint(&jpeg, None),
        JpegBitstreamColorHint::RgbComponentIds012
    );
    assert_eq!(
        tiff_jpeg_color_transform(2, 3, jpeg_bitstream_color_hint(&jpeg, None)),
        SigninumColorTransform::ForceRgb
    );
    assert_eq!(
        tiff_jpeg_color_transform(6, 3, jpeg_bitstream_color_hint(&jpeg, None)),
        SigninumColorTransform::ForceYCbCr
    );
}

#[test]
fn jpeg_rgb_component_ids_ascii_force_rgb() {
    let jpeg = jpeg_sof([b'R', b'G', b'B'], [(1, 1), (1, 1), (1, 1)]);

    assert_eq!(
        jpeg_bitstream_color_hint(&jpeg, None),
        JpegBitstreamColorHint::Rgb
    );
    assert_eq!(
        tiff_jpeg_color_transform(6, 3, jpeg_bitstream_color_hint(&jpeg, None)),
        SigninumColorTransform::ForceRgb
    );
}

#[test]
fn jpeg_rgb_tiff_with_actual_chroma_subsampling_uses_ycbcr_hint() {
    let jpeg = jpeg_sof([1, 2, 3], [(2, 2), (1, 1), (1, 1)]);

    assert_eq!(
        jpeg_bitstream_color_hint(&jpeg, None),
        JpegBitstreamColorHint::YCbCr
    );
    assert_eq!(
        tiff_jpeg_color_transform(2, 3, jpeg_bitstream_color_hint(&jpeg, None)),
        SigninumColorTransform::ForceYCbCr
    );
}

#[test]
fn jpeg_unknown_bitstream_falls_back_to_tiff_photometric() {
    assert_eq!(
        tiff_jpeg_color_transform(2, 3, JpegBitstreamColorHint::Unknown),
        SigninumColorTransform::ForceRgb
    );
    assert_eq!(
        tiff_jpeg_color_transform(6, 3, JpegBitstreamColorHint::Unknown),
        SigninumColorTransform::ForceYCbCr
    );
}

// ── FullDecodeCache tests ─────────────────────────────────────

#[test]
fn full_decode_cache_put_and_get() {
    let mut cache = FullDecodeCache::new(1024);
    let buf = Arc::new(make_sample_buffer(100));
    cache.put(IfdId(1000), buf.clone());

    let result = cache.get(&IfdId(1000));
    assert!(result.is_some());
    assert_eq!(result.unwrap().width, 64);
}

#[test]
fn full_decode_cache_eviction() {
    let mut cache = FullDecodeCache::new(250);
    cache.put(IfdId(100), Arc::new(make_sample_buffer(100)));
    cache.put(IfdId(200), Arc::new(make_sample_buffer(100)));
    // 200 bytes used — both fit
    assert!(cache.get(&IfdId(100)).is_some());
    assert!(cache.get(&IfdId(200)).is_some());

    // Third entry pushes over 250 — LRU (IfdId(100)) should be evicted
    // Note: after the two gets above, access order is 100 then 200,
    // so IfdId(100) is older. But LruCache.get() promotes, so after
    // get(100) then get(200), 100 was accessed first, then 200.
    // The LRU is IfdId(100).
    cache.put(IfdId(300), Arc::new(make_sample_buffer(100)));
    assert!(cache.get(&IfdId(100)).is_none()); // evicted
    assert!(cache.get(&IfdId(200)).is_some());
    assert!(cache.get(&IfdId(300)).is_some());
}

#[test]
fn full_decode_cache_oversize_rejected() {
    let mut cache = FullDecodeCache::new(50);
    let buf = Arc::new(make_sample_buffer(100));
    cache.put(IfdId(1000), buf);

    assert!(cache.get(&IfdId(1000)).is_none());
    assert_eq!(cache.current_bytes, 0);
}

#[test]
fn full_decode_cache_miss() {
    let mut cache = FullDecodeCache::new(1024);
    assert!(cache.get(&IfdId(9999)).is_none());
}

#[test]
fn full_decode_cache_replacement_updates_bytes() {
    let mut cache = FullDecodeCache::new(500);
    cache.put(IfdId(100), Arc::new(make_sample_buffer(100)));
    assert_eq!(cache.current_bytes, 100);

    // Replace with larger buffer
    cache.put(IfdId(100), Arc::new(make_sample_buffer(200)));
    assert_eq!(cache.current_bytes, 200);

    // Still retrievable
    assert!(cache.get(&IfdId(100)).is_some());
}

#[test]
fn clamp_ndpi_strip_crop_limits_edge_requests_to_strip_bounds() {
    assert_eq!(
        TiffPixelReader::clamp_ndpi_strip_crop(112, 0, 136, 240, 104, 240),
        None
    );
    assert_eq!(
        TiffPixelReader::clamp_ndpi_strip_crop(0, 0, 136, 240, 104, 240),
        Some((104, 240))
    );
    assert_eq!(
        TiffPixelReader::clamp_ndpi_strip_crop(112, 16, 136, 240, 248, 240),
        Some((136, 224))
    );
}

fn build_test_ndpi_reader_for_strip_cache(
    width: u32,
    height: u32,
    tiles_across: u32,
) -> (TiffPixelReader, IfdId) {
    let tiles_down = height.div_ceil(16);
    let jpeg = encode_restart_rgb_jpeg(
        &image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0])),
        75,
        8,
    );
    let bitstream_start = find_test_jpeg_bitstream_start(&jpeg).unwrap();
    let jpeg_header = jpeg[..bitstream_start].to_vec();
    let file =
        build_ndpi_full_jpeg_tiff(width, height, &jpeg, (tiles_across * tiles_down) as usize);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(12),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![
                        Level {
                            dimensions: (width as u64, height as u64),
                            downsample: 1.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: width as u64,
                                height: height as u64,
                                virtual_tile_width: 128,
                                virtual_tile_height: 16,
                            },
                        },
                        Level {
                            dimensions: (width as u64, height as u64),
                            downsample: 2.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: width as u64,
                                height: height as u64,
                                virtual_tile_width: 128,
                                virtual_tile_height: 16,
                            },
                        },
                    ],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 1,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header,
                mcu_starts_tag: 65426,
                tiles_across,
                tiles_down,
                restart_interval: 8,
                strip_offset: 8,
                strip_byte_count: jpeg.len() as u64,
            },
        )]),
        associated_sources: HashMap::new(),
    };
    (TiffPixelReader::new(container, layout), ifd_id)
}

struct TestNdpiJpegLayout {
    ifd_id: IfdId,
    dimensions: (u32, u32),
    virtual_tile: (u32, u32),
    tile_grid: (u32, u32),
    jpeg_header: Vec<u8>,
    strip_byte_count: u64,
}

fn build_test_ndpi_layout_from_header(spec: TestNdpiJpegLayout) -> DatasetLayout {
    let (width, height) = spec.dimensions;
    let (virtual_tile_width, virtual_tile_height) = spec.virtual_tile;
    let (tiles_across, tiles_down) = spec.tile_grid;
    DatasetLayout {
        dataset: Dataset {
            id: DatasetId(12),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![Level {
                        dimensions: (width as u64, height as u64),
                        downsample: 1.0,
                        tile_layout: TileLayout::WholeLevel {
                            width: width as u64,
                            height: height as u64,
                            virtual_tile_width,
                            virtual_tile_height,
                        },
                    }],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::NdpiJpeg {
                ifd_id: spec.ifd_id,
                jpeg_header: spec.jpeg_header,
                mcu_starts_tag: 65426,
                tiles_across,
                tiles_down,
                restart_interval: 8,
                strip_offset: 8,
                strip_byte_count: spec.strip_byte_count,
            },
        )]),
        associated_sources: HashMap::new(),
    }
}

fn make_ndpi_strip(width: u32, height: u32, rgb: [u8; 3]) -> Arc<CpuTile> {
    let mut data = vec![0u8; width as usize * height as usize * 3];
    for pixel in data.chunks_exact_mut(3) {
        pixel.copy_from_slice(&rgb);
    }
    Arc::new(CpuTile {
        width,
        height,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(data),
    })
}

#[test]
fn ndpi_display_tile_only_populates_requested_strip_keys() {
    let (reader, ifd_id) = build_test_ndpi_reader_for_strip_cache(680, 72, 5);

    let tile = reader
        .read_display_tile(&TileViewRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
            tile_width: 250,
            tile_height: 32,
        })
        .unwrap();

    assert_eq!(tile.width, 250);
    assert_eq!(tile.height, 32);

    let mut cache = reader.ndpi_strip_cache.lock().unwrap();
    assert!(cache
        .get(&NdpiStripKey {
            ifd_id,
            col: 0,
            native_row: 0
        })
        .is_some());
    assert!(cache
        .get(&NdpiStripKey {
            ifd_id,
            col: 1,
            native_row: 0
        })
        .is_some());
    assert!(cache
        .get(&NdpiStripKey {
            ifd_id,
            col: 0,
            native_row: 1
        })
        .is_some());
    assert!(cache
        .get(&NdpiStripKey {
            ifd_id,
            col: 1,
            native_row: 1
        })
        .is_some());
    assert!(cache
        .get(&NdpiStripKey {
            ifd_id,
            col: 2,
            native_row: 1
        })
        .is_none());
}

#[test]
fn ndpi_display_tile_composites_from_strip_cache_across_rows_and_columns() {
    let (reader, ifd_id) = build_test_ndpi_reader_for_strip_cache(256, 48, 2);
    {
        let mut cache = reader.ndpi_strip_cache.lock().unwrap();
        cache.put(
            NdpiStripKey {
                ifd_id,
                col: 0,
                native_row: 0,
            },
            make_ndpi_strip(128, 16, [10, 0, 0]),
        );
        cache.put(
            NdpiStripKey {
                ifd_id,
                col: 1,
                native_row: 0,
            },
            make_ndpi_strip(128, 16, [20, 0, 0]),
        );
        cache.put(
            NdpiStripKey {
                ifd_id,
                col: 0,
                native_row: 1,
            },
            make_ndpi_strip(128, 16, [30, 0, 0]),
        );
        cache.put(
            NdpiStripKey {
                ifd_id,
                col: 1,
                native_row: 1,
            },
            make_ndpi_strip(128, 16, [40, 0, 0]),
        );
    }

    let tile = reader
        .read_display_tile(&TileViewRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            row: 0,
            col: 0,
            tile_width: 200,
            tile_height: 32,
        })
        .unwrap();

    let CpuTileData::U8(rgb) = tile.data else {
        panic!("expected RGB data");
    };
    assert_eq!(&rgb[0..3], &[10, 0, 0]);
    let right = 128 * 3;
    assert_eq!(&rgb[right..right + 3], &[20, 0, 0]);
    let lower = (16 * tile.width as usize) * 3;
    assert_eq!(&rgb[lower..lower + 3], &[30, 0, 0]);
    let lower_right = ((16 * tile.width as usize) + 128) * 3;
    assert_eq!(&rgb[lower_right..lower_right + 3], &[40, 0, 0]);
}

#[test]
fn ndpi_display_tile_composites_across_multiple_strip_rows_and_columns() {
    let (reader, ifd_id) = build_test_ndpi_reader_for_strip_cache(320, 600, 3);
    {
        let mut cache = reader.ndpi_strip_cache.lock().unwrap();
        for native_row in 16..=31 {
            for col in 0..=1 {
                cache.put(
                    NdpiStripKey {
                        ifd_id,
                        col,
                        native_row,
                    },
                    make_ndpi_strip(128, 16, [(col * 50) as u8, native_row as u8, 7]),
                );
            }
        }
    }

    let tile = reader
        .read_display_tile(&TileViewRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 1,
            tile_width: 256,
            tile_height: 256,
        })
        .unwrap();

    assert_eq!(tile.width, 256);
    assert_eq!(tile.height, 256);
    let rgb = tile.data.as_u8().unwrap();
    let pixel = |x: usize, y: usize| -> [u8; 3] {
        let idx = (y * tile.width as usize + x) * 3;
        [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
    };

    assert_eq!(pixel(50, 4), [0, 16, 7]);
    assert_eq!(pixel(50, 20), [0, 17, 7]);
    assert_eq!(pixel(200, 20), [50, 17, 7]);
}

#[test]
fn ndpi_restart_tile_decodes_target_strip_via_public_read_path() {
    let (file, jpeg_header, strip_byte_count) = build_ndpi_scan_data_tiff_from_blobs(
        128,
        16,
        &[[240, 20, 20], [20, 220, 20], [20, 20, 230], [220, 220, 30]],
        false,
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = build_test_ndpi_layout_from_header(TestNdpiJpegLayout {
        ifd_id,
        dimensions: (128, 16),
        virtual_tile: (64, 8),
        tile_grid: (2, 2),
        jpeg_header,
        strip_byte_count,
    });
    let reader = TiffPixelReader::new(container, layout);

    let tile = reader
        .read_tile_cpu(&TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 1,
            row: 1,
        })
        .unwrap();

    assert_eq!(tile.width, 64);
    assert_eq!(tile.height, 8);
    let CpuTileData::U8(rgb) = tile.data else {
        panic!("expected RGB data");
    };
    let pixel = [rgb[0], rgb[1], rgb[2]];
    assert!(
        pixel[0] > 170,
        "expected red channel dominance, got {pixel:?}"
    );
    assert!(
        pixel[1] > 170,
        "expected green channel dominance, got {pixel:?}"
    );
    assert!(
        pixel[2] < 120,
        "expected blue channel to stay lower, got {pixel:?}"
    );

    let mut cache = reader.ndpi_strip_cache.lock().unwrap();
    assert!(cache
        .get(&NdpiStripKey {
            ifd_id,
            col: 1,
            native_row: 1,
        })
        .is_some());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal device decode"]
fn ndpi_restart_tile_decodes_to_metal_device_tile() {
    let Ok(jpeg_session) = signinum_jpeg_metal::MetalBackendSession::system_default() else {
        return;
    };
    let Ok(j2k_session) = signinum_j2k_metal::MetalBackendSession::system_default() else {
        return;
    };
    std::env::set_var(JPEG_DEVICE_DECODE_ENV, "1");
    let (file, jpeg_header, strip_byte_count) = build_ndpi_scan_data_tiff_from_blobs(
        128,
        16,
        &[[240, 20, 20], [20, 220, 20], [20, 20, 230], [220, 220, 30]],
        false,
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = build_test_ndpi_layout_from_header(TestNdpiJpegLayout {
        ifd_id,
        dimensions: (128, 16),
        virtual_tile: (64, 8),
        tile_grid: (2, 2),
        jpeg_header,
        strip_byte_count,
    });
    let reader = TiffPixelReader::new(container, layout);

    let tiles = reader
        .read_tiles(
            &[TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 1,
                row: 1,
            }],
            TileOutputPreference::prefer_device_auto_with_metal(
                crate::output::metal::MetalBackendSessions::new(jpeg_session, j2k_session),
            ),
        )
        .unwrap();

    assert_eq!(tiles.len(), 1);
    let TilePixels::Device(DeviceTile::Metal(tile)) = &tiles[0] else {
        panic!("expected NDPI tile to decode to Metal");
    };
    assert_eq!((tile.width, tile.height), (64, 8));
    assert_eq!(tile.format, SigninumPixelFormat::Rgb8);
}

#[test]
fn ndpi_restart_tile_does_not_silently_fallback_to_full_decode_on_bad_mcu_table() {
    let jpeg = {
        let mut encoded = Vec::new();
        let image = image::RgbImage::new(8, 8);
        JpegEncoder::new(&mut encoded, 75)
            .encode(
                image.as_raw().as_slice(),
                image.width() as u16,
                image.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        encoded
    };
    let file = build_stripped_jpeg_tiff(8, 8, &jpeg);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(3),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![Level {
                        dimensions: (8, 8),
                        downsample: 1.0,
                        tile_layout: TileLayout::WholeLevel {
                            width: 8,
                            height: 8,
                            virtual_tile_width: 8,
                            virtual_tile_height: 8,
                        },
                    }],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::NdpiJpeg {
                ifd_id,
                jpeg_header: Vec::new(),
                mcu_starts_tag: 65426,
                tiles_across: 1,
                tiles_down: 1,
                restart_interval: 1,
                strip_offset: 8,
                strip_byte_count: jpeg.len() as u64,
            },
        )]),
        associated_sources: HashMap::new(),
    };
    let reader = TiffPixelReader::new(container, layout);

    let err = reader
        .read_tile_cpu(&TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("65426") || err.to_string().contains("MCU"),
        "unexpected error: {err}"
    );
}

#[test]
fn ndpi_restart_tile_decodes_zero_sof_segment_from_mcu_table() {
    let (file, jpeg_header, strip_byte_count) = build_ndpi_scan_data_tiff_from_blobs(
        128,
        16,
        &[[240, 20, 20], [20, 220, 20], [20, 20, 230], [220, 220, 30]],
        true,
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = build_test_ndpi_layout_from_header(TestNdpiJpegLayout {
        ifd_id,
        dimensions: (128, 16),
        virtual_tile: (64, 8),
        tile_grid: (2, 2),
        jpeg_header,
        strip_byte_count,
    });
    let reader = TiffPixelReader::new(container, layout);

    let tile = reader
        .read_tile_cpu(&TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();

    assert_eq!(tile.width, 64);
    assert_eq!(tile.height, 8);
    let rgb = tile.data.as_u8().expect("expected RGB tile");
    assert!(
        rgb[0] > 180 && rgb[1] < 80 && rgb[2] < 80,
        "unexpected first pixel for zero-SOF NDPI tile: {:?}",
        &rgb[0..3]
    );
}

#[test]
fn synthetic_ndpi_levels_downsample_smallest_physical_level() {
    let mut image = image::RgbImage::new(4, 4);
    let source_pixels = [
        [10, 20, 30],
        [30, 40, 50],
        [50, 60, 70],
        [70, 80, 90],
        [90, 100, 110],
        [110, 120, 130],
        [130, 140, 150],
        [150, 160, 170],
        [20, 30, 40],
        [40, 50, 60],
        [60, 70, 80],
        [80, 90, 100],
        [100, 110, 120],
        [120, 130, 140],
        [140, 150, 160],
        [160, 170, 180],
    ];
    for (pixel, rgb) in image.pixels_mut().zip(source_pixels) {
        *pixel = image::Rgb(rgb);
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new(&mut jpeg, 100)
        .encode(
            image.as_raw().as_slice(),
            image.width() as u16,
            image.height() as u16,
            JpegColorType::Rgb,
        )
        .unwrap();
    let file = build_stripped_jpeg_tiff(4, 4, &jpeg);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(99),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![
                        Level {
                            dimensions: (4, 4),
                            downsample: 1.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: 4,
                                height: 4,
                                virtual_tile_width: 4,
                                virtual_tile_height: 4,
                            },
                        },
                        Level {
                            dimensions: (2, 2),
                            downsample: 2.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: 2,
                                height: 2,
                                virtual_tile_width: 2,
                                virtual_tile_height: 2,
                            },
                        },
                        Level {
                            dimensions: (1, 1),
                            downsample: 4.0,
                            tile_layout: TileLayout::WholeLevel {
                                width: 1,
                                height: 1,
                                virtual_tile_width: 1,
                                virtual_tile_height: 1,
                            },
                        },
                    ],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([
            (
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 0,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::NdpiFullDecode {
                    ifd_id,
                    jpeg_header: Vec::new(),
                    strip_offset: 8,
                    strip_byte_count: jpeg.len() as u64,
                },
            ),
            (
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 1,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::SyntheticDownsample {
                    base_level: 0,
                    factor: 2,
                },
            ),
            (
                TileSourceKey {
                    scene: 0,
                    series: 0,
                    level: 2,
                    z: 0,
                    c: 0,
                    t: 0,
                },
                TileSource::SyntheticDownsample {
                    base_level: 0,
                    factor: 4,
                },
            ),
        ]),
        associated_sources: HashMap::new(),
    };
    let reader = TiffPixelReader::new(container, layout);

    let level1 = reader
        .read_tile_cpu(&TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();
    assert_eq!(level1.width, 2);
    assert_eq!(level1.height, 2);
    let level1_rgb = level1.data.as_u8().unwrap();
    assert_rgb_close(&level1_rgb[0..3], &[60, 70, 80], 1);
    assert_rgb_close(&level1_rgb[3..6], &[100, 110, 120], 1);
    assert_rgb_close(&level1_rgb[6..9], &[70, 80, 90], 1);
    assert_rgb_close(&level1_rgb[9..12], &[110, 120, 130], 1);

    let level2 = reader
        .read_tile_cpu(&TileRequest {
            scene: 0,
            series: 0,
            level: 2,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();
    assert_eq!(level2.width, 1);
    assert_eq!(level2.height, 1);
    let level2_rgb = level2.data.as_u8().unwrap();
    assert_rgb_close(&level2_rgb[0..3], &[85, 95, 105], 1);
}

fn assert_rgb_close(actual: &[u8], expected: &[u8; 3], tolerance: u8) {
    assert_eq!(actual.len(), 3);
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            actual.abs_diff(*expected) <= tolerance,
            "actual RGB channel {actual} differs from expected {expected} by more than {tolerance}"
        );
    }
}

fn synthetic_ndpi_base_pixel(x: u32, y: u32) -> [u8; 3] {
    [
        (10 + x.saturating_mul(7) + y.saturating_mul(3)).min(255) as u8,
        (20 + x.saturating_mul(5) + y.saturating_mul(11)).min(255) as u8,
        (30 + x.saturating_mul(13) + y.saturating_mul(2)).min(255) as u8,
    ]
}

fn synthetic_ndpi_base_image(width: u32, height: u32) -> image::RgbImage {
    image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb(synthetic_ndpi_base_pixel(x, y))
    })
}

fn crop_rgb_with_zero_fill(source: &CpuTile, x: i64, y: i64, w: u32, h: u32) -> CpuTile {
    assert_eq!(source.channels, 3);
    assert_eq!(source.color_space, ColorSpace::Rgb);
    assert_eq!(source.layout, CpuTileLayout::Interleaved);
    let src = source.data.as_u8().unwrap();
    let mut out = vec![0u8; w as usize * h as usize * 3];
    let clipped_x0 = x.max(0).min(i64::from(source.width));
    let clipped_y0 = y.max(0).min(i64::from(source.height));
    let clipped_x1 = x
        .saturating_add(i64::from(w))
        .max(0)
        .min(i64::from(source.width));
    let clipped_y1 = y
        .saturating_add(i64::from(h))
        .max(0)
        .min(i64::from(source.height));
    if clipped_x1 <= clipped_x0 || clipped_y1 <= clipped_y0 {
        return CpuTile {
            width: w,
            height: h,
            channels: 3,
            color_space: ColorSpace::Rgb,
            layout: CpuTileLayout::Interleaved,
            data: CpuTileData::u8(out),
        };
    }

    let copy_w = (clipped_x1 - clipped_x0) as usize;
    let copy_h = (clipped_y1 - clipped_y0) as usize;
    let dst_x = (clipped_x0 - x) as usize;
    let dst_y = (clipped_y0 - y) as usize;
    let src_stride = source.width as usize * 3;
    let dst_stride = w as usize * 3;
    for row in 0..copy_h {
        let src_off = (clipped_y0 as usize + row) * src_stride + clipped_x0 as usize * 3;
        let dst_off = (dst_y + row) * dst_stride + dst_x * 3;
        out[dst_off..dst_off + copy_w * 3].copy_from_slice(&src[src_off..src_off + copy_w * 3]);
    }

    CpuTile {
        width: w,
        height: h,
        channels: 3,
        color_space: ColorSpace::Rgb,
        layout: CpuTileLayout::Interleaved,
        data: CpuTileData::u8(out),
    }
}

fn expected_synthetic_ndpi_region(
    reader: &TiffPixelReader,
    factor: u32,
    x: i64,
    y: i64,
    w: u32,
    h: u32,
) -> CpuTile {
    let tile_req = TileRequest {
        scene: 0,
        series: 0,
        level: 1,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };
    let full = if let Some(image) = reader
        .try_decode_synthetic_level_with_signinum(&tile_req, 0, factor)
        .unwrap()
    {
        image
    } else {
        let mut base = reader
            .read_tile_cpu(&TileRequest {
                scene: 0,
                series: 0,
                level: 0,
                plane: PlaneSelection::default(),
                col: 0,
                row: 0,
            })
            .unwrap();
        if base.layout != CpuTileLayout::Interleaved
            || base.channels != 3
            || base.color_space != ColorSpace::Rgb
            || base.data.as_u8().is_none()
        {
            base = rgba_image_to_sample_buffer(base.to_rgba().unwrap());
        }
        let target = &reader.layout.dataset.scenes[0].series[0].levels[1];
        fit_synthetic_rgb_tile_to_dimensions(
            downsample_rgb_pow2_box(&base, factor).unwrap(),
            target.dimensions.0 as u32,
            target.dimensions.1 as u32,
        )
        .unwrap()
    };
    crop_rgb_with_zero_fill(&full, x, y, w, h)
}

fn assert_tile_eq(actual: &CpuTile, expected: &CpuTile) {
    assert_eq!(
        (actual.width, actual.height),
        (expected.width, expected.height)
    );
    assert_eq!(actual.channels, expected.channels);
    assert_eq!(actual.color_space, expected.color_space);
    assert_eq!(actual.layout, expected.layout);
    assert_eq!(actual.data.as_u8().unwrap(), expected.data.as_u8().unwrap());
}

fn read_synthetic_ndpi_region(reader: &TiffPixelReader, x: i64, y: i64, w: u32, h: u32) -> CpuTile {
    let req = region_request(0, 0, 1, PlaneSelection::default(), x, y, w, h);
    let mut ctx = crate::core::registry::SlideReadContext::new(
        None,
        TileOutputPreference::cpu(),
        256 * 1024 * 1024,
    );
    reader
        .read_region_fastpath(&mut ctx, &req)
        .expect("synthetic level should have a region fast path")
        .expect("synthetic region fast path should produce pixels")
}

fn build_synthetic_ndpi_reader(
    width: u32,
    height: u32,
    synthetic: &[(u64, u64, u32)],
) -> TiffPixelReader {
    let image = synthetic_ndpi_base_image(width, height);
    let mut jpeg = Vec::new();
    JpegEncoder::new(&mut jpeg, 95)
        .encode(
            image.as_raw().as_slice(),
            image.width() as u16,
            image.height() as u16,
            JpegColorType::Rgb,
        )
        .unwrap();
    let file = build_stripped_jpeg_tiff(width, height, &jpeg);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();

    let mut levels = vec![Level {
        dimensions: (u64::from(width), u64::from(height)),
        downsample: 1.0,
        tile_layout: TileLayout::WholeLevel {
            width: u64::from(width),
            height: u64::from(height),
            virtual_tile_width: width,
            virtual_tile_height: height,
        },
    }];
    let mut tile_sources = HashMap::from([(
        TileSourceKey {
            scene: 0,
            series: 0,
            level: 0,
            z: 0,
            c: 0,
            t: 0,
        },
        TileSource::NdpiFullDecode {
            ifd_id,
            jpeg_header: Vec::new(),
            strip_offset: 8,
            strip_byte_count: jpeg.len() as u64,
        },
    )]);

    for (idx, (level_width, level_height, factor)) in synthetic.iter().copied().enumerate() {
        let level_idx = (idx + 1) as u32;
        levels.push(Level {
            dimensions: (level_width, level_height),
            downsample: f64::from(factor),
            tile_layout: TileLayout::WholeLevel {
                width: level_width,
                height: level_height,
                virtual_tile_width: level_width as u32,
                virtual_tile_height: level_height as u32,
            },
        });
        tile_sources.insert(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: level_idx,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::SyntheticDownsample {
                base_level: 0,
                factor,
            },
        );
    }

    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(100),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels,
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources,
        associated_sources: HashMap::new(),
    };
    TiffPixelReader::new(container, layout)
}

#[test]
fn synthetic_ndpi_level_source_kind_marks_generated_downsamples() {
    let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);

    assert_eq!(
        reader.level_source_kind(0, 0, 0).unwrap(),
        LevelSourceKind::Physical
    );
    assert_eq!(
        reader.level_source_kind(0, 0, 1).unwrap(),
        LevelSourceKind::SyntheticDownsample
    );
}

#[test]
fn synthetic_ndpi_subregion_fastpath_matches_center_roi_without_materializing_level() {
    let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
    let tile = read_synthetic_ndpi_region(&reader, 1, 1, 2, 2);
    let expected = expected_synthetic_ndpi_region(&reader, 2, 1, 1, 2, 2);

    assert_tile_eq(&tile, &expected);
    assert_eq!(
        reader.synthetic_level_cache.lock().unwrap().current_bytes,
        0,
        "ROI reads must not materialize the whole synthetic level"
    );
    assert_eq!(
        reader.synthetic_region_cache.lock().unwrap().current_bytes,
        0,
        "ROI reads must not populate full synthetic region cache entries"
    );
}

#[test]
fn synthetic_ndpi_display_tile_uses_roi_fastpath_without_materializing_level() {
    let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
    let tile = reader
        .read_display_tile(&TileViewRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 1,
            row: 1,
            tile_width: 2,
            tile_height: 2,
        })
        .unwrap();
    let expected = expected_synthetic_ndpi_region(&reader, 2, 2, 2, 2, 2);

    assert_tile_eq(&tile, &expected);
    assert_eq!(
        reader.synthetic_level_cache.lock().unwrap().current_bytes,
        0,
        "display-tile reads must not materialize the whole synthetic level"
    );
}

#[test]
fn synthetic_ndpi_subregion_fastpath_zero_fills_negative_origin() {
    let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
    let tile = read_synthetic_ndpi_region(&reader, -1, -1, 3, 3);
    let expected = expected_synthetic_ndpi_region(&reader, 2, -1, -1, 3, 3);

    assert_tile_eq(&tile, &expected);
}

#[test]
fn synthetic_ndpi_subregion_fastpath_keeps_odd_ceil_edge_pixels() {
    let reader = build_synthetic_ndpi_reader(5, 5, &[(3, 3, 2)]);
    let tile = read_synthetic_ndpi_region(&reader, 2, 2, 1, 1);
    let expected = expected_synthetic_ndpi_region(&reader, 2, 2, 2, 1, 1);

    assert_tile_eq(&tile, &expected);
}

#[test]
fn synthetic_ndpi_subregion_fastpath_respects_cropped_synthetic_dimensions() {
    let reader = build_synthetic_ndpi_reader(5, 5, &[(2, 2, 2)]);
    let tile = read_synthetic_ndpi_region(&reader, 1, 1, 1, 1);
    let expected = expected_synthetic_ndpi_region(&reader, 2, 1, 1, 1, 1);

    assert_tile_eq(&tile, &expected);
}

#[test]
fn synthetic_ndpi_subregion_fastpath_does_not_prime_deepest_synthetic_level() {
    let reader = build_synthetic_ndpi_reader(8, 8, &[(3, 3, 2), (2, 2, 4)]);
    let tile = read_synthetic_ndpi_region(&reader, 1, 1, 1, 1);
    let expected = expected_synthetic_ndpi_region(&reader, 2, 1, 1, 1, 1);

    assert_tile_eq(&tile, &expected);
    assert_eq!(
        reader.synthetic_level_cache.lock().unwrap().current_bytes,
        0,
        "ROI reads must not materialize the requested synthetic level"
    );
    assert_eq!(
        reader.synthetic_region_cache.lock().unwrap().current_bytes,
        0,
        "ROI reads must not prime unrelated full synthetic levels"
    );
}

#[test]
fn synthetic_ndpi_subregion_fastpath_matches_factor_four_repeated_box_edges() {
    let reader = build_synthetic_ndpi_reader(9, 7, &[(3, 2, 4)]);
    let tile = read_synthetic_ndpi_region(&reader, 1, 1, 2, 1);
    let expected = expected_synthetic_ndpi_region(&reader, 4, 1, 1, 2, 1);

    assert_tile_eq(&tile, &expected);
}

#[test]
fn synthetic_ndpi_tile_path_uses_signinum_downscale_when_dimensions_match() {
    let reader = build_synthetic_ndpi_reader(8, 8, &[(4, 4, 2)]);
    let direct_req = TileRequest {
        scene: 0,
        series: 0,
        level: 1,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };
    let direct = reader
        .try_decode_synthetic_level_with_signinum(&direct_req, 0, 2)
        .expect("signinum synthetic downscale should decode")
        .expect("matching synthetic dimensions should use signinum downscale");
    assert_eq!((direct.width, direct.height), (4, 4));

    let tile = reader
        .read_tile_cpu(&TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();

    assert_eq!((tile.width, tile.height), (4, 4));
}

#[test]
fn synthetic_ndpi_region_fastpath_falls_back_when_signinum_scaled_dims_do_not_match() {
    let reader = build_synthetic_ndpi_reader(5, 5, &[(2, 2, 2)]);
    let direct_req = TileRequest {
        scene: 0,
        series: 0,
        level: 1,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };
    assert!(
        reader
            .try_decode_synthetic_level_with_signinum(&direct_req, 0, 2)
            .expect("signinum synthetic downscale should decode")
            .is_none(),
        "odd source dimensions should reject signinum result with mismatched target dimensions"
    );

    let req = region_request(0, 0, 1, PlaneSelection::default(), 0, 0, 2, 2);
    let mut ctx = crate::core::registry::SlideReadContext::new(
        None,
        TileOutputPreference::cpu(),
        256 * 1024 * 1024,
    );
    let tile = reader
        .read_region_fastpath(&mut ctx, &req)
        .expect("synthetic fast path should handle whole-level region")
        .expect("odd-dimension signinum downscale mismatch should fall back");

    assert_eq!((tile.width, tile.height), (2, 2));
}

fn le_u16(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}

fn le_u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

fn short_in_u32(v: u16) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    bytes[..2].copy_from_slice(&le_u16(v));
    bytes
}

fn build_tiled_associated_tiff(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles: &[Vec<u8>],
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let mut tile_offsets = Vec::with_capacity(tiles.len());
    let mut tile_byte_counts = Vec::with_capacity(tiles.len());
    for tile in tiles {
        tile_offsets.push(buf.len() as u32);
        tile_byte_counts.push(tile.len() as u32);
        buf.extend_from_slice(tile);
    }

    let tile_offsets_array_offset = if tile_offsets.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_offsets {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let tile_byte_counts_array_offset = if tile_byte_counts.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_byte_counts {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (258u16, 3u16, 1u32, short_in_u32(8)),
        (259u16, 3u16, 1u32, short_in_u32(1)),
        (262u16, 3u16, 1u32, short_in_u32(1)),
        (277u16, 3u16, 1u32, short_in_u32(1)),
        (322u16, 4u16, 1u32, le_u32(tile_width)),
        (323u16, 4u16, 1u32, le_u32(tile_height)),
        (
            324u16,
            4u16,
            tile_offsets.len() as u32,
            tile_offsets_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_offsets[0])),
        ),
        (
            325u16,
            4u16,
            tile_byte_counts.len() as u32,
            tile_byte_counts_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

#[allow(clippy::too_many_arguments)]
fn build_tiled_encoded_tiff(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles: &[Vec<u8>],
    compression_tag: u16,
    samples_per_pixel: u16,
    photometric: u16,
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let mut tile_offsets = Vec::with_capacity(tiles.len());
    let mut tile_byte_counts = Vec::with_capacity(tiles.len());
    for tile in tiles {
        tile_offsets.push(buf.len() as u32);
        tile_byte_counts.push(tile.len() as u32);
        buf.extend_from_slice(tile);
    }

    let tile_offsets_array_offset = if tile_offsets.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_offsets {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let tile_byte_counts_array_offset = if tile_byte_counts.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_byte_counts {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (258u16, 3u16, 1u32, short_in_u32(8)),
        (259u16, 3u16, 1u32, short_in_u32(compression_tag)),
        (262u16, 3u16, 1u32, short_in_u32(photometric)),
        (277u16, 3u16, 1u32, short_in_u32(samples_per_pixel)),
        (322u16, 4u16, 1u32, le_u32(tile_width)),
        (323u16, 4u16, 1u32, le_u32(tile_height)),
        (
            324u16,
            4u16,
            tile_offsets.len() as u32,
            tile_offsets_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_offsets[0])),
        ),
        (
            325u16,
            4u16,
            tile_byte_counts.len() as u32,
            tile_byte_counts_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

fn build_stripped_jpeg_tiff(width: u32, height: u32, jpeg_data: &[u8]) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let strip_offset = buf.len() as u32;
    buf.extend_from_slice(jpeg_data);
    let strip_byte_count = jpeg_data.len() as u32;

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (259u16, 3u16, 1u32, short_in_u32(7)),
        (262u16, 3u16, 1u32, short_in_u32(6)),
        (273u16, 4u16, 1u32, le_u32(strip_offset)),
        (277u16, 3u16, 1u32, short_in_u32(3)),
        (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

fn build_stripped_uncompressed_tiff(
    width: u32,
    height: u32,
    pixels: &[u8],
    samples_per_pixel: u16,
    photometric: Option<u16>,
) -> NamedTempFile {
    build_stripped_uncompressed_tiff_with_predictor(
        width,
        height,
        pixels,
        samples_per_pixel,
        photometric,
        None,
    )
}

fn build_stripped_uncompressed_tiff_with_predictor(
    width: u32,
    height: u32,
    pixels: &[u8],
    samples_per_pixel: u16,
    photometric: Option<u16>,
    predictor: Option<u16>,
) -> NamedTempFile {
    build_stripped_tiff(
        width,
        height,
        pixels,
        samples_per_pixel,
        photometric,
        predictor,
        1,
    )
}

fn build_stripped_tiff(
    width: u32,
    height: u32,
    payload: &[u8],
    samples_per_pixel: u16,
    photometric: Option<u16>,
    predictor: Option<u16>,
    compression: u16,
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let strip_offset = buf.len() as u32;
    buf.extend_from_slice(payload);
    let strip_byte_count = payload.len() as u32;

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (258u16, 3u16, 1u32, short_in_u32(8)),
        (259u16, 3u16, 1u32, short_in_u32(compression)),
        (273u16, 4u16, 1u32, le_u32(strip_offset)),
        (277u16, 3u16, 1u32, short_in_u32(samples_per_pixel)),
        (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
    ];
    if let Some(value) = photometric {
        tags.push((262u16, 3u16, 1u32, short_in_u32(value)));
    }
    if let Some(value) = predictor {
        tags.push((317u16, 3u16, 1u32, short_in_u32(value)));
    }
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

fn build_multi_stripped_jpeg_tiff(
    width: u32,
    height: u32,
    rows_per_strip: u32,
    strips: &[Vec<u8>],
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let mut strip_offsets = Vec::with_capacity(strips.len());
    let mut strip_byte_counts = Vec::with_capacity(strips.len());
    for strip in strips {
        strip_offsets.push(buf.len() as u32);
        buf.extend_from_slice(strip);
        strip_byte_counts.push(strip.len() as u32);
    }

    let strip_offsets_array_offset = buf.len() as u32;
    for value in &strip_offsets {
        buf.extend_from_slice(&le_u32(*value));
    }
    let strip_byte_counts_array_offset = buf.len() as u32;
    for value in &strip_byte_counts {
        buf.extend_from_slice(&le_u32(*value));
    }

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (259u16, 3u16, 1u32, short_in_u32(7)),
        (262u16, 3u16, 1u32, short_in_u32(6)),
        (
            273u16,
            4u16,
            strip_offsets.len() as u32,
            le_u32(strip_offsets_array_offset),
        ),
        (277u16, 3u16, 1u32, short_in_u32(3)),
        (278u16, 4u16, 1u32, le_u32(rows_per_strip)),
        (
            279u16,
            4u16,
            strip_byte_counts.len() as u32,
            le_u32(strip_byte_counts_array_offset),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

fn encode_solid_rgb_jpeg(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    let mut encoded = Vec::new();
    JpegEncoder::new(&mut encoded, 95)
        .encode(
            image.as_raw().as_slice(),
            image.width() as u16,
            image.height() as u16,
            JpegColorType::Rgb,
        )
        .unwrap();
    encoded
}

fn encode_restart_rgb_jpeg(image: &image::RgbImage, quality: u8, restart_interval: u16) -> Vec<u8> {
    let mut encoded = Vec::new();
    let mut encoder = JpegEncoder::new(&mut encoded, quality);
    encoder.set_restart_interval(restart_interval);
    encoder
        .encode(
            image.as_raw().as_slice(),
            image.width() as u16,
            image.height() as u16,
            JpegColorType::Rgb,
        )
        .unwrap();
    encoded
}

fn find_test_jpeg_bitstream_start(data: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < data.len().saturating_sub(1) {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        if marker == 0xD8 || marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        if i + 3 >= data.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        if marker == 0xDA {
            return Some(i + 2 + seg_len);
        }
        i += 2 + seg_len;
    }
    None
}

fn test_jpeg_restart_segment_starts(data: &[u8]) -> Vec<u32> {
    let mut starts = Vec::new();
    if let Some(entropy_start) = find_test_jpeg_bitstream_start(data) {
        starts.push(entropy_start as u32);
    }
    let mut i = starts.first().copied().unwrap_or(0) as usize;
    while i + 1 < data.len() {
        if data[i] == 0xFF && (0xD0..=0xD7).contains(&data[i + 1]) {
            starts.push(i as u32);
            i += 2;
            continue;
        }
        i += 1;
    }
    starts
}

fn zero_test_jpeg_sof_dimensions(data: &mut [u8]) {
    let sof = data
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0xC0])
        .expect("test JPEG has SOF0");
    data[sof + 5..sof + 9].copy_from_slice(&[0, 0, 0, 0]);
}

fn build_tiled_jpeg_reader(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles: &[Vec<u8>],
) -> TiffPixelReader {
    let file = build_tiled_associated_tiff(width, height, tile_width, tile_height, tiles);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let level = Level {
        dimensions: (u64::from(width), u64::from(height)),
        downsample: 1.0,
        tile_layout: TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across: u64::from(width.div_ceil(tile_width)),
            tiles_down: u64::from(height.div_ceil(tile_height)),
        },
    };
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(31),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![level],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::Jpeg,
            },
        )]),
        associated_sources: HashMap::new(),
    };
    TiffPixelReader::new(container, layout)
}

fn build_tiled_jpeg_reader_with_tables(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles: &[Vec<u8>],
    jpeg_tables: Vec<u8>,
) -> TiffPixelReader {
    let file = build_tiled_jpeg_tiff_with_tables(
        width,
        height,
        tile_width,
        tile_height,
        tiles,
        &jpeg_tables,
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let level = Level {
        dimensions: (u64::from(width), u64::from(height)),
        downsample: 1.0,
        tile_layout: TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across: u64::from(width.div_ceil(tile_width)),
            tiles_down: u64::from(height.div_ceil(tile_height)),
        },
    };
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(32),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![level],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: Some(jpeg_tables),
                compression: Compression::Jpeg,
            },
        )]),
        associated_sources: HashMap::new(),
    };
    TiffPixelReader::new(container, layout)
}

#[allow(clippy::too_many_arguments)]
fn build_tiled_encoded_reader(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles: &[Vec<u8>],
    compression: Compression,
    compression_tag: u16,
    samples_per_pixel: u16,
    photometric: u16,
) -> TiffPixelReader {
    let file = build_tiled_encoded_tiff(
        width,
        height,
        tile_width,
        tile_height,
        tiles,
        compression_tag,
        samples_per_pixel,
        photometric,
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let level = Level {
        dimensions: (u64::from(width), u64::from(height)),
        downsample: 1.0,
        tile_layout: TileLayout::Regular {
            tile_width,
            tile_height,
            tiles_across: u64::from(width.div_ceil(tile_width)),
            tiles_down: u64::from(height.div_ceil(tile_height)),
        },
    };
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(33),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![level],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: None,
                compression,
            },
        )]),
        associated_sources: HashMap::new(),
    };
    TiffPixelReader::new(container, layout)
}

fn build_tiled_jpeg_tiff_with_tables(
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles: &[Vec<u8>],
    jpeg_tables: &[u8],
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let mut tile_offsets = Vec::with_capacity(tiles.len());
    let mut tile_byte_counts = Vec::with_capacity(tiles.len());
    for tile in tiles {
        tile_offsets.push(buf.len() as u32);
        tile_byte_counts.push(tile.len() as u32);
        buf.extend_from_slice(tile);
    }

    let tile_offsets_array_offset = if tile_offsets.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_offsets {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let tile_byte_counts_array_offset = if tile_byte_counts.len() > 1 {
        let offset = buf.len() as u32;
        for value in &tile_byte_counts {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let jpeg_tables_offset = buf.len() as u32;
    buf.extend_from_slice(jpeg_tables);

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (258u16, 3u16, 1u32, short_in_u32(8)),
        (259u16, 3u16, 1u32, short_in_u32(7)),
        (262u16, 3u16, 1u32, short_in_u32(6)),
        (277u16, 3u16, 1u32, short_in_u32(3)),
        (322u16, 4u16, 1u32, le_u32(tile_width)),
        (323u16, 4u16, 1u32, le_u32(tile_height)),
        (
            324u16,
            4u16,
            tile_offsets.len() as u32,
            tile_offsets_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_offsets[0])),
        ),
        (
            325u16,
            4u16,
            tile_byte_counts.len() as u32,
            tile_byte_counts_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(tile_byte_counts[0])),
        ),
        (
            347u16,
            7u16,
            jpeg_tables.len() as u32,
            le_u32(jpeg_tables_offset),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

fn split_test_jpeg_tables(jpeg: &[u8]) -> (Vec<u8>, Vec<u8>) {
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
    let mut abbreviated = Vec::from(&jpeg[..2]);
    let mut tables = Vec::from(&jpeg[..2]);
    let mut offset = 2usize;
    while offset + 4 <= jpeg.len() {
        assert_eq!(jpeg[offset], 0xFF);
        let marker = jpeg[offset + 1];
        if marker == 0xDA {
            abbreviated.extend_from_slice(&jpeg[offset..]);
            tables.extend_from_slice(&[0xFF, 0xD9]);
            return (abbreviated, tables);
        }
        let len = u16::from_be_bytes([jpeg[offset + 2], jpeg[offset + 3]]) as usize;
        let end = offset + 2 + len;
        assert!(end <= jpeg.len());
        if marker == 0xDB || marker == 0xC4 {
            tables.extend_from_slice(&jpeg[offset..end]);
        } else {
            abbreviated.extend_from_slice(&jpeg[offset..end]);
        }
        offset = end;
    }
    panic!("test JPEG did not contain SOS marker");
}

fn build_ndpi_full_jpeg_tiff(
    width: u32,
    height: u32,
    jpeg_data: &[u8],
    blob_count: usize,
) -> NamedTempFile {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let strip_offset = buf.len() as u32;
    let mut mcu_starts = test_jpeg_restart_segment_starts(jpeg_data);
    if mcu_starts.len() >= blob_count {
        mcu_starts.truncate(blob_count);
    } else {
        mcu_starts = (0..blob_count as u32).collect();
    }
    buf.extend_from_slice(jpeg_data);
    let strip_byte_count = buf.len() as u32 - strip_offset;

    let mcu_starts_array_offset = if mcu_starts.len() > 1 {
        let offset = buf.len() as u32;
        for value in &mcu_starts {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (259u16, 3u16, 1u32, short_in_u32(7)),
        (262u16, 3u16, 1u32, short_in_u32(6)),
        (273u16, 4u16, 1u32, le_u32(strip_offset)),
        (277u16, 3u16, 1u32, short_in_u32(3)),
        (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
        (
            65426u16,
            4u16,
            mcu_starts.len() as u32,
            mcu_starts_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(mcu_starts[0])),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

fn build_ndpi_scan_data_tiff_from_blobs(
    width: u32,
    height: u32,
    colors: &[[u8; 3]],
    zero_sof_dimensions: bool,
) -> (NamedTempFile, Vec<u8>, u64) {
    let test_tile_width = 64;
    let test_tile_height = 8;
    let tiles_across = width.div_ceil(test_tile_width);
    let mut image = image::RgbImage::new(width, height);
    for (idx, rgb) in colors.iter().enumerate() {
        let tile_col = (idx as u32) % tiles_across;
        let tile_row = (idx as u32) / tiles_across;
        let x0 = tile_col * test_tile_width;
        let y0 = tile_row * test_tile_height;
        for y in y0..(y0 + test_tile_height).min(height) {
            for x in x0..(x0 + test_tile_width).min(width) {
                image.put_pixel(x, y, image::Rgb(*rgb));
            }
        }
    }
    let mut encoded = encode_restart_rgb_jpeg(&image, 95, 8);
    if zero_sof_dimensions {
        zero_test_jpeg_sof_dimensions(&mut encoded);
    }
    let bitstream_start = find_test_jpeg_bitstream_start(&encoded).unwrap();
    let jpeg_header = encoded[..bitstream_start].to_vec();
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&le_u16(42));
    let first_ifd_pos = buf.len();
    buf.extend_from_slice(&le_u32(0));

    let strip_offset = buf.len() as u32;
    let mut mcu_starts = test_jpeg_restart_segment_starts(&encoded);
    mcu_starts.truncate(colors.len());
    assert_eq!(mcu_starts.len(), colors.len());
    buf.extend_from_slice(&encoded);
    let strip_byte_count = buf.len() as u32 - strip_offset;

    let mcu_starts_array_offset = if mcu_starts.len() > 1 {
        let offset = buf.len() as u32;
        for value in &mcu_starts {
            buf.extend_from_slice(&le_u32(*value));
        }
        Some(offset)
    } else {
        None
    };

    let ifd_offset = buf.len() as u32;
    buf[first_ifd_pos..first_ifd_pos + 4].copy_from_slice(&le_u32(ifd_offset));

    let mut tags = vec![
        (256u16, 4u16, 1u32, le_u32(width)),
        (257u16, 4u16, 1u32, le_u32(height)),
        (259u16, 3u16, 1u32, short_in_u32(7)),
        (262u16, 3u16, 1u32, short_in_u32(6)),
        (273u16, 4u16, 1u32, le_u32(strip_offset)),
        (277u16, 3u16, 1u32, short_in_u32(3)),
        (279u16, 4u16, 1u32, le_u32(strip_byte_count)),
        (
            65426u16,
            4u16,
            mcu_starts.len() as u32,
            mcu_starts_array_offset
                .map(le_u32)
                .unwrap_or_else(|| le_u32(mcu_starts[0])),
        ),
    ];
    tags.sort_by_key(|tag| tag.0);

    buf.extend_from_slice(&le_u16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        buf.extend_from_slice(&le_u16(*tag));
        buf.extend_from_slice(&le_u16(*typ));
        buf.extend_from_slice(&le_u32(*count));
        buf.extend_from_slice(value);
    }
    buf.extend_from_slice(&le_u32(0));

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    (file, jpeg_header, strip_byte_count as u64)
}

// ── TiffPixelReader tests ─────────────────────────────────────

// Note: Testing TiffPixelReader with NdpiJpeg requires a synthetic NDPI
// file with valid MCU-starts tags. Since building such files is complex,
// we test the TiffPixelReader through the full interpret -> read path in
// Task 9's integration tests. Here we test the FullDecodeCache directly
// (above) and add integration tests in Task 9.

#[test]
fn read_associated_composites_tiled_ifd_images() {
    let tiles = [vec![10u8; 4], vec![20u8; 4], vec![30u8; 4], vec![40u8; 4]];
    let file = build_tiled_associated_tiff(4, 4, 2, 2, &tiles);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(1),
            scenes: vec![],
            associated_images: HashMap::from([(
                "label".to_string(),
                AssociatedImage {
                    dimensions: (4, 4),
                    sample_type: SampleType::Uint8,
                    channels: 1,
                },
            )]),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::from([(
            "label".to_string(),
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::None,
            },
        )]),
    };
    let reader = TiffPixelReader::new(container, layout);

    let image = reader.read_associated("label").unwrap();
    let rgb = image.data.as_u8().unwrap();
    let expected = vec![
        10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20,
        20, 30, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 30, 30, 30, 30, 30, 30, 40, 40, 40, 40,
        40, 40,
    ];
    assert_eq!(rgb, expected.as_slice());
    let pixel = |x: usize, y: usize| -> [u8; 3] {
        let idx = (y * image.width as usize + x) * 3;
        [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
    };

    assert_eq!(pixel(0, 0), [10, 10, 10]);
    assert_eq!(pixel(3, 0), [20, 20, 20]);
    assert_eq!(pixel(0, 3), [30, 30, 30]);
    assert_eq!(pixel(3, 3), [40, 40, 40]);
}

#[test]
fn raw_compressed_tile_returns_standalone_tiled_jpeg_byte_identical() {
    let jpeg = encode_solid_rgb_jpeg(8, 8, [200, 10, 30]);
    let reader = build_tiled_jpeg_reader(8, 8, 8, 8, std::slice::from_ref(&jpeg));

    let raw = reader
        .read_raw_compressed_tile(&TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();

    assert_eq!(raw.compression, Compression::Jpeg);
    assert_eq!((raw.width, raw.height), (8, 8));
    assert_eq!(raw.bits_allocated, 8);
    assert_eq!(raw.samples_per_pixel, 3);
    assert_eq!(raw.data, jpeg);
}

#[test]
fn raw_compressed_tile_rebuilds_tiled_jpeg_with_jpeg_tables_without_reencoding_entropy() {
    let jpeg = encode_solid_rgb_jpeg(8, 8, [40, 180, 90]);
    let (abbreviated_tile, jpeg_tables) = split_test_jpeg_tables(&jpeg);
    let reader = build_tiled_jpeg_reader_with_tables(
        8,
        8,
        8,
        8,
        std::slice::from_ref(&abbreviated_tile),
        jpeg_tables,
    );

    let raw = reader
        .read_raw_compressed_tile(&TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();

    assert_eq!(raw.compression, Compression::Jpeg);
    assert_eq!((raw.width, raw.height), (8, 8));
    assert!(raw.data.len() > abbreviated_tile.len());
    assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xDB]));
    assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xC4]));
    assert!(raw.data.ends_with(&[0xFF, 0xD9]));
    assert!(raw
        .data
        .windows(abbreviated_tile.len().saturating_sub(2))
        .any(|window| window == &abbreviated_tile[2..]));
}

#[test]
fn raw_compressed_tile_returns_tiled_jp2k_rgb_byte_identical() {
    let codestream = include_bytes!("../../../../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
    let expected = load_fixture_rgb(include_bytes!(
        "../../../../tests/fixtures/jp2k/rgb_nomct.ppm"
    ));
    let reader = build_tiled_encoded_reader(
        expected.width(),
        expected.height(),
        expected.width(),
        expected.height(),
        std::slice::from_ref(&codestream),
        Compression::Jp2kRgb,
        33004,
        3,
        2,
    );

    let raw = reader
        .read_raw_compressed_tile(&TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();

    assert_eq!(raw.compression, Compression::Jp2kRgb);
    assert_eq!(
        (raw.width, raw.height),
        (expected.width(), expected.height())
    );
    assert_eq!(raw.bits_allocated, 8);
    assert_eq!(raw.samples_per_pixel, 3);
    assert_eq!(
        raw.photometric_interpretation,
        EncodedTilePhotometricInterpretation::Rgb
    );
    assert_eq!(raw.data, codestream);
}

#[test]
fn raw_compressed_tile_returns_standalone_ndpi_restart_jpeg() {
    let (reader, _) = build_test_ndpi_reader_for_strip_cache(128, 16, 1);

    let raw = reader
        .read_raw_compressed_tile(&TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap();

    assert_eq!(raw.compression, Compression::Jpeg);
    assert_eq!((raw.width, raw.height), (128, 16));
    assert_eq!(raw.bits_allocated, 8);
    assert_eq!(raw.samples_per_pixel, 3);
    assert!(raw.data.starts_with(&[0xFF, 0xD8]));
    assert!(raw.data.ends_with(&[0xFF, 0xD9]));
    assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xC0]));
    assert!(raw.data.windows(2).any(|bytes| bytes == [0xFF, 0xDA]));

    let decoded = decode_jpeg_rgb_with_size_override(
        &raw.data,
        None,
        raw.width,
        raw.height,
        None,
        None,
        SigninumColorTransform::Auto,
    )
    .expect("decode raw NDPI JPEG tile");
    assert_eq!((decoded.width, decoded.height), (128, 16));
}

#[test]
fn raw_compressed_tile_rejects_ndpi_restart_segments_that_cross_rows() {
    let (reader, _) = build_test_ndpi_reader_for_strip_cache(130, 16, 2);

    let err = reader
        .read_raw_compressed_tile(&TileRequest {
            scene: 0,
            series: 0,
            level: 1,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        })
        .unwrap_err();

    assert!(
        err.to_string().contains("align to image rows"),
        "unexpected error: {err}"
    );
}

#[test]
fn read_associated_thumbnail_assembly_matches_expected_rgb_bytes_with_edge_tiles() {
    let tiles = [
        vec![10u8; 4],
        vec![20u8; 4],
        vec![30u8; 2],
        vec![40u8; 2],
        vec![50u8; 2],
        vec![60u8; 1],
    ];
    let file = build_tiled_associated_tiff(5, 3, 2, 2, &tiles);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(1),
            scenes: vec![],
            associated_images: HashMap::from([(
                "label".to_string(),
                AssociatedImage {
                    dimensions: (5, 3),
                    sample_type: SampleType::Uint8,
                    channels: 1,
                },
            )]),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::from([(
            "label".to_string(),
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::None,
            },
        )]),
    };
    let reader = TiffPixelReader::new(container, layout);

    let image = reader.read_associated("label").unwrap();
    let rgb = image.data.as_u8().unwrap();
    let grayscale_pixels = [10u8, 10, 20, 20, 30, 10, 10, 20, 20, 30, 40, 40, 50, 50, 60];
    let expected: Vec<u8> = grayscale_pixels
        .into_iter()
        .flat_map(|value| [value, value, value])
        .collect();

    assert_eq!(image.width, 5);
    assert_eq!(image.height, 3);
    assert_eq!(rgb, expected.as_slice());
}

#[test]
fn read_associated_composes_multi_strip_jpeg_image() {
    let width = 4;
    let height = 4;
    let rows_per_strip = 2;

    let mut top = image::RgbImage::new(width, rows_per_strip);
    for pixel in top.pixels_mut() {
        *pixel = image::Rgb([220, 40, 10]);
    }
    let mut bottom = image::RgbImage::new(width, rows_per_strip);
    for pixel in bottom.pixels_mut() {
        *pixel = image::Rgb([15, 80, 210]);
    }

    let encode_strip = |img: &image::RgbImage| {
        let mut encoded = Vec::new();
        JpegEncoder::new(&mut encoded, 100)
            .encode(
                img.as_raw().as_slice(),
                img.width() as u16,
                img.height() as u16,
                JpegColorType::Rgb,
            )
            .unwrap();
        encoded
    };
    let file = build_multi_stripped_jpeg_tiff(
        width,
        height,
        rows_per_strip,
        &[encode_strip(&top), encode_strip(&bottom)],
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let strip_offsets = container
        .get_u64_array(ifd_id, tags::STRIP_OFFSETS)
        .unwrap();
    let strip_byte_counts = container
        .get_u64_array(ifd_id, tags::STRIP_BYTE_COUNTS)
        .unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(17),
            scenes: vec![],
            associated_images: HashMap::from([(
                "label".to_string(),
                AssociatedImage {
                    dimensions: (width, height),
                    sample_type: SampleType::Uint8,
                    channels: 3,
                },
            )]),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::from([(
            "label".to_string(),
            TileSource::Stripped {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::Jpeg,
                strip_offsets: strip_offsets.to_vec(),
                strip_byte_counts: strip_byte_counts.to_vec(),
            },
        )]),
    };
    let reader = TiffPixelReader::new(container, layout);

    let image = reader.read_associated("label").unwrap();
    let rgb = image.data.as_u8().unwrap();
    let pixel = |x: usize, y: usize| -> [u8; 3] {
        let idx = (y * image.width as usize + x) * 3;
        [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
    };

    let top_left = pixel(0, 0);
    let top_right = pixel((width - 1) as usize, 0);
    let bottom_left = pixel(0, 3);
    let bottom_right = pixel((width - 1) as usize, 3);
    let strip_delta = |a: [u8; 3], b: [u8; 3]| -> u16 {
        a.into_iter()
            .zip(b)
            .map(|(lhs, rhs)| lhs.abs_diff(rhs) as u16)
            .sum()
    };

    assert!(strip_delta(top_left, top_right) < 20);
    assert!(strip_delta(bottom_left, bottom_right) < 20);
    assert!(strip_delta(top_left, bottom_left) > 80);
}

#[test]
fn read_associated_uncompressed_single_sample_rgb_photometric_treated_as_grayscale() {
    let pixels = [12u8, 34, 56, 78, 90, 123, 150, 210];
    let file = build_stripped_uncompressed_tiff(4, 2, &pixels, 1, Some(2));
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(23),
            scenes: vec![],
            associated_images: HashMap::from([(
                "thumbnail".to_string(),
                AssociatedImage {
                    dimensions: (4, 2),
                    sample_type: SampleType::Uint8,
                    channels: 1,
                },
            )]),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::from([(
            "thumbnail".to_string(),
            TileSource::Stripped {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::None,
                strip_offsets: vec![container.get_u64(ifd_id, tags::STRIP_OFFSETS).unwrap()],
                strip_byte_counts: vec![container
                    .get_u64(ifd_id, tags::STRIP_BYTE_COUNTS)
                    .unwrap()],
            },
        )]),
    };
    let reader = TiffPixelReader::new(container, layout);

    let image = reader.read_associated("thumbnail").unwrap();
    assert_eq!(image.width, 4);
    assert_eq!(image.height, 2);
    assert_eq!(image.channels, 1);
    assert_eq!(image.color_space, ColorSpace::Grayscale);
    assert_eq!(image.data.as_u8().unwrap(), pixels.as_slice());
}

#[test]
fn tiff_predictor_reconstructs_8bit_horizontal_deltas() {
    let encoded = [10u8, 5, 5, 1, 2, 3];
    let file = build_stripped_uncompressed_tiff_with_predictor(3, 2, &encoded, 1, Some(1), Some(2));
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(24),
            scenes: vec![],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::new(),
    };
    let reader = TiffPixelReader::new(container, layout);
    let mut data = encoded.to_vec();

    reader
        .apply_tiff_predictor(ifd_id, 3, 2, &mut data)
        .unwrap();

    assert_eq!(data, [10, 15, 20, 1, 3, 6]);
}

#[test]
fn read_associated_deflate_predictor_uses_tilecodec_path() {
    let expected = [10u8, 15, 20, 1, 3, 6];
    let predictor_encoded = [10u8, 5, 5, 1, 2, 3];
    let mut encoder = ZlibEncoder::new(Vec::new(), DeflateCompression::fast());
    encoder.write_all(&predictor_encoded).unwrap();
    let compressed = encoder.finish().unwrap();
    let file = build_stripped_tiff(3, 2, &compressed, 1, Some(1), Some(2), 8);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(25),
            scenes: vec![],
            associated_images: HashMap::from([(
                "thumbnail".to_string(),
                AssociatedImage {
                    dimensions: (3, 2),
                    sample_type: SampleType::Uint8,
                    channels: 1,
                },
            )]),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::from([(
            "thumbnail".to_string(),
            TileSource::Stripped {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::Deflate,
                strip_offsets: vec![container.get_u64(ifd_id, tags::STRIP_OFFSETS).unwrap()],
                strip_byte_counts: vec![container
                    .get_u64(ifd_id, tags::STRIP_BYTE_COUNTS)
                    .unwrap()],
            },
        )]),
    };
    let reader = TiffPixelReader::new(container, layout);

    let image = reader.read_associated("thumbnail").unwrap();

    assert_eq!(image.data.as_u8().unwrap(), expected.as_slice());
}

#[test]
fn read_tiles_classifies_distinct_jpeg_tiled_ifd_requests_as_batchable() {
    let tiles = [
        encode_solid_rgb_jpeg(8, 8, [200, 10, 10]),
        encode_solid_rgb_jpeg(8, 8, [10, 200, 10]),
        encode_solid_rgb_jpeg(8, 8, [10, 10, 200]),
        encode_solid_rgb_jpeg(8, 8, [220, 220, 20]),
    ];
    let reader = build_tiled_jpeg_reader(16, 16, 8, 8, &tiles);
    let reqs = [
        TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 0,
        },
        TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 1,
            row: 0,
        },
        TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 0,
            row: 1,
        },
        TileRequest {
            scene: 0,
            series: 0,
            level: 0,
            plane: PlaneSelection::default(),
            col: 1,
            row: 1,
        },
    ];

    assert_eq!(
        reader.tiled_ifd_batch_compression(&reqs).unwrap(),
        Some(Compression::Jpeg)
    );

    let batched = reader.read_tiles_cpu(&reqs).unwrap();
    let sequential = reqs
        .iter()
        .map(|req| reader.read_tile_cpu(req))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batched.len(), sequential.len());
    for (batched, sequential) in batched.iter().zip(sequential.iter()) {
        assert_eq!((batched.width, batched.height), (8, 8));
        assert_eq!(batched.data.as_u8(), sequential.data.as_u8());
    }
}

#[test]
fn tile_codec_kind_classifies_tiff_jpeg_and_jp2k_sources() {
    let jpeg_tiles = [encode_solid_rgb_jpeg(8, 8, [200, 10, 10])];
    let jpeg_reader = build_tiled_jpeg_reader(8, 8, 8, 8, &jpeg_tiles);
    let req = TileRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };
    assert_eq!(jpeg_reader.tile_codec_kind(&req), TileCodecKind::Jpeg);

    let codestream = include_bytes!("../../../../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
    let expected = load_fixture_rgb(include_bytes!(
        "../../../../tests/fixtures/jp2k/rgb_nomct.ppm"
    ));
    let jp2k_reader = build_tiled_encoded_reader(
        expected.width(),
        expected.height(),
        expected.width(),
        expected.height(),
        &[codestream],
        Compression::Jp2kRgb,
        33004,
        3,
        2,
    );
    assert_eq!(jp2k_reader.tile_codec_kind(&req), TileCodecKind::Jp2k);
}

#[cfg(feature = "metal")]
#[test]
fn prefer_device_empty_tiled_jpeg_falls_back_to_cpu_empty_tile() {
    let tiles = [Vec::new()];
    let reader = build_tiled_jpeg_reader(8, 8, 8, 8, &tiles);
    let req = TileRequest {
        scene: 0,
        series: 0,
        level: 0,
        plane: PlaneSelection::default(),
        col: 0,
        row: 0,
    };

    let tiles = reader
        .read_tiles(&[req], TileOutputPreference::prefer_device_auto())
        .unwrap();

    assert_eq!(tiles.len(), 1);
    let TilePixels::Cpu(tile) = &tiles[0] else {
        panic!("PreferDevice should fall back to CPU for empty tiles");
    };
    assert_eq!((tile.width, tile.height), (8, 8));
    assert_eq!(tile.data.as_u8().unwrap(), &[0u8; 8 * 8 * 3]);
}

#[cfg(feature = "metal")]
#[test]
fn jpeg_device_decode_is_opt_in_by_default() {
    assert!(!jpeg_device_decode_enabled());
}

#[cfg(feature = "metal")]
#[test]
fn jp2k_device_decode_is_opt_in_by_default() {
    assert!(!jp2k_device_decode_enabled());
}

#[test]
fn jp2k_tiled_sources_request_larger_shared_cache_budget() {
    let tiles = [vec![7u8; 4]];
    let file = build_tiled_associated_tiff(2, 2, 2, 2, &tiles);
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(24),
            scenes: vec![Scene {
                id: "s0".into(),
                name: None,
                series: vec![Series {
                    id: "ser0".into(),
                    axes: AxesShape::default(),
                    levels: vec![Level {
                        dimensions: (2, 2),
                        downsample: 1.0,
                        tile_layout: TileLayout::Regular {
                            tile_width: 2,
                            tile_height: 2,
                            tiles_across: 1,
                            tiles_down: 1,
                        },
                    }],
                    sample_type: SampleType::Uint8,
                    channels: vec![],
                }],
            }],
            associated_images: HashMap::new(),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::from([(
            TileSourceKey {
                scene: 0,
                series: 0,
                level: 0,
                z: 0,
                c: 0,
                t: 0,
            },
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: None,
                compression: Compression::Jp2kRgb,
            },
        )]),
        associated_sources: HashMap::new(),
    };
    let reader = TiffPixelReader::new(container, layout);

    assert_eq!(
        reader.recommended_shared_cache_bytes(),
        Some(DEFAULT_JP2K_SHARED_TILE_CACHE_BYTES)
    );
}

fn load_fixture_rgb(ppm_bytes: &[u8]) -> image::RgbImage {
    match image::load(Cursor::new(ppm_bytes), ImageFormat::Pnm).unwrap() {
        DynamicImage::ImageRgb8(image) => image,
        other => other.to_rgb8(),
    }
}

fn build_single_tile_jp2k_layout(
    container: Arc<TiffContainer>,
    compression: Compression,
    width: u32,
    height: u32,
) -> TiffPixelReader {
    let ifd_id = *container.top_ifds().first().unwrap();
    let layout = DatasetLayout {
        dataset: Dataset {
            id: DatasetId(1),
            scenes: vec![],
            associated_images: HashMap::from([(
                "label".to_string(),
                AssociatedImage {
                    dimensions: (width, height),
                    sample_type: SampleType::Uint8,
                    channels: 3,
                },
            )]),
            properties: Properties::new(),
            icc_profiles: HashMap::new(),
        },
        tile_sources: HashMap::new(),
        associated_sources: HashMap::from([(
            "label".to_string(),
            TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: None,
                compression,
            },
        )]),
    };
    TiffPixelReader::new(container, layout)
}

fn assert_sample_buffer_matches_rgb_fixture(image: &CpuTile, expected_rgb: &image::RgbImage) {
    assert_eq!(image.width, expected_rgb.width());
    assert_eq!(image.height, expected_rgb.height());
    let actual = image.data.as_u8().unwrap();
    let expected = expected_rgb.as_raw();
    assert_eq!(actual.len(), expected.len());

    let mut total_delta = 0u64;
    let mut max_delta = 0u8;
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let delta = actual.abs_diff(*expected);
        total_delta += u64::from(delta);
        max_delta = max_delta.max(delta);
    }

    let avg_delta_x100 = if actual.is_empty() {
        0
    } else {
        (total_delta * 100) / actual.len() as u64
    };

    assert!(
        max_delta <= 50,
        "JP2K tiled decode drift too large: max channel delta {max_delta} > 50",
    );
    assert!(
        avg_delta_x100 <= 1600,
        "JP2K tiled decode drift too large: average channel delta {:.2} > 16.00",
        avg_delta_x100 as f64 / 100.0,
    );
}

#[test]
fn read_associated_decodes_jp2k_rgb_tile_from_tiled_ifd() {
    let codestream = include_bytes!("../../../../tests/fixtures/jp2k/rgb_nomct.j2k").to_vec();
    let expected = load_fixture_rgb(include_bytes!(
        "../../../../tests/fixtures/jp2k/rgb_nomct.ppm"
    ));
    let file = build_tiled_associated_tiff(
        expected.width(),
        expected.height(),
        expected.width(),
        expected.height(),
        &[codestream],
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let reader = build_single_tile_jp2k_layout(
        container,
        Compression::Jp2kRgb,
        expected.width(),
        expected.height(),
    );

    let image = reader.read_associated("label").unwrap();
    assert_sample_buffer_matches_rgb_fixture(&image, &expected);
}

#[test]
fn read_associated_decodes_jp2k_ycbcr_tile_from_tiled_ifd() {
    let codestream = include_bytes!("../../../../tests/fixtures/jp2k/ycbcr_420.j2k").to_vec();
    let expected = load_fixture_rgb(include_bytes!(
        "../../../../tests/fixtures/jp2k/ycbcr_420.ppm"
    ));
    let file = build_tiled_associated_tiff(
        expected.width(),
        expected.height(),
        expected.width(),
        expected.height(),
        &[codestream],
    );
    let container = Arc::new(TiffContainer::open(file.path()).unwrap());
    let reader = build_single_tile_jp2k_layout(
        container,
        Compression::Jp2kYcbcr,
        expected.width(),
        expected.height(),
    );

    let image = reader.read_associated("label").unwrap();
    assert_sample_buffer_matches_rgb_fixture(&image, &expected);
}
