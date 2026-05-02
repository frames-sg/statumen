//! Per-codec, per-backend, per-batch-size decode throughput.

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use statumen::{FormatRegistry, PlaneSelection, Slide, TileLayout, TileRequest};

#[path = "../tests/support/mod.rs"]
mod support;

use support::corpus::{load_public, resolve_entry_path};

fn pick_corpus() -> Vec<(String, PathBuf, Vec<String>)> {
    let manifest = match load_public() {
        Ok(manifest) => manifest,
        Err(_) => return Vec::new(),
    };
    manifest
        .slides
        .into_iter()
        .filter_map(|entry| {
            let path = resolve_entry_path(&entry);
            if path.is_file() {
                Some((entry.alias, path, entry.codecs))
            } else {
                None
            }
        })
        .collect()
}

fn build_handle(path: &std::path::Path) -> Slide {
    let registry = FormatRegistry::builtin();
    Slide::open_with_cache_bytes(path, &registry, 256 * 1024 * 1024).expect("open slide")
}

fn bench_decode_throughput(c: &mut Criterion) {
    let slides = pick_corpus();
    if slides.is_empty() {
        eprintln!("[decode_throughput] no corpus slides found; skipping bench");
        return;
    }

    for (alias, path, codecs) in slides {
        let handle = build_handle(&path);
        let level0 = &handle.dataset().scenes[0].series[0].levels[0];
        let (tile_width, tile_height, tiles_across, tiles_down) = match &level0.tile_layout {
            TileLayout::Regular {
                tile_width,
                tile_height,
                tiles_across,
                tiles_down,
            } => (*tile_width, *tile_height, *tiles_across, *tiles_down),
            TileLayout::WholeLevel {
                virtual_tile_width,
                virtual_tile_height,
                width,
                height,
            } => (
                *virtual_tile_width,
                *virtual_tile_height,
                width.div_ceil(u64::from(*virtual_tile_width)),
                height.div_ceil(u64::from(*virtual_tile_height)),
            ),
            TileLayout::Irregular { .. } => continue,
        };
        if tiles_across == 0 || tiles_down == 0 {
            continue;
        }

        for batch_size in [1usize, 4, 16] {
            let reqs: Vec<TileRequest> = (0..batch_size as i64)
                .map(|i| TileRequest {
                    scene: 0,
                    series: 0,
                    level: 0,
                    plane: PlaneSelection::default(),
                    col: i % i64::try_from(tiles_across).unwrap_or(1),
                    row: (i / i64::try_from(tiles_across).unwrap_or(1))
                        .min(i64::try_from(tiles_down.saturating_sub(1)).unwrap_or(0)),
                })
                .collect();

            let codec_label = codecs.first().cloned().unwrap_or_else(|| "unknown".into());
            let mut group = c.benchmark_group(format!("decode_throughput/{codec_label}/{alias}"));
            group.throughput(Throughput::Bytes(
                batch_size as u64 * u64::from(tile_width) * u64::from(tile_height) * 4,
            ));
            group.bench_with_input(
                BenchmarkId::new("signinum_cpu", batch_size),
                &reqs,
                |b, reqs| {
                    b.iter(|| {
                        let _ = handle.source().read_tiles_cpu(reqs).expect("read_tiles");
                    })
                },
            );
            #[cfg(feature = "parity-metal")]
            {
                group.bench_with_input(
                    BenchmarkId::new("signinum_metal_stub", batch_size),
                    &reqs,
                    |b, reqs| {
                        b.iter(|| {
                            let _ = handle.source().read_tiles_cpu(reqs).expect("read_tiles");
                        })
                    },
                );
            }
            group.finish();
        }
    }
}

criterion_group!(benches, bench_decode_throughput);
criterion_main!(benches);
