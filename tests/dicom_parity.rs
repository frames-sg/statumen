//! DICOM-specific parity harness.

mod support;

use support::corpus::{load_public, resolve_entry_path};
use support::oracles::{read_probe, top_left_probe, Oracle, SigninumOracle};

#[test]
#[ignore = "requires public parity corpus; run after scripts/parity-corpus-fetch.sh"]
fn dicom_public_corpus_decodes_with_statumen() {
    let strict_corpus = strict_corpus_required();
    let manifest = match load_public() {
        Ok(manifest) => manifest,
        Err(err) => {
            if strict_corpus {
                panic!("[dicom-parity] public manifest unavailable in strict mode: {err}");
            }
            eprintln!("[dicom-parity] manifest unavailable: {err}; skipping");
            return;
        }
    };

    let mut checked = 0u32;
    let mut missing_slides = 0u32;
    let mut failures = Vec::new();
    for entry in manifest
        .slides
        .iter()
        .filter(|entry| entry.format == "dicom")
    {
        let path = resolve_entry_path(entry);
        if !path.is_file() {
            missing_slides += 1;
            eprintln!(
                "[dicom-parity] {} missing at {}; skipping",
                entry.alias,
                path.display()
            );
            if strict_corpus {
                failures.push(format!(
                    "{}: corpus slide missing at {}",
                    entry.alias,
                    path.display()
                ));
            }
            continue;
        }
        let slide = match SigninumOracle.open(&path) {
            Ok(slide) => slide,
            Err(err) => {
                failures.push(format!("{}: open statumen DICOM: {err}", entry.alias));
                continue;
            }
        };
        for level in 0..slide.level_count.min(3) {
            if !entry.must_decode_level(level) {
                continue;
            }
            let Some(probe) = top_left_probe(&slide, level) else {
                failures.push(format!("{} level={level}: no readable probe", entry.alias));
                continue;
            };
            match read_probe(&slide, probe) {
                Ok(tile) => {
                    checked += 1;
                    eprintln!(
                        "[dicom-parity] {} level={level}: decoded {}x{}",
                        entry.alias, tile.width, tile.height
                    );
                }
                Err(err) => failures.push(format!("{} level={level}: decode: {err}", entry.alias)),
            }
        }
    }

    if missing_slides == 0 && checked == 0 {
        failures.push("DICOM parity decoded zero corpus tiles".to_string());
    }
    eprintln!(
        "[dicom-parity] checked={checked} missing_slides={missing_slides} failures={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "parity-openslide")]
#[test]
#[ignore = "requires public parity corpus and libopenslide"]
fn dicom_public_corpus_matches_openslide_within_tolerance() {
    use support::compare::{compare_rgba, tolerance_failure, Tolerance};
    use support::oracles::OpenSlideOracle;

    let lib = support::openslide_shim::try_load()
        .expect("libopenslide is required for DICOM OpenSlide parity");
    let openslide = OpenSlideOracle { lib };
    let manifest = load_public().expect("load public manifest");
    let mut checked = 0u32;
    let mut failures = Vec::new();

    for entry in manifest
        .slides
        .iter()
        .filter(|entry| entry.format == "dicom")
    {
        let path = resolve_entry_path(entry);
        if !path.is_file() {
            failures.push(format!(
                "{}: corpus slide missing at {}",
                entry.alias,
                path.display()
            ));
            continue;
        }
        let ours = match SigninumOracle.open(&path) {
            Ok(slide) => slide,
            Err(err) => {
                failures.push(format!("{}: open statumen DICOM: {err}", entry.alias));
                continue;
            }
        };
        let theirs = match openslide.open(&path) {
            Ok(slide) => slide,
            Err(err) => {
                failures.push(format!("{}: open OpenSlide: {err}", entry.alias));
                continue;
            }
        };
        for level in 0..ours.level_count.min(theirs.level_count).min(3) {
            if !entry.must_decode_level(level) {
                continue;
            }
            let Some(probe) = top_left_probe(&ours, level) else {
                failures.push(format!("{} level={level}: no readable probe", entry.alias));
                continue;
            };
            let ours_buf = match read_probe(&ours, probe) {
                Ok(buf) => buf,
                Err(err) => {
                    failures.push(format!(
                        "{} level={level}: read statumen: {err}",
                        entry.alias
                    ));
                    continue;
                }
            };
            let theirs_buf = match read_probe(&theirs, probe) {
                Ok(buf) => buf,
                Err(err) => {
                    failures.push(format!(
                        "{} level={level}: read OpenSlide: {err}",
                        entry.alias
                    ));
                    continue;
                }
            };
            let report = compare_rgba(
                &ours_buf.pixels_rgba,
                &theirs_buf.pixels_rgba,
                Tolerance::TOLERANT,
            );
            eprintln!(
                "[dicom-parity] {} level={level}: max_abs={} mean_abs={:.4} passed={}",
                entry.alias, report.max_abs, report.mean_abs, report.passed
            );
            if let Some(failure) = tolerance_failure(
                &format!("{} level={level}: statumen vs OpenSlide", entry.alias),
                &report,
            ) {
                failures.push(failure);
            }
            checked += 1;
        }
    }

    if checked == 0 {
        failures.push("DICOM OpenSlide parity checked zero tiles".to_string());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn strict_corpus_required() -> bool {
    true
}
