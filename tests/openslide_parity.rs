//! OpenSlide compatibility-oracle parity test.

mod support;

use std::path::PathBuf;

use support::compare::{compare_rgba, tolerance_failure, Tolerance};
use support::corpus::{load_public, resolve_entry_path, CorpusEntry};
use support::oracles::{
    is_reference_oracle_unsupported, read_probe, top_left_probe, Oracle, ReferenceOracle,
    SigninumOracle,
};

#[test]
#[ignore = "requires public parity corpus; run after scripts/parity-corpus-fetch.sh"]
fn preflight() {
    let strict_corpus = strict_corpus_required();
    let manifest = match load_public() {
        Ok(manifest) => manifest,
        Err(err) => {
            if strict_corpus {
                panic!("[preflight] public manifest unavailable in strict mode: {err}");
            }
            eprintln!("[preflight] public manifest unavailable: {err}; skipping");
            return;
        }
    };

    #[cfg(feature = "parity-openslide")]
    let openslide = openslide_oracle();
    let mut checked = 0u32;
    let mut missing_slides = 0u32;
    let mut unsupported_reference = 0u32;
    let mut failures = Vec::new();

    for entry in &manifest.slides {
        let path: PathBuf = resolve_entry_path(entry);
        if !path.is_file() {
            missing_slides += 1;
            eprintln!(
                "[preflight] {} not present at {}; run scripts/parity-corpus-fetch.sh; skipping",
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

        let baseline = match ReferenceOracle.open(&path) {
            Ok(slide) => slide,
            Err(err) => {
                failures.push(format!("{}: reference open failed: {err}", entry.alias));
                continue;
            }
        };
        let signinum_report = match SigninumOracle.open(&path) {
            Ok(slide) => Some(slide),
            Err(err) => {
                failures.push(format!("{}: signinum open failed: {err}", entry.alias));
                None
            }
        };

        #[cfg(feature = "parity-openslide")]
        match openslide.open(&path) {
            Ok(os_slide) => {
                if os_slide.level_count != baseline.level_count {
                    failures.push(format!(
                        "{}: OpenSlide level count mismatch openslide={} statumen={}",
                        entry.alias, os_slide.level_count, baseline.level_count
                    ));
                }
                for (level, (ours, theirs)) in baseline
                    .level_dimensions
                    .iter()
                    .zip(os_slide.level_dimensions.iter())
                    .enumerate()
                {
                    if ours != theirs {
                        failures.push(format!(
                            "{}: OpenSlide dimension mismatch at level {level}: statumen={ours:?} openslide={theirs:?}",
                            entry.alias
                        ));
                    }
                }
            }
            Err(err) => eprintln!("[preflight] {} OpenSlide open failed: {err}", entry.alias),
        }

        for level in 0..baseline.level_count.min(3) {
            let required = entry.must_decode_level(level);
            let Some(probe) = top_left_probe(&baseline, level) else {
                eprintln!(
                    "[preflight] {} level={level}: no readable probe available; skipping decode",
                    entry.alias
                );
                if required {
                    failures.push(format!(
                        "{} level={level}: required decode has no readable probe",
                        entry.alias
                    ));
                }
                continue;
            };

            let baseline_buf = match read_probe(&baseline, probe) {
                Ok(buf) => Some(buf),
                Err(err) => {
                    eprintln!(
                        "[preflight] {} level={level}: baseline read failed: {err}; skipping",
                        entry.alias
                    );
                    if is_reference_oracle_unsupported(&err) {
                        unsupported_reference += 1;
                        None
                    } else {
                        if required {
                            failures.push(format!(
                                "{} level={level}: required reference read failed: {err}",
                                entry.alias
                            ));
                        }
                        continue;
                    }
                }
            };

            #[cfg(feature = "parity-openslide")]
            let mut signinum_buf = None;
            if let Some(ref signinum) = signinum_report {
                match read_probe(signinum, probe) {
                    Ok(sc_buf) => {
                        if let Some(ref baseline_buf) = baseline_buf {
                            let tolerance = tolerance_for_entry(entry);
                            let report = compare_rgba(
                                &sc_buf.pixels_rgba,
                                &baseline_buf.pixels_rgba,
                                tolerance,
                            );
                            eprintln!(
                                "[preflight] {} level={level} sc-vs-ref report max_abs={} mean_abs={:.4} psnr={:.2}dB equal_rate={:.4} passed={}",
                                entry.alias,
                                report.max_abs,
                                report.mean_abs,
                                report.psnr_db,
                                report.bytewise_equal_rate,
                                report.passed
                            );
                            if required {
                                record_comparison_failure(
                                    entry,
                                    "signinum-vs-reference",
                                    level,
                                    &format!(
                                        "{} level={level}: signinum vs reference",
                                        entry.alias
                                    ),
                                    &report,
                                    &mut failures,
                                );
                            }
                        } else {
                            eprintln!(
                                "[preflight] {} level={level}: reference oracle unsupported; signinum read succeeded without sc-vs-ref comparison",
                                entry.alias
                            );
                        }
                        #[cfg(feature = "parity-openslide")]
                        {
                            signinum_buf = Some(sc_buf.clone());
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "[preflight] {} level={level} signinum report read failed: {err}",
                            entry.alias
                        );
                        if required {
                            failures.push(format!(
                                "{} level={level}: required signinum read failed: {err}",
                                entry.alias
                            ));
                        }
                    }
                }
            }
            checked += 1;

            #[cfg(feature = "parity-openslide")]
            match openslide
                .open(&path)
                .and_then(|opened| read_probe(&opened, probe))
            {
                Ok(os_buf) => {
                    if let Some(ref baseline_buf) = baseline_buf {
                        let report = compare_rgba(
                            &baseline_buf.pixels_rgba,
                            &os_buf.pixels_rgba,
                            Tolerance::TOLERANT,
                        );
                        eprintln!(
                            "[preflight] {} level={level} reference-vs-openslide max_abs={} mean_abs={:.4} psnr={:.2}dB",
                            entry.alias, report.max_abs, report.mean_abs, report.psnr_db
                        );
                        if required {
                            record_comparison_failure(
                                entry,
                                "reference-vs-openslide",
                                level,
                                &format!("{} level={level}: reference vs OpenSlide", entry.alias),
                                &report,
                                &mut failures,
                            );
                        }
                    }
                    if let Some(ref sc_buf) = signinum_buf {
                        let report = compare_rgba(
                            &sc_buf.pixels_rgba,
                            &os_buf.pixels_rgba,
                            Tolerance::TOLERANT,
                        );
                        eprintln!(
                            "[preflight] {} level={level} signinum-vs-openslide max_abs={} mean_abs={:.4} psnr={:.2}dB",
                            entry.alias, report.max_abs, report.mean_abs, report.psnr_db
                        );
                        if required {
                            record_comparison_failure(
                                entry,
                                "signinum-vs-openslide",
                                level,
                                &format!("{} level={level}: signinum vs OpenSlide", entry.alias),
                                &report,
                                &mut failures,
                            );
                        }
                    }
                }
                Err(err) => {
                    eprintln!("[preflight] {} OpenSlide read failed: {err}", entry.alias);
                    if required {
                        failures.push(format!(
                            "{} level={level}: required OpenSlide read failed: {err}",
                            entry.alias
                        ));
                    }
                }
            }
        }
    }

    if missing_slides == 0 && checked == 0 {
        failures.push("preflight decoded zero tiles".to_string());
    }
    eprintln!(
        "[preflight] summary: checked={checked} unsupported_reference={unsupported_reference} missing_slides={missing_slides} failures={}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "preflight failures:\n  {}",
        failures.join("\n  ")
    );
}

fn strict_corpus_required() -> bool {
    true
}

fn record_comparison_failure(
    entry: &CorpusEntry,
    pair: &str,
    level: u32,
    label: &str,
    report: &support::compare::CompareReport,
    failures: &mut Vec<String>,
) {
    let Some(failure) = tolerance_failure(label, report) else {
        return;
    };
    if entry.expected_failure(pair, level) {
        eprintln!("[preflight] expected parity failure: {failure}");
    } else {
        failures.push(failure);
    }
}

fn tolerance_for_entry(entry: &CorpusEntry) -> Tolerance {
    if entry.codecs.iter().any(|codec| codec == "j2k") || entry.format == "leica" {
        Tolerance::TOLERANT
    } else {
        Tolerance::JPEG_TIGHT
    }
}

#[cfg(feature = "parity-openslide")]
fn openslide_oracle() -> support::oracles::OpenSlideOracle {
    let lib = support::openslide_shim::try_load()
        .expect("libopenslide is required when parity-openslide is enabled");
    support::oracles::OpenSlideOracle { lib }
}
