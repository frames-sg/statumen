//! Ashlar CPU vs reference parity harness.

mod support;

use support::compare::{compare_rgba, tolerance_failure, Tolerance};
use support::corpus::{load_public, resolve_entry_path, CorpusEntry};
use support::oracles::{
    is_reference_oracle_unsupported, read_probe, top_left_probe, AshlarOracle, Oracle,
    ReferenceOracle,
};

#[test]
fn ashlar_cpu_vs_reference_within_tolerance() {
    let strict_corpus = strict_corpus_required();
    let manifest = match load_public() {
        Ok(manifest) => manifest,
        Err(err) => {
            if strict_corpus {
                panic!("[sc-parity] public manifest unavailable in strict mode: {err}");
            }
            eprintln!("[sc-parity] manifest unavailable: {err}; skipping");
            return;
        }
    };
    let mut checked = 0u32;
    let mut missing_slides = 0u32;
    let mut unsupported_reference = 0u32;
    let mut failures = Vec::new();

    for entry in &manifest.slides {
        let path = resolve_entry_path(entry);
        if !path.is_file() {
            missing_slides += 1;
            eprintln!(
                "[sc-parity] {} missing at {}; skipping",
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

        let sc = match AshlarOracle.open(&path) {
            Ok(slide) => slide,
            Err(err) => {
                failures.push(format!("{}: open ashlar: {err}", entry.alias));
                continue;
            }
        };
        let reference = match ReferenceOracle.open(&path) {
            Ok(slide) => slide,
            Err(err) => {
                failures.push(format!("{}: open reference: {err}", entry.alias));
                continue;
            }
        };

        for level in 0..sc.level_count.min(reference.level_count).min(3) {
            let required = entry.must_decode_level(level);
            let Some(probe) = top_left_probe(&reference, level) else {
                if required {
                    failures.push(format!(
                        "{} level={level}: required decode has no readable probe",
                        entry.alias
                    ));
                }
                continue;
            };
            let sc_buf = match read_probe(&sc, probe) {
                Ok(buf) => buf,
                Err(err) => {
                    eprintln!(
                        "[sc-parity] {} level={level}: read ashlar failed: {err}; skipping",
                        entry.alias
                    );
                    if required {
                        failures.push(format!(
                            "{} level={level}: required ashlar read failed: {err}",
                            entry.alias
                        ));
                    }
                    continue;
                }
            };
            let ref_buf = match read_probe(&reference, probe) {
                Ok(buf) => buf,
                Err(err) => {
                    eprintln!(
                        "[sc-parity] {} level={level}: read reference failed: {err}; skipping",
                        entry.alias
                    );
                    if is_reference_oracle_unsupported(&err) {
                        unsupported_reference += 1;
                    } else if required {
                        failures.push(format!(
                            "{} level={level}: required reference read failed: {err}",
                            entry.alias
                        ));
                    }
                    continue;
                }
            };
            let tolerance = tolerance_for_entry(entry);
            let cmp = compare_rgba(&sc_buf.pixels_rgba, &ref_buf.pixels_rgba, tolerance);
            eprintln!(
                "[sc-parity] {} level={level}: max_abs={} mean_abs={:.4} passed={}",
                entry.alias, cmp.max_abs, cmp.mean_abs, cmp.passed
            );
            if required {
                record_comparison_failure(
                    entry,
                    "ashlar-vs-reference",
                    level,
                    &format!("{} level={level}: ashlar vs reference", entry.alias),
                    &cmp,
                    &mut failures,
                );
            }
            checked += 1;
        }
    }

    if missing_slides == 0 && checked == 0 {
        failures
            .push("ashlar parity decoded zero independently reference-backed tiles".to_string());
    }
    eprintln!(
        "[sc-parity] checked={checked} unsupported_reference={unsupported_reference} missing_slides={missing_slides} failures={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn strict_corpus_required() -> bool {
    std::env::var_os("ZIGGURAT_PARITY_REQUIRE_CORPUS").is_some() || std::env::var_os("CI").is_some()
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
        eprintln!("[sc-parity] expected parity failure: {failure}");
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

#[cfg(feature = "parity-metal")]
#[test]
fn ashlar_metal_vs_cpu_within_tolerance() {
    eprintln!(
        "[sc-parity-metal] Phase 0 stub: no production Metal split yet; harness wired for Phase 5"
    );
}
