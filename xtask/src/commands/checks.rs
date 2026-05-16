use std::ffi::OsString;

use super::process::{ensure_clean_worktree, run_cargo, run_cargo_with_env, run_program};

pub(super) fn ci() -> Result<(), String> {
    validate()?;
    package()
}

pub(super) fn validate() -> Result<(), String> {
    fmt()?;
    clippy()?;
    bench_check()?;
    nextest()?;
    doc()
}

pub(super) fn fmt() -> Result<(), String> {
    run_cargo(&["fmt", "--all", "--", "--check"])
}

pub(super) fn clippy() -> Result<(), String> {
    run_cargo(&["clippy", "--all-targets", "--", "-D", "warnings"])
}

pub(super) fn test() -> Result<(), String> {
    run_cargo(&["test", "--lib", "--tests"])?;
    run_cargo(&["test", "--lib", "--tests", "--features", "parity-openslide"])?;
    if cfg!(target_os = "macos") {
        run_cargo(&[
            "test",
            "--lib",
            "--tests",
            "--features",
            "metal",
            "--no-run",
        ])?;
    }
    Ok(())
}

pub(super) fn nextest() -> Result<(), String> {
    run_cargo(&["nextest", "run", "--lib", "--tests"])?;
    run_cargo(&[
        "nextest",
        "run",
        "--lib",
        "--tests",
        "--features",
        "parity-openslide",
    ])?;
    if cfg!(target_os = "macos") {
        run_cargo(&[
            "test",
            "--lib",
            "--tests",
            "--features",
            "metal",
            "--no-run",
        ])?;
    }
    Ok(())
}

pub(super) fn bench_check() -> Result<(), String> {
    run_cargo(&["bench", "--benches", "--no-run"])?;
    run_cargo(&[
        "bench",
        "--benches",
        "--features",
        "parity-openslide",
        "--no-run",
    ])
}

pub(super) fn bench() -> Result<(), String> {
    run_cargo(&[
        "bench",
        "--bench",
        "read_paths",
        "--",
        "synthetic_read_paths",
        "--sample-size",
        "10",
        "--warm-up-time",
        "1",
        "--measurement-time",
        "1",
    ])
}

pub(super) fn feature_check() -> Result<(), String> {
    run_cargo(&[
        "hack",
        "check",
        "--workspace",
        "--all-targets",
        "--feature-powerset",
        "--exclude-features",
        "openslide-bench,metal,parity-metal,cuda",
    ])
}

pub(super) fn parity_corpus_test() -> Result<(), String> {
    run_cargo(&[
        "test",
        "--test",
        "openslide_parity",
        "preflight",
        "--",
        "--exact",
        "--ignored",
    ])?;
    run_cargo(&[
        "test",
        "--test",
        "signinum_parity",
        "signinum_cpu_vs_reference_within_tolerance",
        "--",
        "--exact",
        "--ignored",
    ])?;
    run_cargo(&[
        "test",
        "--test",
        "dicom_parity",
        "dicom_public_corpus_decodes_with_statumen",
        "--",
        "--exact",
        "--ignored",
    ])?;
    run_cargo(&["test", "--test", "real_wsi_behavior", "--", "--ignored"])
}

pub(super) fn doc() -> Result<(), String> {
    run_cargo_with_env(&["doc", "--no-deps"], &[("RUSTDOCFLAGS", "-D warnings")])
}

pub(super) fn typos() -> Result<(), String> {
    run_program(OsString::from("typos"), &[], &[])
}

pub(super) fn deny() -> Result<(), String> {
    run_cargo(&["deny", "check", "advisories", "bans", "licenses", "sources"])
}

pub(super) fn unused_deps() -> Result<(), String> {
    run_program(OsString::from("cargo-machete"), &["."], &[])
}

pub(super) fn deps() -> Result<(), String> {
    deny()?;
    unused_deps()
}

pub(super) fn release_test() -> Result<(), String> {
    run_cargo(&["test", "--lib", "--tests", "--release"])
}

pub(super) fn coverage() -> Result<(), String> {
    run_cargo(&[
        "llvm-cov",
        "--lib",
        "--tests",
        "--lcov",
        "--output-path",
        "lcov.info",
    ])
}

pub(super) fn package() -> Result<(), String> {
    ensure_clean_worktree()?;
    run_cargo(&["package", "--no-verify"])
}
