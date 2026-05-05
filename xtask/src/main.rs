use std::env;
use std::ffi::OsString;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_string());
    match task.as_str() {
        "fmt" => fmt(),
        "clippy" => clippy(),
        "test" => test(),
        "parity-corpus-test" => parity_corpus_test(),
        "doc" | "docs" => doc(),
        "typos" => typos(),
        "release-test" => release_test(),
        "coverage" => coverage(),
        "package" => package(),
        "ci" => ci(),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown task `{other}`")),
    }
}

fn ci() -> Result<(), String> {
    fmt()?;
    clippy()?;
    test()?;
    doc()?;
    package()
}

fn fmt() -> Result<(), String> {
    run_cargo(&["fmt", "--all", "--", "--check"])
}

fn clippy() -> Result<(), String> {
    run_cargo(&["clippy", "--all-targets", "--", "-D", "warnings"])
}

fn test() -> Result<(), String> {
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

fn parity_corpus_test() -> Result<(), String> {
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

fn doc() -> Result<(), String> {
    run_cargo_with_env(&["doc", "--no-deps"], &[("RUSTDOCFLAGS", "-D warnings")])
}

fn typos() -> Result<(), String> {
    run_program(OsString::from("typos"), &[], &[])
}

fn release_test() -> Result<(), String> {
    run_cargo(&["test", "--lib", "--tests", "--release"])
}

fn coverage() -> Result<(), String> {
    run_cargo(&[
        "llvm-cov",
        "--lib",
        "--tests",
        "--lcov",
        "--output-path",
        "lcov.info",
    ])
}

fn package() -> Result<(), String> {
    ensure_clean_worktree()?;
    run_cargo(&["package", "--no-verify"])
}

fn ensure_clean_worktree() -> Result<(), String> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|err| format!("failed to start `git status --porcelain`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git status --porcelain` exited with {}",
            output.status
        ));
    }

    let status = String::from_utf8_lossy(&output.stdout);
    if status.trim().is_empty() {
        Ok(())
    } else {
        Err(format!(
            "working tree must be clean before packaging:\n{status}"
        ))
    }
}

fn run_cargo(args: &[&str]) -> Result<(), String> {
    run_cargo_with_env(args, &[])
}

fn run_cargo_with_env(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    run_program(cargo(), args, envs)
}

fn run_program(program: OsString, args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    let display = program.to_string_lossy();
    eprintln!("+ {} {}", display, args.join(" "));
    let mut command = Command::new(&program);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start `{}`: {err}", display))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{}` exited with {status}", display))
    }
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn print_help() {
    println!(
        "usage: cargo xtask <task>\n\n\
         tasks:\n\
           ci           fmt, clippy, tests, docs, and package\n\
           fmt          check rustfmt\n\
           clippy       run clippy with warnings denied\n\
           test         run library and integration tests\n\
           parity-corpus-test run strict corpus-backed ignored integration tests\n\
           doc          build docs with warnings denied\n\
           typos        run typos\n\
           release-test run release-mode library and integration tests\n\
           coverage     generate lcov.info with cargo-llvm-cov\n\
           package      package the crate from a clean worktree without verification"
    );
}
