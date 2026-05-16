mod checks;
mod process;

use std::env;

pub(crate) fn run() -> Result<(), String> {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_string());
    match task.as_str() {
        "fmt" => checks::fmt(),
        "clippy" => checks::clippy(),
        "test" => checks::test(),
        "nextest" => checks::nextest(),
        "bench-check" => checks::bench_check(),
        "bench" => checks::bench(),
        "feature-check" => checks::feature_check(),
        "parity-corpus-test" => checks::parity_corpus_test(),
        "doc" | "docs" => checks::doc(),
        "typos" => checks::typos(),
        "deny" => checks::deny(),
        "unused-deps" => checks::unused_deps(),
        "deps" => checks::deps(),
        "release-test" => checks::release_test(),
        "coverage" => checks::coverage(),
        "package" => checks::package(),
        "validate" => checks::validate(),
        "ci" => checks::ci(),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown task `{other}`")),
    }
}

fn print_help() {
    println!("{}", help_text());
}

fn help_text() -> &'static str {
    "usage: cargo xtask <task>\n\n\
     tasks:\n\
       ci           validate plus package\n\
       validate     fmt, clippy, bench-check, nextest, and docs\n\
       fmt          check rustfmt\n\
       clippy       run clippy with warnings denied\n\
       test         run library and integration tests\n\
       nextest      run library and integration tests with cargo-nextest\n\
       bench-check  compile Rust benchmark targets without running timings\n\
       bench        run synthetic Rust Criterion benchmarks locally\n\
       feature-check check supported feature combinations with cargo-hack\n\
       parity-corpus-test run strict corpus-backed ignored integration tests\n\
       doc          build docs with warnings denied\n\
       typos        run typos\n\
       deny         run cargo-deny advisories, bans, licenses, and sources checks\n\
       unused-deps  run cargo-machete for unused dependency checks\n\
       deps         run deny and unused-deps\n\
       release-test run release-mode library and integration tests\n\
       coverage     generate lcov.info with cargo-llvm-cov\n\
       package      package the crate from a clean worktree without verification"
}

#[cfg(test)]
mod tests {
    use super::help_text;

    #[test]
    fn help_lists_benchmark_tasks() {
        let help = help_text();

        assert!(help.contains("bench-check"));
        assert!(help.contains("bench        "));
    }
}
