//! `bench_driver` — runs one named workload in fresh subprocesses for
//! comparable benchmark targets and emits a gate-oriented comparison summary.
//!
//! Usage:
//!   `bench_driver SLIDE_PATH WORKLOAD_NAME`
//!
//! Iris is optional because it is a Python package and consumes pre-encoded
//! `.iris` files. Set `WSI_BENCH_INCLUDE_IRIS=1`; for non-`.iris` source
//! slides also set `WSI_IRIS_SLIDE_PATH=/path/to/encoded.iris` or
//! `WSI_IRIS_SLIDE_DIR=/path/to/encoded-slides`.

#[allow(dead_code)]
mod bench_common;

use bench_common::{workload_spec, SCHEMA_VERSION};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const REPEAT_COUNT: usize = 5;
const MACOS_RSS_METHOD: &str = "macos:/usr/bin/time -l";
const DEBUG_BENCH_OVERRIDE_ENV: &str = "WSI_ALLOW_DEBUG_BENCH";
const PEAK_RSS_GATE_RATIO: f64 = 1.05;
const INCLUDE_IRIS_ENV: &str = "WSI_BENCH_INCLUDE_IRIS";
const GATE_IRIS_ENV: &str = "WSI_BENCH_GATE_IRIS";
const IRIS_SLIDE_PATH_ENV: &str = "WSI_IRIS_SLIDE_PATH";
const IRIS_SLIDE_DIR_ENV: &str = "WSI_IRIS_SLIDE_DIR";
const IRIS_BENCH_SCRIPT_ENV: &str = "WSI_IRIS_BENCH_SCRIPT";
const IRIS_PYTHON_ENV: &str = "WSI_IRIS_PYTHON";

#[derive(Debug)]
struct ChildRunSummary {
    repeat_index: usize,
    p50_us: Option<u64>,
    p99_us: Option<u64>,
    peak_rss_bytes: Option<u64>,
    error: Option<String>,
    exit_code: Option<i32>,
}

#[derive(Debug, Clone)]
struct BenchmarkTarget {
    library: &'static str,
    command: BenchCommand,
    slide_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct GateMetrics<'a> {
    p50_us: Option<u64>,
    p99_us: Option<u64>,
    peak_rss_bytes: Option<u64>,
    status: Option<&'a str>,
}

#[derive(Debug, Clone)]
enum BenchCommand {
    Binary(PathBuf),
    PythonScript { python: PathBuf, script: PathBuf },
}

impl BenchCommand {
    fn display(&self) -> String {
        match self {
            BenchCommand::Binary(path) => path.display().to_string(),
            BenchCommand::PythonScript { python, script } => {
                format!("{} {}", python.display(), script.display())
            }
        }
    }
}

fn main() {
    if let Err(err) = ensure_release_profile("bench_driver") {
        eprintln!("{err}");
        std::process::exit(2);
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: bench_driver <slide-path> <workload-name>");
        std::process::exit(2);
    }

    let slide_path = PathBuf::from(&args[1]);
    if !slide_path.is_file() {
        eprintln!("slide path is not a file: {}", slide_path.display());
        std::process::exit(2);
    }

    let workload = &args[2];
    let spec = match workload_spec(workload) {
        Some(spec) => spec,
        None => {
            eprintln!(
                "invalid workload {:?}; valid workloads: {}",
                workload,
                bench_common::valid_workload_names()
            );
            std::process::exit(2);
        }
    };

    let targets = match benchmark_targets(&slide_path) {
        Ok(targets) => targets,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let mut libraries = Vec::new();
    for target in targets {
        let mut runs = Vec::with_capacity(REPEAT_COUNT);
        for repeat_index in 0..REPEAT_COUNT {
            match run_child(&target.command, &target.slide_path, workload, repeat_index) {
                Ok(run) => runs.push(run),
                Err(err) => {
                    eprintln!("{} repeat {repeat_index} failed: {err}", target.library);
                    std::process::exit(1);
                }
            }
        }

        let median_p50_us = median_u64(runs.iter().filter_map(|run| run.p50_us).collect());
        let median_p99_us = median_u64(runs.iter().filter_map(|run| run.p99_us).collect());
        let median_peak_rss_bytes =
            median_u64(runs.iter().filter_map(|run| run.peak_rss_bytes).collect());
        let had_errors = runs.iter().any(|run| run.error.is_some());

        libraries.push(json!({
            "library": target.library,
            "binary": target.command.display(),
            "slide_path": target.slide_path.display().to_string(),
            "runs": runs.iter().map(|run| {
                json!({
                    "repeat_index": run.repeat_index,
                    "p50_us": run.p50_us,
                    "p99_us": run.p99_us,
                    "peak_rss_bytes": run.peak_rss_bytes,
                    "error": run.error,
                    "exit_code": run.exit_code,
                })
            }).collect::<Vec<_>>(),
            "median_p50_us": median_p50_us,
            "median_p99_us": median_p99_us,
            "median_peak_rss_bytes": median_peak_rss_bytes,
            "status": if had_errors { "error" } else { "ok" },
        }));
    }

    let wsi = library_object(&libraries, "statumen").expect("statumen library object");
    let openslide = library_object(&libraries, "openslide").expect("openslide library object");
    let iris = library_object(&libraries, "iris");
    let gate_iris = env_flag(GATE_IRIS_ENV);
    let failures = gate_failures(
        spec.gate_mode,
        GateMetrics::from_library(wsi),
        GateMetrics::from_library(openslide),
        iris.map(GateMetrics::from_library),
        gate_iris,
    );

    let summary = json!({
        "schema_version": SCHEMA_VERSION,
        "driver": "bench_driver",
        "slide_path": slide_path.display().to_string(),
        "selected_workload": workload,
        "target_repeats": REPEAT_COUNT,
        "gate_mode": spec.gate_mode,
        "comparability": spec.comparability,
        "comparability_note": spec.comparability_note,
        "rss_method": rss_method(),
        "libraries": libraries,
        "comparison": {
            "p50_ratio_wsi_over_openslide": ratio_json(
                value_as_u64(wsi.get("median_p50_us")),
                value_as_u64(openslide.get("median_p50_us")),
            ),
            "p50_ratio_wsi_over_iris": ratio_json(
                value_as_u64(wsi.get("median_p50_us")),
                iris.and_then(|entry| value_as_u64(entry.get("median_p50_us"))),
            ),
            "p99_ratio_wsi_over_openslide": ratio_json(
                value_as_u64(wsi.get("median_p99_us")),
                value_as_u64(openslide.get("median_p99_us")),
            ),
            "p99_ratio_wsi_over_iris": ratio_json(
                value_as_u64(wsi.get("median_p99_us")),
                iris.and_then(|entry| value_as_u64(entry.get("median_p99_us"))),
            ),
            "peak_rss_ratio_wsi_over_openslide": ratio_json(
                value_as_u64(wsi.get("median_peak_rss_bytes")),
                value_as_u64(openslide.get("median_peak_rss_bytes")),
            ),
            "peak_rss_ratio_wsi_over_iris": ratio_json(
                value_as_u64(wsi.get("median_peak_rss_bytes")),
                iris.and_then(|entry| value_as_u64(entry.get("median_peak_rss_bytes"))),
            ),
            "iris_gate_enabled": gate_iris,
            "status": gate_status(spec.gate_mode, &failures),
            "failures": failures,
        }
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("summary json")
    );
}

fn ensure_release_profile(tool: &str) -> Result<(), String> {
    if cfg!(debug_assertions) && std::env::var_os(DEBUG_BENCH_OVERRIDE_ENV).is_none() {
        return Err(format!(
            "{tool} must be run with --release for meaningful perf results; set {DEBUG_BENCH_OVERRIDE_ENV}=1 only for local smoke tests"
        ));
    }
    Ok(())
}

fn sibling_binary(name: &str) -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|err| format!("current_exe failed: {err}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| format!("current_exe has no parent: {}", exe.display()))?;
    let binary_name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let path = dir.join(binary_name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "missing benchmark binary {}; build statumen with --features 'bench openslide-bench'",
            path.display()
        ))
    }
}

fn benchmark_targets(source_slide_path: &Path) -> Result<Vec<BenchmarkTarget>, String> {
    let mut targets = vec![
        BenchmarkTarget {
            library: "statumen",
            command: BenchCommand::Binary(sibling_binary("wsi_bench")?),
            slide_path: source_slide_path.to_path_buf(),
        },
        BenchmarkTarget {
            library: "openslide",
            command: BenchCommand::Binary(sibling_binary("openslide_bench")?),
            slide_path: source_slide_path.to_path_buf(),
        },
    ];

    if env_flag(INCLUDE_IRIS_ENV) {
        let iris_slide_path =
            iris_slide_path_for_source(source_slide_path, std::env::var_os(IRIS_SLIDE_PATH_ENV))?;
        if !iris_slide_path.is_file() {
            return Err(format!(
                "{IRIS_SLIDE_PATH_ENV} is set to a non-file path: {}",
                iris_slide_path.display()
            ));
        }
        let script = std::env::var_os(IRIS_BENCH_SCRIPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(default_iris_bench_script);
        if !script.is_file() {
            return Err(format!(
                "Iris benchmark script not found at {}; set {IRIS_BENCH_SCRIPT_ENV}",
                script.display()
            ));
        }
        let python = std::env::var_os(IRIS_PYTHON_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(default_iris_python);
        targets.push(BenchmarkTarget {
            library: "iris",
            command: BenchCommand::PythonScript { python, script },
            slide_path: iris_slide_path,
        });
    }

    Ok(targets)
}

fn iris_slide_path_for_source(
    source_slide_path: &Path,
    override_path: Option<std::ffi::OsString>,
) -> Result<PathBuf, String> {
    iris_slide_path_for_source_with_dir(
        source_slide_path,
        override_path,
        std::env::var_os(IRIS_SLIDE_DIR_ENV),
    )
}

fn iris_slide_path_for_source_with_dir(
    source_slide_path: &Path,
    override_path: Option<std::ffi::OsString>,
    override_dir: Option<std::ffi::OsString>,
) -> Result<PathBuf, String> {
    if let Some(path) = override_path {
        return Ok(PathBuf::from(path));
    }
    if let Some(dir) = override_dir {
        let stem = source_slide_path.file_stem().ok_or_else(|| {
            format!(
                "cannot derive Iris slide name from source path {}",
                source_slide_path.display()
            )
        })?;
        return Ok(PathBuf::from(dir).join(stem).with_extension("iris"));
    }
    if source_slide_path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("iris"))
    {
        return Ok(source_slide_path.to_path_buf());
    }
    Err(format!(
        "{INCLUDE_IRIS_ENV}=1 requires {IRIS_SLIDE_PATH_ENV} or {IRIS_SLIDE_DIR_ENV} for non-.iris source slides"
    ))
}

fn default_iris_bench_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("iris_bench.py")
}

fn default_iris_python() -> PathBuf {
    let workspace_python = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".venv")
        .join("bin")
        .join("python");
    if workspace_python.is_file() {
        workspace_python
    } else {
        PathBuf::from("python3")
    }
}

fn run_child(
    command: &BenchCommand,
    slide_path: &Path,
    workload: &str,
    repeat_index: usize,
) -> Result<ChildRunSummary, String> {
    let mut child = child_command(command, slide_path);
    let output = child
        .env("WSI_BENCH_ONLY", workload)
        .env("WSI_BENCH_REPEAT_INDEX", repeat_index.to_string())
        .output()
        .map_err(|err| format!("spawn failed: {err}"))?;

    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("child stdout was not utf-8: {err}"))?;
    let stderr = String::from_utf8(output.stderr)
        .map_err(|err| format!("child stderr was not utf-8: {err}"))?;
    let child_json: Value =
        serde_json::from_str(&stdout).map_err(|err| format!("invalid child json: {err}"))?;
    let workloads = child_json["workloads"]
        .as_array()
        .ok_or_else(|| "child json missing workloads array".to_string())?;
    if workloads.len() != 1 {
        return Err(format!(
            "expected one workload in single-workload mode, got {}",
            workloads.len()
        ));
    }

    let workload_json = &workloads[0];
    let workload_name = workload_json["name"]
        .as_str()
        .ok_or_else(|| "child workload missing name".to_string())?;
    if workload_name != workload {
        return Err(format!(
            "expected workload {:?}, child returned {:?}",
            workload, workload_name
        ));
    }

    Ok(ChildRunSummary {
        repeat_index,
        p50_us: workload_json["p50_us"].as_u64(),
        p99_us: workload_json["p99_us"].as_u64(),
        peak_rss_bytes: parse_peak_rss_bytes(&stderr),
        error: workload_json["error"]
            .as_str()
            .map(|value| value.to_string()),
        exit_code: output.status.code(),
    })
}

fn child_command(command: &BenchCommand, slide_path: &Path) -> Command {
    if cfg!(target_os = "macos") {
        let mut child = Command::new("/usr/bin/time");
        child.arg("-l");
        append_command_invocation(&mut child, command);
        child.arg(slide_path);
        return child;
    }

    match command {
        BenchCommand::Binary(binary) => {
            let mut child = Command::new(binary);
            child.arg(slide_path);
            child
        }
        BenchCommand::PythonScript { python, script } => {
            let mut child = Command::new(python);
            child.arg(script).arg(slide_path);
            child
        }
    }
}

fn append_command_invocation(child: &mut Command, command: &BenchCommand) {
    match command {
        BenchCommand::Binary(binary) => {
            child.arg(binary);
        }
        BenchCommand::PythonScript { python, script } => {
            child.arg(python).arg(script);
        }
    }
}

fn parse_peak_rss_bytes(stderr: &str) -> Option<u64> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    stderr.lines().find_map(|line| {
        if line.contains("maximum resident set size") {
            line.split_whitespace().next()?.parse::<u64>().ok()
        } else {
            None
        }
    })
}

fn median_u64(mut values: Vec<u64>) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

fn value_as_u64(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

fn library_object<'a>(
    libraries: &'a [Value],
    name: &str,
) -> Option<&'a serde_json::Map<String, Value>> {
    libraries
        .iter()
        .filter_map(Value::as_object)
        .find(|entry| entry.get("library").and_then(Value::as_str) == Some(name))
}

impl<'a> GateMetrics<'a> {
    fn from_library(library: &'a serde_json::Map<String, Value>) -> Self {
        Self {
            p50_us: value_as_u64(library.get("median_p50_us")),
            p99_us: value_as_u64(library.get("median_p99_us")),
            peak_rss_bytes: value_as_u64(library.get("median_peak_rss_bytes")),
            status: library.get("status").and_then(Value::as_str),
        }
    }
}

fn ratio_json(lhs: Option<u64>, rhs: Option<u64>) -> Value {
    match (lhs, rhs) {
        (_, Some(0)) | (None, _) | (_, None) => Value::Null,
        (Some(lhs), Some(rhs)) => json!(lhs as f64 / rhs as f64),
    }
}

fn rss_method() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some(MACOS_RSS_METHOD)
    } else {
        None
    }
}

fn gate_status(gate_mode: &str, failures: &[String]) -> &'static str {
    if gate_mode != "gating" {
        "informational"
    } else if failures.is_empty() {
        "pass"
    } else {
        "fail"
    }
}

fn gate_failures(
    gate_mode: &str,
    wsi: GateMetrics<'_>,
    openslide: GateMetrics<'_>,
    iris: Option<GateMetrics<'_>>,
    gate_iris: bool,
) -> Vec<String> {
    if gate_mode != "gating" {
        return Vec::new();
    }

    let mut failures = Vec::new();
    if wsi.status != Some("ok") {
        failures.push("statumen child runs reported errors".to_string());
    }
    if openslide.status != Some("ok") {
        failures.push("openslide child runs reported errors".to_string());
    }
    if gate_iris && iris.and_then(|metrics| metrics.status) != Some("ok") {
        failures.push("iris child runs reported errors".to_string());
    }

    compare_metric(&mut failures, "p50", wsi.p50_us, openslide.p50_us);
    compare_metric(&mut failures, "p99", wsi.p99_us, openslide.p99_us);
    compare_rss_metric(&mut failures, wsi.peak_rss_bytes, openslide.peak_rss_bytes);
    if gate_iris {
        let iris = iris.unwrap_or(GateMetrics {
            p50_us: None,
            p99_us: None,
            peak_rss_bytes: None,
            status: None,
        });
        compare_metric(&mut failures, "iris p50", wsi.p50_us, iris.p50_us);
        compare_metric(&mut failures, "iris p99", wsi.p99_us, iris.p99_us);
        compare_rss_metric_named(
            &mut failures,
            "iris peak_rss_bytes",
            wsi.peak_rss_bytes,
            iris.peak_rss_bytes,
        );
    }
    failures
}

fn compare_metric(
    failures: &mut Vec<String>,
    label: &str,
    wsi: Option<u64>,
    openslide: Option<u64>,
) {
    match (wsi, openslide) {
        (Some(wsi), Some(openslide)) if wsi <= openslide => {}
        (Some(wsi), Some(openslide)) => failures.push(format!(
            "{label} gate failed: statumen {wsi} > openslide {openslide}"
        )),
        _ => failures.push(format!("{label} gate missing comparable values")),
    }
}

fn compare_rss_metric(failures: &mut Vec<String>, wsi: Option<u64>, openslide: Option<u64>) {
    compare_rss_metric_named(failures, "peak_rss_bytes", wsi, openslide);
}

fn compare_rss_metric_named(
    failures: &mut Vec<String>,
    label: &str,
    wsi: Option<u64>,
    baseline: Option<u64>,
) {
    match (wsi, baseline) {
        (Some(wsi), Some(baseline)) if baseline > 0 => {
            let ratio = wsi as f64 / baseline as f64;
            if ratio <= PEAK_RSS_GATE_RATIO {
                return;
            }
            failures.push(format!(
                "{label} gate failed: statumen {wsi} > {PEAK_RSS_GATE_RATIO:.2}x baseline {baseline} (ratio {ratio:.3}x)"
            ));
        }
        _ => failures.push(format!("{label} gate missing comparable values")),
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var_os(name)
        .and_then(|value| value.into_string().ok())
        .is_some_and(|value| env_flag_value(&value))
}

fn env_flag_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(
        p50_us: Option<u64>,
        p99_us: Option<u64>,
        peak_rss_bytes: Option<u64>,
        status: Option<&str>,
    ) -> GateMetrics<'_> {
        GateMetrics {
            p50_us,
            p99_us,
            peak_rss_bytes,
            status,
        }
    }

    #[test]
    fn parses_peak_rss_from_macos_time_output() {
        let stderr = "        344064  maximum resident set size\n";
        let expected = if cfg!(target_os = "macos") {
            Some(344064)
        } else {
            None
        };
        assert_eq!(parse_peak_rss_bytes(stderr), expected);
    }

    #[test]
    fn median_uses_sorted_middle_value() {
        assert_eq!(median_u64(vec![5, 1, 3]), Some(3));
        assert_eq!(median_u64(Vec::new()), None);
    }

    #[test]
    fn rss_gate_allows_up_to_five_percent_over_openslide() {
        let failures = gate_failures(
            "gating",
            metrics(Some(10), Some(20), Some(105), Some("ok")),
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            None,
            false,
        );
        assert!(failures.is_empty(), "unexpected failures: {failures:?}");
    }

    #[test]
    fn rss_gate_fails_above_five_percent_over_openslide() {
        let failures = gate_failures(
            "gating",
            metrics(Some(10), Some(20), Some(106), Some("ok")),
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            None,
            false,
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("peak_rss_bytes")),
            "expected RSS failure, got {failures:?}"
        );
    }

    #[test]
    fn latency_metrics_remain_exact_gates() {
        let failures = gate_failures(
            "gating",
            metrics(Some(11), Some(20), Some(100), Some("ok")),
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            None,
            false,
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("p50 gate failed")),
            "expected exact p50 failure, got {failures:?}"
        );
    }

    #[test]
    fn iris_gate_is_optional_but_enforced_when_enabled() {
        let without_iris_gate = gate_failures(
            "gating",
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            Some(metrics(Some(9), Some(19), Some(100), Some("ok"))),
            false,
        );
        assert!(without_iris_gate.is_empty());

        let with_iris_gate = gate_failures(
            "gating",
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            metrics(Some(10), Some(20), Some(100), Some("ok")),
            Some(metrics(Some(9), Some(19), Some(100), Some("ok"))),
            true,
        );
        assert!(
            with_iris_gate
                .iter()
                .any(|failure| failure.contains("iris p50")),
            "expected Iris p50 failure, got {with_iris_gate:?}"
        );
    }

    #[test]
    fn iris_source_path_requires_override_for_vendor_slide() {
        let err = iris_slide_path_for_source(Path::new("/tmp/a.svs"), None).unwrap_err();
        assert!(err.contains(IRIS_SLIDE_PATH_ENV));
    }

    #[test]
    fn iris_source_path_accepts_iris_slide_without_override() {
        let path = iris_slide_path_for_source(Path::new("/tmp/a.iris"), None).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/a.iris"));
    }

    #[test]
    fn iris_source_path_uses_override() {
        let path = iris_slide_path_for_source_with_dir(
            Path::new("/tmp/a.svs"),
            Some(std::ffi::OsString::from("/tmp/a.iris")),
            None,
        )
        .unwrap();
        assert_eq!(path, PathBuf::from("/tmp/a.iris"));
    }

    #[test]
    fn iris_source_path_derives_from_override_dir() {
        let path = iris_slide_path_for_source_with_dir(
            Path::new("/tmp/source/a.svs"),
            None,
            Some(std::ffi::OsString::from("/tmp/iris")),
        )
        .unwrap();
        assert_eq!(path, PathBuf::from("/tmp/iris/a.iris"));
    }

    #[test]
    fn env_flag_value_accepts_common_truthy_values() {
        assert!(env_flag_value("1"));
        assert!(env_flag_value("true"));
        assert!(env_flag_value("YES"));
        assert!(env_flag_value("on"));
        assert!(!env_flag_value("0"));
        assert!(!env_flag_value("false"));
    }
}
