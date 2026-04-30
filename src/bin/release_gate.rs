//! `release_gate` — runs parity + bench_driver across the frozen V1 corpus and
//! writes canonical scoreboard artifacts.
//!
//! Usage:
//!   release_gate [manifest-path]

#[allow(dead_code)]
mod bench_common;

use bench_common::workload_specs;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_MANIFEST: &str = "perf/v1-corpus.json";
const SCOREBOARD_JSON: &str = "perf/release-scoreboard.json";
const SCOREBOARD_MD: &str = "perf/release-scoreboard.md";
const RAW_DIR: &str = "perf/raw/release-gate";
const DEBUG_BENCH_OVERRIDE_ENV: &str = "WSI_ALLOW_DEBUG_BENCH";

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    version: u32,
    slides: Vec<CorpusSlide>,
}

#[derive(Debug, Deserialize)]
struct CorpusSlide {
    id: String,
    format: String,
    path: String,
}

#[derive(Debug, Serialize)]
struct ParitySummary {
    status: String,
    command: Vec<String>,
    output_path: String,
}

#[derive(Debug, Serialize)]
struct WorkloadSummary {
    name: String,
    status: String,
    output_path: String,
    summary: Value,
}

#[derive(Debug, Serialize)]
struct SlideSummary {
    id: String,
    format: String,
    resolved_path: String,
    parity: ParitySummary,
    workloads: Vec<WorkloadSummary>,
}

#[derive(Debug, Serialize)]
struct ReleaseSummary {
    version: u32,
    manifest_path: String,
    overall_status: String,
    slides: Vec<SlideSummary>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    ensure_release_profile("release_gate")?;

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 2 {
        return Err("usage: release_gate [manifest-path]".into());
    }

    let workspace_root = workspace_root()?;
    let manifest_path = if let Some(arg) = args.get(1) {
        workspace_root.join(arg)
    } else {
        workspace_root.join(DEFAULT_MANIFEST)
    };
    let manifest: CorpusManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?,
    )
    .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;

    let raw_dir = workspace_root.join(RAW_DIR);
    fs::create_dir_all(&raw_dir)
        .map_err(|err| format!("failed to create {}: {err}", raw_dir.display()))?;

    let bench_driver = ensure_sibling_binary(&workspace_root, "bench_driver")?;
    let mut slides = Vec::with_capacity(manifest.slides.len());
    let mut overall_ok = true;

    for slide in &manifest.slides {
        let resolved = resolve_manifest_path(&workspace_root, &slide.path)?;
        let parity_output_path = raw_dir.join(format!("{}-parity.txt", slide.id));
        let parity = run_parity_check(&workspace_root, &resolved, &parity_output_path)?;
        overall_ok &= parity.status == "pass";

        let mut workloads = Vec::with_capacity(workload_specs().len());
        for spec in workload_specs() {
            let output_path = raw_dir.join(format!("{}-{}.json", slide.id, spec.name));
            let summary = run_bench_driver(&bench_driver, &resolved, spec.name, &output_path)?;
            overall_ok &= summary.status == "pass";
            workloads.push(summary);
        }

        slides.push(SlideSummary {
            id: slide.id.clone(),
            format: slide.format.clone(),
            resolved_path: resolved.display().to_string(),
            parity,
            workloads,
        });
    }

    let summary = ReleaseSummary {
        version: manifest.version,
        manifest_path: manifest_path.display().to_string(),
        overall_status: if overall_ok { "pass" } else { "fail" }.into(),
        slides,
    };

    let json_path = workspace_root.join(SCOREBOARD_JSON);
    let md_path = workspace_root.join(SCOREBOARD_MD);
    fs::write(
        &json_path,
        serde_json::to_vec_pretty(&summary).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("failed to write {}: {err}", json_path.display()))?;
    fs::write(&md_path, render_markdown(&summary))
        .map_err(|err| format!("failed to write {}: {err}", md_path.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|err| err.to_string())?
    );
    if overall_ok {
        Ok(())
    } else {
        Err("release gate failed".into())
    }
}

fn ensure_release_profile(tool: &str) -> Result<(), String> {
    if cfg!(debug_assertions) && std::env::var_os(DEBUG_BENCH_OVERRIDE_ENV).is_none() {
        return Err(format!(
            "{tool} must be run with --release for meaningful perf results; set {DEBUG_BENCH_OVERRIDE_ENV}=1 only for local smoke tests"
        ));
    }
    Ok(())
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .map_err(|err| format!("failed to resolve workspace root: {err}"))
}

fn ensure_sibling_binary(workspace_root: &Path, name: &str) -> Result<PathBuf, String> {
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
        return Ok(path);
    }

    let status = Command::new("cargo")
        .current_dir(workspace_root)
        .args({
            let mut args = vec![
                "build",
                "-p",
                "ziggurat",
                "--features",
                "bench openslide-bench",
            ];
            if !cfg!(debug_assertions) {
                args.push("--release");
            }
            args.push("--bin");
            args.push(name);
            args
        })
        .status()
        .map_err(|err| format!("failed to build {name}: {err}"))?;
    if !status.success() {
        return Err(format!("cargo build failed for {name}"));
    }
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!("missing benchmark binary {}", path.display()))
    }
}

fn resolve_manifest_path(workspace_root: &Path, raw: &str) -> Result<PathBuf, String> {
    let path = if let Some(downloads_rel) = raw.strip_prefix("downloads/") {
        let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
        PathBuf::from(home).join("Downloads").join(downloads_rel)
    } else {
        workspace_root.join(raw)
    };
    let canonical = fs::canonicalize(&path)
        .map_err(|err| format!("failed to resolve {}: {err}", path.display()))?;
    if !canonical.is_file() {
        return Err(format!(
            "resolved path is not a file: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn run_parity_check(
    workspace_root: &Path,
    slide_path: &Path,
    output_path: &Path,
) -> Result<ParitySummary, String> {
    let joined = std::env::join_paths([slide_path])
        .map_err(|err| format!("failed to join parity path list: {err}"))?;
    let output = Command::new("cargo")
        .current_dir(workspace_root)
        .args([
            "test",
            "-p",
            "ziggurat",
            "--test",
            "openslide_compare",
            "--",
            "--nocapture",
        ])
        .env("ZIGGURAT_OPENSLIDE_COMPARE_PATHS", joined)
        .output()
        .map_err(|err| format!("failed to run openslide_compare: {err}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut combined = String::new();
    if !stdout.is_empty() {
        combined.push_str(&stdout);
        if !stderr.is_empty() {
            combined.push('\n');
        }
    }
    if !stderr.is_empty() {
        combined.push_str(&stderr);
    }
    fs::write(output_path, combined)
        .map_err(|err| format!("failed to write {}: {err}", output_path.display()))?;

    Ok(ParitySummary {
        status: if output.status.success() {
            "pass"
        } else {
            "fail"
        }
        .into(),
        command: vec![
            "cargo".into(),
            "test".into(),
            "-p".into(),
            "ziggurat".into(),
            "--test".into(),
            "openslide_compare".into(),
            "--".into(),
            "--nocapture".into(),
        ],
        output_path: output_path.display().to_string(),
    })
}

fn run_bench_driver(
    bench_driver: &Path,
    slide_path: &Path,
    workload: &str,
    output_path: &Path,
) -> Result<WorkloadSummary, String> {
    let output = Command::new(bench_driver)
        .arg(slide_path)
        .arg(workload)
        .output()
        .map_err(|err| format!("failed to run bench_driver for {workload}: {err}"))?;
    fs::write(output_path, &output.stdout)
        .map_err(|err| format!("failed to write {}: {err}", output_path.display()))?;

    let summary: Value = serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("invalid bench_driver json for {workload}: {err}"))?;
    let status = summary["comparison"]["status"]
        .as_str()
        .unwrap_or(if output.status.success() {
            "pass"
        } else {
            "fail"
        })
        .to_string();

    Ok(WorkloadSummary {
        name: workload.to_string(),
        status,
        output_path: output_path.display().to_string(),
        summary,
    })
}

fn render_markdown(summary: &ReleaseSummary) -> String {
    let mut out = String::new();
    out.push_str("# Release Scoreboard\n\n");
    out.push_str(&format!(
        "Status: **{}**\n\n",
        summary.overall_status.to_uppercase()
    ));
    out.push_str(&format!("Manifest: `{}`\n\n", summary.manifest_path));

    for slide in &summary.slides {
        out.push_str(&format!("## {} ({})\n\n", slide.id, slide.format));
        out.push_str(&format!("- Path: `{}`\n", slide.resolved_path));
        out.push_str(&format!(
            "- Parity: **{}** (`{}`)\n\n",
            slide.parity.status.to_uppercase(),
            slide.parity.output_path
        ));
        out.push_str("| workload | status | ziggurat p50 | OpenSlide p50 | Iris p50 | ziggurat p99 | OpenSlide p99 | Iris p99 | OpenSlide RSS ratio | Iris RSS ratio |\n");
        out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for workload in &slide.workloads {
            let wsi = library_summary(&workload.summary, "ziggurat");
            let openslide = library_summary(&workload.summary, "openslide");
            let iris = library_summary(&workload.summary, "iris");
            let openslide_rss_ratio = ratio_string(
                wsi.and_then(|lib| lib.get("median_peak_rss_bytes"))
                    .and_then(Value::as_u64),
                openslide
                    .and_then(|lib| lib.get("median_peak_rss_bytes"))
                    .and_then(Value::as_u64),
            );
            let iris_rss_ratio = ratio_string(
                wsi.and_then(|lib| lib.get("median_peak_rss_bytes"))
                    .and_then(Value::as_u64),
                iris.and_then(|lib| lib.get("median_peak_rss_bytes"))
                    .and_then(Value::as_u64),
            );
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                workload.name,
                workload.status,
                value_string(
                    wsi.and_then(|lib| lib.get("median_p50_us"))
                        .and_then(Value::as_u64)
                ),
                value_string(
                    openslide
                        .and_then(|lib| lib.get("median_p50_us"))
                        .and_then(Value::as_u64)
                ),
                value_string(
                    iris.and_then(|lib| lib.get("median_p50_us"))
                        .and_then(Value::as_u64)
                ),
                value_string(
                    wsi.and_then(|lib| lib.get("median_p99_us"))
                        .and_then(Value::as_u64)
                ),
                value_string(
                    openslide
                        .and_then(|lib| lib.get("median_p99_us"))
                        .and_then(Value::as_u64)
                ),
                value_string(
                    iris.and_then(|lib| lib.get("median_p99_us"))
                        .and_then(Value::as_u64)
                ),
                openslide_rss_ratio,
                iris_rss_ratio,
            ));
        }
        out.push('\n');
    }

    out
}

fn library_summary<'a>(summary: &'a Value, name: &str) -> Option<&'a Value> {
    summary["libraries"]
        .as_array()?
        .iter()
        .find(|entry| entry["library"].as_str() == Some(name))
}

fn value_string(value: Option<u64>) -> String {
    value
        .map(|v| format!("{v}us"))
        .unwrap_or_else(|| "-".into())
}

fn ratio_string(left: Option<u64>, right: Option<u64>) -> String {
    match (left, right) {
        (Some(left), Some(right)) if right > 0 => format!("{:.2}x", left as f64 / right as f64),
        _ => "-".into(),
    }
}
