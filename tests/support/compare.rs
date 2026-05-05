//! Tolerance metrics and diff dumps for parity comparisons.

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub max_abs: u8,
    pub mean_abs: f64,
}

impl Tolerance {
    pub const JPEG_TIGHT: Self = Self {
        max_abs: 1,
        mean_abs: 0.05,
    };

    pub const TOLERANT: Self = Self {
        max_abs: 4,
        mean_abs: 1.0,
    };
}

#[derive(Debug, Clone)]
pub struct CompareReport {
    pub bytewise_equal_rate: f64,
    pub max_abs: u8,
    pub mean_abs: f64,
    pub psnr_db: f64,
    pub passed: bool,
    pub diff_dump: Option<PathBuf>,
}

pub fn compare_rgba(actual: &[u8], expected: &[u8], tol: Tolerance) -> CompareReport {
    assert_eq!(
        actual.len(),
        expected.len(),
        "compare_rgba: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    assert!(
        actual.len().is_multiple_of(4),
        "compare_rgba: not RGBA-aligned"
    );

    if actual.is_empty() {
        return CompareReport {
            bytewise_equal_rate: 1.0,
            max_abs: 0,
            mean_abs: 0.0,
            psnr_db: f64::INFINITY,
            passed: true,
            diff_dump: None,
        };
    }

    let mut equal = 0u64;
    let mut max_abs = 0u8;
    let mut sum_abs = 0u64;
    let mut sum_sq = 0u64;
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        if actual == expected {
            equal += 1;
        }
        let d = actual.abs_diff(*expected);
        max_abs = max_abs.max(d);
        sum_abs += u64::from(d);
        sum_sq += u64::from(d) * u64::from(d);
    }

    let n = actual.len() as f64;
    let mean_abs = sum_abs as f64 / n;
    let mse = sum_sq as f64 / n;
    let psnr_db = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    let passed = max_abs <= tol.max_abs && mean_abs <= tol.mean_abs;

    CompareReport {
        bytewise_equal_rate: equal as f64 / n,
        max_abs,
        mean_abs,
        psnr_db,
        passed,
        diff_dump: None,
    }
}

pub fn tolerance_failure(label: &str, report: &CompareReport) -> Option<String> {
    if report.passed {
        return None;
    }
    Some(format!(
        "{label}: exceeds tolerance (max_abs={} mean_abs={:.4} psnr={:.2}dB equal_rate={:.4})",
        report.max_abs, report.mean_abs, report.psnr_db, report.bytewise_equal_rate
    ))
}
