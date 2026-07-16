// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden byte-parity tests for the accuracy CSV sink.
//!
//! The expected bytes are generated from the real Python exporter's formatting
//! (`csv.writer` excel dialect, CRLF terminator, `f"{ratio:.4f}"`), so a passing
//! test pins byte-identity with `accuracy/accuracy_data_exporter.py`.

use std::collections::BTreeMap;

use super::*;
use crate::metrics_core::{
    AccumulatorSummary, AccuracyAnalysis, AccuracyRollup, AccuracySummary, ConfidenceInterval,
    NativeReport, TaskId,
};

/// Build a rollup with the fields the CSV consumes; ancillary stats are filled
/// with plausible values the sink never reads.
fn rollup(n: usize, correct: usize, unparsed: usize) -> AccuracyRollup {
    let accuracy = (n > 0).then(|| correct as f64 / n as f64);
    let unparsed_rate = (n > 0).then(|| unparsed as f64 / n as f64);
    AccuracyRollup {
        n,
        correct_count: correct,
        unparsed_count: unparsed,
        accuracy,
        unparsed_rate,
        mean_confidence: None,
        ci: Some(ConfidenceInterval {
            low: 0.0,
            high: 1.0,
        }),
    }
}

fn report_with(summary: AccuracySummary) -> NativeReport {
    let metrics = AccumulatorSummary::new();
    let analysis = AccuracyAnalysis {
        summary,
        accuracy_at_load: None,
        correct_answers_per_kwh: None,
    };
    NativeReport::new(&metrics, Some(analysis))
}

fn export_to_temp(report: &NativeReport) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = ExportConfig {
        accuracy_csv: AccuracyCsvExportConfig { enabled: true },
        ..ExportConfig::default()
    };
    AccuracyCsvExporter
        .export(report, dir.path(), &cfg)
        .expect("export");
    let path = dir.path().join(ACCURACY_CSV_FILE);
    (dir, path)
}

#[test]
fn overall_and_sorted_tasks_match_python_bytes() {
    // overall n=10 correct=7 unparsed=2; tasks inserted out of order to prove the
    // BTreeMap yields the alphabetical order Python's sorted() produces.
    let mut per_task = BTreeMap::new();
    per_task.insert(TaskId::new("math"), rollup(6, 5, 1));
    per_task.insert(TaskId::new("code"), rollup(4, 2, 1));
    let summary = AccuracySummary {
        overall: rollup(10, 7, 2),
        per_task,
    };
    let report = report_with(summary);
    let (_dir, path) = export_to_temp(&report);
    let bytes = std::fs::read(&path).expect("read csv");

    // Golden generated from the Python exporter formatting over identical values:
    //   csv.writer(["task","correct","total","unparsed","accuracy"]) + rows,
    //   f"{7/10:.4f}"=0.7000, f"{2/4:.4f}"=0.5000, f"{5/6:.4f}"=0.8333, CRLF.
    let expected = "task,correct,total,unparsed,accuracy\r\n\
         OVERALL,7,10,2,0.7000\r\n\
         code,2,4,1,0.5000\r\n\
         math,5,6,1,0.8333\r\n";
    assert_eq!(String::from_utf8(bytes).unwrap(), expected);
}

#[test]
fn overall_only_no_tasks() {
    let summary = AccuracySummary {
        overall: rollup(3, 1, 0),
        per_task: BTreeMap::new(),
    };
    let report = report_with(summary);
    let (_dir, path) = export_to_temp(&report);
    let bytes = std::fs::read(&path).expect("read csv");

    // f"{1/3:.4f}" == "0.3333".
    let expected = "task,correct,total,unparsed,accuracy\r\n\
         OVERALL,1,3,0,0.3333\r\n";
    assert_eq!(String::from_utf8(bytes).unwrap(), expected);
}

#[test]
fn absent_accuracy_writes_no_file() {
    let metrics = AccumulatorSummary::new();
    let report = NativeReport::new(&metrics, None);
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = ExportConfig {
        accuracy_csv: AccuracyCsvExportConfig { enabled: true },
        ..ExportConfig::default()
    };
    AccuracyCsvExporter
        .export(&report, dir.path(), &cfg)
        .expect("export");
    assert!(!dir.path().join(ACCURACY_CSV_FILE).exists());
}

#[test]
fn empty_population_writes_no_file() {
    let summary = AccuracySummary {
        overall: rollup(0, 0, 0),
        per_task: BTreeMap::new(),
    };
    let report = report_with(summary);
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = ExportConfig {
        accuracy_csv: AccuracyCsvExportConfig { enabled: true },
        ..ExportConfig::default()
    };
    AccuracyCsvExporter
        .export(&report, dir.path(), &cfg)
        .expect("export");
    assert!(!dir.path().join(ACCURACY_CSV_FILE).exists());
}
