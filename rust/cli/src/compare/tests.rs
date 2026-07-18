// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! End-to-end coverage for `aiperf compare`.
//!
//! Small summary-export fixtures are written to a temp dir and diffed through the
//! same helpers the command uses. They exercise a `larger_is_better` metric
//! (`request_throughput`) and a `smaller_is_better` metric (`request_latency`)
//! so the direction-aware verdict is asserted in both polarities.

use std::io::Write;
use std::path::Path;

use super::{Comparison, Verdict, collect_stats, direction_by_tag, read_summary};

/// Summary A: baseline throughput and latency.
fn summary_a() -> &'static str {
    r#"{
        "schema_version": "1.4",
        "aiperf_version": "0.0.0",
        "request_throughput": {"unit": "requests/sec", "avg": 100.0},
        "request_latency": {"unit": "ms", "avg": 50.0, "p50": 48.0},
        "was_cancelled": false,
        "error_summary": []
    }"#
}

/// Summary B: throughput up, latency up.
fn summary_b() -> &'static str {
    r#"{
        "schema_version": "1.4",
        "aiperf_version": "0.0.0",
        "request_throughput": {"unit": "requests/sec", "avg": 120.0},
        "request_latency": {"unit": "ms", "avg": 60.0, "p50": 58.0},
        "was_cancelled": false,
        "error_summary": []
    }"#
}

fn write_fixture(dir: &Path, name: &str, contents: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path).expect("create fixture");
    file.write_all(contents.as_bytes()).expect("write fixture");
    path
}

/// Build the shared-metric comparisons the way `run` does.
fn compare(dir: &Path) -> Vec<Comparison> {
    let file_a = write_fixture(dir, "a_aiperf.json", summary_a());
    let file_b = write_fixture(dir, "b_aiperf.json", summary_b());
    let summary_a = read_summary(&file_a).unwrap();
    let summary_b = read_summary(&file_b).unwrap();
    let left = collect_stats(&summary_a, "avg");
    let right = collect_stats(&summary_b, "avg");
    let direction = direction_by_tag();

    let mut rows = Vec::new();
    for (tag, l) in &left {
        let Some(r) = right.get(tag) else { continue };
        rows.push(Comparison::new(
            tag.clone(),
            l.unit.clone(),
            l.value,
            r.value,
            direction.get(tag.as_str()).copied(),
        ));
    }
    rows
}

#[test]
fn collect_stats_skips_envelope_keys() {
    let summary: serde_json::Value = serde_json::from_str(summary_a()).unwrap();
    let stats = collect_stats(&summary, "avg");
    assert_eq!(stats.len(), 2, "only the two metric objects are collected");
    assert!((stats["request_throughput"].value - 100.0).abs() < 1e-9);
    assert_eq!(stats["request_latency"].unit, "ms");
}

#[test]
fn verdict_follows_metric_direction() {
    let dir = tempfile::tempdir().unwrap();
    let rows = compare(dir.path());

    let throughput = rows
        .iter()
        .find(|r| r.tag == "request_throughput")
        .expect("throughput compared");
    // 100 -> 120, larger_is_better => better.
    assert_eq!(throughput.delta, 20.0);
    assert_eq!(throughput.percent, Some(20.0));
    assert_eq!(throughput.verdict, Verdict::Better);

    let latency = rows
        .iter()
        .find(|r| r.tag == "request_latency")
        .expect("latency compared");
    // 50 -> 60, smaller_is_better => worse.
    assert_eq!(latency.delta, 10.0);
    assert_eq!(latency.percent, Some(20.0));
    assert_eq!(latency.verdict, Verdict::Worse);
}

#[test]
fn read_summary_picks_last_jsonl_record() {
    let dir = tempfile::tempdir().unwrap();
    let stream = format!("{}\n{}\n", summary_a(), summary_b());
    let path = write_fixture(dir.path(), "stream.jsonl", &stream);
    let summary = read_summary(&path).unwrap();
    let stats = collect_stats(&summary, "avg");
    // Final record (120) is selected over the first (100).
    assert!((stats["request_throughput"].value - 120.0).abs() < 1e-9);
}
