// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the offline `.wandb` W&B sink.
//!
//! The run-dir layout, the `wandb.Table` file contract, and the tag/config
//! projection are asserted directly. When a W&B SDK interpreter is available,
//! it additionally decodes the emitted file and checks run, summary, and exit records.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;
use crate::export::ExportConfig;
use crate::metrics_core::catalog::MetricTag;
use crate::metrics_core::{AccumulatorSummary, NativeReport};

/// Create a unique empty scratch directory under the system temp dir.
fn scratch_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "aiperf-wandb-{tag}-{}-{nanos}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A small profiling report with two scalar metrics.
fn sample_report() -> NativeReport {
    let mut summary = AccumulatorSummary::new();
    summary.insert_finite(MetricTag::RequestThroughput, 123.456);
    summary.insert_finite(MetricTag::BenchmarkDuration, 10.0);
    NativeReport::new(&summary, None)
}

/// An enabled export config that projects config + cli_command.
fn sample_cfg() -> ExportConfig {
    ExportConfig {
        wandb: WandbExportConfig {
            project: Some("aiperf-demo".to_string()),
            entity: Some("nv-team".to_string()),
            run_name: None,
            tags: vec!["mytag".to_string()],
            benchmark_id: Some("abcd1234 effgh".to_string()),
            aiperf_version: None,
            config_json: Some(r#"{"a":1,"nested":{"x":"y"}}"#.to_string()),
            cli_command: Some("aiperf profile --secret redacted".to_string()),
            sync_url: None,
        },
        ..ExportConfig::default()
    }
}

/// Locate the single `offline-run-*` directory the exporter creates.
fn find_run_dir(artifact_dir: &Path) -> PathBuf {
    let wandb = artifact_dir.join("wandb");
    let mut runs: Vec<PathBuf> = std::fs::read_dir(&wandb)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("offline-run-"))
        })
        .collect();
    assert_eq!(runs.len(), 1, "expected exactly one offline run dir");
    runs.pop().unwrap()
}

#[test]
fn export_writes_run_dir_layout() {
    let dir = scratch_dir("layout");
    WandbExporter
        .export(&sample_report(), &dir, &sample_cfg())
        .unwrap();

    let run_dir = find_run_dir(&dir);
    let name = run_dir.file_name().unwrap().to_str().unwrap();
    assert!(name.starts_with("offline-run-"));

    // The `.wandb` transaction log is present and non-empty.
    let wandb_files: Vec<_> = std::fs::read_dir(&run_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("wandb"))
        .collect();
    assert_eq!(wandb_files.len(), 1);
    assert!(std::fs::metadata(&wandb_files[0]).unwrap().len() > 7);

    // The media table file uses the W&B table shape.
    let table_path = run_dir
        .join("files")
        .join("media/table/summary_metrics.table.json");
    let table: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&table_path).unwrap()).unwrap();
    let columns = table["columns"].as_array().unwrap();
    assert_eq!(columns[0], serde_json::json!("Metric"));
    let expected_cols: Vec<serde_json::Value> = std::iter::once(serde_json::json!("Metric"))
        .chain(STAT_COLUMN_KEYS.iter().map(|k| serde_json::json!(k)))
        .collect();
    assert_eq!(columns, &expected_cols);

    // One row per report metric; each row is Metric + 7 stat cells.
    let data = table["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "two scalar metrics in the report");
    for row in data {
        let cells = row.as_array().unwrap();
        assert_eq!(cells.len(), 1 + STAT_COLUMN_KEYS.len());
        assert!(cells[0].is_string());
    }
}

#[test]
fn scalar_metric_populates_avg_column_only() {
    // Scalar metrics carry a single value surfaced under avg/min/max/p*, but
    // std has no scalar meaning and stays null.
    let rows = build_metric_rows(&sample_report());
    assert_eq!(rows.len(), 2);
    for row in &rows {
        let avg = row.cells[0];
        assert!(avg.is_some(), "scalar avg must be present");
        // std is index 6 in STAT_COLUMN_KEYS.
        assert_eq!(row.cells[6], None, "scalar std stays null");
    }
}

#[test]
fn round2_matches_python_rounding() {
    assert_eq!(round2(123.456), 123.46);
    assert_eq!(round2(10.0), 10.0);
    assert_eq!(round2(1.005_000_1), 1.01);
}

#[test]
fn tags_prepend_version_and_benchmark() {
    let report = sample_report();
    let cfg = WandbExportConfig {
        tags: vec!["user-a".to_string(), "user-b".to_string()],
        benchmark_id: Some("0123456789abcdef".to_string()),
        ..Default::default()
    };
    let tags = build_tags(&report, &cfg);
    assert_eq!(tags[0], format!("aiperf-{}", report.aiperf_version));
    assert_eq!(tags[1], "benchmark-01234567");
    assert_eq!(&tags[2..], &["user-a".to_string(), "user-b".to_string()]);
}

#[test]
fn config_items_split_object_and_append_cli() {
    let cfg = WandbExportConfig {
        config_json: Some(r#"{"a":1,"nested":{"x":"y"}}"#.to_string()),
        cli_command: Some("aiperf profile".to_string()),
        ..Default::default()
    };
    let items = build_config_items(&cfg).unwrap();
    assert_eq!(items[0].key, "_wandb");
    assert_eq!(items[0].value_json, "{}");
    let by_key: std::collections::HashMap<_, _> = items
        .iter()
        .map(|i| (i.key.as_str(), i.value_json.as_str()))
        .collect();
    assert_eq!(by_key.get("a"), Some(&"1"));
    assert_eq!(by_key.get("nested"), Some(&r#"{"x":"y"}"#));
    assert_eq!(
        by_key.get("aiperf.cli_command"),
        Some(&r#""aiperf profile""#)
    );
}

#[test]
fn config_json_must_be_object() {
    let cfg = WandbExportConfig {
        config_json: Some("[1,2,3]".to_string()),
        ..Default::default()
    };
    assert!(build_config_items(&cfg).is_err());
}

#[test]
fn default_run_name_from_benchmark_id() {
    assert_eq!(default_run_name(Some("abcd1234 effff")), "aiperf-abcd1234");
    assert_eq!(default_run_name(Some("")), "aiperf-run");
    assert_eq!(default_run_name(None), "aiperf-run");
}

#[test]
fn cell_json_null_for_absent() {
    assert_eq!(cell_json(None), serde_json::Value::Null);
    assert_eq!(cell_json(Some(1.5)), serde_json::json!(1.5));
}

#[test]
fn large_record_spans_blocks_and_round_trips() {
    // A record larger than a 32768-byte block must split FIRST/MIDDLE*/LAST and
    // round-trip through the W&B SDK.
    let Some(python) = wandb_python() else {
        eprintln!("skipping: no wandb-bearing python (set AIPERF_WANDB_PYTHON)");
        return;
    };

    let big = "x".repeat(80_000);
    let record = proto::Record {
        num: 1,
        run: Some(proto::RunRecord {
            run_id: "spanid00".to_string(),
            project: "big".to_string(),
            config: Some(proto::ConfigRecord {
                update: vec![proto::ConfigItem {
                    key: "blob".to_string(),
                    nested_key: Vec::new(),
                    value_json: serde_json::to_string(&big).unwrap(),
                }],
                info: None,
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut store = datastore::DataStore::new();
    store.write(&record.to_bytes());

    let dir = scratch_dir("span");
    let path = dir.join("run-spanid00.wandb");
    std::fs::write(&path, store.into_bytes()).unwrap();

    let script = r#"
import sys, json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
ds = DataStore(); ds.open_for_scan(sys.argv[1])
data = ds.scan_data()
r = pb.Record(); r.ParseFromString(data)
print(json.dumps({"len": len(r.run.config.update[0].value_json)}))
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "span decode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let decoded: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    // 80000 chars + 2 quote chars in the JSON string.
    assert_eq!(decoded["len"], 80_002);
}

/// Locate an optional W&B SDK interpreter for independent decoding.
fn wandb_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AIPERF_WANDB_PYTHON") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let candidate = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.venv/bin/python"
    ));
    if candidate.exists() {
        // Confirm wandb actually imports before relying on it.
        let ok = Command::new(&candidate)
            .args(["-c", "import wandb"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate);
        }
    }
    None
}

/// Decode the emitted `.wandb` with the wandb SDK itself and assert record
/// contents. Skipped (not failed) when no wandb interpreter is available.
#[test]
fn wandb_file_decodes_with_sdk() {
    let Some(python) = wandb_python() else {
        eprintln!("skipping: no wandb-bearing python (set AIPERF_WANDB_PYTHON)");
        return;
    };

    let dir = scratch_dir("decode");
    WandbExporter
        .export(&sample_report(), &dir, &sample_cfg())
        .unwrap();
    let run_dir = find_run_dir(&dir);
    let wandb_file = std::fs::read_dir(&run_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().and_then(|x| x.to_str()) == Some("wandb"))
        .unwrap();

    let script = r#"
import sys, json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
ds = DataStore(); ds.open_for_scan(sys.argv[1])
kinds = []
run = None
have_summary = have_history = have_exit = False
while True:
    data = ds.scan_data()
    if data is None: break
    r = pb.Record(); r.ParseFromString(data)
    which = r.WhichOneof("record_type")
    kinds.append(which)
    if which == "run": run = r.run
    if which == "summary": have_summary = True
    if which == "history": have_history = True
    if which == "exit": have_exit = True
assert run is not None, "no run record"
out = {
    "project": run.project,
    "entity": run.entity,
    "display_name": run.display_name,
    "tags": list(run.tags),
    "config_keys": [c.key for c in run.config.update],
    "kinds": kinds,
    "have_summary": have_summary,
    "have_history": have_history,
    "have_exit": have_exit,
}
print(json.dumps(out))
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(&wandb_file)
        .output()
        .expect("run wandb decode");
    assert!(
        output.status.success(),
        "decode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let decoded: serde_json::Value = serde_json::from_slice(&output.stdout).expect("decode json");

    assert_eq!(decoded["project"], "aiperf-demo");
    assert_eq!(decoded["entity"], "nv-team");
    assert_eq!(decoded["display_name"], "aiperf-abcd1234");
    let tags: Vec<String> = decoded["tags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_string())
        .collect();
    assert!(tags.iter().any(|t| t.starts_with("aiperf-")));
    assert!(tags.contains(&"benchmark-abcd1234".to_string()));
    assert!(tags.contains(&"mytag".to_string()));
    let config_keys: Vec<String> = decoded["config_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_string())
        .collect();
    assert!(config_keys.contains(&"_wandb".to_string()));
    assert!(config_keys.contains(&"a".to_string()));
    assert!(config_keys.contains(&"aiperf.cli_command".to_string()));
    assert_eq!(decoded["have_summary"], true);
    assert_eq!(decoded["have_history"], true);
    assert_eq!(decoded["have_exit"], true);
}
