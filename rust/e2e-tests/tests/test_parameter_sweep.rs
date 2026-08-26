// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

const WORKERS_MAX: u32 = 1;
const UI: &str = "simple";

fn jload(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn read_text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn sorted_dirs(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .map(|n| n.to_string_lossy().starts_with(prefix))
                        .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

fn name_of(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().to_string()
}

fn metric_keys_with(metrics: &Value, needle: &str) -> Vec<String> {
    metrics
        .as_object()
        .map(|o| {
            o.keys()
                .filter(|k| k.to_lowercase().contains(needle))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn assert_fields(m: &Value, fields: &[&str], ctx: &str) {
    for f in fields {
        assert!(m.get(f).is_some(), "{ctx}: should have {f} field");
    }
}

fn request_count_avg(json_file: &Path) -> f64 {
    jload(json_file)["request_count"]["avg"]
        .as_f64()
        .expect("request_count.avg numeric")
}

const CONFIDENCE_METRIC_FIELDS: &[&str] = &[
    "mean",
    "std",
    "min",
    "max",
    "cv",
    "se",
    "ci_low",
    "ci_high",
    "t_critical",
    "unit",
];

#[tokio::test]
async fn test_sweep_with_confidence_repeated_mode() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 3 --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    let profile_runs_dir = root.join("profile_runs");
    assert!(
        profile_runs_dir.exists(),
        "profile_runs directory should exist"
    );

    let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
    assert_eq!(trial_dirs.len(), 3, "Should have 3 trial directories");
    assert_eq!(name_of(&trial_dirs[0]), "trial_0001");
    assert_eq!(name_of(&trial_dirs[1]), "trial_0002");
    assert_eq!(name_of(&trial_dirs[2]), "trial_0003");

    let concurrency_values = [2, 4, 6];
    for trial_dir in &trial_dirs {
        for c in concurrency_values {
            let cdir = trial_dir.join(format!("concurrency_{c}"));
            assert!(
                cdir.exists(),
                "{} should have concurrency_{c}",
                name_of(trial_dir)
            );
            let json_file = cdir.join("profile_export_aiperf.json");
            let csv_file = cdir.join("profile_export_aiperf.csv");
            assert!(json_file.exists(), "should have JSON");
            assert!(csv_file.exists(), "should have CSV");
            assert_eq!(request_count_avg(&json_file), 10.0);
        }
    }

    let aggregate_dir = root.join("aggregate");
    assert!(aggregate_dir.exists(), "aggregate directory should exist");

    for c in concurrency_values {
        let cagg = aggregate_dir.join(format!("concurrency_{c}"));
        assert!(cagg.exists(), "aggregate/concurrency_{c} should exist");
        let agg_json = cagg.join("profile_export_aiperf_aggregate.json");
        let agg_csv = cagg.join("profile_export_aiperf_aggregate.csv");
        assert!(agg_json.exists(), "aggregate JSON should exist");
        assert!(agg_csv.exists(), "aggregate CSV should exist");

        let agg = jload(&agg_json);
        assert_eq!(agg["metadata"]["aggregation_type"], json!("confidence"));
        assert_eq!(agg["metadata"]["num_profile_runs"], json!(3));
        assert_eq!(agg["metadata"]["num_successful_runs"], json!(3));
        assert_eq!(agg["metadata"]["failed_runs"].as_array().unwrap().len(), 0);
        assert_eq!(agg["metadata"]["confidence_level"], json!(0.95));

        let metrics = &agg["metrics"];
        assert!(
            !metrics.as_object().unwrap().is_empty(),
            "Should have metrics"
        );
        let throughput = metric_keys_with(metrics, "throughput");
        assert!(!throughput.is_empty(), "Should have throughput metrics");
        assert_fields(&metrics[&throughput[0]], CONFIDENCE_METRIC_FIELDS, "metric");
    }

    let sweep_agg_dir = aggregate_dir.join("sweep_aggregate");
    assert!(
        sweep_agg_dir.exists(),
        "sweep_aggregate directory should exist"
    );
    let sweep_json = sweep_agg_dir.join("profile_export_aiperf_sweep.json");
    let sweep_csv = sweep_agg_dir.join("profile_export_aiperf_sweep.csv");
    assert!(sweep_json.exists(), "Sweep aggregate JSON should exist");
    assert!(sweep_csv.exists(), "Sweep aggregate CSV should exist");

    assert_sweep_json_shape(&sweep_json, "repeated", &[2, 4, 6]);
    assert_sweep_csv_shape(&sweep_csv, &[2, 4, 6]);
}

#[tokio::test]
async fn test_sweep_with_confidence_independent_mode() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 3 \
         --parameter-sweep-mode independent --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    let concurrency_values = [2, 4, 6];
    for c in concurrency_values {
        let cdir = root.join(format!("concurrency_{c}"));
        assert!(cdir.exists(), "concurrency_{c} directory should exist");

        let profile_runs_dir = cdir.join("profile_runs");
        assert!(
            profile_runs_dir.exists(),
            "concurrency_{c}/profile_runs should exist"
        );

        let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
        assert_eq!(trial_dirs.len(), 3, "concurrency_{c} should have 3 trials");
        assert_eq!(name_of(&trial_dirs[0]), "trial_0001");
        assert_eq!(name_of(&trial_dirs[1]), "trial_0002");
        assert_eq!(name_of(&trial_dirs[2]), "trial_0003");

        for trial_dir in &trial_dirs {
            let json_file = trial_dir.join("profile_export_aiperf.json");
            let csv_file = trial_dir.join("profile_export_aiperf.csv");
            assert!(json_file.exists(), "should have JSON");
            assert!(csv_file.exists(), "should have CSV");
            assert_eq!(request_count_avg(&json_file), 10.0);
        }

        let aggregate_dir = cdir.join("aggregate");
        assert!(
            aggregate_dir.exists(),
            "concurrency_{c}/aggregate should exist"
        );
        let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
        let agg_csv = aggregate_dir.join("profile_export_aiperf_aggregate.csv");
        assert!(agg_json.exists(), "aggregate JSON should exist");
        assert!(agg_csv.exists(), "aggregate CSV should exist");

        let agg = jload(&agg_json);
        assert_eq!(agg["metadata"]["aggregation_type"], json!("confidence"));
        assert_eq!(agg["metadata"]["num_profile_runs"], json!(3));
        assert_eq!(agg["metadata"]["num_successful_runs"], json!(3));
        assert_eq!(agg["metadata"]["failed_runs"].as_array().unwrap().len(), 0);
        assert_eq!(agg["metadata"]["confidence_level"], json!(0.95));

        let metrics = &agg["metrics"];
        assert!(
            !metrics.as_object().unwrap().is_empty(),
            "Should have metrics"
        );
        let throughput = metric_keys_with(metrics, "throughput");
        assert!(!throughput.is_empty(), "Should have throughput metrics");
        assert_fields(&metrics[&throughput[0]], CONFIDENCE_METRIC_FIELDS, "metric");
    }

    let sweep_agg_dir = root.join("sweep_aggregate");
    assert!(
        sweep_agg_dir.exists(),
        "sweep_aggregate directory should exist"
    );
    let sweep_json = sweep_agg_dir.join("profile_export_aiperf_sweep.json");
    let sweep_csv = sweep_agg_dir.join("profile_export_aiperf_sweep.csv");
    assert!(sweep_json.exists(), "Sweep aggregate JSON should exist");
    assert!(sweep_csv.exists(), "Sweep aggregate CSV should exist");

    assert_sweep_json_shape(&sweep_json, "independent", &[2, 4, 6]);
    assert_sweep_csv_shape(&sweep_csv, &[2, 4, 6]);
}

#[tokio::test]
async fn test_artifact_directory_structure_repeated_mode() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4 --num-profile-runs 2 --request-count 5 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    let profile_runs_dir = root.join("profile_runs");
    assert!(
        profile_runs_dir.exists(),
        "profile_runs directory must exist"
    );

    let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
    assert_eq!(
        trial_dirs.len(),
        2,
        "Should have exactly 2 trial directories"
    );
    assert_eq!(name_of(&trial_dirs[0]), "trial_0001");
    assert_eq!(name_of(&trial_dirs[1]), "trial_0002");

    let mut trial_names: Vec<String> = trial_dirs.iter().map(|p| name_of(p)).collect();
    let sorted_names = {
        let mut s = trial_names.clone();
        s.sort();
        s
    };
    assert_eq!(
        trial_names, sorted_names,
        "Zero-padded names should sort correctly"
    );
    trial_names.clear();

    let concurrency_values = [2, 4];
    for trial_dir in &trial_dirs {
        for c in concurrency_values {
            let cdir = trial_dir.join(format!("concurrency_{c}"));
            assert!(
                cdir.exists(),
                "{} must contain concurrency_{c}",
                name_of(trial_dir)
            );
            assert!(cdir.join("profile_export_aiperf.json").exists());
            assert!(cdir.join("profile_export_aiperf.csv").exists());
        }
    }

    let aggregate_dir = root.join("aggregate");
    assert!(aggregate_dir.exists(), "aggregate directory must exist");
    for c in concurrency_values {
        let cagg = aggregate_dir.join(format!("concurrency_{c}"));
        assert!(cagg.exists(), "aggregate/concurrency_{c} must exist");
        assert!(cagg.join("profile_export_aiperf_aggregate.json").exists());
        assert!(cagg.join("profile_export_aiperf_aggregate.csv").exists());
    }

    let sweep_agg_dir = aggregate_dir.join("sweep_aggregate");
    assert!(
        sweep_agg_dir.exists(),
        "sweep_aggregate directory must exist"
    );
    assert!(
        sweep_agg_dir
            .join("profile_export_aiperf_sweep.json")
            .exists()
    );
    assert!(
        sweep_agg_dir
            .join("profile_export_aiperf_sweep.csv")
            .exists()
    );

    for d in sorted_dirs(&profile_runs_dir, "") {
        assert!(d.is_dir());
        assert!(name_of(&d).starts_with("trial_"));
    }
    for trial_dir in &trial_dirs {
        for d in sorted_dirs(trial_dir, "") {
            assert!(d.is_dir());
            assert!(name_of(&d).starts_with("concurrency_"));
        }
    }
}

#[tokio::test]
async fn test_artifact_directory_structure_independent_mode() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4 --num-profile-runs 2 \
         --parameter-sweep-mode independent --request-count 5 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    let concurrency_values = [2, 4];
    for c in concurrency_values {
        let cdir = root.join(format!("concurrency_{c}"));
        assert!(
            cdir.exists(),
            "concurrency_{c} directory must exist at top level"
        );

        let profile_runs_dir = cdir.join("profile_runs");
        assert!(
            profile_runs_dir.exists(),
            "concurrency_{c}/profile_runs must exist"
        );

        let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
        assert_eq!(trial_dirs.len(), 2, "concurrency_{c} should have 2 trials");
        assert_eq!(name_of(&trial_dirs[0]), "trial_0001");
        assert_eq!(name_of(&trial_dirs[1]), "trial_0002");

        let names: Vec<String> = trial_dirs.iter().map(|p| name_of(p)).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);

        for trial_dir in &trial_dirs {
            let json_file = trial_dir.join("profile_export_aiperf.json");
            assert!(json_file.exists(), "should have JSON artifact");
            assert!(trial_dir.join("profile_export_aiperf.csv").exists());
            assert_eq!(request_count_avg(&json_file), 5.0);
        }

        let aggregate_dir = cdir.join("aggregate");
        assert!(
            aggregate_dir.exists(),
            "concurrency_{c}/aggregate must exist"
        );
        assert!(
            aggregate_dir
                .join("profile_export_aiperf_aggregate.json")
                .exists()
        );
        assert!(
            aggregate_dir
                .join("profile_export_aiperf_aggregate.csv")
                .exists()
        );
    }

    let sweep_agg_dir = root.join("sweep_aggregate");
    assert!(
        sweep_agg_dir.exists(),
        "sweep_aggregate must exist at top level"
    );
    assert!(
        sweep_agg_dir
            .join("profile_export_aiperf_sweep.json")
            .exists()
    );
    assert!(
        sweep_agg_dir
            .join("profile_export_aiperf_sweep.csv")
            .exists()
    );

    let concurrency_dirs = sorted_dirs(&root, "concurrency_");
    assert_eq!(
        concurrency_dirs.len(),
        2,
        "Should have exactly 2 concurrency dirs"
    );
    let cnames: Vec<String> = concurrency_dirs.iter().map(|p| name_of(p)).collect();
    assert_eq!(cnames, vec!["concurrency_2", "concurrency_4"]);

    for c in concurrency_values {
        let profile_runs_dir = root.join(format!("concurrency_{c}")).join("profile_runs");
        for d in sorted_dirs(&profile_runs_dir, "") {
            assert!(d.is_dir());
            assert!(name_of(&d).starts_with("trial_"));
        }
    }

    assert!(
        !root.join("profile_runs").exists(),
        "Independent mode should NOT have profile_runs at top level"
    );
    assert!(
        root.join("concurrency_2").join("profile_runs").exists(),
        "Independent mode should have profile_runs under each concurrency"
    );
}

#[tokio::test]
async fn test_partial_failure_scenarios() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4 --num-profile-runs 2 --request-count 5 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Sweep should complete successfully");

    let profile_runs_dir = root.join("profile_runs");
    assert!(
        profile_runs_dir.exists(),
        "profile_runs directory should exist"
    );
    let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
    assert_eq!(trial_dirs.len(), 2, "Should have 2 trial directories");

    for trial_dir in &trial_dirs {
        for c in [2, 4] {
            let cdir = trial_dir.join(format!("concurrency_{c}"));
            assert!(cdir.exists());
            assert!(cdir.join("profile_export_aiperf.json").exists());
        }
    }

    let aggregate_dir = root.join("aggregate");
    assert!(aggregate_dir.exists(), "aggregate directory should exist");

    for c in [2, 4] {
        let cagg = aggregate_dir.join(format!("concurrency_{c}"));
        assert!(cagg.exists(), "aggregate/concurrency_{c} should exist");
        let agg_json = cagg.join("profile_export_aiperf_aggregate.json");
        assert!(agg_json.exists());

        let agg = jload(&agg_json);
        let metadata = &agg["metadata"];
        assert!(metadata.get("num_profile_runs").is_some());
        assert!(metadata.get("num_successful_runs").is_some());
        assert!(metadata.get("failed_runs").is_some());

        let num_successful = metadata["num_successful_runs"].as_i64().unwrap();
        let num_failed = metadata["failed_runs"].as_array().unwrap().len() as i64;
        let total = metadata["num_profile_runs"].as_i64().unwrap();
        assert_eq!(num_successful + num_failed, total);
        assert_eq!(num_successful, 2);
        assert_eq!(num_failed, 0);
    }

    let sweep_json = aggregate_dir
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(sweep_json.exists(), "Sweep aggregate JSON should exist");
    let sweep_data = jload(&sweep_json);
    assert!(sweep_data.get("metadata").is_some());
    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();

    let found: Vec<i64> = per_combo
        .iter()
        .map(|c| c["parameters"]["concurrency"].as_i64().unwrap())
        .collect();
    assert!(found.contains(&2));
    assert!(found.contains(&4));

    for combo in per_combo {
        let metrics = combo["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());
        let sample = metrics.values().next().unwrap();
        for f in ["mean", "std", "ci_low", "ci_high"] {
            assert!(sample.get(f).is_some(), "metrics should have {f}");
        }
    }
}

#[tokio::test]
async fn test_single_concurrency_artifact_layout() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 5 --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Single concurrency run should succeed");

    assert!(!root.join("profile_runs").exists());
    assert!(!root.join("concurrency_5").exists());
    assert!(!root.join("sweep_aggregate").exists());
    assert!(!root.join("aggregate").exists());

    let json_file = root.join("profile_export_aiperf.json");
    assert!(json_file.exists(), "Should have JSON artifact at top level");
    assert!(root.join("profile_export_aiperf.csv").exists());

    let run_data = jload(&json_file);
    assert_eq!(run_data["request_count"]["avg"], json!(10.0));
    let meta = run_data.get("metadata").cloned().unwrap_or(Value::Null);
    assert!(meta.get("sweep_index").is_none());
    assert!(meta.get("sweep_mode").is_none());

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 5 --num-profile-runs 3 --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(
        r.exit_code, 0,
        "Single concurrency with confidence should succeed"
    );

    let profile_runs_dir = root.join("profile_runs");
    assert!(profile_runs_dir.exists());
    let run_dirs = sorted_dirs(&profile_runs_dir, "run_");
    assert_eq!(run_dirs.len(), 3, "Should have 3 run directories");
    assert_eq!(name_of(&run_dirs[0]), "run_0001");
    assert_eq!(name_of(&run_dirs[1]), "run_0002");
    assert_eq!(name_of(&run_dirs[2]), "run_0003");

    for run_dir in &run_dirs {
        let jf = run_dir.join("profile_export_aiperf.json");
        assert!(jf.exists());
        assert!(run_dir.join("profile_export_aiperf.csv").exists());
        assert!(sorted_dirs(run_dir, "concurrency_").is_empty());
        let rd = jload(&jf);
        assert_eq!(rd["request_count"]["avg"], json!(10.0));
        let m = rd.get("metadata").cloned().unwrap_or(Value::Null);
        assert!(m.get("sweep_index").is_none());
        assert!(m.get("sweep_mode").is_none());
    }

    let aggregate_dir = root.join("aggregate");
    assert!(aggregate_dir.exists());
    let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
    assert!(agg_json.exists());
    assert!(
        aggregate_dir
            .join("profile_export_aiperf_aggregate.csv")
            .exists()
    );
    assert!(sorted_dirs(&aggregate_dir, "concurrency_").is_empty());
    assert!(!aggregate_dir.join("sweep_aggregate").exists());
    assert!(!root.join("sweep_aggregate").exists());

    let agg = jload(&agg_json);
    assert_eq!(agg["metadata"]["aggregation_type"], json!("confidence"));
    assert_eq!(agg["metadata"]["num_profile_runs"], json!(3));
    assert_eq!(agg["metadata"]["num_successful_runs"], json!(3));
    assert_eq!(agg["metadata"]["failed_runs"].as_array().unwrap().len(), 0);
    assert_eq!(agg["metadata"]["confidence_level"], json!(0.95));
    assert!(agg["metadata"].get("sweep_parameters").is_none());
    assert!(agg["metadata"].get("sweep_mode").is_none());

    let metrics = &agg["metrics"];
    assert!(!metrics.as_object().unwrap().is_empty());
    let throughput = metric_keys_with(metrics, "throughput");
    assert!(!throughput.is_empty());
    assert_fields(&metrics[&throughput[0]], CONFIDENCE_METRIC_FIELDS, "metric");
}

#[tokio::test]
async fn test_aggregate_file_generation() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 3 \
         --parameter-sweep-mode repeated --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Repeated mode sweep should succeed");

    let concurrency_values = [2, 4, 6];
    let aggregate_dir = root.join("aggregate");
    assert!(aggregate_dir.exists(), "aggregate directory must exist");

    let required_csv_columns = [
        "metric",
        "mean",
        "std",
        "min",
        "max",
        "cv",
        "se",
        "ci_low",
        "ci_high",
        "t_critical",
        "unit",
    ];

    for c in concurrency_values {
        let cagg = aggregate_dir.join(format!("concurrency_{c}"));
        assert!(cagg.exists());
        let agg_json = cagg.join("profile_export_aiperf_aggregate.json");
        let agg_csv = cagg.join("profile_export_aiperf_aggregate.csv");
        assert!(agg_json.exists());
        assert!(agg_csv.exists());

        let agg = jload(&agg_json);
        assert!(agg.get("metadata").is_some());
        assert!(agg.get("metrics").is_some());
        let metadata = &agg["metadata"];
        for f in [
            "aggregation_type",
            "num_profile_runs",
            "num_successful_runs",
            "failed_runs",
            "confidence_level",
        ] {
            assert!(
                metadata.get(f).is_some(),
                "aggregate metadata must have {f}"
            );
        }
        assert_eq!(metadata["aggregation_type"], json!("confidence"));
        assert_eq!(metadata["num_profile_runs"], json!(3));
        assert_eq!(metadata["num_successful_runs"], json!(3));
        assert_eq!(metadata["failed_runs"].as_array().unwrap().len(), 0);
        assert_eq!(metadata["confidence_level"], json!(0.95));

        let metrics = agg["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());
        for (name, data) in metrics {
            assert_fields(data, CONFIDENCE_METRIC_FIELDS, name);
        }

        let csv = read_text(&agg_csv);
        let lines: Vec<&str> = csv.trim().split('\n').collect();
        assert!(lines.len() > 1, "CSV must have header and data rows");
        for col in required_csv_columns {
            assert!(lines[0].contains(col), "CSV header must have {col} column");
        }
    }

    let sweep_agg_dir = aggregate_dir.join("sweep_aggregate");
    assert!(sweep_agg_dir.exists());
    let sweep_json = sweep_agg_dir.join("profile_export_aiperf_sweep.json");
    let sweep_csv = sweep_agg_dir.join("profile_export_aiperf_sweep.csv");
    assert!(sweep_json.exists());
    assert!(sweep_csv.exists());

    assert_sweep_json_full(&sweep_json, "repeated", &[2, 4, 6]);
    assert_sweep_csv_shape(&sweep_csv, &[2, 4, 6]);

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 3 \
         --parameter-sweep-mode independent --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Independent mode sweep should succeed");

    for c in concurrency_values {
        let cdir = root.join(format!("concurrency_{c}"));
        assert!(cdir.exists());
        let aggregate_dir = cdir.join("aggregate");
        assert!(aggregate_dir.exists());
        let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
        let agg_csv = aggregate_dir.join("profile_export_aiperf_aggregate.csv");
        assert!(agg_json.exists());
        assert!(agg_csv.exists());

        let agg = jload(&agg_json);
        let metadata = &agg["metadata"];
        assert_eq!(metadata["aggregation_type"], json!("confidence"));
        assert_eq!(metadata["num_profile_runs"], json!(3));
        assert_eq!(metadata["num_successful_runs"], json!(3));
        assert_eq!(metadata["failed_runs"].as_array().unwrap().len(), 0);
        assert_eq!(metadata["confidence_level"], json!(0.95));
        let metrics = agg["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());
        for (name, data) in metrics {
            assert_fields(data, CONFIDENCE_METRIC_FIELDS, name);
        }

        let csv = read_text(&agg_csv);
        let lines: Vec<&str> = csv.trim().split('\n').collect();
        assert!(lines.len() > 1);
        for col in required_csv_columns {
            assert!(lines[0].contains(col), "CSV header must have {col} column");
        }
    }

    let sweep_agg_dir = root.join("sweep_aggregate");
    assert!(sweep_agg_dir.exists());
    let sweep_json = sweep_agg_dir.join("profile_export_aiperf_sweep.json");
    let sweep_csv = sweep_agg_dir.join("profile_export_aiperf_sweep.csv");
    assert!(sweep_json.exists());
    assert!(sweep_csv.exists());

    assert_sweep_json_full(&sweep_json, "independent", &[2, 4, 6]);
    assert_sweep_csv_shape(&sweep_csv, &[2, 4, 6]);
}

#[tokio::test]
async fn test_per_value_confidence_statistics() {
    use std::collections::HashMap;

    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 3 \
         --parameter-sweep-mode repeated --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Repeated mode sweep should succeed");

    let aggregate_dir = root.join("aggregate");
    assert!(aggregate_dir.exists());
    let concurrency_values = [2, 4, 6];

    let mut raw: HashMap<i64, HashMap<String, Vec<f64>>> = HashMap::new();
    for c in concurrency_values {
        raw.insert(c, HashMap::new());
    }

    let profile_runs_dir = root.join("profile_runs");
    let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
    assert_eq!(trial_dirs.len(), 3);

    for trial_dir in &trial_dirs {
        for c in concurrency_values {
            let json_file = trial_dir
                .join(format!("concurrency_{c}"))
                .join("profile_export_aiperf.json");
            let run_data = jload(&json_file);
            if let Some(obj) = run_data.as_object() {
                for (metric_name, metric_value) in obj {
                    if let Some(avg) = metric_value.get("avg").and_then(|v| v.as_f64()) {
                        raw.get_mut(&c)
                            .unwrap()
                            .entry(metric_name.clone())
                            .or_default()
                            .push(avg);
                    }
                }
            }
        }
    }

    for c in concurrency_values {
        let agg_json = aggregate_dir
            .join(format!("concurrency_{c}"))
            .join("profile_export_aiperf_aggregate.json");
        assert!(agg_json.exists());
        let agg = jload(&agg_json);

        let metadata = &agg["metadata"];
        assert_eq!(metadata["aggregation_type"], json!("confidence"));
        assert_eq!(metadata["num_profile_runs"], json!(3));
        assert_eq!(metadata["num_successful_runs"], json!(3));
        assert_eq!(metadata["confidence_level"], json!(0.95));

        let metrics = agg["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());

        for (metric_name, m) in metrics {
            assert_fields(m, CONFIDENCE_METRIC_FIELDS, metric_name);

            let mean = m["mean"].as_f64().unwrap();
            let std = m["std"].as_f64().unwrap();
            let min = m["min"].as_f64().unwrap();
            let max = m["max"].as_f64().unwrap();
            let se = m["se"].as_f64().unwrap();
            let ci_low = m["ci_low"].as_f64().unwrap();
            let ci_high = m["ci_high"].as_f64().unwrap();
            let t_critical = m["t_critical"].as_f64().unwrap();
            assert!(
                m["cv"].is_null() || m["cv"].is_number(),
                "cv numeric or None"
            );
            assert!(m["unit"].is_string());

            let eps = 1e-9;
            assert!(min - eps <= mean && mean <= max + eps, "min<=mean<=max");
            assert!(std >= 0.0, "std non-negative");
            assert!(se >= 0.0, "se non-negative");
            assert!(ci_low <= mean && mean <= ci_high, "ci_low<=mean<=ci_high");
            assert!(t_critical > 0.0, "t_critical positive");

            if mean != 0.0 {
                let expected_cv = std / mean.abs();
                let cv = m["cv"].as_f64().unwrap();
                assert!((cv - expected_cv).abs() < 0.01, "cv == std/mean");
            }

            if let Some(vals) = raw.get(&c).and_then(|mm| mm.get(metric_name)) {
                assert_eq!(vals.len(), 3, "3 raw values for {metric_name}");
                let exp_mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let exp_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let exp_max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let tol = 0.01;
                assert!((mean - exp_mean).abs() < tol, "mean matches raw");
                assert!((min - exp_min).abs() < tol, "min matches raw");
                assert!((max - exp_max).abs() < tol, "max matches raw");
            }
        }
    }

    let sweep_json = aggregate_dir
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(sweep_json.exists());
    let sweep_data = jload(&sweep_json);
    let metadata = &sweep_data["metadata"];
    assert_eq!(metadata["aggregation_type"], json!("sweep"));
    assert_eq!(metadata["num_trials_per_value"], json!(3));
    assert_eq!(metadata["confidence_level"], json!(0.95));

    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();
    let found: Vec<i64> = per_combo
        .iter()
        .map(|c| c["parameters"]["concurrency"].as_i64().unwrap())
        .collect();
    for v in [2, 4, 6] {
        assert!(
            found.contains(&v),
            "Should have metrics for concurrency {v}"
        );
    }

    let sweep_fields = ["mean", "std", "min", "max", "ci_low", "ci_high", "unit"];
    for combo in per_combo {
        let metrics = combo["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());
        for (metric_name, m) in metrics {
            assert_fields(m, &sweep_fields, metric_name);
            let mean = m["mean"].as_f64().unwrap();
            let std = m["std"].as_f64().unwrap();
            let min = m["min"].as_f64().unwrap();
            let max = m["max"].as_f64().unwrap();
            let ci_low = m["ci_low"].as_f64().unwrap();
            let ci_high = m["ci_high"].as_f64().unwrap();
            let eps = 1e-9;
            assert!(min - eps <= mean && mean <= max + eps);
            assert!(std >= 0.0);
            assert!(ci_low <= mean && mean <= ci_high);
        }
    }

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 3 \
         --parameter-sweep-mode independent --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Independent mode sweep should succeed");

    for c in concurrency_values {
        let agg_json = root
            .join(format!("concurrency_{c}"))
            .join("aggregate")
            .join("profile_export_aiperf_aggregate.json");
        assert!(agg_json.exists());
        let agg = jload(&agg_json);
        let metadata = &agg["metadata"];
        assert_eq!(metadata["aggregation_type"], json!("confidence"));
        assert_eq!(metadata["num_profile_runs"], json!(3));
        assert_eq!(metadata["num_successful_runs"], json!(3));
        assert_eq!(metadata["confidence_level"], json!(0.95));

        let metrics = agg["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());
        for (metric_name, m) in metrics {
            assert_fields(m, CONFIDENCE_METRIC_FIELDS, metric_name);
            let mean = m["mean"].as_f64().unwrap();
            let std = m["std"].as_f64().unwrap();
            let min = m["min"].as_f64().unwrap();
            let max = m["max"].as_f64().unwrap();
            let ci_low = m["ci_low"].as_f64().unwrap();
            let ci_high = m["ci_high"].as_f64().unwrap();
            let eps = 1e-9;
            assert!(min - eps <= mean && mean <= max + eps);
            assert!(std >= 0.0);
            assert!(ci_low <= mean && mean <= ci_high);
        }
    }

    let sweep_json = root
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(sweep_json.exists());
    let sweep_data = jload(&sweep_json);
    let metadata = &sweep_data["metadata"];
    assert_eq!(metadata["aggregation_type"], json!("sweep"));
    assert_eq!(metadata["sweep_mode"], json!("independent"));
    assert_eq!(metadata["num_trials_per_value"], json!(3));
    assert_eq!(metadata["confidence_level"], json!(0.95));

    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();
    let found: Vec<i64> = per_combo
        .iter()
        .map(|c| c["parameters"]["concurrency"].as_i64().unwrap())
        .collect();
    for v in [2, 4, 6] {
        assert!(found.contains(&v));
    }
    for combo in per_combo {
        let metrics = combo["metrics"].as_object().unwrap();
        assert!(!metrics.is_empty());
        for (metric_name, m) in metrics {
            assert_fields(m, &sweep_fields, metric_name);
            let mean = m["mean"].as_f64().unwrap();
            let std = m["std"].as_f64().unwrap();
            let min = m["min"].as_f64().unwrap();
            let max = m["max"].as_f64().unwrap();
            let ci_low = m["ci_low"].as_f64().unwrap();
            let ci_high = m["ci_high"].as_f64().unwrap();
            let eps = 1e-9;
            assert!(min - eps <= mean && mean <= max + eps);
            assert!(std >= 0.0);
            assert!(ci_low - eps <= mean && mean <= ci_high + eps);
        }
    }
}

#[tokio::test]
#[ignore = "slow: two 4-value x 3-trial sweeps"]
async fn test_sweep_level_statistics() {
    use std::collections::HashMap;

    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();

    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --concurrency 2,4,6,8 --num-profile-runs 3 \
             --parameter-sweep-mode repeated --request-count 10 \
             --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        420,
    );
    assert_eq!(r.exit_code, 0, "Repeated mode sweep should succeed");

    let sweep_json = root
        .join("aggregate")
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(sweep_json.exists());
    let sweep_data = jload(&sweep_json);

    let metadata = &sweep_data["metadata"];
    assert_eq!(metadata["aggregation_type"], json!("sweep"));
    let sweep_params = metadata["sweep_parameters"].as_array().unwrap();
    assert_eq!(sweep_params.len(), 1);
    assert_eq!(sweep_params[0]["name"], json!("concurrency"));
    assert_eq!(sweep_params[0]["values"], json!([2, 4, 6, 8]));
    assert_eq!(metadata["num_combinations"], json!(4));
    assert_eq!(metadata["sweep_mode"], json!("repeated"));

    let best_configs = &sweep_data["best_configurations"];
    let best_throughput = &best_configs["best_throughput"];
    assert!(best_throughput.get("parameters").is_some());
    assert!(best_throughput.get("metric").is_some());
    assert!(best_throughput.get("unit").is_some());
    assert!(best_throughput["parameters"]["concurrency"].is_number());
    let bt_c = best_throughput["parameters"]["concurrency"]
        .as_i64()
        .unwrap();
    assert!([2, 4, 6, 8].contains(&bt_c));
    assert!(best_throughput["metric"].as_f64().unwrap() > 0.0);
    let bt_unit = best_throughput["unit"].as_str().unwrap().to_lowercase();
    assert!(bt_unit.contains("request") || bt_unit.contains("req"));

    let best_latency = &best_configs["best_latency_p99"];
    assert!(best_latency.get("parameters").is_some());
    assert!(best_latency.get("metric").is_some());
    assert!(best_latency.get("unit").is_some());
    let bl_c = best_latency["parameters"]["concurrency"].as_i64().unwrap();
    assert!([2, 4, 6, 8].contains(&bl_c));
    assert!(best_latency["metric"].as_f64().unwrap() > 0.0);
    let bl_unit = best_latency["unit"].as_str().unwrap().to_lowercase();
    assert!(bl_unit.contains("ms") || bl_unit.contains("sec"));

    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();
    let mut throughput_values: HashMap<i64, f64> = HashMap::new();
    let mut latency_values: HashMap<i64, f64> = HashMap::new();
    for combo in per_combo {
        let c = combo["parameters"]["concurrency"].as_i64().unwrap();
        let metrics = &combo["metrics"];
        let tkeys: Vec<String> = metrics
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| {
                let l = k.to_lowercase();
                l.contains("throughput") && l.contains("request")
            })
            .cloned()
            .collect();
        if let Some(k) = tkeys.first() {
            throughput_values.insert(c, metrics[k]["mean"].as_f64().unwrap());
        }
        let lkeys: Vec<String> = metrics
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| {
                let l = k.to_lowercase();
                l.contains("ttft") && l.contains("p99")
            })
            .cloned()
            .collect();
        if let Some(k) = lkeys.first() {
            latency_values.insert(c, metrics[k]["mean"].as_f64().unwrap());
        }
    }

    if !throughput_values.is_empty() {
        let max_c = *throughput_values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            bt_c, max_c,
            "best_throughput should be max-throughput concurrency"
        );
    }
    if !latency_values.is_empty() {
        let min_c = *latency_values
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            bl_c, min_c,
            "best_latency should be min-latency concurrency"
        );
    }

    let pareto = sweep_data["pareto_optimal"].as_array().unwrap();
    assert!(
        !pareto.is_empty(),
        "Must have at least one Pareto optimal point"
    );
    let pareto_c: Vec<i64> = pareto
        .iter()
        .map(|p| p["concurrency"].as_i64().unwrap())
        .collect();
    for v in &pareto_c {
        assert!([2, 4, 6, 8].contains(v));
    }
    let mut sorted_pc = pareto_c.clone();
    sorted_pc.sort();
    assert_eq!(
        pareto_c, sorted_pc,
        "Pareto optimal points should be sorted"
    );

    if !throughput_values.is_empty() && !latency_values.is_empty() {
        for &pv in &pareto_c {
            for ov in [2, 4, 6, 8] {
                if ov == pv {
                    continue;
                }
                let ot = throughput_values.get(&ov).copied().unwrap_or(0.0);
                let pt = throughput_values.get(&pv).copied().unwrap_or(0.0);
                let ol = latency_values.get(&ov).copied().unwrap_or(f64::INFINITY);
                let pl = latency_values.get(&pv).copied().unwrap_or(f64::INFINITY);
                assert!(!(ot > pt && ol < pl), "Pareto point {pv} dominated by {ov}");
            }
        }
    }
    if pareto_c.len() > 1 {
        let unique: std::collections::HashSet<i64> = pareto_c.iter().copied().collect();
        assert_eq!(
            pareto_c.len(),
            unique.len(),
            "Pareto points should be distinct"
        );
    }

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6,8 --num-profile-runs 3 \
         --parameter-sweep-mode independent --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "Independent mode sweep should succeed");

    let sweep_json = root
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(sweep_json.exists());
    let sweep_data = jload(&sweep_json);
    let metadata = &sweep_data["metadata"];
    assert_eq!(metadata["aggregation_type"], json!("sweep"));
    let sweep_params = metadata["sweep_parameters"].as_array().unwrap();
    assert_eq!(sweep_params.len(), 1);
    assert_eq!(sweep_params[0]["name"], json!("concurrency"));
    assert_eq!(sweep_params[0]["values"], json!([2, 4, 6, 8]));
    assert_eq!(metadata["num_combinations"], json!(4));
    assert_eq!(metadata["sweep_mode"], json!("independent"));

    let best_configs = &sweep_data["best_configurations"];
    let bt = &best_configs["best_throughput"];
    assert!(bt.get("parameters").is_some());
    assert!(bt.get("metric").is_some());
    assert!(bt.get("unit").is_some());
    assert!([2, 4, 6, 8].contains(&bt["parameters"]["concurrency"].as_i64().unwrap()));
    assert!(bt["metric"].as_f64().unwrap() > 0.0);
    let bl = &best_configs["best_latency_p99"];
    assert!(bl.get("parameters").is_some());
    assert!(bl.get("metric").is_some());
    assert!(bl.get("unit").is_some());
    assert!([2, 4, 6, 8].contains(&bl["parameters"]["concurrency"].as_i64().unwrap()));
    assert!(bl["metric"].as_f64().unwrap() > 0.0);

    let pareto = sweep_data["pareto_optimal"].as_array().unwrap();
    assert!(!pareto.is_empty());
    let pareto_c: Vec<i64> = pareto
        .iter()
        .map(|p| p["concurrency"].as_i64().unwrap())
        .collect();
    for v in &pareto_c {
        assert!([2, 4, 6, 8].contains(v));
    }
    let mut sorted_pc = pareto_c.clone();
    sorted_pc.sort();
    assert_eq!(pareto_c, sorted_pc);
}

#[tokio::test]
async fn test_sweep_only_mode_without_confidence() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    for c in [2, 4, 6] {
        let cdir = root.join(format!("concurrency_{c}"));
        assert!(cdir.exists(), "concurrency_{c} directory should exist");
        assert!(
            !cdir.join("profile_runs").exists(),
            "concurrency_{c} should NOT have profile_runs in sweep-only mode"
        );
        assert!(
            !cdir.join("aggregate").exists(),
            "concurrency_{c} should NOT have aggregate in sweep-only mode"
        );

        let json_file = cdir.join("profile_export_aiperf.json");
        assert!(json_file.exists());
        assert!(cdir.join("profile_export_aiperf.csv").exists());
        assert_eq!(request_count_avg(&json_file), 10.0);
    }
    // Single-trial sweeps may omit or emit top-level aggregate directories.
}

#[tokio::test]
async fn test_sweep_directory_structure_consumable_by_plot() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let profile_result = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 2 \
         --parameter-sweep-mode repeated --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(
        profile_result.exit_code, 0,
        "Profile command should succeed"
    );

    let profile_runs_dir = root.join("profile_runs");
    assert!(profile_runs_dir.exists());
    let trial_dirs = sorted_dirs(&profile_runs_dir, "trial_");
    assert_eq!(trial_dirs.len(), 2);
    for trial_dir in &trial_dirs {
        for c in [2, 4, 6] {
            let cdir = trial_dir.join(format!("concurrency_{c}"));
            assert!(cdir.exists());
            assert!(cdir.join("profile_export_aiperf.json").exists());
            assert!(cdir.join("profile_export_aiperf.csv").exists());
        }
    }

    let plot_result = h.run_no_server(&format!("plot --paths {}", root.display()));
    assert_eq!(plot_result.exit_code, 0, "Plot command should succeed");

    let plot_dir = root.join("plots");
    assert!(plot_dir.exists(), "Plot directory should be created");
    let plot_log = plot_dir.join("aiperf_plot.log");
    assert!(plot_log.exists(), "Plot log should be created");

    let log_content = read_text(&plot_log);
    assert!(
        log_content.contains("Found 3 unique run directories"),
        "Plot should detect 3 aggregate cells (one per concurrency value)"
    );
    assert!(
        log_content.contains("MULTI_RUN mode"),
        "Plot should detect multi-run mode for sweep"
    );
}

#[tokio::test]
async fn test_sweep_aggregate_structure_validation() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let profile_result = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --num-profile-runs 2 \
         --parameter-sweep-mode repeated --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(profile_result.exit_code, 0, "Profile should succeed");

    let sweep_json = root
        .join("aggregate")
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(sweep_json.exists(), "Sweep JSON should exist");
    let sweep_data = jload(&sweep_json);

    assert!(sweep_data.get("metadata").is_some());
    assert!(sweep_data.get("per_combination_metrics").is_some());
    assert!(sweep_data.get("best_configurations").is_some());
    assert!(sweep_data.get("pareto_optimal").is_some());

    let metadata = &sweep_data["metadata"];
    let sweep_params = metadata["sweep_parameters"].as_array().unwrap();
    assert!(!sweep_params.is_empty());

    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();
    assert_eq!(per_combo.len(), 3);
    for combo in per_combo {
        assert!(combo.get("parameters").is_some());
        assert!(combo.get("metrics").is_some());
        assert!(combo["parameters"].is_object());
        assert!(combo["parameters"].get("concurrency").is_some());
    }

    let best_configs = &sweep_data["best_configurations"];
    if let Some(bt) = best_configs.get("best_throughput") {
        assert!(bt.get("parameters").is_some());
        assert!(bt["parameters"].is_object());
        assert!(bt.get("metric").is_some());
        assert!(bt.get("unit").is_some());
    }

    let pareto = sweep_data["pareto_optimal"].as_array().unwrap();
    for config in pareto {
        assert!(config.is_object());
        assert!(config.get("concurrency").is_some());
    }
}

#[tokio::test]
async fn test_sweep_with_cooldown_flag_succeeds() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4 --parameter-sweep-cooldown-seconds 1 \
         --request-count 10 --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    for val in [2, 4] {
        let json_file = root
            .join(format!("concurrency_{val}"))
            .join("profile_export_aiperf.json");
        assert!(
            json_file.exists(),
            "concurrency_{val} should have artifacts"
        );
    }
}

#[tokio::test]
async fn test_sweep_with_same_seed_flag_succeeds() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4 --parameter-sweep-same-seed \
         --request-count 10 --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0);

    for val in [2, 4] {
        let json_file = root
            .join(format!("concurrency_{val}"))
            .join("profile_export_aiperf.json");
        assert!(
            json_file.exists(),
            "concurrency_{val} should have artifacts"
        );
    }
}

fn assert_sweep_json_shape(sweep_json: &Path, mode: &str, values: &[i64]) {
    let sweep_data = jload(sweep_json);
    let metadata = &sweep_data["metadata"];
    assert_eq!(metadata["aggregation_type"], json!("sweep"));

    let sweep_params = metadata["sweep_parameters"].as_array().unwrap();
    assert_eq!(sweep_params.len(), 1);
    assert_eq!(sweep_params[0]["name"], json!("concurrency"));
    assert_eq!(sweep_params[0]["values"], Value::from(values.to_vec()));
    assert_eq!(metadata["num_combinations"], json!(values.len()));
    assert_eq!(metadata["num_trials_per_value"], json!(3));
    assert_eq!(metadata["sweep_mode"], json!(mode));
    assert_eq!(metadata["confidence_level"], json!(0.95));

    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();
    assert_eq!(per_combo.len(), values.len());

    let mut found: Vec<i64> = Vec::new();
    for combo in per_combo {
        assert!(combo.get("parameters").is_some());
        assert!(combo.get("metrics").is_some());
        let c = combo["parameters"]["concurrency"].as_i64().unwrap();
        found.push(c);
        let metrics = &combo["metrics"];
        assert!(!metrics.as_object().unwrap().is_empty());
        let tkeys = metric_keys_with(metrics, "throughput");
        assert!(!tkeys.is_empty());
        for f in ["mean", "std", "ci_low", "ci_high"] {
            assert!(
                metrics[&tkeys[0]].get(f).is_some(),
                "combo metric should have {f}"
            );
        }
    }
    found.sort();
    assert_eq!(found, values.to_vec());

    let best_configs = &sweep_data["best_configurations"];
    let bt = &best_configs["best_throughput"];
    assert!(bt.get("parameters").is_some());
    assert!(bt.get("metric").is_some());
    assert!(bt.get("unit").is_some());
    assert!(values.contains(&bt["parameters"]["concurrency"].as_i64().unwrap()));
    let bl = &best_configs["best_latency_p99"];
    assert!(bl.get("parameters").is_some());
    assert!(bl.get("metric").is_some());
    assert!(bl.get("unit").is_some());
    assert!(values.contains(&bl["parameters"]["concurrency"].as_i64().unwrap()));

    let pareto = sweep_data["pareto_optimal"].as_array().unwrap();
    assert!(!pareto.is_empty());
    for params in pareto {
        assert!(params.is_object());
        assert!(params.get("concurrency").is_some());
        assert!(values.contains(&params["concurrency"].as_i64().unwrap()));
    }
}

fn assert_sweep_json_full(sweep_json: &Path, mode: &str, values: &[i64]) {
    let sweep_data = jload(sweep_json);
    let required_top_level = [
        "aggregation_type",
        "num_profile_runs",
        "num_successful_runs",
        "failed_runs",
        "metadata",
        "per_combination_metrics",
        "best_configurations",
        "pareto_optimal",
    ];
    for key in required_top_level {
        assert!(
            sweep_data.get(key).is_some(),
            "Sweep JSON must have {key} key"
        );
    }
    assert!(
        sweep_data.get("trends").is_none(),
        "Sweep aggregate should NOT have trends"
    );

    assert_sweep_json_shape(sweep_json, mode, values);

    let per_combo = sweep_data["per_combination_metrics"].as_array().unwrap();
    for combo in per_combo {
        for (name, m) in combo["metrics"].as_object().unwrap() {
            for f in ["mean", "std", "ci_low", "ci_high", "unit"] {
                assert!(m.get(f).is_some(), "combo metric {name} must have {f}");
            }
        }
    }
}

fn assert_sweep_csv_shape(sweep_csv: &Path, values: &[i64]) {
    let csv = read_text(sweep_csv);
    let lines: Vec<&str> = csv.trim().split('\n').collect();
    assert!(lines.len() > 1, "Sweep CSV should have data rows");
    let header = lines[0];
    assert!(
        header.contains("concurrency"),
        "Sweep CSV should have concurrency column"
    );
    let suffixes = ["_mean", "_std", "_min", "_max", "_cv"];
    assert!(
        suffixes.iter().any(|s| header.contains(s)),
        "Sweep CSV should have metric columns with statistical suffixes"
    );
    let full = lines.join("\n");
    for c in values {
        assert!(
            full.contains(&c.to_string()),
            "Sweep CSV must have data for {c}"
        );
    }
}
