// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::{Path, PathBuf};

use aiperf_mock_server::config::MockServerConfig;

const WORKERS_MAX: u32 = 8;
const UI: &str = "simple";

fn run_dirs(artifact_dir: &Path) -> Vec<PathBuf> {
    let profile_runs = artifact_dir.join("profile_runs");
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(&profile_runs) {
        Ok(rd) => rd
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("run_"))
                        .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    dirs.sort();
    dirs
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).expect("read json artifact");
    serde_json::from_slice(&bytes).expect("parse json artifact")
}

fn error_rate_config(error_rate: f64) -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 8;
    cfg.no_tokenizer = true;
    cfg.error_rate = error_rate;
    cfg
}

#[tokio::test]
async fn test_multi_run_basic() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 3 --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    let dir = h.artifact_path();

    let profile_runs_dir = dir.join("profile_runs");
    assert!(
        profile_runs_dir.exists(),
        "profile_runs directory should exist"
    );

    let dirs = run_dirs(dir);
    assert_eq!(dirs.len(), 3, "Should have 3 run directories");

    assert_eq!(dirs[0].file_name().unwrap(), "run_0001");
    assert_eq!(dirs[1].file_name().unwrap(), "run_0002");
    assert_eq!(dirs[2].file_name().unwrap(), "run_0003");

    for run_dir in &dirs {
        let json_file = run_dir.join("profile_export_aiperf.json");
        let csv_file = run_dir.join("profile_export_aiperf.csv");
        assert!(json_file.exists(), "run dir should have JSON artifact");
        assert!(csv_file.exists(), "run dir should have CSV artifact");

        let run_data = read_json(&json_file);
        assert_eq!(run_data["request_count"]["avg"].as_f64().unwrap(), 10.0);
    }

    let aggregate_dir = dir.join("aggregate");
    assert!(aggregate_dir.exists(), "aggregate directory should exist");

    let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
    let agg_csv = aggregate_dir.join("profile_export_aiperf_aggregate.csv");
    assert!(agg_json.exists(), "Aggregate JSON should exist");
    assert!(agg_csv.exists(), "Aggregate CSV should exist");

    let agg_data = read_json(&agg_json);

    assert_eq!(agg_data["metadata"]["aggregation_type"], "confidence");
    assert_eq!(
        agg_data["metadata"]["num_profile_runs"].as_i64().unwrap(),
        3
    );
    assert_eq!(
        agg_data["metadata"]["num_successful_runs"]
            .as_i64()
            .unwrap(),
        3
    );
    assert_eq!(
        agg_data["metadata"]["failed_runs"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        agg_data["metadata"]["confidence_level"].as_f64().unwrap(),
        0.95
    );
    assert_eq!(
        agg_data["metadata"]["run_labels"].as_array().unwrap().len(),
        3
    );

    let metrics = agg_data["metrics"].as_object().expect("metrics object");
    assert!(!metrics.is_empty(), "Should have aggregated metrics");

    let throughput_metrics: Vec<&String> = metrics
        .keys()
        .filter(|k| k.to_lowercase().contains("throughput"))
        .collect();
    assert!(
        !throughput_metrics.is_empty(),
        "Should have throughput metrics"
    );

    let sample_metric = &metrics[throughput_metrics[0]];
    let required_fields = [
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
    for field in required_fields {
        assert!(
            sample_metric.get(field).is_some(),
            "Metric should have {field} field"
        );
    }
}

#[tokio::test]
async fn test_single_run_uses_root_artifacts() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 1 --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    let dir = h.artifact_path();

    assert!(
        !dir.join("profile_runs").exists(),
        "profile_runs should not exist for single run"
    );
    assert!(
        !dir.join("aggregate").exists(),
        "aggregate should not exist for single run"
    );

    assert!(
        dir.join("profile_export_aiperf.json").exists(),
        "Standard JSON should exist at root"
    );
    assert!(
        dir.join("profile_export_aiperf.csv").exists(),
        "Standard CSV should exist at root"
    );
}

#[tokio::test]
async fn test_multi_run_with_cooldown() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 2 --profile-run-cooldown-seconds 0.5 --request-count 5 \
             --concurrency {DEFAULT_CONCURRENCY} --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        300,
    );

    assert_eq!(r.exit_code, 0);

    let agg_json = h
        .artifact_path()
        .join("aggregate")
        .join("profile_export_aiperf_aggregate.json");
    let agg_data = read_json(&agg_json);
    assert_eq!(
        agg_data["metadata"]["cooldown_seconds"].as_f64().unwrap(),
        0.5
    );
}

#[tokio::test]
async fn test_multi_run_custom_confidence_level() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 3 --confidence-level 0.99 --request-count 10 \
         --concurrency {DEFAULT_CONCURRENCY} --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    let agg_json = h
        .artifact_path()
        .join("aggregate")
        .join("profile_export_aiperf_aggregate.json");
    let agg_data = read_json(&agg_json);
    assert_eq!(
        agg_data["metadata"]["confidence_level"].as_f64().unwrap(),
        0.99
    );
}

#[tokio::test]
async fn test_multi_run_concurrency_mode() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 2 --concurrency 4 --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    let dirs = run_dirs(h.artifact_path());
    assert_eq!(dirs.len(), 2);
}

#[tokio::test]
async fn test_multi_run_request_rate_mode() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 2 --request-rate 5.0 --request-count 10 \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    let dirs = run_dirs(h.artifact_path());
    assert_eq!(dirs.len(), 2);
}

#[tokio::test]
async fn test_multi_run_with_warmup() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 2 --warmup-request-count 5 --request-count 10 \
         --concurrency {DEFAULT_CONCURRENCY} --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    for run_dir in run_dirs(h.artifact_path()) {
        let json_file = run_dir.join("profile_export_aiperf.json");
        let run_data = read_json(&json_file);
        assert_eq!(run_data["request_count"]["avg"].as_f64().unwrap(), 10.0);
    }
}

#[tokio::test]
async fn test_aggregate_csv_format() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 2 --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_eq!(r.exit_code, 0);

    let agg_csv = h
        .artifact_path()
        .join("aggregate")
        .join("profile_export_aiperf_aggregate.csv");
    let csv_content = std::fs::read_to_string(&agg_csv).expect("read aggregate csv");
    let lines: Vec<&str> = csv_content.trim().split('\n').collect();

    let header = lines[0];
    let required_columns = [
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
    for col in required_columns {
        assert!(header.contains(col), "CSV should have {col} column");
    }

    assert!(lines.len() > 1, "CSV should have data rows");
}

#[tokio::test]
async fn test_multi_run_with_partial_failures() {
    let h = AIPerfHarness::new().await;
    let _r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 5 --request-count 100 --concurrency 50 \
             --benchmark-duration 0.1 --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        300,
    );

    let dir = h.artifact_path();
    let profile_runs_dir = dir.join("profile_runs");
    if profile_runs_dir.exists() {
        let dirs = run_dirs(dir);
        assert!(
            !dirs.is_empty(),
            "Should have at least some run directories"
        );

        let aggregate_dir = dir.join("aggregate");
        if aggregate_dir.exists() {
            let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
            if agg_json.exists() {
                let agg_data = read_json(&agg_json);

                assert!(agg_data["metadata"].get("num_successful_runs").is_some());
                assert!(agg_data["metadata"].get("failed_runs").is_some());

                let num_successful = agg_data["metadata"]["num_successful_runs"]
                    .as_i64()
                    .unwrap();
                let failed = agg_data["metadata"]["failed_runs"].as_array().unwrap();
                let num_failed = failed.len() as i64;

                assert_eq!(num_successful + num_failed, 5);

                if num_failed > 0 {
                    for failed_run in failed {
                        assert!(failed_run.get("label").is_some());
                        assert!(failed_run.get("error").is_some());
                        assert!(failed_run["label"].as_str().unwrap().starts_with("run_"));
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn test_multi_run_insufficient_successful_runs() {
    let h = AIPerfHarness::new_with(error_rate_config(100.0)).await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 3 --request-count 5 --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_ne!(
        r.exit_code, 0,
        "Should exit with non-zero code when all runs fail"
    );

    let aggregate_dir = h.artifact_path().join("aggregate");
    if aggregate_dir.exists() {
        let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
        assert!(
            !agg_json.exists(),
            "Aggregate JSON should not exist with insufficient successful runs"
        );
    }
}

#[tokio::test]
async fn test_multi_run_all_runs_fail() {
    let h = AIPerfHarness::new_with(error_rate_config(100.0)).await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-profile-runs 3 --request-count 5 --concurrency {DEFAULT_CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));

    assert_ne!(
        r.exit_code, 0,
        "Should exit with non-zero code when all runs fail"
    );

    let aggregate_dir = h.artifact_path().join("aggregate");
    if aggregate_dir.exists() {
        let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
        assert!(
            !agg_json.exists(),
            "Aggregate JSON should not exist when all runs fail"
        );
    }
}

#[tokio::test]
async fn test_multi_run_single_failure_still_aggregates() {
    let h = AIPerfHarness::new().await;
    let _r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 3 --request-count 20 --concurrency {DEFAULT_CONCURRENCY} \
             --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        300,
    );

    let aggregate_dir = h.artifact_path().join("aggregate");
    if aggregate_dir.exists() {
        let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
        if agg_json.exists() {
            let agg_data = read_json(&agg_json);

            let num_successful = agg_data["metadata"]["num_successful_runs"]
                .as_i64()
                .unwrap();
            assert!(
                num_successful >= 2,
                "Aggregate requires at least 2 successful runs"
            );

            assert!(
                !agg_data["metrics"].as_object().unwrap().is_empty(),
                "Should have aggregated metrics"
            );
        }
    }
}

#[tokio::test]
async fn test_multi_run_preserves_failed_run_artifacts() {
    let h = AIPerfHarness::new().await;
    let _r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 3 --request-count 100 --concurrency 100 \
             --benchmark-duration 0.05 --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        300,
    );

    let dir = h.artifact_path();
    let profile_runs_dir = dir.join("profile_runs");
    if profile_runs_dir.exists() {
        let dirs = run_dirs(dir);
        assert!(!dirs.is_empty(), "Should have run directories");
        for run_dir in &dirs {
            assert!(run_dir.is_dir(), "run dir should be a directory");
        }
    }
}

#[tokio::test]
async fn test_multi_run_invalid_num_profile_runs() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 0 --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
             --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        30,
    );

    assert_ne!(r.exit_code, 0);
    let output = format!("{}{}", r.stdout, r.stderr).to_lowercase();
    assert!(output.contains("num-profile-runs") || output.contains("num_profile_runs"));
}

#[tokio::test]
async fn test_multi_run_exceeds_max_limit() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 11 --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
             --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        30,
    );

    assert_ne!(r.exit_code, 0);
    let output = format!("{}{}", r.stdout, r.stderr).to_lowercase();
    assert!(
        ["10", "limit", "maximum", "validation"]
            .iter()
            .any(|phrase| output.contains(phrase))
    );
}

#[tokio::test]
async fn test_multi_run_invalid_confidence_level() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 3 --confidence-level 1.5 --request-count 10 \
             --concurrency {DEFAULT_CONCURRENCY} --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        30,
    );

    assert_ne!(r.exit_code, 0);
    let output = format!("{}{}", r.stdout, r.stderr).to_lowercase();
    assert!(output.contains("confidence"));
}

#[tokio::test]
async fn test_multi_run_negative_cooldown() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 2 --profile-run-cooldown-seconds -1.0 --request-count 10 \
             --concurrency {DEFAULT_CONCURRENCY} --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        30,
    );

    assert_ne!(r.exit_code, 0);
    let output = format!("{}{}", r.stdout, r.stderr).to_lowercase();
    assert!(output.contains("cooldown") || output.contains("negative"));
}
