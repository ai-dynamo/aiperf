// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::PathBuf;

const WORKERS_MAX: u32 = 1;

fn run_dirs(profile_runs_dir: &std::path::Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(profile_runs_dir)
        .expect("read profile_runs dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("run_"))
                    .unwrap_or(false)
        })
        .collect();
    dirs.sort();
    dirs
}

#[tokio::test]
async fn test_adaptive_ci_width_stops_early() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 5 --convergence-metric time_to_first_token \
             --convergence-mode ci_width --convergence-threshold 0.20 \
             --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
             --workers-max {WORKERS_MAX} --ui none",
            h.mock.url
        ),
        600,
    );

    assert_eq!(r.exit_code, 0);

    let profile_runs_dir = h.artifact_path().join("profile_runs");
    assert!(
        profile_runs_dir.exists(),
        "profile_runs directory should exist"
    );

    let dirs = run_dirs(&profile_runs_dir);
    assert!(
        dirs.len() >= 2,
        "Should have at least 2 run directories (min_runs floor)"
    );
    assert!(
        dirs.len() <= 5,
        "Should have at most 5 run directories (max_runs cap)"
    );

    for run_dir in &dirs {
        let json_file = run_dir.join("profile_export_aiperf.json");
        assert!(
            json_file.exists(),
            "{:?} should have JSON artifact",
            run_dir.file_name()
        );
    }

    let aggregate_dir = h.artifact_path().join("aggregate");
    assert!(aggregate_dir.exists(), "aggregate directory should exist");

    let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
    assert!(agg_json.exists(), "Confidence aggregate JSON should exist");

    let agg_data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&agg_json).unwrap()).unwrap();
    assert_eq!(agg_data["metadata"]["aggregation_type"], "confidence");
    assert!(
        agg_data["metadata"]["num_successful_runs"]
            .as_u64()
            .unwrap()
            >= 2
    );
    assert!(agg_data.get("metrics").is_some());
    assert!(agg_data["metrics"].as_object().unwrap().len() > 0);

    let detailed_json = aggregate_dir.join("profile_export_aiperf_collated.json");
    assert!(
        detailed_json.exists(),
        "Collated aggregate JSON should exist"
    );

    let detailed_data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&detailed_json).unwrap()).unwrap();
    assert_eq!(detailed_data["metadata"]["aggregation_type"], "detailed");
    assert!(
        detailed_data["metadata"]["num_successful_runs"]
            .as_u64()
            .unwrap()
            >= 2
    );
    assert!(detailed_data.get("metrics").is_some());

    let metrics = detailed_data["metrics"].as_object().unwrap();
    if !metrics.is_empty() {
        let sample_metric = metrics.values().next().unwrap();
        assert!(sample_metric.get("combined").is_some());
        let combined = &sample_metric["combined"];
        for field in ["mean", "std", "p50", "p90", "p95", "p99", "count"] {
            assert!(
                combined.get(field).is_some(),
                "Combined stats should have {field}"
            );
        }
        assert!(sample_metric.get("per_run").is_some());
        assert!(sample_metric["per_run"].as_array().unwrap().len() >= 2);
    }
}

#[tokio::test]
async fn test_fixed_trials_without_convergence_flags() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 3 --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
             --workers-max {WORKERS_MAX} --ui none",
            h.mock.url
        ),
        600,
    );

    assert_eq!(r.exit_code, 0);

    let profile_runs_dir = h.artifact_path().join("profile_runs");
    assert!(profile_runs_dir.exists());
    let dirs = run_dirs(&profile_runs_dir);
    assert_eq!(
        dirs.len(),
        3,
        "FixedTrialsStrategy should run exactly 3 times"
    );

    let aggregate_dir = h.artifact_path().join("aggregate");
    assert!(aggregate_dir.exists());

    let agg_json = aggregate_dir.join("profile_export_aiperf_aggregate.json");
    assert!(agg_json.exists(), "Confidence aggregate should exist");

    let detailed_json = aggregate_dir.join("profile_export_aiperf_collated.json");
    assert!(
        !detailed_json.exists(),
        "Collated aggregate should NOT exist without convergence flags"
    );
}

#[tokio::test]
async fn test_adaptive_cv_mode() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 5 --convergence-metric time_to_first_token \
             --convergence-mode cv --convergence-threshold 0.20 \
             --request-count 10 --concurrency {DEFAULT_CONCURRENCY} \
             --workers-max {WORKERS_MAX} --ui none",
            h.mock.url
        ),
        600,
    );

    assert_eq!(r.exit_code, 0);

    let profile_runs_dir = h.artifact_path().join("profile_runs");
    let dirs = run_dirs(&profile_runs_dir);
    assert!(dirs.len() >= 2);
    assert!(dirs.len() <= 5);

    let aggregate_dir = h.artifact_path().join("aggregate");
    assert!(
        aggregate_dir
            .join("profile_export_aiperf_aggregate.json")
            .exists()
    );
    assert!(
        aggregate_dir
            .join("profile_export_aiperf_collated.json")
            .exists()
    );
}

#[tokio::test]
async fn test_adaptive_request_rate_mode() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --num-profile-runs 5 --convergence-metric time_to_first_token \
             --convergence-mode ci_width --convergence-threshold 0.20 \
             --request-rate 5.0 --request-count 10 \
             --workers-max {WORKERS_MAX} --ui none",
            h.mock.url
        ),
        600,
    );

    assert_eq!(r.exit_code, 0);

    let profile_runs_dir = h.artifact_path().join("profile_runs");
    let dirs = run_dirs(&profile_runs_dir);
    assert!(dirs.len() >= 2);
    assert!(dirs.len() <= 5);
}

#[tokio::test]
async fn test_convergence_metric_without_multi_run_fails() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --convergence-metric time_to_first_token --request-count 10 \
             --concurrency {DEFAULT_CONCURRENCY} --workers-max {WORKERS_MAX} --ui none",
            h.mock.url
        ),
        30,
    );

    assert_ne!(r.exit_code, 0);
    let output = format!("{}{}", r.stdout, r.stderr).to_lowercase();
    assert!(output.contains("convergence") || output.contains("num-profile-runs"));
}
