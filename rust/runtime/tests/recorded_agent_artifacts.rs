// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for controller-owned recorded-agent replay artifacts.

use std::fs;

use aiperf_runtime::graph::replay::{
    ReplayArtifactPaths, ReplayCallMeasurement, ReplayTraceSupplement, ToolCallMeasurement,
    write_replay_artifacts,
};

#[test]
fn tool_time_schema_excludes_failed_trace_and_reports_mixed_backend() {
    let output = tempfile::tempdir().expect("temporary output directory");
    let tool_time_path = output.path().join("tool-time.json");
    let trace_summary_path = output.path().join("trace-summary.json");

    write_replay_artifacts(
        &ReplayArtifactPaths {
            tool_time_path: Some(tool_time_path.clone()),
            trace_summary_path: Some(trace_summary_path),
            metrics_json_path: None,
            metrics_csv_path: None,
        },
        &[
            ReplayTraceSupplement {
                trace_id: "successful".into(),
                trajectory_id: "trajectory-successful".into(),
                worker_id: 0,
                completed: true,
                calls: vec![ReplayCallMeasurement::completed("successful", 0)],
                tools: vec![
                    ToolCallMeasurement::new(0.2, "local").with_call_index(0),
                    ToolCallMeasurement::new(0.4, "docker:pinch:latest").with_call_index(1),
                ],
                trace_wall_ms: 1_000.0,
            },
            ReplayTraceSupplement {
                trace_id: "failed".into(),
                trajectory_id: "trajectory-failed".into(),
                worker_id: 1,
                completed: false,
                calls: vec![ReplayCallMeasurement::completed("failed", 0)],
                tools: vec![ToolCallMeasurement::new(99.0, "local")],
                trace_wall_ms: 99_000.0,
            },
        ],
    )
    .expect("controller writes strict artifacts");

    let artifact: serde_json::Value =
        serde_json::from_slice(&fs::read(tool_time_path).expect("tool-time artifact exists"))
            .expect("strict JSON");
    assert_eq!(
        artifact,
        serde_json::json!({
            "command_count": 2,
            "trace_count": 1,
            "backend": "mixed",
            "total_s": 0.6,
            "mean_s": 0.3,
            "median_s": 0.3,
            "max_s": 0.4,
            "durations_s": [0.2, 0.4]
        })
    );
}

#[test]
fn strict_writer_rejects_nonfinite_overflow_and_unknown_backend_before_json() {
    let output = tempfile::tempdir().expect("temporary output directory");
    let path = output.path().join("tool-time.json");
    let paths = ReplayArtifactPaths {
        tool_time_path: Some(path.clone()),
        ..ReplayArtifactPaths::default()
    };
    for tool in [
        ToolCallMeasurement::new(f64::NAN, "local"),
        ToolCallMeasurement::new(f64::MAX, "local"),
        ToolCallMeasurement::new(1.0, "remote"),
    ] {
        let trace = ReplayTraceSupplement {
            trace_id: "strict".into(),
            trajectory_id: "trajectory-strict".into(),
            worker_id: 0,
            completed: true,
            calls: Vec::new(),
            tools: vec![tool.clone().with_call_index(0), tool.with_call_index(1)],
            trace_wall_ms: 1.0,
        };
        assert!(write_replay_artifacts(&paths, &[trace]).is_err());
        assert!(
            !path.exists(),
            "strict failure must not serialize JSON null"
        );
    }
}
