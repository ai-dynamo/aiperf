// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "engine")]

//! Strict protocol-v2 projection coverage for recorded-agent replay.

use aiperf_runtime::engine::protocol_v2::BenchmarkRunWireV2;
use serde_json::{Value, json};

fn run_with_artifacts(artifacts: Value) -> BenchmarkRunWireV2 {
    serde_json::from_value(json!({
        "benchmark_id": "recorded-agent-protocol",
        "artifact_dir": "/tmp/recorded-agent-protocol",
        "cfg": {
            "models": {"strategy": "round_robin", "items": [{"name": "model"}]},
            "endpoint": {
                "type": "chat", "urls": ["http://127.0.0.1:8000"], "streaming": true,
                "use_legacy_max_tokens": false, "use_server_token_count": false,
                "timeout_seconds": 30.0, "connection_reuse": "pooled", "ssl_verify": true,
                "connection_limit": 1, "keepalive_timeout": 1.0,
                "download_video_content": false, "extra": {}, "headers": {}, "http2": false,
                "wait_for_model_timeout": 0.0, "wait_for_model_interval": 1.0,
                "wait_for_model_mode": "inference"
            },
            "tokenizer": {"name": "model", "revision": "main", "trust_remote_code": false, "apply_chat_template": false},
            "transport": {"type": "http"},
            "runtime": {"workers": 1, "workers_min": null, "cells": 1},
            "datasets": [{
                "type": "file", "format": "agent_recording", "path": "/tmp/recording.json",
                "sampling": "sequential", "options": {},
                "graph": {
                    "replay_root": "/tmp/replay", "source_format": "claude_code",
                    "include_subagents": false, "execute_tools": false,
                    "command_timeout_seconds": 9.5, "container_stop_timeout_seconds": 4.0,
                    "session_close_grace_seconds": 1.5, "use_family_sampling": false,
                    "emit_warmup": true, "resume": false, "stop_on_failure": true
                }
            }],
            "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 1, "exclude_from_results": false, "seamless": false, "requests": 1}],
            "metadata": {"hardware": "unknown", "endpoint_placement": "remote"},
            "artifacts": artifacts
        }
    }))
    .expect("recorded-agent run must decode")
}

#[test]
fn recorded_agent_import_contract_round_trips_graph_source_configuration() {
    let artifacts = json!({
        "trace": false, "inputs_path": "inputs.json",
        "graph_tool_time_path": "tool-time.json",
        "graph_trace_summary_path": "trace-summary.json",
        "graph_replay_metrics_path": "metrics.json",
        "graph_replay_metrics_csv_path": "metrics.csv",
        "graph_replay_failures_path": "failures.tsv",
        "graph_replay_provenance_path": "provenance.json",
        "graph_replay_backend_metadata_path": "backend.json"
    });
    let authored = run_with_artifacts(artifacts)
        .into_authored()
        .expect("typed run projects to protocol");
    authored.validate_outer().expect("artifact paths validate");
    assert_eq!(authored.workload.id.as_str(), "graph");
    let graph: Value =
        serde_json::from_str(authored.workload.config.get()).expect("graph config JSON");
    assert_eq!(graph["dataset"]["graph"]["source_format"], "claude_code");
    assert_eq!(graph["dataset"]["graph"]["include_subagents"], false);
    assert_eq!(graph["dataset"]["graph"]["execute_tools"], false);
    assert_eq!(graph["dataset"]["graph"]["command_timeout_seconds"], 9.5);
    assert_eq!(
        authored.artifacts.graph_replay_metrics_csv_path.as_deref(),
        Some(std::path::Path::new("metrics.csv"))
    );
    assert_eq!(
        authored
            .artifacts
            .graph_replay_backend_metadata_path
            .as_deref(),
        Some(std::path::Path::new("backend.json"))
    );

    let bad_artifacts = json!({
        "trace": false, "inputs_path": "inputs.json",
        "graph_tool_time_path": "../tool-time.json"
    });
    let error = run_with_artifacts(bad_artifacts)
        .into_authored()
        .expect("typed run projects")
        .validate_outer()
        .expect_err("parent traversal is rejected")
        .to_string();
    assert!(error.contains("graph_tool_time_path") && error.contains("normal relative"));
}
