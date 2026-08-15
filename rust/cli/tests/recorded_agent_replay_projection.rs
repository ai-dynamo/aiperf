// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI and YAML projection parity for recorded-agent replay configuration.

use std::fs;

use aiperf_cli::flags::ProfileFlags;
use clap::Parser;
use serde_json::Value;

fn graph_projection(run: &aiperf_cli::model::BenchmarkRun) -> Value {
    let value = serde_json::to_value(run).expect("run serializes");
    json_subset(&value["cfg"], &["datasets", "metadata", "artifacts"])
}

fn json_subset(value: &Value, keys: &[&str]) -> Value {
    let object = value.as_object().expect("config is an object");
    keys.iter()
        .map(|key| ((*key).to_string(), object.get(*key).cloned().expect("projected key")))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

#[test]
fn cli_and_yaml_project_identical_agent_recording_graph_config() {
    let flags = ProfileFlags::try_parse_from([
        "aiperf", "--model", "model", "--url", "http://127.0.0.1:8000", "--endpoint-type", "chat",
        "--input-file", "/tmp/recording.json", "--graph-format", "agent_recording",
        "--graph-replay-root", "/tmp/replay", "--graph-execute-tools", "--graph-tool-image", "tools:latest",
        "--graph-pinch-image", "pinch:latest", "--graph-tool-command-timeout", "9.5",
        "--graph-tool-container-stop-timeout", "4", "--graph-tool-session-close-grace", "1.5",
        "--no-graph-use-family-sampling", "--graph-emit-warmup", "--graph-stop-on-failure",
        "--hardware-description", "unknown", "--endpoint-placement", "remote",
    ])
    .expect("profile flags parse");
    let cli_run = aiperf_cli::load::resolve(&flags).expect("CLI resolves");

    let yaml = r#"
model: model
url: http://127.0.0.1:8000
endpoint:
  type: chat
  streaming: false
dataset:
  type: file
  path: /tmp/recording.json
  format: agent_recording
  graph:
    replay_root: /tmp/replay
    execute_tools: true
    tool_image: tools:latest
    pinch_image: pinch:latest
    command_timeout_seconds: 9.5
    container_stop_timeout_seconds: 4
    session_close_grace_seconds: 1.5
    use_family_sampling: false
    emit_warmup: true
    stop_on_failure: true
metadata:
  hardware: unknown
  endpoint_placement: remote
"#;
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("recorded-agent.yaml");
    fs::write(&path, yaml).expect("write config");
    let yaml_run = aiperf_cli::yaml::resolve(&path, Some(directory.path().join("artifacts")))
        .expect("YAML resolves");

    assert_eq!(graph_projection(&cli_run), graph_projection(&yaml_run));
}
