// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI and YAML projection parity for recorded-agent replay configuration.

use std::fs;

use aiperf_cli::flags::ProfileFlags;
use clap::Parser;
use serde_json::Value;

/// Run a test body on a stack large enough for clap's derived `ProfileFlags` parser.
fn on_big_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .expect("spawn worker")
        .join()
        .expect("worker panicked");
}

fn graph_projection(run: &aiperf_cli::model::BenchmarkRun) -> Value {
    let value = serde_json::to_value(run).expect("run serializes");
    json_subset(&value["cfg"], &["datasets", "metadata", "artifacts"])
}

fn json_subset(value: &Value, keys: &[&str]) -> Value {
    let object = value.as_object().expect("config is an object");
    keys.iter()
        .map(|key| {
            (
                (*key).to_string(),
                object.get(*key).cloned().expect("projected key"),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

#[test]
fn cli_and_yaml_project_identical_agent_recording_graph_config() {
    on_big_stack(cli_and_yaml_project_identical_agent_recording_graph_config_body);
}

fn cli_and_yaml_project_identical_agent_recording_graph_config_body() {
    let flags = ProfileFlags::try_parse_from([
        "aiperf",
        "--model",
        "model",
        "--url",
        "http://127.0.0.1:8000",
        "--endpoint-type",
        "chat",
        "--input-file",
        "/tmp/recording.json",
        "--graph-format",
        "agent_recording",
        "--graph-recording-source",
        "claude-code",
        "--graph-include-subagents=false",
        "--graph-replay-root",
        "/tmp/replay",
        "--graph-execute-tools",
        "--graph-tool-image",
        "tools:latest",
        "--graph-pinch-image",
        "pinch:latest",
        "--graph-tool-command-timeout",
        "9.5",
        "--graph-tool-container-stop-timeout",
        "4",
        "--graph-tool-session-close-grace",
        "1.5",
        "--no-graph-use-family-sampling",
        "--graph-emit-warmup",
        "--graph-stop-on-failure",
        "--hardware-description",
        "unknown",
        "--endpoint-placement",
        "remote",
    ])
    .expect("profile flags parse");
    let cli_run = aiperf_cli::load::resolve(&flags).expect("CLI resolves");

    let yaml = r#"
benchmark:
  model: model
  endpoint:
    url: http://127.0.0.1:8000
    type: chat
    streaming: false
  dataset:
    type: file
    path: /tmp/recording.json
    format: agent_recording
    graph:
      source_format: claude_code
      include_subagents: false
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
  profiling:
    concurrency: 1
    requests: 1
"#;
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("recorded-agent.yaml");
    fs::write(&path, yaml).expect("write config");
    let yaml_run = aiperf_cli::yaml::resolve(&path, Some(directory.path().join("artifacts")))
        .expect("YAML resolves");

    assert_eq!(graph_projection(&cli_run), graph_projection(&yaml_run));
    assert_eq!(
        graph_projection(&cli_run)["datasets"][0]["graph"]["source_format"],
        "claude_code"
    );
    assert_eq!(
        graph_projection(&cli_run)["datasets"][0]["graph"]["include_subagents"],
        false
    );
}

#[test]
fn graph_recording_source_and_subagent_flags_accept_the_documented_spellings() {
    on_big_stack(graph_recording_source_and_subagent_flags_accept_the_documented_spellings_body);
}

fn graph_recording_source_and_subagent_flags_accept_the_documented_spellings_body() {
    for args in [
        vec!["--graph-include-subagents"],
        vec!["--graph-include-subagents=true"],
        vec!["--graph-include-subagents=false"],
        vec!["--graph-recording-source", "mini-swe-agent"],
    ] {
        ProfileFlags::try_parse_from(std::iter::once("aiperf").chain(args))
            .expect("documented graph import flag parses");
    }

    for args in [
        vec!["--graph-recording-source", "claude"],
        vec!["--graph-recording-source", "mini_swe"],
    ] {
        assert!(
            ProfileFlags::try_parse_from(std::iter::once("aiperf").chain(args.iter().copied()),)
                .is_err(),
            "unsupported graph recording source parsed: {args:?}"
        );
    }
}
