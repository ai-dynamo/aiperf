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

fn profile_flags_with_minimum_endpoint_and(args: &[&str]) -> ProfileFlags {
    ProfileFlags::try_parse_from(
        std::iter::once("aiperf")
            .chain([
                "--model",
                "model",
                "--url",
                "http://127.0.0.1:8000",
                "--endpoint-type",
                "chat",
                "--concurrency",
                "1",
                "--request-count",
                "1",
            ])
            .chain(args.iter().copied()),
    )
    .expect("profile flags parse")
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
        "--graph-resume",
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
      resume: true
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
fn recorded_agent_replay_flags_require_agent_recording_format() {
    on_big_stack(recorded_agent_replay_flags_require_agent_recording_format_body);
}

fn recorded_agent_replay_flags_require_agent_recording_format_body() {
    for (args, option) in [
        (
            vec!["--graph-recording-source", "codex"],
            "--graph-recording-source",
        ),
        (
            vec!["--graph-include-subagents"],
            "--graph-include-subagents",
        ),
        (
            vec!["--graph-include-subagents=false"],
            "--graph-include-subagents",
        ),
        (
            vec!["--graph-replay-root", "/tmp/replay"],
            "--graph-replay-root",
        ),
        (vec!["--graph-execute-tools"], "--graph-execute-tools"),
        (vec!["--graph-execute-tools=false"], "--graph-execute-tools"),
        (
            vec!["--graph-tool-image", "tools:latest"],
            "--graph-tool-image",
        ),
        (
            vec!["--graph-pinch-image", "pinch:latest"],
            "--graph-pinch-image",
        ),
        (
            vec!["--graph-tool-command-timeout", "9"],
            "--graph-tool-command-timeout",
        ),
        (
            vec!["--graph-tool-container-stop-timeout", "4"],
            "--graph-tool-container-stop-timeout",
        ),
        (
            vec!["--graph-tool-session-close-grace", "1.5"],
            "--graph-tool-session-close-grace",
        ),
        (
            vec!["--graph-use-family-sampling"],
            "--graph-use-family-sampling",
        ),
        (
            vec!["--graph-use-family-sampling=false"],
            "--graph-use-family-sampling",
        ),
        (
            vec!["--no-graph-use-family-sampling"],
            "--no-graph-use-family-sampling",
        ),
        (
            vec!["--no-graph-use-family-sampling=false"],
            "--no-graph-use-family-sampling",
        ),
        (vec!["--graph-emit-warmup"], "--graph-emit-warmup"),
        (vec!["--graph-emit-warmup=false"], "--graph-emit-warmup"),
        (vec!["--graph-resume"], "--graph-resume"),
        (vec!["--graph-resume=false"], "--graph-resume"),
        (vec!["--graph-stop-on-failure"], "--graph-stop-on-failure"),
        (
            vec!["--graph-stop-on-failure=false"],
            "--graph-stop-on-failure",
        ),
    ] {
        let flags = profile_flags_with_minimum_endpoint_and(&args);
        let error = aiperf_cli::load::resolve(&flags)
            .expect_err("recorded-agent replay option must not disappear");
        assert!(error.to_string().contains(option), "{args:?}: {error}");
        assert!(
            error.to_string().contains("--graph-format agent_recording"),
            "{args:?}: {error}"
        );
    }
}

#[test]
fn resolve_inputs_rejects_explicit_false_recorded_agent_flags() {
    on_big_stack(resolve_inputs_rejects_explicit_false_recorded_agent_flags_body);
}

fn resolve_inputs_rejects_explicit_false_recorded_agent_flags_body() {
    let flags = profile_flags_with_minimum_endpoint_and(&["--no-graph-use-family-sampling=false"]);
    let error = aiperf_cli::load::resolve_inputs(&flags)
        .expect_err("single-run input projection must not discard replay-only flags");
    assert!(
        error.to_string().contains("--no-graph-use-family-sampling"),
        "{error}"
    );
    assert!(
        error.to_string().contains("--graph-format agent_recording"),
        "{error}"
    );
}

#[test]
fn graph_recording_source_and_subagent_flags_accept_the_documented_spellings() {
    on_big_stack(graph_recording_source_and_subagent_flags_accept_the_documented_spellings_body);
}

fn graph_recording_source_and_subagent_flags_accept_the_documented_spellings_body() {
    for (args, expected) in [
        (&[][..], None),
        (&["--graph-include-subagents"][..], Some(true)),
        (&["--graph-include-subagents=true"][..], Some(true)),
        (&["--graph-include-subagents=false"][..], Some(false)),
    ] {
        let flags =
            ProfileFlags::try_parse_from(std::iter::once("aiperf").chain(args.iter().copied()))
                .expect("documented graph import flag parses");
        assert_eq!(flags.graph_include_subagents, expected, "{args:?}");
    }

    for (cli, wire) in [
        ("auto", "auto"),
        ("mini-swe-agent", "mini_swe_agent"),
        ("codex", "codex"),
        ("claude-code", "claude_code"),
    ] {
        let flags = ProfileFlags::try_parse_from([
            "aiperf",
            "--model",
            "model",
            "--url",
            "http://127.0.0.1:8000",
            "--endpoint-type",
            "chat",
            "--input-file",
            "/tmp/session.jsonl",
            "--graph-format",
            "agent_recording",
            "--graph-recording-source",
            cli,
            "--concurrency",
            "1",
            "--request-count",
            "1",
            "--hardware-description",
            "unknown",
            "--endpoint-placement",
            "remote",
        ])
        .expect("documented source spelling parses");
        let run = aiperf_cli::load::resolve(&flags).expect("documented source projects");
        let projected = serde_json::to_value(run).expect("projected run serializes");
        assert_eq!(
            projected["cfg"]["datasets"][0]["graph"]["source_format"],
            wire,
        );
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
