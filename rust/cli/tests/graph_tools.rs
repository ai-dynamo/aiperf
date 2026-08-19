// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process contracts for the native graph-inspection command namespace.

use std::path::Path;
use std::process::{Command, Output};

use aiperf_cli::graph::report::{GraphCommandErrorCode, GraphErrorReport, GraphOperation};
use aiperf_runtime::config::model::workload_kind::GRAPH_FORMATS;
use serde_json::json;

fn command_without_python(fixture_bin: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_aiperf"));
    command
        .env_remove("PYTHONPATH")
        .env_remove("AIPERF_PYTHON")
        .env("PATH", fixture_bin);
    command
}

fn output(args: &[&str]) -> Output {
    let fixture_bin = tempfile::tempdir().expect("create empty PATH directory");
    command_without_python(fixture_bin.path())
        .args(args)
        .output()
        .expect("run native aiperf binary")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

fn json_error(args: &[&str]) -> GraphErrorReport {
    let output = output(args);
    assert_eq!(output.status.code(), Some(2));
    assert!(!stderr(&output).contains("aiperf:"));
    serde_json::from_slice(&output.stdout).expect("fatal JSON error report on stdout")
}

fn assert_json_error(report: GraphErrorReport, code: &str, message: &str, source: Option<&str>) {
    let value = serde_json::to_value(report).expect("serialize graph error report");
    assert_eq!(value["schema_version"], "aiperf.graph.error.v1");
    assert_eq!(value["operation"], "validate");
    assert_eq!(value["code"], code);
    assert_eq!(value["message"], message);
    assert_eq!(value["source"], json!(source));
}

#[test]
fn graph_without_subcommand_prints_help_and_exits_two() {
    let output = output(&["graph"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = stderr(&output);
    assert!(stderr.contains("validate"));
    assert!(stderr.contains("explain"));
    assert!(stderr.contains("visualize"));
}

#[test]
fn graph_help_lists_the_stock_built_in_formats_without_python() {
    let output = output(&["graph", "--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = String::from_utf8(output.stdout).expect("help is UTF-8");
    for format in GRAPH_FORMATS {
        assert!(help.contains(format), "missing built-in format {format}");
    }
}

#[test]
fn unknown_non_graph_commands_still_delegate_to_python() {
    let output = output(&["legacy-python-command"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("failed to delegate"));
}

#[test]
fn fatal_source_errors_serialize_their_stable_codes() {
    let missing = "/definitely/not/a/graph.json";
    assert_json_error(
        json_error(&[
            "graph",
            "validate",
            missing,
            "--graph-format",
            "dag_jsonl",
            "--output-format",
            "json",
        ]),
        "source-not-found",
        "graph source does not exist locally",
        Some(missing),
    );
    for path in ["https://example.invalid/input.json", "-"] {
        assert_json_error(
            json_error(&[
                "graph",
                "validate",
                path,
                "--graph-format",
                "dag_jsonl",
                "--output-format=json",
            ]),
            "source-not-local",
            "graph source must be an existing local file or directory",
            None,
        );
    }
}

#[cfg(unix)]
#[test]
fn non_file_non_directory_sources_are_rejected_before_adapter_loading() {
    let directory = tempfile::tempdir().expect("create socket directory");
    let socket = directory.path().join("graph.sock");
    let _listener = std::os::unix::net::UnixListener::bind(&socket).expect("bind socket");
    let output = output(&[
        "graph",
        "validate",
        socket.to_str().expect("UTF-8 socket path"),
        "--graph-format",
        "dag_jsonl",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("local file or directory"));
}

#[test]
fn fatal_invalid_arguments_and_tokenizer_errors_serialize_their_stable_codes() {
    let fixture = tempfile::NamedTempFile::new().expect("create graph fixture");
    std::fs::write(fixture.path(), "{}").expect("write graph fixture");
    let source = fixture
        .path()
        .canonicalize()
        .expect("canonical fixture path")
        .display()
        .to_string();
    for output_flag in ["--output-format", "--output-format=json"] {
        let mut args = vec![
            "graph",
            "validate",
            fixture.path().to_str().expect("UTF-8 path"),
            "--graph-format",
            "unregistered",
        ];
        if output_flag == "--output-format" {
            args.extend(["--output-format", "json"]);
        } else {
            args.push(output_flag);
        }
        assert_json_error(
            json_error(&args),
            "invalid-arguments",
            "invalid arguments: one of the values isn't valid for an argument",
            None,
        );
    }
    assert_json_error(
        json_error(&[
            "graph",
            "validate",
            fixture.path().to_str().expect("UTF-8 path"),
            "--graph-format",
            "dag_jsonl",
            "--tokenizer",
            "acme/missing-tokenizer",
            "--output-format=json",
        ]),
        "tokenizer-unsupported",
        "tokenizer must be a built-in encoding or an existing local path",
        Some(&source),
    );

    let tokenizer = tempfile::NamedTempFile::new().expect("create tokenizer fixture");
    std::fs::write(tokenizer.path(), "not a tokenizer").expect("write tokenizer fixture");
    assert_json_error(
        json_error(&[
            "graph",
            "validate",
            fixture.path().to_str().expect("UTF-8 path"),
            "--graph-format",
            "dag_jsonl",
            "--tokenizer",
            tokenizer.path().to_str().expect("UTF-8 tokenizer path"),
            "--output-format",
            "json",
        ]),
        "tokenizer-load-failed",
        "local tokenizer could not be loaded",
        Some(&source),
    );
}

#[test]
fn malformed_input_serializes_input_decode_failed() {
    let fixture = tempfile::NamedTempFile::new().expect("create malformed fixture");
    std::fs::write(fixture.path(), "not json").expect("write malformed fixture");
    let source = fixture
        .path()
        .canonicalize()
        .expect("canonical fixture path")
        .display()
        .to_string();
    assert_json_error(
        json_error(&[
            "graph",
            "validate",
            fixture.path().to_str().expect("UTF-8 path"),
            "--graph-format",
            "dag_jsonl",
            "--output-format=json",
        ]),
        "input-decode-failed",
        "graph input could not be decoded",
        Some(&source),
    );
}

#[test]
fn structurally_invalid_input_serializes_input_lowering_failed() {
    let fixture = tempfile::NamedTempFile::new().expect("create invalid graph fixture");
    std::fs::write(
        fixture.path(),
        r#"{"session_id":"root","turns":[{"messages":[{"role":"user","content":"q"}],"spawns":["missing"]}]}"#,
    )
    .expect("write invalid graph fixture");
    let source = fixture
        .path()
        .canonicalize()
        .expect("canonical fixture path")
        .display()
        .to_string();
    assert_json_error(
        json_error(&[
            "graph",
            "validate",
            fixture.path().to_str().expect("UTF-8 path"),
            "--graph-format",
            "dag_jsonl",
            "--output-format",
            "json",
        ]),
        "input-lowering-failed",
        "graph input could not be lowered",
        Some(&source),
    );
}

#[test]
fn json_fatal_envelope_has_no_generic_fallback() {
    let output = output(&[
        "graph",
        "validate",
        "/definitely/not/a/graph.json",
        "--graph-format",
        "dag_jsonl",
        "--output-format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(!stderr(&output).contains("aiperf:"));
    let report: GraphErrorReport =
        serde_json::from_slice(&output.stdout).expect("fatal JSON error report on stdout");
    assert_eq!(report.schema_version, "aiperf.graph.error.v1");
    assert!(matches!(report.operation, GraphOperation::Validate));
    assert!(matches!(report.code, GraphCommandErrorCode::SourceNotFound));
    assert!(!report.message.is_empty());
    assert!(report.source.is_some());
}
