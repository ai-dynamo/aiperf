// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process contracts for the native graph-inspection command namespace.

use std::path::Path;
use std::process::{Command, Output};

use aiperf_cli::graph::report::{GraphCommandErrorCode, GraphErrorReport, GraphOperation};

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
    for format in [
        "dag_jsonl",
        "conditional_graph",
        "weka_trace",
        "dynamo_trace",
        "aiperf_trace",
        "agent_recording",
        "otlp_genai",
    ] {
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
fn fatal_source_errors_are_typed_and_exit_two() {
    for (path, code) in [
        ("/definitely/not/a/graph.json", "source-not-found"),
        ("https://example.invalid/input.json", "source-not-local"),
        ("-", "source-not-local"),
    ] {
        let output = output(&["graph", "validate", path, "--graph-format", "dag_jsonl"]);
        assert_eq!(output.status.code(), Some(2), "{path}");
        let diagnostic = stderr(&output);
        assert!(diagnostic.starts_with("aiperf graph validate:"), "{path}");
        assert!(
            diagnostic.contains(if code == "source-not-found" {
                "does not exist locally"
            } else {
                "existing local file or directory"
            }),
            "{path}"
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
fn fatal_invalid_arguments_and_tokenizer_errors_are_typed() {
    let unsupported_format = output(&[
        "graph",
        "validate",
        "/definitely/not/a/graph.json",
        "--graph-format",
        "unregistered",
    ]);
    assert_eq!(unsupported_format.status.code(), Some(2));
    let format_error = stderr(&unsupported_format);
    assert!(format_error.contains("invalid value"));
    assert!(format_error.contains("dag_jsonl"));

    let fixture = tempfile::NamedTempFile::new().expect("create graph fixture");
    std::fs::write(fixture.path(), "{}").expect("write graph fixture");
    let source = fixture.path().to_str().expect("UTF-8 path");
    let unsupported_tokenizer = output(&[
        "graph",
        "validate",
        source,
        "--graph-format",
        "dag_jsonl",
        "--tokenizer",
        "acme/missing-tokenizer",
    ]);
    assert_eq!(unsupported_tokenizer.status.code(), Some(2));
    assert!(stderr(&unsupported_tokenizer).contains("built-in encoding"));

    let tokenizer = tempfile::NamedTempFile::new().expect("create tokenizer fixture");
    std::fs::write(tokenizer.path(), "not a tokenizer").expect("write tokenizer fixture");
    let malformed_tokenizer = output(&[
        "graph",
        "validate",
        source,
        "--graph-format",
        "dag_jsonl",
        "--tokenizer",
        tokenizer.path().to_str().expect("UTF-8 tokenizer path"),
    ]);
    assert_eq!(malformed_tokenizer.status.code(), Some(2));
    assert!(stderr(&malformed_tokenizer).contains("could not be loaded"));
}

#[test]
fn malformed_input_is_a_typed_adapter_error() {
    let fixture = tempfile::NamedTempFile::new().expect("create malformed fixture");
    std::fs::write(fixture.path(), "not json").expect("write malformed fixture");
    let output = output(&[
        "graph",
        "validate",
        fixture.path().to_str().expect("UTF-8 path"),
        "--graph-format",
        "dag_jsonl",
    ]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = stderr(&output);
    assert!(stderr.contains("could not be decoded") || stderr.contains("could not be lowered"));
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
