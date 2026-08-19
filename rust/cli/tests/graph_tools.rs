// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process contracts for the native graph-inspection command namespace.

use std::path::Path;
use std::process::{Command, Output};

use aiperf_cli::graph::report::{
    GraphCommandErrorCode, GraphErrorReport, GraphExplainReport, GraphIssueReport,
    GraphIssueSeverityReport, GraphIssueSummary, GraphOperation, GraphPlanPhaseReport,
    GraphValidateReport,
};
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

fn fixture(path: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(path)
        .canonicalize()
        .expect("canonical graph fixture")
        .display()
        .to_string()
}

fn validate_output(path: &str, format: &str, extra: &[&str]) -> Output {
    let mut args = vec!["graph", "validate", path, "--graph-format", format];
    args.extend_from_slice(extra);
    output(&args)
}

fn explain_output(path: &str, format: &str, extra: &[&str]) -> Output {
    let mut args = vec!["graph", "explain", path, "--graph-format", format];
    args.extend_from_slice(extra);
    output(&args)
}

#[test]
fn graph_explain_report_preserves_the_versioned_safe_wire_shape() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = explain_output(&source, "dag_jsonl", &["--output-format", "json"]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    let report: GraphExplainReport =
        serde_json::from_slice(&output.stdout).expect("explain JSON is versioned");
    assert_eq!(report.schema_version, "aiperf.graph.explain.v1");
    assert_eq!(report.input.format, "dag_jsonl");
    assert_eq!(report.input.root_count, 1);
    assert_eq!(report.programs.len(), 1);
    assert_eq!(report.programs[0].driver, "static_graph");
    assert!(report.programs[0].profiling.readiness_waves.is_some());
    assert!(report.programs[0].profiling.readiness_unavailable.is_none());

    let json = String::from_utf8(output.stdout).expect("explain JSON UTF-8");
    for forbidden in [
        "segment_handle",
        "prompt_payload",
        "tool_command",
        "environment_data",
        "driver_data",
    ] {
        assert!(
            !json.contains(forbidden),
            "explain report must not expose {forbidden}"
        );
    }
}

#[test]
fn explain_small_input_matches_the_human_golden() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = explain_output(&source, "dag_jsonl", &[]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    assert_eq!(
        String::from_utf8(output.stdout).expect("explain text UTF-8"),
        include_str!("goldens/graph_tools/explain-small.txt").replace("$SOURCE", &source)
    );
}

#[test]
fn explain_invalid_lowered_input_is_best_effort_and_matches_the_human_golden() {
    let source = fixture("tests/fixtures/graph_tools/mixed-anchor.conditional.yaml");
    let output = explain_output(&source, "conditional_graph", &[]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    assert_eq!(
        String::from_utf8(output.stdout).expect("explain text UTF-8"),
        include_str!("goldens/graph_tools/explain-invalid.txt").replace("$SOURCE", &source)
    );

    let output = explain_output(&source, "conditional_graph", &["--output-format=json"]);
    assert_eq!(output.status.code(), Some(0));
    let report: GraphExplainReport =
        serde_json::from_slice(&output.stdout).expect("best-effort explain JSON");
    assert_eq!(report.programs[0].profiling.readiness_waves, None);
    assert_eq!(
        report.programs[0]
            .profiling
            .readiness_unavailable
            .as_ref()
            .map(|unavailable| unavailable.code.as_str()),
        Some("validation-errors")
    );
}

#[test]
fn graph_validate_report_preserves_the_versioned_wire_shape() {
    let report = GraphValidateReport {
        schema_version: "aiperf.graph.validate.v1".to_owned(),
        source: "/tmp/input.graph".to_owned(),
        format: "dag_jsonl".to_owned(),
        root_count: 1,
        node_count: 2,
        issues: vec![GraphIssueReport {
            code: "node-unreachable".to_owned(),
            severity: GraphIssueSeverityReport::Error,
            trace_id: Some("t-1".to_owned()),
            phase: Some(GraphPlanPhaseReport::Profiling),
            location: Some("graph.nodes.foo".to_owned()),
            message: "node is unreachable from START".to_owned(),
            context: [("node_id".to_owned(), "foo".to_owned())]
                .into_iter()
                .collect(),
        }],
        summary: GraphIssueSummary {
            errors: 1,
            warnings: 0,
        },
    };

    let value = serde_json::to_value(&report).expect("serialize validation report");
    assert_eq!(value["schema_version"], "aiperf.graph.validate.v1");
    assert_eq!(value["issues"][0]["severity"], "error");
    assert_eq!(value["issues"][0]["phase"], "profiling");
    assert_eq!(value["summary"]["errors"], 1);
    let round_trip: GraphValidateReport =
        serde_json::from_value(value).expect("deserialize validation report");
    assert_eq!(round_trip.issues.len(), 1);
}

#[test]
fn validate_clean_dag_reports_success() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = validate_output(&source, "dag_jsonl", &[]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    assert!(
        String::from_utf8(output.stdout)
            .expect("stdout UTF-8")
            .ends_with("OK: 0 errors, 0 warning(s).\n")
    );
}

#[test]
fn validate_mixed_anchor_input_matches_the_human_golden() {
    let source = fixture("tests/fixtures/graph_tools/mixed-anchor.conditional.yaml");
    let output = validate_output(&source, "conditional_graph", &[]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).is_empty());
    let expected = include_str!("goldens/graph_tools/validate-mixed-anchor.txt");
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout UTF-8"),
        expected
    );
}

#[test]
fn validate_arrival_pace_requires_offsets() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = validate_output(&source, "dag_jsonl", &["--pace", "arrival"]);

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).expect("stdout UTF-8");
    assert!(stdout.contains("[arrival-offset-missing]"));
    assert!(stdout.ends_with("warning(s).\n"));
}

#[test]
fn validate_collapsed_replay_warning_does_not_fail() {
    let source = fixture("tests/fixtures/graph_tools/collapsed-replay.otlp.json");
    let output = validate_output(&source, "otlp_genai", &[]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    let stdout = String::from_utf8(output.stdout).expect("stdout UTF-8");
    assert!(stdout.contains("[adapter-warning.otlp_genai_replay_spans_collapsed]"));
    assert!(stdout.ends_with("OK: 0 errors, 1 warning(s).\n"));
}

#[test]
fn validate_json_reports_only_the_versioned_document() {
    let clean_source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let clean = validate_output(&clean_source, "dag_jsonl", &["--output-format", "json"]);
    assert_eq!(clean.status.code(), Some(0));
    assert!(stderr(&clean).is_empty());
    let clean_json: GraphValidateReport =
        serde_json::from_slice(&clean.stdout).expect("clean validation JSON");
    assert_eq!(clean_json.schema_version, "aiperf.graph.validate.v1");
    assert_eq!(clean_json.summary.errors, 0);
    assert_eq!(clean_json.summary.warnings, 0);
    assert!(
        !String::from_utf8(clean.stdout)
            .expect("stdout UTF-8")
            .contains("OK:")
    );

    let invalid_source = fixture("tests/fixtures/graph_tools/mixed-anchor.conditional.yaml");
    let invalid = validate_output(
        &invalid_source,
        "conditional_graph",
        &["--output-format=json"],
    );
    assert_eq!(invalid.status.code(), Some(1));
    assert!(stderr(&invalid).is_empty());
    let invalid_json: GraphValidateReport =
        serde_json::from_slice(&invalid.stdout).expect("invalid validation JSON");
    assert_eq!(invalid_json.summary.errors, 1);
    assert_eq!(invalid_json.issues[0].code, "mixed-anchor-fan-in");
    assert!(matches!(
        invalid_json.issues[0].severity,
        GraphIssueSeverityReport::Error
    ));
    assert_eq!(invalid_json.issues[0].trace_id.as_deref(), Some("t-mixed"));
    assert!(matches!(
        invalid_json.issues[0].phase,
        Some(GraphPlanPhaseReport::Profiling)
    ));
    assert_eq!(
        invalid_json.issues[0].location.as_deref(),
        Some("graph.nodes.c")
    );
    assert!(
        !String::from_utf8(invalid.stdout)
            .expect("stdout UTF-8")
            .contains("FAIL:")
    );
}

#[test]
fn validate_json_malformed_input_stays_a_fatal_error_document() {
    let fixture = tempfile::NamedTempFile::new().expect("create malformed fixture");
    std::fs::write(fixture.path(), "not json").expect("write malformed fixture");
    let output = validate_output(
        fixture.path().to_str().expect("UTF-8 fixture path"),
        "dag_jsonl",
        &["--output-format", "json"],
    );

    assert_eq!(output.status.code(), Some(2));
    let fatal: GraphErrorReport = serde_json::from_slice(&output.stdout).expect("fatal JSON");
    assert_eq!(fatal.schema_version, "aiperf.graph.error.v1");
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
