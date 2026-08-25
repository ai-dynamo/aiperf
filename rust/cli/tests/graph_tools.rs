// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process contracts for the native graph-inspection command namespace.

use std::fs;
#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::io::Read;
use std::path::Path;
use std::process::{Command, Output};

use aiperf_cli::graph::report::{
    GraphCommandErrorCode, GraphErrorReport, GraphExplainReport, GraphIssueReport,
    GraphIssueSeverityReport, GraphIssueSummary, GraphOperation, GraphPlanPhaseReport,
    GraphValidateReport,
};
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

fn imported_agent_fixture(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../runtime/tests/fixtures/recorded_agent_session_import")
        .join(path)
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

struct BuiltInGraphSource {
    format: &'static str,
    path: String,
    static_compatible: bool,
}

fn built_in_graph_sources() -> Vec<BuiltInGraphSource> {
    vec![
        BuiltInGraphSource {
            format: "dag_jsonl",
            path: fixture("../../tests/fixtures/dag/small.dag.jsonl"),
            static_compatible: true,
        },
        BuiltInGraphSource {
            format: "conditional_graph",
            path: fixture("../e2e-tests/tests/fixtures/conditional/conditional_shopping.yaml"),
            static_compatible: true,
        },
        BuiltInGraphSource {
            format: "weka_trace",
            path: fixture("../../tests/fixtures/weka_traces/simple.json"),
            static_compatible: true,
        },
        BuiltInGraphSource {
            format: "dynamo_trace",
            path: fixture("../runtime/tests/fixtures/graph_inspection/dynamo-trace.jsonl"),
            static_compatible: true,
        },
        BuiltInGraphSource {
            format: "aiperf_trace",
            path: fixture("../runtime/tests/fixtures/graph_inspection/aiperf-trace.json"),
            static_compatible: true,
        },
        BuiltInGraphSource {
            format: "agent_recording",
            path: fixture("../runtime/tests/fixtures/recorded_agent_replay/recordings"),
            static_compatible: false,
        },
        BuiltInGraphSource {
            format: "otlp_genai",
            path: fixture("tests/fixtures/graph_tools/collapsed-replay.otlp.json"),
            static_compatible: false,
        },
    ]
}

fn graph_help_formats(help: &str) -> Vec<&str> {
    let values = help
        .lines()
        .find_map(|line| line.strip_prefix("Supported built-in graph formats: "))
        .expect("graph help possible-value list");
    values.split(", ").collect()
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

fn static_timing_fixture() -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().expect("create static timing fixture directory");
    let source = directory.path().join("static-timing.conditional.yaml");
    fs::write(
        &source,
        r#"graph:
  state:
    a: {type: text}
    b: {type: text}
    c: {type: text}
  nodes:
    source:
      node_type: llm
      prompt: ["source"]
      output: a
      min_start_delay_us: 7.0
    combined:
      node_type: llm
      prompt: ["combined"]
      output: b
    first_only:
      node_type: llm
      prompt: ["first-only"]
      output: c
  edges:
    - {source: START, target: source}
    - {source: source, target: combined, min_start_delay_us: 0.0, delay_after_predecessor_start_us: 20.0, delay_after_predecessor_first_token_us: 5.0}
    - {source: source, target: first_only, delay_after_predecessor_first_token_us: 3.0}
traces:
  - id: static-timing
"#,
    )
    .expect("write static timing fixture");
    (directory, source.display().to_string())
}

fn static_channel_readiness_fixture() -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().expect("create readiness fixture directory");
    let source = directory
        .path()
        .join("static-channel-readiness.conditional.yaml");
    fs::write(
        &source,
        r#"graph:
  state:
    produced: {type: messages, reducer: add_messages}
    done: {type: messages, reducer: add_messages}
  nodes:
    reader:
      node_type: llm
      prompt: ["reader"]
      output: done
      inputs: [{channel: produced, count: 1}]
    producer:
      node_type: llm
      prompt: ["producer"]
      output: produced
  edges:
    - {source: START, target: reader}
    - {source: reader, target: producer}
traces:
  - id: static-channel-readiness
"#,
    )
    .expect("write readiness fixture");
    (directory, source.display().to_string())
}

fn conditional_count_fixture(count: &str) -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().expect("create channel count fixture directory");
    let source = directory.path().join("channel-count.conditional.yaml");
    fs::write(
        &source,
        format!(
            r#"graph:
  state:
    produced: {{type: text}}
    done: {{type: text}}
  nodes:
    source:
      node_type: llm
      prompt: ["source"]
      output: produced
    reader:
      node_type: llm
      prompt: ["reader"]
      output: done
      inputs: [{{channel: produced, count: {count}}}]
  edges:
    - {{source: START, target: source}}
    - {{source: source, target: reader}}
traces:
  - id: channel-count
"#
        ),
    )
    .expect("write channel count fixture");
    (directory, source.display().to_string())
}

fn static_channel_readiness_anchor_fixture(edge: &str) -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().expect("create readiness anchor fixture directory");
    let source = directory
        .path()
        .join("static-channel-readiness-anchor.conditional.yaml");
    fs::write(
        &source,
        format!(
            r#"graph:
  state:
    produced: {{type: messages, reducer: add_messages}}
    done: {{type: messages, reducer: add_messages}}
  nodes:
    producer:
      node_type: llm
      prompt: ["producer"]
      output: produced
    reader:
      node_type: llm
      prompt: ["reader"]
      output: done
  edges:
    - {{source: START, target: producer}}
    - {{source: producer, target: reader{edge}}}
traces:
  - id: static-channel-readiness-anchor
"#
        ),
    )
    .expect("write readiness anchor fixture");
    (directory, source.display().to_string())
}

fn visualize_mermaid_output(source: &str, destination: Option<&Path>) -> Output {
    let fixture_bin = tempfile::tempdir().expect("create empty PATH directory");
    let mut command = command_without_python(fixture_bin.path());
    command.args([
        "graph",
        "visualize",
        source,
        "--graph-format",
        "dag_jsonl",
        "--output-format",
        "mermaid",
    ]);
    if let Some(destination) = destination {
        command.arg("--output").arg(destination);
    }
    command.output().expect("run native aiperf binary")
}

#[test]
fn output_new_file_receives_rendered_bytes_and_keeps_stdout_empty() {
    let directory = tempfile::tempdir().expect("create output directory");
    let destination = directory.path().join("graph.mmd");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).is_empty());
    assert_eq!(
        fs::read(&destination).expect("read graph output"),
        include_bytes!("goldens/graph_tools/visualize-small.mmd")
    );
}

#[test]
fn output_without_destination_writes_mermaid_only_to_stdout() {
    let directory = tempfile::tempdir().expect("create output directory");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, None);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    assert_eq!(
        output.stdout,
        include_bytes!("goldens/graph_tools/visualize-small.mmd")
    );
    assert!(
        fs::read_dir(directory.path())
            .expect("read untouched output directory")
            .next()
            .is_none()
    );
}

#[cfg(unix)]
#[test]
fn output_existing_regular_file_is_replaced_atomically() {
    let directory = tempfile::tempdir().expect("create output directory");
    let destination = directory.path().join("graph.mmd");
    fs::write(&destination, b"old graph bytes").expect("write previous graph");
    let mut previous = File::open(&destination).expect("hold previous graph open");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).is_empty());
    let mut previous_bytes = Vec::new();
    previous
        .read_to_end(&mut previous_bytes)
        .expect("read held previous graph");
    assert_eq!(previous_bytes, b"old graph bytes");
    assert_eq!(
        fs::read(&destination).expect("read replacement graph"),
        include_bytes!("goldens/graph_tools/visualize-small.mmd")
    );
}

#[cfg(not(unix))]
#[test]
fn output_existing_regular_file_is_rejected_when_atomic_replacement_is_unavailable() {
    let directory = tempfile::tempdir().expect("create output directory");
    let destination = directory.path().join("graph.mmd");
    fs::write(&destination, b"old graph bytes").expect("write previous graph");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("output-invalid"));
    assert_eq!(
        fs::read(&destination).expect("read preserved graph"),
        b"old graph bytes"
    );
}

#[test]
fn output_directory_target_is_rejected_without_a_temporary_file() {
    let directory = tempfile::tempdir().expect("create output directory");
    let destination = directory.path().join("output-directory");
    fs::create_dir(&destination).expect("create directory target");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("output-invalid"));
    assert!(destination.is_dir());
    assert_eq!(
        fs::read_dir(directory.path())
            .expect("read output directory")
            .count(),
        1
    );
}

#[test]
fn output_missing_parent_reports_write_failure_without_creating_a_destination() {
    let directory = tempfile::tempdir().expect("create output directory");
    let destination = directory.path().join("missing").join("graph.mmd");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("output-write-failed"));
    assert!(!destination.exists());
}

#[test]
fn output_non_directory_parent_preserves_the_existing_parent_file() {
    let directory = tempfile::tempdir().expect("create output directory");
    let parent = directory.path().join("not-a-directory");
    fs::write(&parent, b"keep this file").expect("write parent file");
    let destination = parent.join("graph.mmd");
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("output-write-failed"));
    assert_eq!(
        fs::read(&parent).expect("read preserved parent file"),
        b"keep this file"
    );
}

#[cfg(unix)]
#[test]
fn output_read_only_parent_preserves_existing_destination() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("create output directory");
    let output_directory = directory.path().join("read-only");
    fs::create_dir(&output_directory).expect("create read-only output directory");
    let destination = output_directory.join("graph.mmd");
    fs::write(&destination, b"old graph bytes").expect("write previous graph");
    let mut permissions = fs::metadata(&output_directory)
        .expect("read output directory permissions")
        .permissions();
    let original_mode = permissions.mode();
    permissions.set_mode(0o555);
    fs::set_permissions(&output_directory, permissions).expect("make output directory read-only");

    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = visualize_mermaid_output(&source, Some(&destination));

    let mut restore = fs::metadata(&output_directory)
        .expect("read output directory permissions")
        .permissions();
    restore.set_mode(original_mode);
    fs::set_permissions(&output_directory, restore).expect("restore output directory permissions");

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("output-write-failed"));
    assert_eq!(
        fs::read(&destination).expect("read preserved graph"),
        b"old graph bytes"
    );
}

#[test]
fn visualize_emits_mermaid_for_a_lowered_graph() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "dag_jsonl",
        "--output-format",
        "mermaid",
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(
        String::from_utf8(output.stdout)
            .expect("Mermaid is UTF-8")
            .starts_with("flowchart LR\n")
    );
}

#[test]
fn visualize_small_mermaid_matches_the_deterministic_golden() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "dag_jsonl",
        "--output-format",
        "mermaid",
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    assert_eq!(
        String::from_utf8(output.stdout).expect("Mermaid is UTF-8"),
        include_str!("goldens/graph_tools/visualize-small.mmd")
    );
}

#[test]
fn visualize_markdown_describes_the_selected_resolved_plan() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = output(&["graph", "visualize", &source, "--graph-format", "dag_jsonl"]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    let markdown = String::from_utf8(output.stdout).expect("Markdown is UTF-8");
    assert!(markdown.contains("## Graph topology\n\n```mermaid\nflowchart LR\n"));
    assert!(markdown.contains("This is the selected trace's resolved Graph-IR topology."));
    assert!(markdown.contains("- Trace: root\n- Driver: static_graph\n"));
    assert!(
        markdown.contains("## Illustrative readiness waves\n\n| Wave | Nodes ready | Trigger |")
    );
    assert!(markdown.ends_with('\n'));
    assert_eq!(
        markdown.replace(&source, "$REPO/tests/fixtures/dag/small.dag.jsonl"),
        include_str!("goldens/graph_tools/visualize-small.md")
    );
}

#[test]
fn visualize_uses_authored_program_order_and_exact_profiling_trace_selection() {
    let source = fixture("../../tests/fixtures/dag/multi_root_single_turn.dag.jsonl");
    let first = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "dag_jsonl",
        "--output-format",
        "mermaid",
    ]);
    assert_eq!(first.status.code(), Some(0));
    let first = String::from_utf8(first.stdout).expect("Mermaid is UTF-8");
    assert!(first.contains("n0[\"n00000000\"]"));
    assert!(!first.contains("n00000003"));

    let later = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "dag_jsonl",
        "--trace",
        "r2",
        "--output-format",
        "mermaid",
    ]);
    assert_eq!(later.status.code(), Some(0));
    let later = String::from_utf8(later.stdout).expect("Mermaid is UTF-8");
    assert!(later.contains("n0[\"n00000000\"]"));
    assert!(later.contains("n2[\"n00000002\"]"));
}

#[test]
fn visualize_missing_trace_reports_authored_available_trace_order() {
    let source = fixture("../../tests/fixtures/dag/multi_root_single_turn.dag.jsonl");
    let output = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "dag_jsonl",
        "--trace",
        "missing",
    ]);

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = stderr(&output);
    assert!(stderr.contains("trace-not-found"));
    assert!(stderr.contains("available traces: r1, r2"));
}

#[test]
fn visualize_blocks_invalid_graphs_unless_best_effort_is_requested() {
    let source = fixture("tests/fixtures/graph_tools/mixed-anchor.conditional.yaml");
    let blocked = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "conditional_graph",
    ]);

    assert_eq!(blocked.status.code(), Some(1));
    assert!(blocked.stdout.is_empty());
    let diagnostics = stderr(&blocked);
    assert!(diagnostics.contains("[mixed-anchor-fan-in]"));
    assert!(diagnostics.contains("rerun with --no-validate to render best-effort topology"));

    let rendered = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "conditional_graph",
        "--no-validate",
        "--output-format",
        "mermaid",
    ]);
    assert_eq!(rendered.status.code(), Some(0));
    assert!(stderr(&rendered).is_empty());
    assert!(
        String::from_utf8(rendered.stdout)
            .expect("Mermaid is UTF-8")
            .starts_with("flowchart LR\n")
    );
}

#[test]
fn visualize_retains_every_explicit_edge_anchor_in_mermaid() {
    let source = fixture("tests/fixtures/graph_tools/mixed-anchor.conditional.yaml");
    let output = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "conditional_graph",
        "--no-validate",
        "--output-format",
        "mermaid",
    ]);

    assert_eq!(output.status.code(), Some(0));
    let mermaid = String::from_utf8(output.stdout).expect("Mermaid is UTF-8");
    assert!(mermaid.contains("start -->|\"schedule completion\"| n0"));
    assert!(mermaid.contains("n0 -->|\"schedule completion\"| n2"));
    assert!(mermaid.contains("n1 -->|\"schedule dispatch, dispatch +0us\"| n2"));
    assert!(mermaid.contains("n2 -->|\"schedule completion\"| end_node"));
}

#[test]
fn all_static_timing_gates_are_exhaustive_in_graph_command_outputs() {
    let (_directory, source) = static_timing_fixture();

    let json = explain_output(&source, "conditional_graph", &["--output-format=json"]);
    assert_eq!(json.status.code(), Some(0), "{}", stderr(&json));
    let json: serde_json::Value =
        serde_json::from_slice(&json.stdout).expect("explain JSON is valid");
    let topology = &json["programs"][0]["profiling"]["topology"];
    let combined = topology["edges"]
        .as_array()
        .expect("edge array")
        .iter()
        .find(|edge| edge["source"] == "source" && edge["target"] == "combined")
        .expect("combined edge");
    assert_eq!(combined["schedule_anchor"], "dispatch");
    assert_eq!(combined["completion_delay_us"], serde_json::Value::Null);
    assert_eq!(combined["min_start_delay_us"], 0.0);
    assert_eq!(combined["dispatch_delay_us"], 20.0);
    assert_eq!(combined["first_token_delay_us"], 5.0);
    let first_only = topology["edges"]
        .as_array()
        .expect("edge array")
        .iter()
        .find(|edge| edge["source"] == "source" && edge["target"] == "first_only")
        .expect("first-token-only edge");
    assert_eq!(first_only["schedule_anchor"], "completion");
    assert_eq!(first_only["first_token_delay_us"], 3.0);
    let source_node = topology["nodes"]
        .as_array()
        .expect("node array")
        .iter()
        .find(|node| node["id"] == "source")
        .expect("source node");
    assert_eq!(source_node["min_start_delay_us"], 7.0);

    let text = explain_output(&source, "conditional_graph", &[]);
    assert_eq!(text.status.code(), Some(0), "{}", stderr(&text));
    let text = String::from_utf8(text.stdout).expect("explain text is UTF-8");
    assert!(text.contains("schedule=dispatch"));
    assert!(text.contains("dispatch_delay_us=20"));
    assert!(text.contains("first_token_delay_us=5"));
    assert!(text.contains("edge_min_start_delay_us=0"));
    assert!(text.contains("node_min_start_delay_us=7"));

    let mermaid = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "conditional_graph",
        "--output-format",
        "mermaid",
    ]);
    assert_eq!(mermaid.status.code(), Some(0), "{}", stderr(&mermaid));
    let mermaid = String::from_utf8(mermaid.stdout).expect("Mermaid is UTF-8");
    assert!(
        mermaid.contains("schedule dispatch, dispatch +20us, first-token +5us, min-start +0us")
    );
    assert!(mermaid.contains("source<br/>min-start +7us"));
    assert!(mermaid.contains("schedule completion, first-token +3us"));

    let markdown = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "conditional_graph",
    ]);
    assert_eq!(markdown.status.code(), Some(0), "{}", stderr(&markdown));
    let markdown = String::from_utf8(markdown.stdout).expect("Markdown is UTF-8");
    assert!(
        markdown.contains("schedule dispatch, dispatch +20us, first-token +5us, min-start +0us")
    );
    assert!(markdown.contains("source<br/>min-start +7us"));
    assert!(markdown.contains("schedule completion, first-token +3us"));
}

#[test]
fn static_channel_readiness_rejects_deadlocks_and_marks_explain_unavailable() {
    let (_directory, source) = static_channel_readiness_fixture();
    let validate = validate_output(&source, "conditional_graph", &["--output-format", "json"]);

    assert_eq!(
        validate.status.code(),
        Some(1),
        "stderr: {}\nstdout: {}",
        stderr(&validate),
        String::from_utf8_lossy(&validate.stdout)
    );
    let validate: GraphValidateReport =
        serde_json::from_slice(&validate.stdout).expect("validation JSON");
    assert_eq!(validate.summary.errors, 1);
    assert_eq!(validate.issues.len(), 1);
    assert_eq!(validate.issues[0].code, "static-channel-readiness-deadlock");
    assert_eq!(
        validate.issues[0].context.get("blocked_nodes"),
        Some(&"[\"reader\",\"producer\"]".to_owned())
    );

    let explain = explain_output(&source, "conditional_graph", &["--output-format=json"]);
    assert_eq!(explain.status.code(), Some(0), "{}", stderr(&explain));
    let explain: GraphExplainReport =
        serde_json::from_slice(&explain.stdout).expect("explain JSON");
    assert_eq!(explain.programs[0].profiling.readiness_waves, None);
    assert_eq!(
        explain.programs[0]
            .profiling
            .readiness_unavailable
            .as_ref()
            .map(|unavailable| unavailable.code.as_str()),
        Some("validation-errors")
    );
}

#[test]
fn conditional_cycle_remains_available_to_graph_validate() {
    let fixture = tempfile::NamedTempFile::new().expect("create cycle fixture");
    fs::write(
        fixture.path(),
        r#"
graph:
  state: {a: {type: text}, b: {type: text}}
  nodes:
    a: {node_type: llm, prompt: [a], output: a}
    b: {node_type: llm, prompt: [b], output: b}
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: a}
traces: [{id: cycle}]
"#,
    )
    .expect("write cycle fixture");

    let output = validate_output(
        fixture.path().to_str().expect("UTF-8 fixture"),
        "conditional_graph",
        &["--output-format", "json"],
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "stderr: {}\nstdout: {}",
        stderr(&output),
        String::from_utf8_lossy(&output.stdout)
    );
    let report: GraphValidateReport =
        serde_json::from_slice(&output.stdout).expect("validation JSON");
    assert_eq!(report.issues.len(), 1);
    assert_eq!(report.issues[0].code, "graph-cycle");
    assert_eq!(
        report.issues[0].context.get("node_ids"),
        Some(&"a,b,a".to_owned())
    );
}

#[test]
fn conditional_non_finite_timing_is_refused_before_inspection_rendering() {
    for value in [".nan", ".inf", "-.inf"] {
        let fixture = tempfile::NamedTempFile::new().expect("create non-finite fixture");
        fs::write(
            fixture.path(),
            r#"graph:
  state: {out: {type: text}}
  nodes:
    node: {node_type: llm, prompt: [x], output: out, min_start_delay_us: VALUE}
  edges:
    - {source: START, target: node}
traces: [{id: non-finite}]
"#
            .replace("VALUE", value),
        )
        .expect("write non-finite fixture");

        let output = explain_output(
            fixture.path().to_str().expect("fixture path"),
            "conditional_graph",
            &["--output-format=json"],
        );
        assert_eq!(output.status.code(), Some(2), "{value}");
        let report_json = String::from_utf8(output.stdout).expect("JSON error");
        assert!(
            report_json.contains("graph input could not be lowered"),
            "{value}"
        );
        assert!(!report_json.contains("\"topology\""), "{value}");
        assert!(
            !report_json.contains("\"min_start_delay_us\":null"),
            "{value}"
        );
    }
}

#[test]
fn conditional_non_finite_replay_duration_and_arrival_time_are_refused() {
    for (field, value, source) in [
        (
            "duration_ms",
            ".nan",
            r#"graph:
  nodes:
    replay: {node_type: replay, outputs: [out], duration_ms: VALUE}
"#,
        ),
        (
            "arrival_time",
            "-.inf",
            r#"graph:
  nodes:
    node: {node_type: llm, prompt: [x], output: out}
  edges:
    - {source: START, target: node}
traces: [{id: trace, arrival_time: VALUE}]
"#,
        ),
    ] {
        let fixture = tempfile::NamedTempFile::new().expect("create non-finite fixture");
        fs::write(fixture.path(), source.replace("VALUE", value)).expect("write fixture");
        let output = explain_output(
            fixture.path().to_str().expect("fixture path"),
            "conditional_graph",
            &["--output-format=json"],
        );
        assert_eq!(output.status.code(), Some(2), "{field}");
        let report_json = String::from_utf8(output.stdout).expect("JSON error");
        assert!(
            report_json.contains("graph input could not be lowered"),
            "{field}"
        );
        assert!(!report_json.contains("\"topology\""), "{field}");
    }
}

#[test]
fn conditional_replay_folding_refuses_cycles_and_unrepresentable_timing() {
    for source in [
        r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    replay: {node_type: replay, outputs: [out]}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: source}
    - {source: source, target: replay, min_start_delay_us: 1}
    - {source: replay, target: target}
traces: [{id: trace, replay_outputs: {replay: {out: replayed}}}]
"#,
        r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    r1: {node_type: replay, outputs: [out]}
    r2: {node_type: replay, outputs: [out]}
  edges:
    - {source: START, target: source}
    - {source: source, target: r1}
    - {source: r1, target: r2}
    - {source: r2, target: r1}
traces: [{id: trace, replay_outputs: {r1: {out: one}, r2: {out: two}}}]
"#,
    ] {
        let fixture = tempfile::NamedTempFile::new().expect("create replay fixture");
        fs::write(fixture.path(), source).expect("write replay fixture");
        let output = explain_output(
            fixture.path().to_str().expect("fixture path"),
            "conditional_graph",
            &["--output-format=json"],
        );
        assert_eq!(output.status.code(), Some(2));
        let report_json = String::from_utf8(output.stdout).expect("JSON error");
        assert!(report_json.contains("graph input could not be lowered"));
        assert!(!report_json.contains("\"topology\""));
    }
}

#[test]
fn conditional_graph_rejects_invalid_channel_counts_before_readiness() {
    for count in ["-1", "any"] {
        let (_directory, source) = conditional_count_fixture(count);
        let report = json_error(&[
            "graph",
            "validate",
            &source,
            "--graph-format",
            "conditional_graph",
            "--output-format=json",
        ]);
        assert_eq!(report.code, GraphCommandErrorCode::InputLoweringFailed);
        assert_eq!(report.message, "graph input could not be lowered");
    }

    for count in ["all", "0"] {
        let (_directory, source) = conditional_count_fixture(count);
        let output = validate_output(&source, "conditional_graph", &["--output-format=json"]);
        assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
        let report: GraphValidateReport =
            serde_json::from_slice(&output.stdout).expect("validation JSON");
        assert_eq!(report.summary.errors, 0);
    }
}

#[test]
fn static_channel_readiness_reports_scheduler_dispatch_and_completion_routes() {
    for (edge, trigger) in [
        (
            ", delay_after_predecessor_first_token_us: 3.0",
            "completed: producer",
        ),
        (
            ", delay_after_predecessor_start_us: 20.0, delay_after_predecessor_first_token_us: 5.0",
            "dispatched: producer",
        ),
    ] {
        let (_directory, source) = static_channel_readiness_anchor_fixture(edge);
        let explain = explain_output(&source, "conditional_graph", &["--output-format=json"]);

        assert_eq!(explain.status.code(), Some(0), "{}", stderr(&explain));
        let explain: GraphExplainReport =
            serde_json::from_slice(&explain.stdout).expect("explain JSON");
        let waves = explain.programs[0]
            .profiling
            .readiness_waves
            .as_ref()
            .expect("static readiness waves");
        assert_eq!(waves[1].trigger, trigger);
    }
}

#[test]
fn visualize_escapes_unsafe_authored_identifiers_in_visible_labels_only() {
    let source = fixture("tests/fixtures/graph_tools/unsafe-node-ids.conditional.yaml");
    let output = output(&[
        "graph",
        "visualize",
        &source,
        "--graph-format",
        "conditional_graph",
        "--output-format",
        "mermaid",
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    let mermaid = String::from_utf8(output.stdout).expect("Mermaid is UTF-8");
    assert!(mermaid.contains("n0[\"first | space &quot; [x] &amp; &lt;tag&gt;<br/>line\"]:::llm"));
    assert!(mermaid.contains("n1[\"second [x] &quot; &amp; &lt;&gt;<br/>line\"]:::llm"));
    assert!(mermaid.contains("n1 -. implicit .-> end_node"));
    assert!(!mermaid.contains("first space \" [x] & <tag>"));
    assert_eq!(
        mermaid,
        include_str!("goldens/graph_tools/visualize-unsafe-ids.mmd")
    );
}

#[test]
fn visualize_markdown_escapes_source_trace_and_readiness_values() {
    let fixture = fixture("tests/fixtures/graph_tools/unsafe-node-ids.conditional.yaml");
    let directory = tempfile::tempdir().expect("create unsafe source directory");
    let source = directory.path().join("unsafe|<source>&\nnext.yaml");
    std::fs::copy(&fixture, &source).expect("copy unsafe graph fixture");
    let source = source.to_str().expect("unsafe source path is UTF-8");
    let output = output(&[
        "graph",
        "visualize",
        source,
        "--graph-format",
        "conditional_graph",
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    let markdown = String::from_utf8(output.stdout).expect("Markdown is UTF-8");
    assert!(markdown.contains("unsafe\\|&lt;source&gt;&amp;<br/>next.yaml"));
    assert!(markdown.contains("- Trace: unsafe\\|&lt;trace&gt;&amp;<br/>next"));
    assert!(markdown.contains("first \\| space \" \\[x\\] &amp; &lt;tag&gt;<br/>line"));
    assert!(markdown.contains("second \\[x\\] \" &amp; &lt;&gt;<br/>line"));
    assert!(!markdown.contains("unsafe|<trace>&\nnext"));
    assert!(!markdown.contains("| space \" [x] & <tag>\nline"));
}

#[test]
fn visualize_markdown_neutralizes_backslashes_links_and_images() {
    let fixture = fixture("tests/fixtures/graph_tools/markdown-link-escape.conditional.yaml");
    let directory = tempfile::tempdir().expect("create unsafe source directory");
    let source = directory
        .path()
        .join("source\\|[text](URL)![alt](URL)<source>&\nnext.yaml");
    std::fs::copy(&fixture, &source).expect("copy unsafe graph fixture");
    let source = source.to_str().expect("unsafe source path is UTF-8");
    let output = output(&[
        "graph",
        "visualize",
        source,
        "--graph-format",
        "conditional_graph",
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).is_empty());
    let markdown = String::from_utf8(output.stdout).expect("Markdown is UTF-8");
    let interpolated = markdown
        .split("## Resolved plan")
        .nth(1)
        .expect("Markdown retains the resolved-plan section");
    for active_syntax in ["[text](URL)", "![alt](URL)", "|[text](URL)"] {
        assert!(
            !interpolated.contains(active_syntax),
            "Markdown must not retain active syntax {active_syntax:?}: {interpolated}"
        );
    }
    assert!(
        markdown.contains(
            r"source\\\|\[text\]\(URL\)\!\[alt\]\(URL\)&lt;source&gt;&amp;<br/>next.yaml"
        )
    );
    assert!(
        markdown.contains(
            r"- Trace: trace\\\|\[text\]\(URL\)\!\[alt\]\(URL\)&lt;trace&gt;&amp;<br/>next"
        )
    );
    assert!(
        markdown.contains(r"ready\\\|\[text\]\(URL\)\!\[alt\]\(URL\)&lt;tag&gt;&amp;<br/>line")
    );
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
fn source_format_does_not_change_non_agent_graph_lowering() {
    let source = fixture("../../tests/fixtures/dag/small.dag.jsonl");
    let output = validate_output(&source, "dag_jsonl", &["--source-format", "codex"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
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
    assert_eq!(
        graph_help_formats(&help),
        vec![
            "dag_jsonl",
            "conditional_graph",
            "weka_trace",
            "dynamo_trace",
            "aiperf_trace",
            "agent_recording",
            "otlp_genai",
        ]
    );
}

#[test]
fn built_in_format_commands_agree_with_the_real_adapter_inventory() {
    for source in built_in_graph_sources() {
        let validate = validate_output(&source.path, source.format, &["--output-format", "json"]);
        assert_eq!(
            validate.status.code(),
            Some(0),
            "{}: {}",
            source.format,
            stderr(&validate)
        );
        let validate: GraphValidateReport =
            serde_json::from_slice(&validate.stdout).expect("validate JSON");
        assert_eq!(validate.format, source.format);

        let explain = explain_output(&source.path, source.format, &["--output-format", "json"]);
        assert_eq!(
            explain.status.code(),
            Some(0),
            "{}: {}",
            source.format,
            stderr(&explain)
        );
        let explain: GraphExplainReport =
            serde_json::from_slice(&explain.stdout).expect("explain JSON");
        assert_eq!(explain.input.format, source.format);
        assert!(!explain.programs.is_empty(), "{}", source.format);

        if source.static_compatible {
            let visualize = output(&[
                "graph",
                "visualize",
                &source.path,
                "--graph-format",
                source.format,
                "--no-validate",
                "--output-format",
                "mermaid",
            ]);
            assert_eq!(
                visualize.status.code(),
                Some(0),
                "{}: {}",
                source.format,
                stderr(&visualize)
            );
            assert!(
                String::from_utf8(visualize.stdout)
                    .expect("Mermaid UTF-8")
                    .starts_with("flowchart LR\n")
            );
        }
    }
}

#[test]
fn unknown_non_graph_commands_refuse_without_python_delegation() {
    let output = output(&["legacy-python-command"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("unsupported native aiperf command"));
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
fn conditional_yaml_decode_failures_use_the_decode_envelope() {
    let decode_fixture = tempfile::NamedTempFile::new().expect("create decode fixture");
    std::fs::write(decode_fixture.path(), "traces: [").expect("write malformed YAML fixture");
    let decode_report = json_error(&[
        "graph",
        "validate",
        decode_fixture.path().to_str().expect("UTF-8 path"),
        "--graph-format",
        "conditional_graph",
        "--output-format=json",
    ]);
    assert_eq!(decode_report.code, GraphCommandErrorCode::InputDecodeFailed);

    let lowering_fixture = tempfile::NamedTempFile::new().expect("create lowering fixture");
    std::fs::write(
        lowering_fixture.path(),
        r#"
graph:
  state:
    result: {type: text, reducer: unsupported}
  nodes: {}
  edges: []
traces: []
"#,
    )
    .expect("write semantic YAML fixture");
    let lowering_report = json_error(&[
        "graph",
        "validate",
        lowering_fixture.path().to_str().expect("UTF-8 path"),
        "--graph-format",
        "conditional_graph",
        "--output-format=json",
    ]);
    assert_eq!(
        lowering_report.code,
        GraphCommandErrorCode::InputLoweringFailed
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

#[test]
fn graph_inspection_authors_imported_endpoint_and_source_format() {
    let claude = imported_agent_fixture("claude_code/linear.jsonl");
    let claude_path = claude.to_str().expect("UTF-8 Claude fixture");
    let messages = output(&[
        "graph",
        "validate",
        claude_path,
        "--graph-format",
        "agent_recording",
        "--source-format",
        "claude_code",
        "--endpoint-type",
        "messages",
    ]);
    assert_eq!(messages.status.code(), Some(0), "{}", stderr(&messages));
    let chat = output(&[
        "graph",
        "validate",
        claude_path,
        "--graph-format",
        "agent_recording",
        "--source-format",
        "claude_code",
    ]);
    assert_eq!(chat.status.code(), Some(2));
    assert!(stderr(&chat).contains("graph input could not be lowered"));

    let codex_directory = imported_agent_fixture("codex");
    let directory = output(&[
        "graph",
        "validate",
        codex_directory.to_str().expect("UTF-8 Codex fixture"),
        "--graph-format",
        "agent_recording",
        "--source-format",
        "codex",
    ]);
    assert_eq!(directory.status.code(), Some(0), "{}", stderr(&directory));
}
