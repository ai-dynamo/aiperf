// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native graph explanation presentation boundary.

use std::fmt::Write as _;
use std::io::{self, Write};

use aiperf_runtime::graph::inspect::{GraphInspectionOptions, inspect_bundle};

use super::report::{
    GraphChannelTypeReport, GraphCommandError, GraphCommandErrorCode, GraphEdgeAnchorReport,
    GraphExplainReport, GraphIssueReport, GraphIssueSeverityReport, GraphNodeKindReport,
    GraphPlanReport, GraphReducerReport,
};
use super::{LoadedGraphInput, TextJsonFormat};

/// Inspect one retained graph input and render its explanation report.
pub(super) fn run(
    input: LoadedGraphInput,
    output_format: TextJsonFormat,
) -> Result<(), GraphCommandError> {
    let LoadedGraphInput { source, prepared } = input;
    let inspection = inspect_bundle(&prepared.bundle, GraphInspectionOptions::default());
    let report = GraphExplainReport::from_inspection(source.display().to_string(), inspection);
    let stdout = io::stdout();
    emit_report(&report, output_format, &mut stdout.lock())
}

/// Render an explanation report with one final output write.
pub(super) fn emit_report<W: Write>(
    report: &GraphExplainReport,
    output_format: TextJsonFormat,
    output: &mut W,
) -> Result<(), GraphCommandError> {
    let rendered = match output_format {
        TextJsonFormat::Text => render_text(report),
        TextJsonFormat::Json => serde_json::to_string_pretty(report)
            .map(|mut text| {
                text.push('\n');
                text
            })
            .map_err(|cause| {
                output_error(report, "could not render graph explanation report", cause)
            })?,
    };
    output
        .write_all(rendered.as_bytes())
        .map_err(|cause| output_error(report, "could not write graph explanation report", cause))
}

fn output_error(
    report: &GraphExplainReport,
    message: &'static str,
    cause: impl Into<anyhow::Error>,
) -> GraphCommandError {
    GraphCommandError::with_cause(
        GraphCommandErrorCode::OutputWriteFailed,
        message,
        Some(report.input.source.clone()),
        cause.into(),
    )
}

fn render_text(report: &GraphExplainReport) -> String {
    let mut text = format!(
        "Input\n  source: {}\n  format: {}\n  roots: {}\n  nodes: {}\n  segments: {}\n",
        report.input.source,
        report.input.format,
        report.input.root_count,
        report.input.node_count,
        report.input.segment_count,
    );
    text.push_str("  Adapter warnings\n");
    render_issues(&mut text, 4, &report.input.adapter_warnings);
    text.push_str("  Bundle findings\n");
    render_issues(&mut text, 4, &report.input.bundle_findings);
    text.push_str("Traces\n");
    let mut wrote_wave_note = false;
    for program in &report.programs {
        let _ = writeln!(text, "Trace {}", program.trace_id);
        let _ = writeln!(text, "  driver: {}", program.driver);
        let _ = writeln!(
            text,
            "  arrival offset: {}",
            program
                .arrival_offset_ns
                .map_or_else(|| "none".to_owned(), |value| value.to_string())
        );
        let _ = writeln!(text, "  warmup: {}", yes_no(program.has_warmup));
        let _ = writeln!(text, "  environment: {}", yes_no(program.has_environment));
        let _ = writeln!(text, "  replay: {}", yes_no(program.has_replay));
        render_plan(
            &mut text,
            "  Profiling plan",
            &program.profiling,
            &mut wrote_wave_note,
        );
        if let Some(warmup) = &program.warmup {
            render_plan(&mut text, "  Warmup plan", warmup, &mut wrote_wave_note);
        }
    }
    text
}

fn render_plan(
    text: &mut String,
    heading: &str,
    plan: &GraphPlanReport,
    wrote_wave_note: &mut bool,
) {
    let summary = &plan.summary;
    let _ = writeln!(text, "{heading}");
    let _ = writeln!(
        text,
        "    Counts: nodes={}, llm_nodes={}, tool_nodes={}, edges={}, channels={}",
        summary.node_count,
        summary.llm_node_count,
        summary.tool_node_count,
        summary.edge_count,
        summary.channel_count
    );
    text.push_str("    Nodes\n");
    if plan.topology.nodes.is_empty() {
        text.push_str("      - none\n");
    }
    for node in &plan.topology.nodes {
        let kind = match node.kind {
            GraphNodeKindReport::Llm => "llm",
            GraphNodeKindReport::Tool => "tool",
        };
        let inputs = join_or_dash(
            node.inputs
                .iter()
                .map(|input| format!("{}:{}", input.channel, input.count)),
        );
        let splices = join_or_dash(node.prompt_splice_channels.iter().cloned());
        let streaming = node.streaming.map_or_else(|| "-".to_owned(), yes_no);
        let model = node.model_override.as_deref().unwrap_or("-");
        let tokens = node
            .max_tokens
            .map_or_else(|| "-".to_owned(), |value| value.to_string());
        let _ = writeln!(
            text,
            "      {} kind={} output={} inputs={} splices={} streaming={} model={} max_tokens={}",
            node.id, kind, node.output, inputs, splices, streaming, model, tokens
        );
    }
    text.push_str("    Channels\n");
    if plan.topology.channels.is_empty() {
        text.push_str("      - none\n");
    }
    for channel in &plan.topology.channels {
        let _ = writeln!(
            text,
            "      {} type={} reducer={}",
            channel.name,
            channel_type_name(channel.channel_type),
            reducer_name(channel.reducer)
        );
    }
    text.push_str("    Edges\n");
    if plan.topology.edges.is_empty() {
        text.push_str("      - none\n");
    }
    for edge in &plan.topology.edges {
        let _ = writeln!(
            text,
            "      {} -> {} anchor={} delay_us={} min_start_delay_us={}",
            edge.source,
            edge.target,
            anchor_name(edge.anchor),
            nonzero_delay(edge.delay_us),
            nonzero_delay(edge.min_start_delay_us)
        );
    }
    text.push_str("    Validation issues\n");
    render_issues(text, 6, &plan.issues);
    if let Some(waves) = &plan.readiness_waves {
        text.push_str("    Illustrative readiness waves\n");
        if !*wrote_wave_note {
            text.push_str(
                "      Waves are dependency levels, not runtime barriers or timing predictions.\n",
            );
            *wrote_wave_note = true;
        }
        for wave in waves {
            let _ = writeln!(
                text,
                "      wave {} trigger={} nodes={}",
                wave.wave,
                wave.trigger,
                wave.node_ids.join(",")
            );
        }
    } else if let Some(unavailable) = &plan.readiness_unavailable {
        let _ = writeln!(
            text,
            "    Illustrative readiness waves unavailable [{}]: {}",
            unavailable.code, unavailable.message
        );
    }
}

fn render_issues(text: &mut String, indent: usize, issues: &[GraphIssueReport]) {
    let padding = " ".repeat(indent);
    if issues.is_empty() {
        let _ = writeln!(text, "{padding}- none");
        return;
    }
    for issue in issues {
        let severity = match issue.severity {
            GraphIssueSeverityReport::Error => "ERROR",
            GraphIssueSeverityReport::Warning => "WARNING",
        };
        let _ = writeln!(
            text,
            "{padding}- {severity} [{}]: {}",
            issue.code, issue.message
        );
    }
}

fn join_or_dash(values: impl Iterator<Item = String>) -> String {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        "-".to_owned()
    } else {
        values.join(",")
    }
}

fn yes_no(value: bool) -> String {
    if value { "yes" } else { "no" }.to_owned()
}

fn channel_type_name(channel_type: GraphChannelTypeReport) -> &'static str {
    match channel_type {
        GraphChannelTypeReport::Text => "text",
        GraphChannelTypeReport::Messages => "messages",
    }
}

fn reducer_name(reducer: GraphReducerReport) -> &'static str {
    match reducer {
        GraphReducerReport::Overwrite => "overwrite",
        GraphReducerReport::AddMessages => "add_messages",
    }
}

fn anchor_name(anchor: GraphEdgeAnchorReport) -> &'static str {
    match anchor {
        GraphEdgeAnchorReport::Completion => "completion",
        GraphEdgeAnchorReport::Dispatch => "dispatch",
        GraphEdgeAnchorReport::FirstToken => "first_token",
    }
}

fn nonzero_delay(value: Option<f64>) -> String {
    value
        .filter(|delay| *delay != 0.0)
        .map_or_else(|| "-".to_owned(), |delay| delay.to_string())
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};

    use super::{TextJsonFormat, emit_report};
    use crate::graph::report::{GraphExplainInputReport, GraphExplainReport};

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "persistent test failure",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn report() -> GraphExplainReport {
        GraphExplainReport {
            schema_version: "aiperf.graph.explain.v1".to_owned(),
            input: GraphExplainInputReport {
                source: "/tmp/canonical.graph".to_owned(),
                format: "dag_jsonl".to_owned(),
                root_count: 0,
                node_count: 0,
                segment_count: 0,
                adapter_warnings: Vec::new(),
                bundle_findings: Vec::new(),
            },
            programs: Vec::new(),
        }
    }

    #[test]
    fn persistent_output_failure_is_a_typed_graph_error() {
        let mut writer = FailingWriter;
        let error = emit_report(&report(), TextJsonFormat::Json, &mut writer)
            .expect_err("writer must reject final explanation output");
        assert_eq!(error.code.as_str(), "output-write-failed");
        assert_eq!(error.source.as_deref(), Some("/tmp/canonical.graph"));
    }
}
