// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native graph validation presentation boundary.

use std::fmt::Write as _;
use std::io::{self, Write};

use aiperf_runtime::graph::inspect::{GraphInspectionOptions, inspect_bundle};

use super::report::{
    GraphCommandError, GraphCommandErrorCode, GraphIssueReport, GraphIssueSeverityReport,
    GraphPlanPhaseReport, GraphValidateReport,
};
use super::{LoadedGraphInput, TextJsonFormat};

/// Inspect one retained graph input and render its validation report.
pub(super) fn run(
    input: LoadedGraphInput,
    output_format: TextJsonFormat,
    requires_arrival_offsets: bool,
) -> Result<i32, GraphCommandError> {
    let LoadedGraphInput { source, prepared } = input;
    let inspection = inspect_bundle(
        &prepared.bundle,
        GraphInspectionOptions {
            requires_arrival_offsets,
        },
    );
    let report = GraphValidateReport::from_inspection(source.display().to_string(), inspection);
    let stdout = io::stdout();
    let mut output = stdout.lock();
    render_report(&report, output_format, &mut output)
        .map_err(|error| output_write_error(&report.source, error))?;
    Ok(if report.summary.errors > 0 { 1 } else { 0 })
}

fn output_write_error(source: &str, cause: anyhow::Error) -> GraphCommandError {
    GraphCommandError::with_cause(
        GraphCommandErrorCode::OutputWriteFailed,
        "could not write graph validation report",
        Some(source.to_owned()),
        cause,
    )
}

fn render_report<W: Write>(
    report: &GraphValidateReport,
    output_format: TextJsonFormat,
    output: &mut W,
) -> anyhow::Result<()> {
    match output_format {
        TextJsonFormat::Text => render_text(report, output),
        TextJsonFormat::Json => render_json(report, output),
    }
}

fn render_json<W: Write>(report: &GraphValidateReport, output: &mut W) -> anyhow::Result<()> {
    serde_json::to_writer_pretty(&mut *output, report)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn render_text<W: Write>(report: &GraphValidateReport, output: &mut W) -> anyhow::Result<()> {
    let mut rendered = String::new();
    for issue in &report.issues {
        rendered.push_str(&issue_line(issue));
        rendered.push('\n');
    }
    let status = if report.summary.errors == 0 {
        "OK"
    } else {
        "FAIL"
    };
    let error_count = if report.summary.errors == 0 {
        "0 errors".to_owned()
    } else {
        format!("{} error(s)", report.summary.errors)
    };
    rendered.push_str(&format!(
        "{status}: {error_count}, {} warning(s).\n",
        report.summary.warnings,
    ));
    output.write_all(rendered.as_bytes())?;
    Ok(())
}

fn issue_line(issue: &GraphIssueReport) -> String {
    let severity = match issue.severity {
        GraphIssueSeverityReport::Error => "ERROR",
        GraphIssueSeverityReport::Warning => "WARNING",
    };
    let mut output = format!("{severity} [{}]", issue.code);
    if let Some(trace_id) = &issue.trace_id {
        let _ = write!(output, " trace={trace_id}");
    }
    if let Some(phase) = issue.phase {
        let phase = match phase {
            GraphPlanPhaseReport::Profiling => "profiling",
            GraphPlanPhaseReport::Warmup => "warmup",
        };
        let _ = write!(output, " phase={phase}");
    }
    if let Some(location) = &issue.location {
        let _ = write!(output, " {location}");
    }
    let message = issue_message(issue);
    let _ = write!(output, ": {message}");
    if !issue.context.is_empty() {
        let context = issue
            .context
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(output, " ({context})");
    }
    output
}

fn issue_message(issue: &GraphIssueReport) -> String {
    issue.code.strip_prefix("adapter-warning.").map_or_else(
        || issue.message.clone(),
        |code| format!("adapter warning {code}"),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{self, Write};

    use super::{TextJsonFormat, output_write_error, render_report};
    use crate::graph::report::{
        GraphIssueReport, GraphIssueSeverityReport, GraphIssueSummary, GraphOperation,
        GraphValidateReport,
    };

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "test broken pipe",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn report() -> GraphValidateReport {
        GraphValidateReport {
            schema_version: "aiperf.graph.validate.v1".to_owned(),
            source: "/tmp/graph.json".to_owned(),
            format: "dag_jsonl".to_owned(),
            root_count: 1,
            node_count: 1,
            issues: vec![GraphIssueReport {
                code: "test".to_owned(),
                severity: GraphIssueSeverityReport::Error,
                trace_id: None,
                phase: None,
                location: None,
                message: "test issue".to_owned(),
                context: BTreeMap::new(),
            }],
            summary: GraphIssueSummary {
                errors: 1,
                warnings: 0,
            },
        }
    }

    #[test]
    fn output_write_failure_is_a_typed_graph_error() {
        let report = report();
        let mut writer = FailingWriter;
        let cause = render_report(&report, TextJsonFormat::Json, &mut writer)
            .expect_err("failing writer must reject JSON output");
        let error = output_write_error(&report.source, cause);

        assert_eq!(error.code.as_str(), "output-write-failed");
        assert_eq!(error.message, "could not write graph validation report");
        assert_eq!(error.source.as_deref(), Some("/tmp/graph.json"));
        let value = serde_json::to_value(error.report(GraphOperation::Validate))
            .expect("serialize output failure");
        assert_eq!(value["schema_version"], "aiperf.graph.error.v1");
        assert_eq!(value["code"], "output-write-failed");
    }
}
