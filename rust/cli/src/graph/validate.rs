// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native graph validation presentation boundary.

use std::fmt::Write as _;
use std::io::{self, Write as _};

use aiperf_runtime::graph::inspect::{GraphInspectionOptions, inspect_bundle};

use super::report::{
    GraphIssueReport, GraphIssueSeverityReport, GraphPlanPhaseReport, GraphValidateReport,
};
use super::{LoadedGraphInput, TextJsonFormat};

/// Inspect one retained graph input and render its validation report.
pub(super) fn run(
    input: LoadedGraphInput,
    output_format: TextJsonFormat,
    requires_arrival_offsets: bool,
) -> anyhow::Result<i32> {
    let LoadedGraphInput { source, prepared } = input;
    let inspection = inspect_bundle(
        &prepared.bundle,
        GraphInspectionOptions {
            requires_arrival_offsets,
        },
    );
    let report = GraphValidateReport::from_inspection(source.display().to_string(), inspection);
    match output_format {
        TextJsonFormat::Text => print_text(&report)?,
        TextJsonFormat::Json => print_json(&report)?,
    }
    Ok(if report.summary.errors > 0 { 1 } else { 0 })
}

fn print_json(report: &GraphValidateReport) -> anyhow::Result<()> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, report)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn print_text(report: &GraphValidateReport) -> anyhow::Result<()> {
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
    io::stdout().lock().write_all(rendered.as_bytes())?;
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
