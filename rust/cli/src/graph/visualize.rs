// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native graph visualization presentation boundary.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{self, Write};

use aiperf_runtime::graph::inspect::{
    GraphBundleInspection, GraphEdgeAnchor, GraphInspectionOptions, GraphInspectionSeverity,
    GraphProgramInspection, GraphTopologyInspection, ReadinessInspection, inspect_bundle,
};
use aiperf_runtime::graph::model::{END_NODE_ID, START_NODE_ID};

use super::report::{GraphCommandError, GraphCommandErrorCode};
use super::{LoadedGraphInput, VisualizeFormat};

/// Render one selected profiling topology after the single retained load.
pub(super) fn run(
    input: LoadedGraphInput,
    requested_trace: Option<&str>,
    output_format: VisualizeFormat,
    no_validate: bool,
) -> Result<i32, GraphCommandError> {
    let LoadedGraphInput { source, prepared } = input;
    let inspection = inspect_bundle(&prepared.bundle, GraphInspectionOptions::default());
    let program = select_program(&inspection, requested_trace, &source)?;
    let issues = validation_errors(&inspection, program);
    if !no_validate && !issues.is_empty() {
        write_validation_errors(&issues)?;
        return Ok(1);
    }

    let mermaid = render_mermaid(&program.profiling.topology);
    let rendered = match output_format {
        VisualizeFormat::Mermaid => mermaid,
        VisualizeFormat::Markdown => render_markdown(
            source.display().to_string(),
            &inspection.format,
            program,
            &mermaid,
        ),
    };
    let stdout = io::stdout();
    stdout
        .lock()
        .write_all(rendered.as_bytes())
        .map_err(|cause| {
            GraphCommandError::with_cause(
                GraphCommandErrorCode::OutputWriteFailed,
                "could not write graph visualization",
                Some(source.display().to_string()),
                anyhow::Error::new(cause),
            )
        })?;
    Ok(0)
}

fn select_program<'a>(
    inspection: &'a GraphBundleInspection,
    requested_trace: Option<&str>,
    source: &std::path::Path,
) -> Result<&'a GraphProgramInspection, GraphCommandError> {
    let program = match requested_trace {
        Some(trace_id) => inspection
            .programs
            .iter()
            .find(|program| program.trace_id == trace_id),
        None => inspection.programs.first(),
    };
    program.ok_or_else(|| {
        let available = inspection
            .programs
            .iter()
            .map(|program| program.trace_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let detail = if requested_trace.is_some() {
            format!("[trace-not-found] trace was not found; available traces: {available}")
        } else {
            "[bundle-empty] graph input contains no programs".to_owned()
        };
        GraphCommandError::new(
            if requested_trace.is_some() {
                GraphCommandErrorCode::TraceNotFound
            } else {
                GraphCommandErrorCode::InputLoweringFailed
            },
            detail,
            Some(source.display().to_string()),
        )
    })
}

fn validation_errors<'a>(
    inspection: &'a GraphBundleInspection,
    program: &'a GraphProgramInspection,
) -> Vec<&'a aiperf_runtime::graph::inspect::GraphInspectionIssue> {
    inspection
        .issues
        .iter()
        .chain(program.profiling.issues.iter())
        .filter(|issue| issue.severity == GraphInspectionSeverity::Error)
        .collect()
}

fn write_validation_errors(
    issues: &[&aiperf_runtime::graph::inspect::GraphInspectionIssue],
) -> Result<(), GraphCommandError> {
    let stderr = io::stderr();
    let mut output = stderr.lock();
    for issue in issues {
        writeln!(output, "ERROR [{}]: {}", issue.code, issue.message).map_err(|cause| {
            GraphCommandError::with_cause(
                GraphCommandErrorCode::OutputWriteFailed,
                "could not write graph visualization diagnostics",
                None,
                anyhow::Error::new(cause),
            )
        })?;
    }
    writeln!(
        output,
        "rerun with --no-validate to render best-effort topology"
    )
    .map_err(|cause| {
        GraphCommandError::with_cause(
            GraphCommandErrorCode::OutputWriteFailed,
            "could not write graph visualization diagnostics",
            None,
            anyhow::Error::new(cause),
        )
    })
}

/// Render one normalized resolved Graph-IR topology as Mermaid source.
pub(super) fn render_mermaid(topology: &GraphTopologyInspection) -> String {
    let node_ids = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id.as_str(), format!("n{index}")))
        .collect::<BTreeMap<_, _>>();
    let mut unknown_ids = BTreeMap::new();
    for edge in &topology.edges {
        for endpoint in [&edge.source, &edge.target] {
            if endpoint != START_NODE_ID
                && endpoint != END_NODE_ID
                && !node_ids.contains_key(endpoint.as_str())
            {
                let next = unknown_ids.len();
                unknown_ids
                    .entry(endpoint.as_str())
                    .or_insert_with(|| format!("u{next}"));
            }
        }
    }

    let mut text = "flowchart LR\n  start([\"START\"])\n".to_owned();
    for (index, node) in topology.nodes.iter().enumerate() {
        let identifier = format!("n{index}");
        let class = match node.kind {
            aiperf_runtime::graph::inspect::GraphNodeKind::Llm => "llm",
            aiperf_runtime::graph::inspect::GraphNodeKind::Tool => "tool",
        };
        let _ = writeln!(
            text,
            "  {identifier}[\"{}\"]:::{class}",
            escape_label(&node.id)
        );
    }
    for (node_id, identifier) in &unknown_ids {
        let _ = writeln!(
            text,
            "  {identifier}[\"{}\"]:::invalid",
            escape_label(node_id)
        );
    }
    text.push_str("  end_node([\"END\"])\n");

    for edge in &topology.edges {
        let Some(source) = endpoint_identifier(&edge.source, &node_ids, &unknown_ids) else {
            continue;
        };
        let Some(target) = endpoint_identifier(&edge.target, &node_ids, &unknown_ids) else {
            continue;
        };
        let _ = writeln!(text, "  {source} -->|\"{}\"| {target}", edge_label(edge));
    }
    for node in &topology.nodes {
        let has_explicit_successor = topology.edges.iter().any(|edge| {
            edge.source == node.id
                && (edge.target == END_NODE_ID || node_ids.contains_key(edge.target.as_str()))
        });
        if !has_explicit_successor {
            if let Some(identifier) = node_ids.get(node.id.as_str()) {
                let _ = writeln!(text, "  {identifier} -. implicit .-> end_node");
            }
        }
    }
    text.push_str("  classDef llm fill:#76b900,color:#000,stroke:#333\n");
    text.push_str("  classDef tool fill:#5b8def,color:#fff,stroke:#333\n");
    text.push_str("  classDef invalid fill:#d9534f,color:#fff,stroke:#333\n");
    text
}

fn endpoint_identifier<'a>(
    endpoint: &'a str,
    node_ids: &'a BTreeMap<&str, String>,
    unknown_ids: &'a BTreeMap<&str, String>,
) -> Option<&'a str> {
    match endpoint {
        START_NODE_ID => Some("start"),
        END_NODE_ID => Some("end_node"),
        _ => node_ids
            .get(endpoint)
            .or_else(|| unknown_ids.get(endpoint))
            .map(String::as_str),
    }
}

fn edge_label(edge: &aiperf_runtime::graph::inspect::GraphEdgeInspection) -> String {
    let delay = edge.delay_us.filter(|value| *value != 0.0);
    let min_start = edge.min_start_delay_us.filter(|value| *value != 0.0);
    let anchor = match edge.anchor {
        GraphEdgeAnchor::Completion => "completion",
        GraphEdgeAnchor::Dispatch => "dispatch",
        GraphEdgeAnchor::FirstToken => "first-token",
    };
    let mut label = delay.map_or_else(|| anchor.to_owned(), |value| format!("{anchor} {value}us"));
    if let Some(value) = min_start {
        let _ = write!(label, ", min-start {value}us");
    }
    label
}

fn escape_label(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace("\r\n", "<br/>")
        .replace('\r', "<br/>")
        .replace('\n', "<br/>")
}

fn render_markdown(
    source: String,
    format: &str,
    program: &GraphProgramInspection,
    mermaid: &str,
) -> String {
    let mut text = format!(
        "## Graph topology\n\n```mermaid\n{mermaid}```\n\nThis is the selected trace's resolved Graph-IR topology.\n\n## Resolved plan\n\n- Source: {}\n- Format: {}\n- Trace: {}\n- Driver: {}\n\n## Illustrative readiness waves\n\n",
        escape_markdown(&source),
        escape_markdown(format),
        escape_markdown(&program.trace_id),
        escape_markdown(&program.driver),
    );
    match &program.profiling.readiness {
        ReadinessInspection::Available { waves } => {
            text.push_str("| Wave | Nodes ready | Trigger |\n|---:|---|---|\n");
            for wave in waves {
                let _ = writeln!(
                    text,
                    "| {} | {} | {} |",
                    wave.wave,
                    wave.node_ids
                        .iter()
                        .map(|node_id| escape_markdown(node_id))
                        .collect::<Vec<_>>()
                        .join(", "),
                    escape_markdown(&normalize_readiness_trigger(&wave.trigger))
                );
            }
        }
        ReadinessInspection::Unavailable { code, message } => {
            let _ = writeln!(
                text,
                "Unavailable [{}]: {}.",
                escape_markdown(code),
                escape_markdown(message)
            );
        }
    }
    text
}

fn normalize_readiness_trigger(trigger: &str) -> String {
    trigger
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(", ")
}

fn escape_markdown(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "\\|")
        .replace('!', "\\!")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('(', "\\(")
        .replace(')', "\\)")
        .replace("\r\n", "<br/>")
        .replace('\r', "<br/>")
        .replace('\n', "<br/>")
}
