// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Stable fatal-error vocabulary for native graph commands.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use aiperf_runtime::graph::inspect::{
    GraphBundleInspection, GraphInspectionIssue, GraphInspectionSeverity, GraphPlanPhase,
};
use serde::{Deserialize, Serialize};

/// Native graph operation selected by the caller.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphOperation {
    /// Validate one graph input.
    Validate,
    /// Explain one graph input.
    Explain,
    /// Visualize one graph input.
    Visualize,
}

impl GraphOperation {
    /// Return the stable operation name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Validate => "validate",
            Self::Explain => "explain",
            Self::Visualize => "visualize",
        }
    }
}

/// Stable class of a graph-command fatal error.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphCommandErrorCode {
    /// Clap rejected the public argument shape.
    InvalidArguments,
    /// The local source does not exist.
    SourceNotFound,
    /// The source is not a local path.
    SourceNotLocal,
    /// The requested adapter format is unavailable.
    FormatUnsupported,
    /// The tokenizer is neither built in nor local.
    TokenizerUnsupported,
    /// The local tokenizer could not be loaded.
    TokenizerLoadFailed,
    /// The adapter could not decode the source.
    InputDecodeFailed,
    /// The adapter could not lower the source.
    InputLoweringFailed,
    /// A selected trace does not exist.
    TraceNotFound,
    /// The output target is invalid.
    OutputInvalid,
    /// The output target could not be written.
    OutputWriteFailed,
}

impl GraphCommandErrorCode {
    /// Return the stable kebab-case code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArguments => "invalid-arguments",
            Self::SourceNotFound => "source-not-found",
            Self::SourceNotLocal => "source-not-local",
            Self::FormatUnsupported => "format-unsupported",
            Self::TokenizerUnsupported => "tokenizer-unsupported",
            Self::TokenizerLoadFailed => "tokenizer-load-failed",
            Self::InputDecodeFailed => "input-decode-failed",
            Self::InputLoweringFailed => "input-lowering-failed",
            Self::TraceNotFound => "trace-not-found",
            Self::OutputInvalid => "output-invalid",
            Self::OutputWriteFailed => "output-write-failed",
        }
    }
}

/// Versioned JSON envelope for expected graph-command failures.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphErrorReport {
    /// Schema identifier.
    pub schema_version: String,
    /// Selected operation.
    pub operation: GraphOperation,
    /// Stable error class.
    pub code: GraphCommandErrorCode,
    /// Bounded, content-safe message.
    pub message: String,
    /// Canonical local source if it was reached.
    pub source: Option<String>,
}

/// Versioned JSON document returned by `aiperf graph validate --output-format json`.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphValidateReport {
    /// Schema identifier.
    pub schema_version: String,
    /// Canonical local source that was lowered.
    pub source: String,
    /// Adapter-reported input format.
    pub format: String,
    /// Adapter-reported root trace count.
    pub root_count: usize,
    /// Adapter-reported aggregate LLM node count.
    pub node_count: usize,
    /// Deterministically ordered inspection issues.
    pub issues: Vec<GraphIssueReport>,
    /// Aggregate issue counts.
    pub summary: GraphIssueSummary,
}

/// One stable graph-validation issue suitable for public output.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphIssueReport {
    /// Stable issue code.
    pub code: String,
    /// Issue severity.
    pub severity: GraphIssueSeverityReport,
    /// Trace scope, when applicable.
    pub trace_id: Option<String>,
    /// Plan phase, when applicable.
    pub phase: Option<GraphPlanPhaseReport>,
    /// Stable graph location, when applicable.
    pub location: Option<String>,
    /// Bounded, content-safe issue message.
    pub message: String,
    /// Deterministically ordered issue details.
    pub context: BTreeMap<String, String>,
}

/// Public severity vocabulary for graph-validation reports.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphIssueSeverityReport {
    /// The graph cannot be executed as authored.
    Error,
    /// The graph remains executable but has a notable condition.
    Warning,
}

/// Public plan-phase vocabulary for graph-validation reports.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphPlanPhaseReport {
    /// Profiling plan.
    Profiling,
    /// Warmup plan.
    Warmup,
}

/// Aggregate issue counts for a graph-validation report.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphIssueSummary {
    /// Number of error-severity issues.
    pub errors: usize,
    /// Number of warning-severity issues.
    pub warnings: usize,
}

impl GraphValidateReport {
    /// Convert one retained inspection into the stable validation wire document.
    pub fn from_inspection(source: String, inspection: GraphBundleInspection) -> Self {
        let GraphBundleInspection {
            format,
            root_count,
            node_count,
            mut issues,
            programs,
            ..
        } = inspection;
        for program in programs {
            issues.extend(program.profiling.issues);
            if let Some(warmup) = program.warmup {
                issues.extend(warmup.issues);
            }
        }
        let issues = issues
            .into_iter()
            .map(GraphIssueReport::from)
            .collect::<Vec<_>>();
        let summary = GraphIssueSummary {
            errors: issues
                .iter()
                .filter(|issue| matches!(issue.severity, GraphIssueSeverityReport::Error))
                .count(),
            warnings: issues
                .iter()
                .filter(|issue| matches!(issue.severity, GraphIssueSeverityReport::Warning))
                .count(),
        };
        Self {
            schema_version: "aiperf.graph.validate.v1".to_owned(),
            source,
            format,
            root_count,
            node_count,
            issues,
            summary,
        }
    }
}

impl From<GraphInspectionIssue> for GraphIssueReport {
    fn from(issue: GraphInspectionIssue) -> Self {
        Self {
            code: issue.code,
            severity: match issue.severity {
                GraphInspectionSeverity::Error => GraphIssueSeverityReport::Error,
                GraphInspectionSeverity::Warning => GraphIssueSeverityReport::Warning,
            },
            trace_id: issue.trace_id,
            phase: issue.phase.map(|phase| match phase {
                GraphPlanPhase::Profiling => GraphPlanPhaseReport::Profiling,
                GraphPlanPhase::Warmup => GraphPlanPhaseReport::Warmup,
            }),
            location: issue.location,
            message: issue.message,
            context: issue.context,
        }
    }
}

/// An expected graph-command failure retained until dispatcher-owned rendering.
#[derive(Debug)]
pub struct GraphCommandError {
    /// Stable error class.
    pub code: GraphCommandErrorCode,
    /// Bounded, content-safe message.
    pub message: String,
    /// Canonical local source if it was reached.
    pub source: Option<String>,
    /// Opaque expected-failure chain retained for logging and callers only.
    cause: Option<anyhow::Error>,
}

impl GraphCommandError {
    /// Build an expected command failure.
    pub fn new(
        code: GraphCommandErrorCode,
        message: impl Into<String>,
        source: Option<String>,
    ) -> Self {
        Self {
            code,
            message: bound_message(message.into()),
            source,
            cause: None,
        }
    }

    /// Build an expected command failure while retaining its opaque source chain.
    pub fn with_cause(
        code: GraphCommandErrorCode,
        message: impl Into<String>,
        source: Option<String>,
        cause: anyhow::Error,
    ) -> Self {
        Self {
            code,
            message: bound_message(message.into()),
            source,
            cause: Some(cause),
        }
    }

    /// Convert to the public JSON envelope.
    pub fn report(&self, operation: GraphOperation) -> GraphErrorReport {
        GraphErrorReport {
            schema_version: "aiperf.graph.error.v1".to_owned(),
            operation,
            code: self.code,
            message: self.message.clone(),
            source: self.source.clone(),
        }
    }
}

impl fmt::Display for GraphCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for GraphCommandError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.cause.as_ref().map(|cause| cause.as_ref())
    }
}

/// Restrict public messages to 1024 Unicode scalar values.
pub fn bound_message(message: String) -> String {
    const MAX_SCALARS: usize = 1024;
    if message.chars().count() <= MAX_SCALARS {
        return message;
    }
    let prefix: String = message.chars().take(MAX_SCALARS - 1).collect();
    format!("{prefix}…")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error as _;

    use aiperf_runtime::graph::inspect::{
        GraphBundleInspection, GraphInspectionIssue, GraphInspectionSeverity, GraphPlanInspection,
        GraphPlanPhase, GraphPlanSummary, GraphProgramInspection, GraphTopologyInspection,
        ReadinessInspection,
    };

    use super::{
        GraphCommandError, GraphCommandErrorCode, GraphIssueSeverityReport, GraphOperation,
        GraphPlanPhaseReport, GraphValidateReport,
    };

    fn issue(
        code: &str,
        severity: GraphInspectionSeverity,
        trace_id: Option<&str>,
        phase: Option<GraphPlanPhase>,
        location: Option<&str>,
        context: BTreeMap<String, String>,
    ) -> GraphInspectionIssue {
        GraphInspectionIssue {
            code: code.to_owned(),
            severity,
            trace_id: trace_id.map(str::to_owned),
            phase,
            location: location.map(str::to_owned),
            message: format!("{code} message"),
            context,
        }
    }

    fn plan(phase: GraphPlanPhase, issues: Vec<GraphInspectionIssue>) -> GraphPlanInspection {
        GraphPlanInspection {
            phase,
            summary: GraphPlanSummary {
                node_count: 0,
                llm_node_count: 0,
                tool_node_count: 0,
                edge_count: 0,
                channel_count: 0,
            },
            topology: GraphTopologyInspection {
                nodes: Vec::new(),
                channels: Vec::new(),
                edges: Vec::new(),
            },
            issues,
            readiness: ReadinessInspection::Unavailable {
                code: "test".to_owned(),
                message: "test only".to_owned(),
            },
        }
    }

    #[test]
    fn opaque_cause_is_retained_without_reaching_the_public_report() {
        let error = GraphCommandError::with_cause(
            GraphCommandErrorCode::InputLoweringFailed,
            "graph input could not be lowered",
            Some("/tmp/input.json".to_owned()),
            anyhow::anyhow!("sensitive adapter context"),
        );

        assert_eq!(
            error.source().map(ToString::to_string).as_deref(),
            Some("sensitive adapter context")
        );
        let serialized = serde_json::to_string(&error.report(GraphOperation::Validate))
            .expect("serialize public error report");
        assert!(!serialized.contains("sensitive adapter context"));
    }

    #[test]
    fn validation_report_flattens_bundle_then_profiling_then_warmup() {
        let bundle_issue = issue(
            "bundle-warning",
            GraphInspectionSeverity::Warning,
            None,
            None,
            None,
            BTreeMap::from([
                ("z".to_owned(), "last".to_owned()),
                ("a".to_owned(), "first".to_owned()),
            ]),
        );
        let profiling_issue = issue(
            "profiling-error",
            GraphInspectionSeverity::Error,
            Some("trace-1"),
            Some(GraphPlanPhase::Profiling),
            Some("graph.nodes.p"),
            BTreeMap::new(),
        );
        let warmup_issue = issue(
            "warmup-warning",
            GraphInspectionSeverity::Warning,
            Some("trace-1"),
            Some(GraphPlanPhase::Warmup),
            Some("graph.nodes.w"),
            BTreeMap::new(),
        );
        let inspection = GraphBundleInspection {
            format: "conditional_graph".to_owned(),
            root_count: 1,
            node_count: 2,
            segment_count: 0,
            issues: vec![bundle_issue],
            programs: vec![GraphProgramInspection {
                trace_id: "trace-1".to_owned(),
                driver: "static_graph".to_owned(),
                arrival_offset_ns: None,
                has_environment: false,
                has_replay: false,
                profiling: plan(GraphPlanPhase::Profiling, vec![profiling_issue]),
                warmup: Some(plan(GraphPlanPhase::Warmup, vec![warmup_issue])),
            }],
        };

        let report = GraphValidateReport::from_inspection("/tmp/source".to_owned(), inspection);

        assert_eq!(
            report
                .issues
                .iter()
                .map(|issue| issue.code.as_str())
                .collect::<Vec<_>>(),
            ["bundle-warning", "profiling-error", "warmup-warning"]
        );
        assert!(matches!(
            report.issues[0].severity,
            GraphIssueSeverityReport::Warning
        ));
        assert_eq!(report.issues[0].trace_id, None);
        assert!(report.issues[0].phase.is_none());
        assert_eq!(report.issues[0].location, None);
        assert_eq!(
            report.issues[0]
                .context
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["a", "z"]
        );
        assert!(matches!(
            report.issues[2].phase,
            Some(GraphPlanPhaseReport::Warmup)
        ));
        assert_eq!(report.summary.errors, 1);
        assert_eq!(report.summary.warnings, 2);
        let json = serde_json::to_value(report).expect("serialize validation report");
        assert!(json["issues"][0]["trace_id"].is_null());
        assert!(json["issues"][0]["phase"].is_null());
        assert!(json["issues"][0]["location"].is_null());
    }
}
