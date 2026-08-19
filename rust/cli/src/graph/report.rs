// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Stable fatal-error vocabulary for native graph commands.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use aiperf_runtime::graph::inspect::{
    GraphBundleInspection, GraphChannelInspection, GraphEdgeAnchor, GraphEdgeInspection,
    GraphInspectionIssue, GraphInspectionSeverity, GraphNodeInspection, GraphNodeKind,
    GraphPlanInspection, GraphPlanPhase, GraphPlanSummary, GraphProgramInspection,
    GraphTopologyInspection, ReadinessInspection, ReadinessWave,
};
use aiperf_runtime::graph::model::{ChannelType, ReducerName};
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

/// Versioned JSON document returned by `aiperf graph explain --output-format json`.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphExplainReport {
    /// Schema identifier.
    pub schema_version: String,
    /// Input-wide facts safe for public presentation.
    pub input: GraphExplainInputReport,
    /// Resolved programs in adapter-authored order.
    pub programs: Vec<GraphProgramReport>,
}

/// Public input facts shared by all explained programs.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphExplainInputReport {
    /// Canonical local source that was lowered.
    pub source: String,
    /// Adapter-reported input format.
    pub format: String,
    /// Adapter-reported root trace count.
    pub root_count: usize,
    /// Adapter-reported aggregate LLM node count.
    pub node_count: usize,
    /// Number of retained immutable segments, without exposing their handles or payloads.
    pub segment_count: usize,
    /// Bundle-level warning findings only.
    pub adapter_warnings: Vec<GraphIssueReport>,
}

/// One resolved trace program safe for public explanation.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphProgramReport {
    /// Profiling trace identifier.
    pub trace_id: String,
    /// Registered runtime driver kind.
    pub driver: String,
    /// Profiling arrival offset, when authored.
    pub arrival_offset_ns: Option<i64>,
    /// Whether a warmup plan is retained.
    pub has_warmup: bool,
    /// Whether an environment recipe is retained, without exposing its contents.
    pub has_environment: bool,
    /// Whether recorded replay metadata is retained, without exposing its contents.
    pub has_replay: bool,
    /// Profiling-plan topology and analysis.
    pub profiling: GraphPlanReport,
    /// Warmup-plan topology and analysis, when retained.
    pub warmup: Option<GraphPlanReport>,
}

/// One resolved graph-plan explanation.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphPlanReport {
    /// Aggregate topology facts.
    pub summary: GraphPlanSummaryReport,
    /// Deterministically normalized runtime topology.
    pub topology: GraphTopologyReport,
    /// Plan-scoped validation findings.
    pub issues: Vec<GraphIssueReport>,
    /// Illustrative readiness waves, when static analysis is available.
    pub readiness_waves: Option<Vec<ReadinessWaveReport>>,
    /// Typed reason illustrative readiness waves are unavailable.
    pub readiness_unavailable: Option<ReadinessUnavailableReport>,
}

/// Aggregate count facts for one graph plan.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphPlanSummaryReport {
    /// Number of executable nodes, including tools.
    pub node_count: usize,
    /// Number of LLM nodes.
    pub llm_node_count: usize,
    /// Number of tool nodes.
    pub tool_node_count: usize,
    /// Number of static edges.
    pub edge_count: usize,
    /// Number of declared state channels.
    pub channel_count: usize,
}

/// Deterministically normalized topology safe for public serialization.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphTopologyReport {
    /// Nodes in inspection order.
    pub nodes: Vec<GraphNodeReport>,
    /// Channels in lexical name order.
    pub channels: Vec<GraphChannelReport>,
    /// Edges in normalized dependency order.
    pub edges: Vec<GraphEdgeReport>,
}

/// Public executable-node kind.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKindReport {
    /// A measured LLM invocation.
    Llm,
    /// A tool observation producer.
    Tool,
}

/// Public state-channel value type.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphChannelTypeReport {
    /// Text channel values.
    Text,
    /// Message channel values.
    Messages,
}

/// Public state-channel reducer.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphReducerReport {
    /// Last-write-wins reducer.
    Overwrite,
    /// Message append reducer.
    AddMessages,
}

/// Public edge timing anchor.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeAnchorReport {
    /// The predecessor completion gates the successor.
    Completion,
    /// The predecessor dispatch gates the successor.
    Dispatch,
    /// The predecessor first token gates the successor.
    FirstToken,
}

/// One input-channel requirement.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphNodeInputReport {
    /// Required channel name.
    pub channel: String,
    /// Authored count rendered as decimal or `all`.
    pub count: String,
}

/// One presentation-safe executable-node record.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphNodeReport {
    /// Node identity.
    pub id: String,
    /// Runtime node kind.
    pub kind: GraphNodeKindReport,
    /// Declared output channel.
    pub output: String,
    /// Declared firing-gate inputs.
    pub inputs: Vec<GraphNodeInputReport>,
    /// Dynamic prompt splice-channel names.
    pub prompt_splice_channels: Vec<String>,
    /// LLM streaming setting, absent for tools.
    pub streaming: Option<bool>,
    /// LLM request model override, absent for tools.
    pub model_override: Option<String>,
    /// LLM generation cap, absent for tools.
    pub max_tokens: Option<usize>,
}

/// One declared state channel.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphChannelReport {
    /// Channel name.
    pub name: String,
    /// Runtime channel value type.
    #[serde(rename = "type")]
    pub channel_type: GraphChannelTypeReport,
    /// Runtime channel reducer.
    pub reducer: GraphReducerReport,
}

/// One normalized static dependency edge.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphEdgeReport {
    /// Source node identity, START, or END.
    pub source: String,
    /// Target node identity, START, or END.
    pub target: String,
    /// Runtime timing anchor.
    pub anchor: GraphEdgeAnchorReport,
    /// Selected anchor delay, retaining an authored zero.
    pub delay_us: Option<f64>,
    /// Minimum start delay, when authored.
    pub min_start_delay_us: Option<f64>,
}

/// One deterministic illustrative readiness wave.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ReadinessWaveReport {
    /// Zero-based wave index.
    pub wave: usize,
    /// Nodes admitted in this wave.
    pub node_ids: Vec<String>,
    /// Illustrative trigger summary.
    pub trigger: String,
}

/// Typed reason illustrative readiness waves cannot be rendered.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ReadinessUnavailableReport {
    /// Stable analysis reason identifier.
    pub code: String,
    /// Bounded, content-safe explanation.
    pub message: String,
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

impl GraphExplainReport {
    /// Convert one retained inspection into the stable explanation wire document.
    pub fn from_inspection(source: String, inspection: GraphBundleInspection) -> Self {
        let GraphBundleInspection {
            format,
            root_count,
            node_count,
            segment_count,
            issues,
            programs,
        } = inspection;
        Self {
            schema_version: "aiperf.graph.explain.v1".to_owned(),
            input: GraphExplainInputReport {
                source,
                format,
                root_count,
                node_count,
                segment_count,
                adapter_warnings: issues
                    .into_iter()
                    .filter(|issue| issue.severity == GraphInspectionSeverity::Warning)
                    .map(GraphIssueReport::from)
                    .collect(),
            },
            programs: programs.into_iter().map(GraphProgramReport::from).collect(),
        }
    }
}

impl From<GraphProgramInspection> for GraphProgramReport {
    fn from(program: GraphProgramInspection) -> Self {
        Self {
            trace_id: program.trace_id,
            driver: program.driver,
            arrival_offset_ns: program.arrival_offset_ns,
            has_warmup: program.warmup.is_some(),
            has_environment: program.has_environment,
            has_replay: program.has_replay,
            profiling: program.profiling.into(),
            warmup: program.warmup.map(GraphPlanReport::from),
        }
    }
}

impl From<GraphPlanInspection> for GraphPlanReport {
    fn from(plan: GraphPlanInspection) -> Self {
        let (readiness_waves, readiness_unavailable) = match plan.readiness {
            ReadinessInspection::Available { waves } => (
                Some(waves.into_iter().map(ReadinessWaveReport::from).collect()),
                None,
            ),
            ReadinessInspection::Unavailable { code, message } => {
                (None, Some(ReadinessUnavailableReport { code, message }))
            }
        };
        Self {
            summary: plan.summary.into(),
            topology: plan.topology.into(),
            issues: plan
                .issues
                .into_iter()
                .map(GraphIssueReport::from)
                .collect(),
            readiness_waves,
            readiness_unavailable,
        }
    }
}

impl From<GraphPlanSummary> for GraphPlanSummaryReport {
    fn from(summary: GraphPlanSummary) -> Self {
        Self {
            node_count: summary.node_count,
            llm_node_count: summary.llm_node_count,
            tool_node_count: summary.tool_node_count,
            edge_count: summary.edge_count,
            channel_count: summary.channel_count,
        }
    }
}

impl From<GraphTopologyInspection> for GraphTopologyReport {
    fn from(topology: GraphTopologyInspection) -> Self {
        Self {
            nodes: topology
                .nodes
                .into_iter()
                .map(GraphNodeReport::from)
                .collect(),
            channels: topology
                .channels
                .into_iter()
                .map(GraphChannelReport::from)
                .collect(),
            edges: topology
                .edges
                .into_iter()
                .map(GraphEdgeReport::from)
                .collect(),
        }
    }
}

impl From<GraphNodeInspection> for GraphNodeReport {
    fn from(node: GraphNodeInspection) -> Self {
        Self {
            id: node.id,
            kind: match node.kind {
                GraphNodeKind::Llm => GraphNodeKindReport::Llm,
                GraphNodeKind::Tool => GraphNodeKindReport::Tool,
            },
            output: node.output,
            inputs: node
                .inputs
                .into_iter()
                .map(|input| GraphNodeInputReport {
                    channel: input.channel,
                    count: input.count,
                })
                .collect(),
            prompt_splice_channels: node.prompt_splice_channels,
            streaming: node.streaming,
            model_override: node.model_override,
            max_tokens: node.max_tokens,
        }
    }
}

impl From<GraphChannelInspection> for GraphChannelReport {
    fn from(channel: GraphChannelInspection) -> Self {
        Self {
            name: channel.name,
            channel_type: match channel.channel_type {
                ChannelType::Text => GraphChannelTypeReport::Text,
                ChannelType::Messages => GraphChannelTypeReport::Messages,
            },
            reducer: match channel.reducer {
                ReducerName::Overwrite => GraphReducerReport::Overwrite,
                ReducerName::AddMessages => GraphReducerReport::AddMessages,
            },
        }
    }
}

impl From<GraphEdgeInspection> for GraphEdgeReport {
    fn from(edge: GraphEdgeInspection) -> Self {
        Self {
            source: edge.source,
            target: edge.target,
            anchor: match edge.anchor {
                GraphEdgeAnchor::Completion => GraphEdgeAnchorReport::Completion,
                GraphEdgeAnchor::Dispatch => GraphEdgeAnchorReport::Dispatch,
                GraphEdgeAnchor::FirstToken => GraphEdgeAnchorReport::FirstToken,
            },
            delay_us: edge.delay_us,
            min_start_delay_us: edge.min_start_delay_us,
        }
    }
}

impl From<ReadinessWave> for ReadinessWaveReport {
    fn from(wave: ReadinessWave) -> Self {
        Self {
            wave: wave.wave,
            node_ids: wave.node_ids,
            trigger: wave.trigger,
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
    use std::sync::Arc;

    use aiperf_runtime::dataset::{Payload, SegmentPool};
    use aiperf_runtime::graph::driver::{
        ReplayTaskIdentity, ReplayTraceMetadata, TraceDriverSpec, TraceEnvironmentSpec,
    };
    use aiperf_runtime::graph::input::{GraphInputBundle, GraphInputMetadata};
    use aiperf_runtime::graph::inspect::{
        GraphBundleInspection, GraphInspectionIssue, GraphInspectionOptions,
        GraphInspectionSeverity, GraphPlanInspection, GraphPlanPhase, GraphPlanSummary,
        GraphProgramInspection, GraphTopologyInspection, ReadinessInspection, inspect_bundle,
    };
    use aiperf_runtime::graph::model::{
        ExecutableGraphNode, GraphRecord, GraphTracePlan, GraphTraceProgram, LlmNode,
        LlmRequestSpec, PromptItem, StaticEdge, ToolNode, TraceRecord,
    };

    use super::{
        GraphCommandError, GraphCommandErrorCode, GraphExplainReport, GraphIssueSeverityReport,
        GraphOperation, GraphPlanPhaseReport, GraphValidateReport,
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
        let second_profiling_issue = issue(
            "second-profiling-warning",
            GraphInspectionSeverity::Warning,
            Some("trace-2"),
            Some(GraphPlanPhase::Profiling),
            Some("graph.nodes.p2"),
            BTreeMap::new(),
        );
        let second_warmup_issue = issue(
            "second-warmup-error",
            GraphInspectionSeverity::Error,
            Some("trace-2"),
            Some(GraphPlanPhase::Warmup),
            Some("graph.nodes.w2"),
            BTreeMap::new(),
        );
        let inspection = GraphBundleInspection {
            format: "conditional_graph".to_owned(),
            root_count: 1,
            node_count: 2,
            segment_count: 0,
            issues: vec![bundle_issue],
            programs: vec![
                GraphProgramInspection {
                    trace_id: "trace-1".to_owned(),
                    driver: "static_graph".to_owned(),
                    arrival_offset_ns: None,
                    has_environment: false,
                    has_replay: false,
                    profiling: plan(GraphPlanPhase::Profiling, vec![profiling_issue]),
                    warmup: Some(plan(GraphPlanPhase::Warmup, vec![warmup_issue])),
                },
                GraphProgramInspection {
                    trace_id: "trace-2".to_owned(),
                    driver: "static_graph".to_owned(),
                    arrival_offset_ns: None,
                    has_environment: false,
                    has_replay: false,
                    profiling: plan(GraphPlanPhase::Profiling, vec![second_profiling_issue]),
                    warmup: Some(plan(GraphPlanPhase::Warmup, vec![second_warmup_issue])),
                },
            ],
        };

        let report = GraphValidateReport::from_inspection("/tmp/source".to_owned(), inspection);

        assert_eq!(
            report
                .issues
                .iter()
                .map(|issue| issue.code.as_str())
                .collect::<Vec<_>>(),
            [
                "bundle-warning",
                "profiling-error",
                "warmup-warning",
                "second-profiling-warning",
                "second-warmup-error",
            ]
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
        assert!(matches!(
            report.issues[3].severity,
            GraphIssueSeverityReport::Warning
        ));
        assert!(matches!(
            report.issues[3].phase,
            Some(GraphPlanPhaseReport::Profiling)
        ));
        assert!(matches!(
            report.issues[4].severity,
            GraphIssueSeverityReport::Error
        ));
        assert!(matches!(
            report.issues[4].phase,
            Some(GraphPlanPhaseReport::Warmup)
        ));
        assert_eq!(report.summary.errors, 2);
        assert_eq!(report.summary.warnings, 3);
        let json = serde_json::to_value(report).expect("serialize validation report");
        assert!(json["issues"][0]["trace_id"].is_null());
        assert!(json["issues"][0]["phase"].is_null());
        assert!(json["issues"][0]["location"].is_null());
    }

    #[test]
    fn recorded_replay_inspection_never_serializes_retained_secret_payloads() {
        let secret_payload = "secret-prompt-payload";
        let secret_tool = "secret-tool-command";
        let secret_environment = "secret-environment-value";
        let secret_driver = "secret-driver-value";
        let mut pool = SegmentPool::new();
        let prompt = pool
            .intern(
                None,
                Payload::Raw {
                    wire: secret_payload.into(),
                },
            )
            .expect("prompt segment");
        let tools = pool
            .intern(
                None,
                Payload::Raw {
                    wire: secret_tool.into(),
                },
            )
            .expect("tool segment");
        let mut graph = GraphRecord::default();
        graph.nodes.insert(
            "llm".into(),
            ExecutableGraphNode::Llm(LlmNode {
                output: "reply".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(7),
                items: vec![PromptItem::Seg { seg: prompt }],
                request: Some(LlmRequestSpec {
                    tools: Some(tools),
                    model: Some("recorded-model".into()),
                    additional_body: None,
                }),
                metadata: BTreeMap::new(),
            }),
        );
        graph.nodes.insert(
            "tool".into(),
            ExecutableGraphNode::Tool(ToolNode {
                output: "tool_out".into(),
                commands: vec![secret_tool.into()],
                timeout_ns: None,
            }),
        );
        graph.edges = vec![
            StaticEdge {
                source: "START".into(),
                target: "llm".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: "llm".into(),
                target: "tool".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: Some(42.0),
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: Some(9.0),
            },
            StaticEdge {
                source: "tool".into(),
                target: "END".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
        ];
        let program = GraphTraceProgram {
            profiling: GraphTracePlan {
                graph,
                trace: TraceRecord {
                    id: "recorded-trace".into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: Some(11),
            },
            warmup: None,
            environment: Some(TraceEnvironmentSpec {
                kind: "recorded".into(),
                data: BTreeMap::from([("secret".into(), secret_environment.into())]),
            }),
            replay: Some(ReplayTraceMetadata {
                manifest_ordinal: 0,
                identity: ReplayTaskIdentity {
                    adapter: "recorded".into(),
                    family: "fixture".into(),
                    task_id: "secret-task".into(),
                    primary_role: None,
                },
                source_digest: "secret-source-digest".into(),
                normalization_target_digest: None,
                target_output_tokens: vec![7],
                expected_llm_node_count: 1,
                expected_tool_node_count: 1,
                request_profile_identity: "secret-profile".into(),
                comparability_annotations: BTreeMap::new(),
            }),
            driver: TraceDriverSpec::with_data(
                "recorded_replay".into(),
                BTreeMap::from([("secret".into(), secret_driver.into())]),
            ),
        };
        let warmup = GraphTracePlan {
            graph: program.profiling.graph.clone(),
            trace: TraceRecord {
                id: "warmup-trace".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        };
        let second = GraphTraceProgram {
            profiling: GraphTracePlan {
                graph: program.profiling.graph.clone(),
                trace: TraceRecord {
                    id: "second-trace".into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: Some(12),
            },
            warmup: Some(warmup),
            environment: None,
            replay: None,
            driver: TraceDriverSpec::static_graph(),
        };
        let bundle = GraphInputBundle {
            programs: vec![program, second],
            segments: Arc::new(pool.freeze()),
            metadata: GraphInputMetadata {
                format: "agent_recording".into(),
                root_count: 2,
                node_count: 2,
                warning_facts: Vec::new(),
            },
        };
        let report = GraphExplainReport::from_inspection(
            "/tmp/recording.json".into(),
            inspect_bundle(&bundle, GraphInspectionOptions::default()),
        );
        assert_eq!(report.programs[0].driver, "recorded_replay");
        assert_eq!(report.programs[0].profiling.readiness_waves, None);
        assert_eq!(
            report.programs[0]
                .profiling
                .readiness_unavailable
                .as_ref()
                .map(|value| value.code.as_str()),
            Some("non-static-driver")
        );
        assert_eq!(report.programs[0].profiling.summary.tool_node_count, 1);
        assert_eq!(
            report
                .programs
                .iter()
                .map(|program| program.trace_id.as_str())
                .collect::<Vec<_>>(),
            ["recorded-trace", "second-trace"]
        );
        assert!(report.programs[1].warmup.is_some());
        let timed_edge = report.programs[0]
            .profiling
            .topology
            .edges
            .iter()
            .find(|edge| edge.source == "llm" && edge.target == "tool")
            .expect("timed edge");
        assert!(matches!(
            timed_edge.anchor,
            super::GraphEdgeAnchorReport::FirstToken
        ));
        assert_eq!(timed_edge.delay_us, Some(9.0));
        assert_eq!(timed_edge.min_start_delay_us, Some(42.0));
        let json = serde_json::to_string(&report).expect("serialize report");
        for forbidden in [
            secret_payload,
            secret_tool,
            secret_environment,
            secret_driver,
            "source_digest",
            "request_profile_identity",
            "commands",
            "items",
            "data",
            "table",
        ] {
            assert!(!json.contains(forbidden), "public JSON leaked {forbidden}");
        }
    }
}
