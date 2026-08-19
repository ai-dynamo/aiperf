// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared graph-inspection compatibility vocabulary.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::graph::driver::TraceDriverSpec;
use crate::graph::input::GraphInputBundle;
use crate::graph::model::{
    ChannelType, Count, END_NODE_ID, ExecutableGraphNode, GraphRecord, GraphTracePlan, PromptItem,
    ReducerName, START_NODE_ID, StaticEdge,
};
use crate::graph::scheduler::{AnchorFanInKind, Scheduler};

/// Validates graph structure and returns detailed inspection findings.
pub use crate::graph::validate::validate_detailed;

/// Severity assigned to an inspection finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphInspectionSeverity {
    /// The graph cannot be executed as authored.
    Error,
    /// The graph can be executed, but has a notable condition.
    Warning,
}

/// Execution-plan phase to which an inspection finding is scoped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphPlanPhase {
    /// The profiling phase.
    Profiling,
    /// The warmup phase.
    Warmup,
}

/// A machine-readable graph-inspection finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphInspectionIssue {
    /// Stable identifier for the kind of finding.
    pub code: String,
    /// Severity of the finding.
    pub severity: GraphInspectionSeverity,
    /// Trace scope, when the finding applies to one trace.
    pub trace_id: Option<String>,
    /// Plan-phase scope, when the finding applies to one phase.
    pub phase: Option<GraphPlanPhase>,
    /// Source location within the graph payload, when available.
    pub location: Option<String>,
    /// Human-readable compatibility message.
    pub message: String,
    /// Deterministic, machine-readable details for the finding.
    pub context: BTreeMap<String, String>,
}

/// Options that select additional inspection rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphInspectionOptions {
    /// Require an authored profiling-plan arrival offset for every program.
    pub requires_arrival_offsets: bool,
}

/// Filesystem-free inspection output for one retained graph-input bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphBundleInspection {
    /// Adapter-authored graph-input format.
    pub format: String,
    /// Adapter-reported root trace count.
    pub root_count: usize,
    /// Adapter-reported aggregate LLM node count.
    pub node_count: usize,
    /// Number of retained immutable segments.
    pub segment_count: usize,
    /// Bundle-level issues and adapter warnings.
    pub issues: Vec<GraphInspectionIssue>,
    /// Per-program inspection in authored order.
    pub programs: Vec<GraphProgramInspection>,
}

/// Inspection output for one retained trace program.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphProgramInspection {
    /// Trace identity from the profiling plan.
    pub trace_id: String,
    /// Registered driver identifier.
    pub driver: String,
    /// Profiling-plan arrival offset.
    pub arrival_offset_ns: Option<i64>,
    /// Whether the program retains an environment recipe.
    pub has_environment: bool,
    /// Whether the program retains recorded-replay metadata.
    pub has_replay: bool,
    /// Profiling-plan inspection.
    pub profiling: GraphPlanInspection,
    /// Warmup-plan inspection, when retained.
    pub warmup: Option<GraphPlanInspection>,
}

/// Inspection output for one profiling or warmup graph plan.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphPlanInspection {
    /// The plan phase.
    pub phase: GraphPlanPhase,
    /// Aggregate runtime-topology facts.
    pub summary: GraphPlanSummary,
    /// Deterministically normalized topology.
    pub topology: GraphTopologyInspection,
    /// Plan-scoped structural and scheduler issues.
    pub issues: Vec<GraphInspectionIssue>,
    /// Illustrative readiness waves, when statically available.
    pub readiness: ReadinessInspection,
}

/// Aggregate executable topology counts for one graph plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphPlanSummary {
    /// Number of executable nodes, including tools.
    pub node_count: usize,
    /// Number of LLM nodes.
    pub llm_node_count: usize,
    /// Number of tool nodes.
    pub tool_node_count: usize,
    /// Number of retained static edges.
    pub edge_count: usize,
    /// Number of declared state channels.
    pub channel_count: usize,
}

/// Normalized runtime topology safe for presentation.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphTopologyInspection {
    /// Nodes in START traversal then lexical fallback order.
    pub nodes: Vec<GraphNodeInspection>,
    /// Channels in lexical name order.
    pub channels: Vec<GraphChannelInspection>,
    /// Edges in normalized source/target order.
    pub edges: Vec<GraphEdgeInspection>,
}

/// Executable node kind safe for presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphNodeKind {
    /// A measured LLM invocation.
    Llm,
    /// A tool observation producer.
    Tool,
}

/// Edge timing anchor selected by runtime semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphEdgeAnchor {
    /// The predecessor completion gates the successor.
    Completion,
    /// The predecessor dispatch gates the successor.
    Dispatch,
    /// The predecessor first token gates the successor.
    FirstToken,
}

/// One declared node input gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphNodeInputInspection {
    /// Required channel name.
    pub channel: String,
    /// Authored count as decimal or `all`.
    pub count: String,
}

/// Presentation-safe executable node facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphNodeInspection {
    /// Node identity.
    pub id: String,
    /// Runtime node kind.
    pub kind: GraphNodeKind,
    /// Declared output channel.
    pub output: String,
    /// Declared firing-gate inputs.
    pub inputs: Vec<GraphNodeInputInspection>,
    /// Dynamic prompt splice-channel names for LLM nodes.
    pub prompt_splice_channels: Vec<String>,
    /// LLM streaming setting, absent for tools.
    pub streaming: Option<bool>,
    /// LLM request model override, absent for tools.
    pub model_override: Option<String>,
    /// LLM generation cap, absent for tools.
    pub max_tokens: Option<usize>,
}

/// One declared state channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphChannelInspection {
    /// Channel name.
    pub name: String,
    /// Runtime channel value type.
    pub channel_type: ChannelType,
    /// Runtime channel reducer.
    pub reducer: ReducerName,
}

/// One normalized static dependency edge.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphEdgeInspection {
    /// Source node identity, START, or END.
    pub source: String,
    /// Target node identity, START, or END.
    pub target: String,
    /// Runtime timing anchor.
    pub anchor: GraphEdgeAnchor,
    /// Selected anchor delay, retaining zero when authored.
    pub delay_us: Option<f64>,
    /// Minimum start delay, when authored.
    pub min_start_delay_us: Option<f64>,
}

/// Static readiness analysis output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadinessInspection {
    /// Every node appeared in a finite illustrative readiness wave.
    Available {
        /// Waves in deterministic admission order.
        waves: Vec<ReadinessWave>,
    },
    /// Static readiness cannot be derived safely.
    Unavailable {
        /// Stable reason identifier.
        code: String,
        /// Bounded human-readable explanation.
        message: String,
    },
}

/// One deterministic readiness wave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadinessWave {
    /// Zero-based wave index.
    pub wave: usize,
    /// Nodes admitted in this wave.
    pub node_ids: Vec<String>,
    /// Illustrative trigger summary.
    pub trigger: String,
}

/// Inspect a retained Graph-IR bundle without reading source content or executing it.
pub fn inspect_bundle(
    bundle: &GraphInputBundle,
    options: GraphInspectionOptions,
) -> GraphBundleInspection {
    let issues = bundle_issues(bundle, options);
    let programs = bundle.programs.iter().map(inspect_program).collect();
    GraphBundleInspection {
        format: bundle.metadata.format.clone(),
        root_count: bundle.metadata.root_count,
        node_count: bundle.metadata.node_count,
        segment_count: bundle.segments.len(),
        issues,
        programs,
    }
}

fn bundle_issues(
    bundle: &GraphInputBundle,
    options: GraphInspectionOptions,
) -> Vec<GraphInspectionIssue> {
    let mut issues = Vec::new();
    if bundle.programs.is_empty() {
        issues.push(error_issue(
            "bundle-empty",
            None,
            None,
            None,
            "graph input contains no programs",
            BTreeMap::new(),
        ));
    }
    if bundle.metadata.root_count != bundle.programs.len() {
        issues.push(error_issue(
            "metadata-root-count-mismatch",
            None,
            None,
            None,
            "adapter root count does not match retained program count",
            BTreeMap::from([
                (
                    "metadata_root_count".into(),
                    bundle.metadata.root_count.to_string(),
                ),
                ("program_count".into(), bundle.programs.len().to_string()),
            ]),
        ));
    }
    let profiling_node_count: usize = bundle
        .programs
        .iter()
        .map(|program| program.profiling.graph.llm_node_count())
        .sum();
    if bundle.metadata.node_count != profiling_node_count {
        issues.push(error_issue(
            "metadata-node-count-mismatch",
            None,
            None,
            None,
            "adapter node count does not match retained profiling LLM node count",
            BTreeMap::from([
                (
                    "metadata_node_count".into(),
                    bundle.metadata.node_count.to_string(),
                ),
                (
                    "profiling_llm_node_count".into(),
                    profiling_node_count.to_string(),
                ),
            ]),
        ));
    }
    for warning in &bundle.metadata.warning_facts {
        let trace_id = warning.context.get("trace_id").cloned();
        issues.push(GraphInspectionIssue {
            code: format!("adapter-warning.{}", warning.code),
            severity: GraphInspectionSeverity::Warning,
            trace_id,
            phase: None,
            location: None,
            message: format!("adapter warning: {}", warning.code),
            context: warning.context.clone(),
        });
    }
    let mut seen = BTreeSet::new();
    for program in &bundle.programs {
        let trace_id = &program.profiling.trace.id;
        if trace_id.is_empty() {
            issues.push(error_issue(
                "trace-id-empty",
                Some(trace_id.clone()),
                Some(GraphPlanPhase::Profiling),
                None,
                "profiling trace id must not be empty",
                BTreeMap::new(),
            ));
        }
        if !seen.insert(trace_id.clone()) {
            issues.push(error_issue(
                "trace-id-duplicate",
                Some(trace_id.clone()),
                Some(GraphPlanPhase::Profiling),
                None,
                "profiling trace id is duplicated",
                BTreeMap::from([("trace_id".into(), trace_id.clone())]),
            ));
        }
        if options.requires_arrival_offsets && program.profiling.arrival_offset_ns.is_none() {
            issues.push(error_issue(
                "arrival-offset-missing",
                Some(trace_id.clone()),
                Some(GraphPlanPhase::Profiling),
                None,
                "profiling plan is missing an arrival offset",
                BTreeMap::new(),
            ));
        }
        if let Some(warmup) = &program.warmup {
            if warmup.trace.id.is_empty() {
                issues.push(error_issue(
                    "trace-id-empty",
                    Some(trace_id.clone()),
                    Some(GraphPlanPhase::Warmup),
                    None,
                    "warmup trace id must not be empty",
                    BTreeMap::new(),
                ));
            }
        }
    }
    issues
}

fn inspect_program(program: &crate::graph::model::GraphTraceProgram) -> GraphProgramInspection {
    let trace_id = program.profiling.trace.id.clone();
    let profiling = inspect_plan(
        &program.profiling,
        &trace_id,
        GraphPlanPhase::Profiling,
        &program.driver,
    );
    let warmup = program
        .warmup
        .as_ref()
        .map(|plan| inspect_plan(plan, &trace_id, GraphPlanPhase::Warmup, &program.driver));
    GraphProgramInspection {
        trace_id,
        driver: program.driver.kind.clone(),
        arrival_offset_ns: program.profiling.arrival_offset_ns,
        has_environment: program.environment.is_some(),
        has_replay: program.replay.is_some(),
        profiling,
        warmup,
    }
}

fn inspect_plan(
    plan: &GraphTracePlan,
    trace_id: &str,
    phase: GraphPlanPhase,
    driver: &TraceDriverSpec,
) -> GraphPlanInspection {
    let mut issues = validate_detailed(&plan.graph)
        .into_iter()
        .map(|mut issue| {
            issue.trace_id = Some(trace_id.to_string());
            issue.phase = Some(phase);
            issue
        })
        .collect::<Vec<_>>();
    let scheduler = Scheduler::new(&plan.graph);
    if let Err(error) = &scheduler {
        let code = match error.kind() {
            AnchorFanInKind::Mixed => "mixed-anchor-fan-in",
            AnchorFanInKind::MultipleStartAnchored => "multi-start-anchor-fan-in",
        };
        issues.push(error_issue(
            code,
            Some(trace_id.to_string()),
            Some(phase),
            Some(format!("graph.nodes.{}", error.target())),
            error.to_string(),
            BTreeMap::from([("target".into(), error.target().to_string())]),
        ));
    }
    let topology = inspect_topology(&plan.graph);
    let summary = GraphPlanSummary {
        node_count: plan.graph.total_node_count(),
        llm_node_count: plan.graph.llm_node_count(),
        tool_node_count: plan.graph.total_node_count() - plan.graph.llm_node_count(),
        edge_count: plan.graph.edges.len(),
        channel_count: plan.graph.state.len(),
    };
    let readiness = inspect_readiness(&plan.graph, driver, &issues, scheduler.ok());
    GraphPlanInspection {
        phase,
        summary,
        topology,
        issues,
        readiness,
    }
}

fn error_issue(
    code: &str,
    trace_id: Option<String>,
    phase: Option<GraphPlanPhase>,
    location: Option<String>,
    message: impl Into<String>,
    context: BTreeMap<String, String>,
) -> GraphInspectionIssue {
    GraphInspectionIssue {
        code: code.into(),
        severity: GraphInspectionSeverity::Error,
        trace_id,
        phase,
        location,
        message: message.into(),
        context,
    }
}

fn inspect_topology(graph: &GraphRecord) -> GraphTopologyInspection {
    let node_order = normalized_node_order(graph);
    let ranks = node_order
        .iter()
        .enumerate()
        .map(|(rank, node)| (node.clone(), rank.saturating_add(1)))
        .collect::<BTreeMap<_, _>>();
    let nodes = node_order
        .into_iter()
        .filter_map(|id| graph.nodes.get(&id).map(|node| inspect_node(id, node)))
        .collect();
    let channels = graph
        .state
        .iter()
        .map(|(name, spec)| GraphChannelInspection {
            name: name.clone(),
            channel_type: spec.channel_type,
            reducer: spec.reducer,
        })
        .collect();
    let mut edges = graph
        .edges
        .iter()
        .enumerate()
        .map(|(ordinal, edge)| (edge_sort_key(edge, ordinal, &ranks), inspect_edge(edge)))
        .collect::<Vec<_>>();
    edges.sort_by(|(left, _), (right, _)| left.cmp(right));
    GraphTopologyInspection {
        nodes,
        channels,
        edges: edges.into_iter().map(|(_, edge)| edge).collect(),
    }
}

fn normalized_node_order(graph: &GraphRecord) -> Vec<String> {
    let mut successors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &graph.edges {
        successors
            .entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }
    let mut queue = VecDeque::new();
    if let Some(entries) = successors.get(START_NODE_ID) {
        queue.extend(entries.iter().copied());
    }
    let mut seen = BTreeSet::new();
    let mut nodes = Vec::new();
    while let Some(node) = queue.pop_front() {
        if !graph.nodes.contains_key(node) || !seen.insert(node) {
            continue;
        }
        nodes.push(node.to_string());
        if let Some(next) = successors.get(node) {
            queue.extend(next.iter().copied());
        }
    }
    for node in graph.nodes.keys() {
        if seen.insert(node.as_str()) {
            nodes.push(node.clone());
        }
    }
    nodes
}

fn inspect_node(id: String, node: &ExecutableGraphNode) -> GraphNodeInspection {
    match node {
        ExecutableGraphNode::Llm(llm) => GraphNodeInspection {
            id,
            kind: GraphNodeKind::Llm,
            output: llm.output.clone(),
            inputs: llm
                .inputs
                .iter()
                .map(|input| GraphNodeInputInspection {
                    channel: input.channel.clone(),
                    count: count_text(&input.count),
                })
                .collect(),
            prompt_splice_channels: llm
                .items
                .iter()
                .filter_map(|item| match item {
                    PromptItem::Splice { splice } => Some(splice.clone()),
                    _ => None,
                })
                .collect(),
            streaming: Some(llm.streaming),
            model_override: llm
                .request
                .as_ref()
                .and_then(|request| request.model.clone()),
            max_tokens: llm.max_tokens,
        },
        ExecutableGraphNode::Tool(tool) => GraphNodeInspection {
            id,
            kind: GraphNodeKind::Tool,
            output: tool.output.clone(),
            inputs: Vec::new(),
            prompt_splice_channels: Vec::new(),
            streaming: None,
            model_override: None,
            max_tokens: None,
        },
    }
}

fn count_text(count: &Count) -> String {
    match count {
        Count::N(count) => count.to_string(),
        Count::Word(count) => count.clone(),
    }
}

fn edge_sort_key(
    edge: &StaticEdge,
    ordinal: usize,
    ranks: &BTreeMap<String, usize>,
) -> (usize, String, usize, String, usize) {
    let end_rank = ranks.len().saturating_add(1);
    let unknown_rank = end_rank.saturating_add(1);
    let source_rank = if edge.source == START_NODE_ID {
        0
    } else if edge.source == END_NODE_ID {
        end_rank
    } else {
        ranks.get(&edge.source).copied().unwrap_or(unknown_rank)
    };
    let target_rank = if edge.target == START_NODE_ID {
        0
    } else if edge.target == END_NODE_ID {
        end_rank
    } else {
        ranks.get(&edge.target).copied().unwrap_or(unknown_rank)
    };
    (
        source_rank,
        edge.source.clone(),
        target_rank,
        edge.target.clone(),
        ordinal,
    )
}

fn inspect_edge(edge: &StaticEdge) -> GraphEdgeInspection {
    let (anchor, delay_us) = if let Some(delay) = edge.delay_after_predecessor_start_us {
        (GraphEdgeAnchor::Dispatch, Some(delay))
    } else if let Some(delay) = edge.delay_after_predecessor_first_token_us {
        (GraphEdgeAnchor::FirstToken, Some(delay))
    } else {
        (GraphEdgeAnchor::Completion, edge.delay_after_predecessor_us)
    };
    GraphEdgeInspection {
        source: edge.source.clone(),
        target: edge.target.clone(),
        anchor,
        delay_us,
        min_start_delay_us: edge.min_start_delay_us,
    }
}

fn inspect_readiness(
    graph: &GraphRecord,
    driver: &TraceDriverSpec,
    issues: &[GraphInspectionIssue],
    scheduler: Option<Scheduler>,
) -> ReadinessInspection {
    if !driver.is_static_graph() {
        return unavailable(
            "non-static-driver",
            "readiness waves require the built-in static_graph driver",
        );
    }
    if issues
        .iter()
        .any(|issue| issue.severity == GraphInspectionSeverity::Error)
    {
        return unavailable(
            "validation-errors",
            "readiness waves are unavailable while validation errors remain",
        );
    }
    let Some(scheduler) = scheduler else {
        return unavailable(
            "validation-errors",
            "readiness waves are unavailable while scheduler validation errors remain",
        );
    };
    match readiness_waves(graph, &scheduler) {
        Some(waves) => ReadinessInspection::Available { waves },
        None => unavailable(
            "analysis-incomplete",
            "static readiness analysis did not admit every graph node",
        ),
    }
}

fn unavailable(code: &str, message: &str) -> ReadinessInspection {
    ReadinessInspection::Unavailable {
        code: code.into(),
        message: message.into(),
    }
}

fn readiness_waves(graph: &GraphRecord, scheduler: &Scheduler) -> Option<Vec<ReadinessWave>> {
    let order = normalized_node_order(graph);
    let writers = channel_writer_counts(graph);
    let mut emitted = BTreeSet::new();
    let mut completed_channels = BTreeMap::<String, usize>::new();
    let mut waves = Vec::new();
    let mut prior_wave = Vec::<String>::new();

    for wave in 0..=graph.nodes.len() {
        let candidates = if wave == 0 {
            scheduler
                .entry_nodes()
                .filter(|node| graph.nodes.contains_key(*node))
                .filter(|node| {
                    graph.nodes.get(*node).is_some_and(|graph_node| {
                        channels_ready(graph_node, &completed_channels, &writers)
                    })
                })
                .map(str::to_string)
                .collect::<Vec<_>>()
        } else {
            order
                .iter()
                .filter(|node| !emitted.contains(node.as_str()))
                .filter(|node| {
                    let Some(graph_node) = graph.nodes.get(node.as_str()) else {
                        return false;
                    };
                    predecessors_ready(scheduler, node, &emitted)
                        && channels_ready(graph_node, &completed_channels, &writers)
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        let node_ids = deduplicate(candidates);
        if node_ids.is_empty() {
            break;
        }
        let trigger = if wave == 0 {
            "START".to_string()
        } else {
            readiness_trigger(graph, &node_ids, &prior_wave, &completed_channels, &writers)
        };
        for node_id in &node_ids {
            emitted.insert(node_id.clone());
            if let Some(node) = graph.nodes.get(node_id) {
                *completed_channels
                    .entry(node.output().to_string())
                    .or_default() += 1;
            }
        }
        prior_wave = node_ids.clone();
        waves.push(ReadinessWave {
            wave,
            node_ids,
            trigger,
        });
    }
    (emitted.len() == graph.nodes.len()).then_some(waves)
}

fn channel_writer_counts(graph: &GraphRecord) -> BTreeMap<String, usize> {
    let mut writers = BTreeMap::new();
    for node in graph.nodes.values() {
        *writers.entry(node.output().to_string()).or_default() += 1;
    }
    writers
}

fn predecessors_ready(scheduler: &Scheduler, node: &str, emitted: &BTreeSet<String>) -> bool {
    scheduler.incoming_static_edges(node).iter().all(|edge| {
        edge.source == START_NODE_ID
            || (edge.delay_after_predecessor_start_us.is_some() && emitted.contains(&edge.source))
            || (edge.delay_after_predecessor_start_us.is_none() && emitted.contains(&edge.source))
    })
}

fn channels_ready(
    node: &ExecutableGraphNode,
    completed_channels: &BTreeMap<String, usize>,
    writers: &BTreeMap<String, usize>,
) -> bool {
    node.input_requirements().iter().all(|requirement| {
        let Some(needed) = required_count(requirement, writers) else {
            return false;
        };
        completed_channels
            .get(&requirement.channel)
            .copied()
            .unwrap_or(0)
            >= needed
    })
}

fn required_count(
    requirement: &crate::graph::model::ChannelRequirement,
    writers: &BTreeMap<String, usize>,
) -> Option<usize> {
    match &requirement.count {
        Count::N(count) => usize::try_from(*count).ok(),
        Count::Word(word) if word == "all" => {
            Some(writers.get(&requirement.channel).copied().unwrap_or(0))
        }
        Count::Word(_) => None,
    }
}

fn readiness_trigger(
    graph: &GraphRecord,
    node_ids: &[String],
    prior_wave: &[String],
    completed_channels: &BTreeMap<String, usize>,
    writers: &BTreeMap<String, usize>,
) -> String {
    let current = node_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let prior = prior_wave
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let dispatch_sources = graph
        .edges
        .iter()
        .filter(|edge| {
            current.contains(edge.target.as_str())
                && prior.contains(edge.source.as_str())
                && edge.delay_after_predecessor_start_us.is_some()
        })
        .map(|edge| edge.source.as_str())
        .collect::<Vec<_>>();
    let completion_edge_unlocks = graph.edges.iter().any(|edge| {
        current.contains(edge.target.as_str())
            && prior.contains(edge.source.as_str())
            && edge.delay_after_predecessor_start_us.is_none()
    });
    let channel_unlocks = node_ids.iter().any(|node_id| {
        let Some(node) = graph.nodes.get(node_id) else {
            return false;
        };
        node.input_requirements().iter().any(|requirement| {
            let Some(needed) = required_count(requirement, writers) else {
                return false;
            };
            let completed = completed_channels
                .get(&requirement.channel)
                .copied()
                .unwrap_or(0);
            let prior_completed = prior_wave
                .iter()
                .filter_map(|prior_node| graph.nodes.get(prior_node))
                .filter(|prior_node| prior_node.output() == requirement.channel)
                .count();
            completed >= needed && completed.saturating_sub(prior_completed) < needed
        })
    });
    if completion_edge_unlocks || channel_unlocks || dispatch_sources.is_empty() {
        return format!("completed: {}", prior_wave.join(","));
    }
    if !dispatch_sources.is_empty() {
        return format!(
            "dispatched: {}",
            deduplicate_refs(dispatch_sources).join(",")
        );
    }
    format!("completed: {}", prior_wave.join(","))
}

fn deduplicate(nodes: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    nodes
        .into_iter()
        .filter(|node| seen.insert(node.clone()))
        .collect()
}

fn deduplicate_refs(nodes: Vec<&str>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    nodes
        .into_iter()
        .filter(|node| seen.insert((*node).to_string()))
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::*;
    use crate::graph::driver::TraceDriverSpec;
    use crate::graph::input::{GraphInputBundle, GraphInputMetadata, GraphInputWarning};
    use crate::graph::model::{GraphRecord, GraphTracePlan, GraphTraceProgram, TraceRecord};
    use crate::graph::segment::SegmentPool;

    fn graph(json: &str) -> GraphRecord {
        serde_json::from_str(json).unwrap()
    }

    fn program(id: &str, graph: GraphRecord) -> GraphTraceProgram {
        GraphTraceProgram {
            profiling: GraphTracePlan {
                graph,
                trace: TraceRecord {
                    id: id.into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: Some(0),
            },
            warmup: None,
            environment: None,
            replay: None,
            driver: TraceDriverSpec::static_graph(),
        }
    }

    fn bundle(programs: Vec<GraphTraceProgram>) -> GraphInputBundle {
        GraphInputBundle {
            metadata: GraphInputMetadata {
                format: "test".into(),
                root_count: programs.len(),
                node_count: programs
                    .iter()
                    .map(|program| program.profiling.graph.llm_node_count())
                    .sum(),
                warning_facts: Vec::new(),
            },
            programs,
            segments: Arc::new(SegmentPool::new().freeze()),
        }
    }

    #[test]
    fn bundle_inspection_api_is_available() {
        let _ = GraphInspectionOptions::default();
    }

    #[test]
    fn bundle_invariants_and_warnings_are_deterministic() {
        let valid = graph(
            r#"{"state":{"out":{}},"nodes":{"n":{"output":"out"}},"edges":[{"source":"START","target":"n"}]}"#,
        );
        let mut first = program("duplicate", valid.clone());
        first.warmup = Some(GraphTracePlan {
            graph: valid.clone(),
            trace: TraceRecord {
                id: String::new(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        });
        let mut second = program("duplicate", valid);
        second.profiling.arrival_offset_ns = None;
        let mut input = bundle(vec![first, second]);
        input.metadata.root_count = 1;
        input.metadata.node_count = 0;
        input.metadata.warning_facts = vec![
            GraphInputWarning::new(
                "missing-model",
                BTreeMap::from([("trace_id".into(), "duplicate".into())]),
            ),
            GraphInputWarning::new("generic", BTreeMap::new()),
        ];
        let inspection = inspect_bundle(
            &input,
            GraphInspectionOptions {
                requires_arrival_offsets: true,
            },
        );
        assert_eq!(
            inspection
                .issues
                .iter()
                .map(|issue| issue.code.as_str())
                .collect::<Vec<_>>(),
            vec![
                "metadata-root-count-mismatch",
                "metadata-node-count-mismatch",
                "adapter-warning.missing-model",
                "adapter-warning.generic",
                "trace-id-empty",
                "trace-id-duplicate",
                "arrival-offset-missing"
            ],
        );
        assert_eq!(
            inspection.issues[2].severity,
            GraphInspectionSeverity::Warning
        );
        assert_eq!(inspection.issues[2].trace_id.as_deref(), Some("duplicate"));
        assert_eq!(inspection.issues[3].trace_id, None);
        assert_eq!(inspection.issues[4].phase, Some(GraphPlanPhase::Warmup));
        assert_eq!(inspection.issues[6].phase, Some(GraphPlanPhase::Profiling));
        assert_eq!(
            inspect_bundle(&bundle(Vec::new()), GraphInspectionOptions::default()).issues[0].code,
            "bundle-empty"
        );
    }

    #[test]
    fn topology_and_waves_are_normalized_without_tool_secrets() {
        let graph = graph(
            r#"{
            "state":{"out":{},"joined":{}},
            "nodes":{"a":{"output":"out","streaming":false,"max_tokens":4,"request":{"model":"m"}},"b":{"kind":"tool","output":"out","commands":["secret"]},"join":{"output":"joined","inputs":[{"channel":"out","count":2}]}},
            "edges":[{"source":"START","target":"b"},{"source":"START","target":"a"},{"source":"a","target":"join","delay_after_predecessor_first_token_us":3.0},{"source":"b","target":"join"}]}"#,
        );
        let input = bundle(vec![program("t", graph)]);
        let plan = &inspect_bundle(&input, GraphInspectionOptions::default()).programs[0].profiling;
        assert_eq!(
            plan.topology
                .nodes
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "a", "join"]
        );
        let tool = &plan.topology.nodes[0];
        assert_eq!(tool.kind, GraphNodeKind::Tool);
        assert_eq!(tool.streaming, None);
        assert_eq!(tool.model_override, None);
        let llm = &plan.topology.nodes[1];
        assert_eq!(llm.streaming, Some(false));
        assert_eq!(llm.model_override.as_deref(), Some("m"));
        assert_eq!(llm.max_tokens, Some(4));
        assert!(
            plan.topology.edges.iter().any(
                |edge| edge.anchor == GraphEdgeAnchor::FirstToken && edge.delay_us == Some(3.0)
            )
        );
        let ReadinessInspection::Available { waves } = &plan.readiness else {
            panic!("valid graph must have waves")
        };
        assert_eq!(waves[0].node_ids, vec!["b", "a"]);
        assert_eq!(waves[0].trigger, "START");
        assert_eq!(waves[1].node_ids, vec!["join"]);
        assert_eq!(waves[1].trigger, "completed: b,a");
    }

    #[test]
    fn readiness_distinguishes_dispatch_non_static_and_invalid_graphs() {
        let dispatch = graph(
            r#"{"state":{"a":{},"b":{}},"nodes":{"source":{"output":"a"},"child":{"output":"b"}},"edges":[{"source":"START","target":"source"},{"source":"source","target":"child","delay_after_predecessor_start_us":0.0}]}"#,
        );
        let input = bundle(vec![program("t", dispatch)]);
        let plan = &inspect_bundle(&input, GraphInspectionOptions::default()).programs[0].profiling;
        let ReadinessInspection::Available { waves } = &plan.readiness else {
            panic!("static graph must have waves")
        };
        assert_eq!(waves[1].trigger, "dispatched: source");

        let mut non_static = program("t", graph(r#"{"nodes":{}}"#));
        non_static.driver = TraceDriverSpec::recorded_replay();
        let unavailable =
            &inspect_bundle(&bundle(vec![non_static]), GraphInspectionOptions::default()).programs
                [0]
            .profiling
            .readiness;
        assert!(
            matches!(unavailable, ReadinessInspection::Unavailable { code, .. } if code == "non-static-driver")
        );

        let invalid = graph(
            r#"{"nodes":{"n":{"output":"missing"}},"edges":[{"source":"START","target":"n"}]}"#,
        );
        let unavailable = &inspect_bundle(
            &bundle(vec![program("t", invalid)]),
            GraphInspectionOptions::default(),
        )
        .programs[0]
            .profiling
            .readiness;
        assert!(
            matches!(unavailable, ReadinessInspection::Unavailable { code, .. } if code == "validation-errors")
        );
    }

    #[test]
    fn readiness_waits_for_entry_channel_gates_and_reports_completion_unlocks() {
        let graph = graph(
            r#"{
                "state":{"out":{},"done":{}},
                "nodes":{
                    "source":{"output":"out"},
                    "child":{"output":"done","inputs":[{"channel":"out","count":1}]}},
                "edges":[
                    {"source":"START","target":"source"},
                    {"source":"START","target":"child"}
                ]
            }"#,
        );
        let readiness = &inspect_bundle(
            &bundle(vec![program("t", graph)]),
            GraphInspectionOptions::default(),
        )
        .programs[0]
            .profiling
            .readiness;
        let ReadinessInspection::Available { waves } = readiness else {
            panic!("valid graph must have waves")
        };
        assert_eq!(waves[0].node_ids, vec!["source"]);
        assert_eq!(waves[1].node_ids, vec!["child"]);
        assert_eq!(waves[1].trigger, "completed: source");
    }

    #[test]
    fn dispatch_edge_with_a_completion_gate_reports_completion() {
        let graph = graph(
            r#"{
                "state":{"out":{},"done":{}},
                "nodes":{
                    "source":{"output":"out"},
                    "child":{"output":"done","inputs":[{"channel":"out","count":1}]}},
                "edges":[
                    {"source":"START","target":"source"},
                    {"source":"source","target":"child","delay_after_predecessor_start_us":0.0}
                ]
            }"#,
        );
        let readiness = &inspect_bundle(
            &bundle(vec![program("t", graph)]),
            GraphInspectionOptions::default(),
        )
        .programs[0]
            .profiling
            .readiness;
        let ReadinessInspection::Available { waves } = readiness else {
            panic!("valid graph must have waves")
        };
        assert_eq!(waves[1].trigger, "completed: source");
    }

    #[test]
    fn warmup_findings_keep_the_program_trace_scope() {
        let mut trace = program(
            "profile-id",
            graph(r#"{"state":{"out":{}},"nodes":{"n":{"output":"out"}}}"#),
        );
        trace.warmup = Some(GraphTracePlan {
            graph: graph(r#"{"nodes":{"bad":{"output":"missing"}}}"#),
            trace: TraceRecord {
                id: "different-warmup-id".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        });
        let inspection = inspect_bundle(&bundle(vec![trace]), GraphInspectionOptions::default());
        let warmup = inspection.programs[0].warmup.as_ref().unwrap();
        assert!(
            warmup
                .issues
                .iter()
                .all(|issue| issue.trace_id.as_deref() == Some("profile-id"))
        );
        assert!(
            warmup
                .issues
                .iter()
                .all(|issue| issue.phase == Some(GraphPlanPhase::Warmup))
        );
    }

    #[test]
    fn edge_ranks_reserve_start_and_end_and_keep_exact_order() {
        let graph = graph(
            r#"{
                "nodes":{"a":{"output":"a"},"b":{"output":"b"}},
                "edges":[
                    {"source":"a","target":"END"},
                    {"source":"START","target":"a"},
                    {"source":"START","target":"b"},
                    {"source":"b","target":"END"}
                ]
            }"#,
        );
        let topology = &inspect_bundle(
            &bundle(vec![program("t", graph)]),
            GraphInspectionOptions::default(),
        )
        .programs[0]
            .profiling
            .topology;
        assert_eq!(
            topology
                .edges
                .iter()
                .map(|edge| (edge.source.as_str(), edge.target.as_str()))
                .collect::<Vec<_>>(),
            vec![("START", "a"), ("START", "b"), ("a", "END"), ("b", "END")],
        );
    }

    #[test]
    fn topology_uses_start_order_then_unreachable_fallback_and_exact_edges() {
        let graph = graph(
            r#"{
                "state":{"alpha":{},"zeta":{}},
                "nodes":{"a":{"output":"alpha"},"b":{"output":"alpha"},"m":{"output":"zeta"},"z":{"output":"zeta"}},
                "edges":[
                    {"source":"m","target":"END","delay_after_predecessor_us":0.0},
                    {"source":"z","target":"a","delay_after_predecessor_start_us":0.0},
                    {"source":"START","target":"z"},
                    {"source":"a","target":"m","delay_after_predecessor_first_token_us":3.0}
                ]
            }"#,
        );
        let topology = &inspect_bundle(
            &bundle(vec![program("t", graph)]),
            GraphInspectionOptions::default(),
        )
        .programs[0]
            .profiling
            .topology;
        assert_eq!(
            topology
                .nodes
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            vec!["z", "a", "m", "b"],
        );
        assert_eq!(
            topology
                .channels
                .iter()
                .map(|channel| channel.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"],
        );
        assert_eq!(
            topology
                .edges
                .iter()
                .map(|edge| (
                    edge.source.as_str(),
                    edge.target.as_str(),
                    edge.anchor,
                    edge.delay_us
                ))
                .collect::<Vec<_>>(),
            vec![
                ("START", "z", GraphEdgeAnchor::Completion, None),
                ("z", "a", GraphEdgeAnchor::Dispatch, Some(0.0)),
                ("a", "m", GraphEdgeAnchor::FirstToken, Some(3.0)),
                ("m", "END", GraphEdgeAnchor::Completion, Some(0.0)),
            ],
        );
    }

    #[test]
    fn profiling_trace_id_and_warning_contexts_are_preserved_exactly() {
        let mut trace = program("", graph(r#"{"nodes":{}}"#));
        trace.profiling.arrival_offset_ns = None;
        let mut input = bundle(vec![trace]);
        let warning_context = BTreeMap::from([
            ("trace_id".into(), "source-trace".into()),
            ("detail".into(), "kept".into()),
        ]);
        input.metadata.warning_facts = vec![
            GraphInputWarning::new("scoped", warning_context.clone()),
            GraphInputWarning::new(
                "unscoped",
                BTreeMap::from([("detail".into(), "also-kept".into())]),
            ),
        ];
        let inspection = inspect_bundle(
            &input,
            GraphInspectionOptions {
                requires_arrival_offsets: true,
            },
        );
        let empty_id = inspection
            .issues
            .iter()
            .find(|issue| issue.code == "trace-id-empty")
            .unwrap();
        assert_eq!(empty_id.trace_id.as_deref(), Some(""));
        assert_eq!(empty_id.phase, Some(GraphPlanPhase::Profiling));
        assert_eq!(inspection.issues[0].context, warning_context);
        assert_eq!(
            inspection.issues[0].trace_id.as_deref(),
            Some("source-trace")
        );
        assert_eq!(inspection.issues[1].trace_id, None);
        assert_eq!(
            inspection.issues[1]
                .context
                .get("detail")
                .map(String::as_str),
            Some("also-kept")
        );
    }

    #[test]
    fn readiness_rejects_every_non_static_driver_shape_and_reports_incomplete_analysis() {
        use crate::graph::driver::AgentContinuationSpec;

        let empty_graph = graph(r#"{"nodes":{}}"#);
        let drivers = vec![
            TraceDriverSpec::with_data("other".into(), BTreeMap::new()),
            TraceDriverSpec::with_data(
                "static_graph".into(),
                BTreeMap::from([("extra".into(), serde_json::json!(true))]),
            ),
            TraceDriverSpec::static_graph().with_continuation(AgentContinuationSpec::Load {
                trajectory: "resume".into(),
            }),
            TraceDriverSpec::static_graph().with_delegation(),
        ];
        for driver in drivers {
            let mut trace = program("t", empty_graph.clone());
            trace.driver = driver;
            assert!(matches!(
                inspect_bundle(&bundle(vec![trace]), GraphInspectionOptions::default()).programs[0].profiling.readiness,
                ReadinessInspection::Unavailable { ref code, .. } if code == "non-static-driver"
            ));
        }

        let incomplete = graph(
            r#"{
                "state":{"out":{},"done":{}},
                "nodes":{"source":{"output":"out"},"blocked":{"output":"done","inputs":[{"channel":"out","count":"not-all"}]}},
                "edges":[{"source":"START","target":"source"},{"source":"START","target":"blocked"}]
            }"#,
        );
        assert!(matches!(
            inspect_bundle(&bundle(vec![program("t", incomplete)]), GraphInspectionOptions::default()).programs[0].profiling.readiness,
            ReadinessInspection::Unavailable { ref code, .. } if code == "analysis-incomplete"
        ));
    }
}
