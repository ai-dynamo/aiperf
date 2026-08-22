// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict NativeGraph source lowering into the shared Graph-IR program.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::num::NonZeroU32;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::eval::ArtifactDigest;
use crate::eval::semantic::{
    GraphLowererCapabilities, GraphLowererFactory, GraphLoweringError, GraphLoweringRequest,
};
use crate::graph::driver::TraceDriverSpec;
use crate::graph::model::{
    ChannelRequirement, ChannelSpec, ChannelType, ExecutableGraphNode, GraphRecord, GraphTracePlan,
    GraphTraceProgram, LlmNode, ReducerName, START_NODE_ID, StaticEdge, ToolNode, TraceRecord,
};
use crate::graph::supplement::DeclaredDynamicControlName;

use super::package::{AdapterRole, GenerationDefaults, NativeGraphPackagePlan, NativeGraphProfile};

pub(crate) const NATIVE_GRAPH_SOURCE_SCHEMA: &str = "native_graph/1.0";
pub(crate) const NATIVE_GRAPH_EXECUTION_PROFILE: &str = "native_graph";
const LIVE_DRIVER_KIND: &str = "native_graph_live";
pub(crate) const BINDING_METADATA_KEY: &str = "native_graph.binding";
pub(crate) const GENERATION_METADATA_KEY: &str = "native_graph.generation";

/// Immutable Task-6 static projection facts retained from one imported source.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGraphControlContract {
    /// Digest of the Task-1 immutable program snapshot.
    pub source_snapshot_digest: String,
    /// Digest of the canonical static acyclic projection executed by Task 6.
    pub static_projection_digest: String,
    /// Digest of the complete declared source graph before live stage projection.
    pub source_graph_digest: String,
    /// Independent cap on emitted acyclic stages.
    pub stage_bound: NonZeroU32,
    /// Declared terminal channel identifiers.
    pub terminal_outputs: Vec<String>,
    /// Exact node identifiers permitted in this slice's one acyclic stage.
    pub stage_node_ids: Vec<String>,
    /// Exact channel identifiers permitted in this slice's one acyclic stage.
    pub stage_channel_ids: Vec<String>,
    /// Declared model-selected branch facts.
    pub branches: Vec<ReservedNativeGraphBranch>,
    /// Declared branch-join facts.
    pub joins: Vec<ReservedNativeGraphJoin>,
    /// Declared bounded loop and retry facts.
    pub loops: Vec<ReservedNativeGraphLoop>,
}

/// One declared conditional edge retained in the immutable control contract.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGraphControlEdge {
    /// Source node identifier.
    pub source: String,
    /// Target node identifier.
    pub target: String,
}

/// One model-result candidate permitted by a declared branch.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGraphBranchCandidate {
    /// Stable candidate identifier scoped to its branch.
    pub id: String,
    /// Exact model result that selects this candidate.
    #[serde(rename = "match")]
    pub match_value: String,
    /// Declared graph edge enabled by this candidate.
    pub edge: NativeGraphControlEdge,
    /// Complete acyclic stage nodes permitted for this candidate.
    pub nodes: Vec<String>,
    /// Channels produced by the candidate before the next model stage.
    pub channels: Vec<String>,
}

/// Typed model-result branch contract.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservedNativeGraphBranch {
    /// Stable branch identifier.
    pub id: String,
    /// Model node whose completed output is interpreted by this branch.
    pub selector_node: String,
    /// Declared model-output channel carrying the selector result.
    pub selector_channel: String,
    /// Exhaustive model-result candidates allowed by this branch.
    pub candidates: Vec<NativeGraphBranchCandidate>,
}

/// Declared policy for selecting one branch workspace candidate at a join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeGraphJoinReduction {
    /// Select the immutable workspace candidate chosen by the named branch.
    SelectedCandidate,
}

/// Typed branch-join contract.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservedNativeGraphJoin {
    /// Stable join identifier.
    pub id: String,
    /// Branch whose model decision selects the merge candidate.
    pub selector: String,
    /// Candidate identifiers accepted by the merge.
    pub candidates: Vec<String>,
    /// Declared channel receiving the selected immutable candidate fact.
    pub output_channel: String,
    /// Explicit merge reduction policy.
    pub reduction: NativeGraphJoinReduction,
}

/// Typed bounded loop and retry contract.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservedNativeGraphLoop {
    /// Stable loop identifier.
    pub id: String,
    /// Model node whose output selects continuation or exit.
    pub selector_node: String,
    /// Declared model-output channel carrying the loop decision.
    pub selector_channel: String,
    /// Exact model result which requests another iteration.
    pub continue_match: String,
    /// Exact model result which spends one declared retry before re-entering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_match: Option<String>,
    /// Complete declared loop member set re-enabled between bounded iterations.
    pub members: Vec<String>,
    /// Declared edge entering one loop iteration.
    pub entry: NativeGraphControlEdge,
    /// Declared feedback edge re-entering the loop.
    pub backedge: NativeGraphControlEdge,
    /// Declared edge that leaves the loop.
    pub exit: NativeGraphControlEdge,
    /// Rust-owned maximum number of emitted loop iterations.
    pub max_iterations: NonZeroU32,
    /// Rust-owned retry budget consumed before another loop iteration.
    pub max_retries: u32,
}

/// Backwards-compatible name for the Task-6 NativeGraph control contract.
pub type BoundedControlFlowContract = NativeGraphControlContract;

/// Exact lowering disposition for one source node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeGraphNodeFidelity {
    /// The source node maps directly to one existing Graph-IR node.
    Exact,
}

/// Source-to-Graph-IR account for one NativeGraph node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeGraphNodeLowering {
    node_id: String,
    fidelity: NativeGraphNodeFidelity,
}

impl NativeGraphNodeLowering {
    /// Borrows the source node identifier retained by the lowered graph.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Reports whether no source operation was substituted or omitted.
    pub const fn is_exact(&self) -> bool {
        matches!(self.fidelity, NativeGraphNodeFidelity::Exact)
    }
}

/// Immutable account of a source-faithful NativeGraph lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeGraphLoweringReport {
    source_digest: String,
    nodes: Vec<NativeGraphNodeLowering>,
}

impl NativeGraphLoweringReport {
    /// Borrows the immutable source digest that was lowered.
    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    /// Iterates source-node lowering facts in authored order.
    pub fn nodes(&self) -> impl Iterator<Item = &NativeGraphNodeLowering> {
        self.nodes.iter()
    }
}

/// Typed refusal at the NativeGraph source boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeGraphLoweringError {
    /// The package selected a profile that does not own native graph traversal.
    UnsupportedProfile,
    /// The package did not retain a native graph program source.
    MissingProgram,
    /// The source document is not the supported strict JSON grammar.
    InvalidSource(String),
    /// A declared cycle does not include a finite iteration budget.
    UnboundedCycle {
        /// Source cycle identifier retained for diagnostics.
        cycle: String,
    },
    /// A source model node references no declared package model binding.
    UnknownModelBinding {
        /// Source node identifier.
        node_id: String,
        /// Unresolved model binding identifier.
        binding: String,
    },
    /// A source tool node references no declared tool adapter.
    UnknownToolAdapter {
        /// Source node identifier.
        node_id: String,
        /// Unresolved adapter identifier.
        adapter: String,
    },
    /// A source graph cannot be closed over the shared static Graph-IR executor.
    InvalidClosure(String),
}

impl Display for NativeGraphLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProfile => {
                formatter.write_str("NativeGraph lowering requires the native_graph profile")
            }
            Self::MissingProgram => formatter.write_str("NativeGraph package has no graph program"),
            Self::InvalidSource(message) => {
                write!(formatter, "invalid NativeGraph source: {message}")
            }
            Self::UnboundedCycle { cycle } => {
                write!(
                    formatter,
                    "NativeGraph cycle {cycle:?} has no finite iteration bound"
                )
            }
            Self::UnknownModelBinding { node_id, binding } => write!(
                formatter,
                "NativeGraph model node {node_id:?} references undeclared binding {binding:?}"
            ),
            Self::UnknownToolAdapter { node_id, adapter } => write!(
                formatter,
                "NativeGraph tool node {node_id:?} references undeclared tool adapter {adapter:?}"
            ),
            Self::InvalidClosure(message) => {
                write!(formatter, "NativeGraph graph is not closed: {message}")
            }
        }
    }
}

impl Error for NativeGraphLoweringError {}

/// Factory retaining one immutable NativeGraph package's declared bindings.
#[derive(Clone)]
pub struct NativeGraphLowererFactory {
    package: NativeGraphPackagePlan,
}

impl NativeGraphLowererFactory {
    /// Freeze one package's source snapshot and declared binding authority.
    pub fn new(package: &NativeGraphPackagePlan) -> Self {
        Self {
            package: package.clone(),
        }
    }
}

impl GraphLowererFactory for NativeGraphLowererFactory {
    fn capabilities(&self) -> GraphLowererCapabilities {
        GraphLowererCapabilities::new(
            [NATIVE_GRAPH_SOURCE_SCHEMA.to_owned()],
            [NATIVE_GRAPH_EXECUTION_PROFILE.to_owned()],
        )
    }

    fn lower(
        &self,
        request: GraphLoweringRequest<'_>,
    ) -> Result<GraphTraceProgram, GraphLoweringError> {
        let capabilities = self.capabilities();
        if !capabilities.supports_source_schema(request.source_schema)
            || !capabilities.supports_execution_profile(request.execution_profile)
        {
            return Err(GraphLoweringError::new(format!(
                "NativeGraph lowerer does not support source schema {:?} and execution profile {:?}",
                request.source_schema, request.execution_profile
            )));
        }
        let source = self.package.program_source().ok_or_else(|| {
            GraphLoweringError::new("NativeGraph lowerer package is missing its graph source")
        })?;
        if source.bytes() != request.source {
            return Err(GraphLoweringError::new(
                "NativeGraph lowerer request bytes do not match the imported source snapshot",
            ));
        }
        lower_native_graph(&self.package)
            .map(|(program, _)| program)
            .map_err(|error| GraphLoweringError::new(error.to_string()))
    }
}

/// Lower an imported NativeGraph package into the existing graph trace program.
///
/// This consumes only the immutable source bytes and declared bindings retained
/// by package import. It never rereads a task path or accepts caller-supplied
/// model or adapter authority.
pub fn lower_native_graph(
    package: &NativeGraphPackagePlan,
) -> Result<(GraphTraceProgram, NativeGraphLoweringReport), NativeGraphLoweringError> {
    if package.profile() != NativeGraphProfile::NativeGraph {
        return Err(NativeGraphLoweringError::UnsupportedProfile);
    }
    let source = package
        .program_source()
        .ok_or(NativeGraphLoweringError::MissingProgram)?;
    let document = serde_json::from_slice::<NativeGraphSourceDto>(source.bytes())
        .map_err(|error| NativeGraphLoweringError::InvalidSource(error.to_string()))?;
    let NativeGraphSourceDto {
        schema_version,
        trace_id,
        stage_bound,
        channels,
        nodes: source_nodes,
        edges,
        branches,
        joins,
        loops,
        terminal_outputs,
        initial_state,
        arrival_offset_ns,
    } = document;
    if schema_version != "1.0" {
        return Err(NativeGraphLoweringError::InvalidSource(format!(
            "unsupported schema_version {:?}",
            schema_version
        )));
    }
    let produced_channels = source_nodes
        .iter()
        .map(|node| match node {
            NativeGraphSourceNode::Model { output, .. }
            | NativeGraphSourceNode::Tool { output, .. } => output.clone(),
        })
        .collect::<BTreeSet<_>>();
    let mut nodes = BTreeMap::new();
    let mut facts = Vec::with_capacity(source_nodes.len());
    for source_node in source_nodes {
        let node_id = source_node.id().to_owned();
        if node_id.is_empty() || node_id == START_NODE_ID || node_id == "END" {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "node identifier {node_id:?} is reserved or empty"
            )));
        }
        let node = match source_node {
            NativeGraphSourceNode::Model {
                id,
                binding,
                output,
                streaming,
                inputs,
                max_tokens,
            } => {
                let binding_spec = package
                    .model_bindings()
                    .iter()
                    .find(|candidate| candidate.id.as_str() == binding)
                    .ok_or_else(|| NativeGraphLoweringError::UnknownModelBinding {
                        node_id: id.clone(),
                        binding: binding.clone(),
                    })?;
                let mut metadata = BTreeMap::new();
                metadata.insert(BINDING_METADATA_KEY.into(), Value::String(binding));
                metadata.insert("model".into(), Value::String(binding_spec.model.clone()));
                metadata.insert(
                    "endpoint".into(),
                    Value::String(binding_spec.endpoint_profile_id.clone()),
                );
                let generation = generation_request_body(&binding_spec.generation);
                if !generation.is_empty() {
                    metadata.insert(GENERATION_METADATA_KEY.into(), Value::Object(generation));
                }
                ExecutableGraphNode::Llm(LlmNode {
                    output,
                    streaming,
                    inputs: lower_input_requirements(&inputs, &initial_state, &produced_channels),
                    min_start_delay_us: None,
                    max_tokens: max_tokens.or_else(|| {
                        binding_spec
                            .generation
                            .max_tokens
                            .map(|value| value as usize)
                    }),
                    items: inputs
                        .into_iter()
                        .map(|splice| crate::graph::model::PromptItem::Splice { splice })
                        .collect(),
                    request: None,
                    metadata,
                })
            }
            NativeGraphSourceNode::Tool {
                id,
                adapter,
                operation,
                output,
                timeout_ns,
            } => {
                let is_declared_tool = package.adapters().iter().any(|candidate| {
                    candidate.id.as_str() == adapter && candidate.role == AdapterRole::Tool
                });
                if !is_declared_tool {
                    return Err(NativeGraphLoweringError::UnknownToolAdapter {
                        node_id: id,
                        adapter,
                    });
                }
                ExecutableGraphNode::Tool(ToolNode {
                    output,
                    commands: vec![operation],
                    timeout_ns,
                })
            }
        };
        if nodes.insert(node_id.clone(), node).is_some() {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "source declares duplicate node {node_id:?}"
            )));
        }
        facts.push(NativeGraphNodeLowering {
            node_id,
            fidelity: NativeGraphNodeFidelity::Exact,
        });
    }
    let graph = GraphRecord {
        version: Some(schema_version),
        system: None,
        state: channels
            .into_iter()
            .map(|(name, channel)| {
                (
                    name,
                    ChannelSpec {
                        channel_type: channel.channel_type,
                        reducer: channel.reducer,
                    },
                )
            })
            .collect(),
        nodes,
        edges: edges
            .into_iter()
            .map(|edge| StaticEdge {
                source: edge.source,
                target: edge.target,
                delay_after_predecessor_us: edge.delay_after_predecessor_us,
                min_start_delay_us: edge.min_start_delay_us,
                delay_after_predecessor_start_us: edge.delay_after_predecessor_start_us,
                delay_after_predecessor_first_token_us: edge.delay_after_predecessor_first_token_us,
            })
            .collect(),
    };
    for output in &terminal_outputs {
        let produced = graph.nodes.values().any(|node| node.output() == output);
        if !graph.state.contains_key(output) || !produced {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "terminal output {output:?} is not a declared produced channel"
            )));
        }
    }
    let plan = GraphTracePlan {
        graph,
        trace: TraceRecord {
            id: trace_id,
            graph_ref: None,
            initial_state,
        },
        arrival_offset_ns,
    };
    let source_graph_digest = canonical_static_projection_digest(&plan)?;
    let static_projection = (!branches.is_empty() || !joins.is_empty() || !loops.is_empty())
        .then(|| initial_dynamic_stage(&plan))
        .transpose()?
        .unwrap_or_else(|| plan.clone());
    let static_projection_digest = canonical_static_projection_digest(&static_projection)?;
    let control_flow = NativeGraphControlContract {
        source_snapshot_digest: source.digest().as_str().to_owned(),
        static_projection_digest: static_projection_digest.clone(),
        source_graph_digest,
        stage_bound,
        terminal_outputs: terminal_outputs.clone(),
        stage_node_ids: plan.graph.nodes.keys().cloned().collect(),
        stage_channel_ids: plan.graph.state.keys().cloned().collect(),
        branches,
        joins,
        loops,
    };
    validate_control_flow_contract(&control_flow)?;
    validate_control_bindings(&control_flow, &plan)?;
    if control_flow.branches.is_empty()
        && control_flow.joins.is_empty()
        && control_flow.loops.is_empty()
    {
        validate_native_graph_trace_plan(&plan)?;
    } else {
        validate_dynamic_native_graph_source(&plan, &control_flow)?;
    }
    let control_digest = canonical_control_digest(&control_flow)?;
    let mut driver_data = BTreeMap::new();
    driver_data.insert(
        "control_flow".into(),
        serde_json::to_value(&control_flow).map_err(|error| {
            NativeGraphLoweringError::InvalidSource(format!("serializing control flow: {error}"))
        })?,
    );
    let program = GraphTraceProgram {
        profiling: plan,
        warmup: None,
        environment: None,
        replay: None,
        driver: TraceDriverSpec::with_data(LIVE_DRIVER_KIND.into(), driver_data)
            .with_source_provenance(
                source.digest().as_str().to_owned(),
                source.bytes(),
                static_projection_digest,
                control_digest,
            ),
    };
    Ok((
        program,
        NativeGraphLoweringReport {
            source_digest: source.digest().as_str().to_owned(),
            nodes: facts,
        },
    ))
}

fn generation_request_body(generation: &GenerationDefaults) -> Map<String, Value> {
    let mut body = Map::new();
    if let Some(value) = generation.min_tokens {
        body.insert("min_tokens".into(), Value::from(value));
    }
    if let Some(value) = generation.temperature {
        body.insert("temperature".into(), Value::from(value));
    }
    if let Some(value) = generation.top_p {
        body.insert("top_p".into(), Value::from(value));
    }
    if let Some(value) = generation.top_k {
        body.insert("top_k".into(), Value::from(value));
    }
    if let Some(value) = generation.seed {
        body.insert("seed".into(), Value::from(value));
    }
    if let Some(value) = generation.presence_penalty {
        body.insert("presence_penalty".into(), Value::from(value));
    }
    if let Some(value) = generation.frequency_penalty {
        body.insert("frequency_penalty".into(), Value::from(value));
    }
    if let Some(value) = generation.repetition_penalty {
        body.insert("repetition_penalty".into(), Value::from(value));
    }
    body
}

/// Validate closure over one driver-produced NativeGraph stage before execution.
///
/// The validator is intentionally independent of the original source document:
/// every driver-authored rewrite must satisfy these same shared Graph-IR facts.
pub fn validate_native_graph_trace_plan(
    plan: &GraphTracePlan,
) -> Result<(), NativeGraphLoweringError> {
    let node_ids = plan.graph.nodes.keys().cloned().collect::<BTreeSet<_>>();
    for (node_id, node) in &plan.graph.nodes {
        if node_id.is_empty() || node_id == START_NODE_ID || node_id == "END" {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "node identifier {node_id:?} is reserved or empty"
            )));
        }
        if !plan.graph.state.contains_key(node.output()) {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "node {node_id:?} writes undeclared channel {:?}",
                node.output()
            )));
        }
        for channel in node.read_channels() {
            if !plan.graph.state.contains_key(channel) {
                return Err(NativeGraphLoweringError::InvalidClosure(format!(
                    "node {node_id:?} reads undeclared channel {channel:?}"
                )));
            }
        }
    }

    let mut successors = HashMap::<String, Vec<String>>::new();
    let mut predecessors = HashMap::<String, Vec<String>>::new();
    for edge in &plan.graph.edges {
        let source_is_valid = edge.source == START_NODE_ID || node_ids.contains(&edge.source);
        let target_is_valid = edge.target == "END" || node_ids.contains(&edge.target);
        if !source_is_valid
            || !target_is_valid
            || edge.source == "END"
            || edge.target == START_NODE_ID
        {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "edge {:?} -> {:?} references an invalid graph endpoint",
                edge.source, edge.target
            )));
        }
        successors
            .entry(edge.source.clone())
            .or_default()
            .push(edge.target.clone());
        predecessors
            .entry(edge.target.clone())
            .or_default()
            .push(edge.source.clone());
    }
    let reachable = walk_graph(START_NODE_ID, &successors);
    if !node_ids.iter().all(|node_id| reachable.contains(node_id)) {
        let missing = node_ids
            .iter()
            .filter(|node_id| !reachable.contains(*node_id))
            .cloned()
            .collect::<Vec<_>>();
        return Err(NativeGraphLoweringError::InvalidClosure(format!(
            "nodes are unreachable from START: {}",
            missing.join(", ")
        )));
    }
    let reverse = predecessors;
    let reaches_end = walk_graph("END", &reverse);
    if !node_ids.iter().all(|node_id| reaches_end.contains(node_id)) {
        let missing = node_ids
            .iter()
            .filter(|node_id| !reaches_end.contains(*node_id))
            .cloned()
            .collect::<Vec<_>>();
        return Err(NativeGraphLoweringError::InvalidClosure(format!(
            "nodes cannot reach END: {}",
            missing.join(", ")
        )));
    }
    if let Some(cycle) = find_cycle(&node_ids, &successors) {
        return Err(NativeGraphLoweringError::UnboundedCycle { cycle });
    }
    Ok(())
}

pub(crate) fn validate_native_graph_stage(
    plan: &GraphTracePlan,
    control_flow: &NativeGraphControlContract,
) -> Result<(), NativeGraphLoweringError> {
    validate_native_graph_trace_plan(plan)?;
    validate_control_flow_contract(control_flow)?;
    let stage_node_ids = plan.graph.nodes.keys().cloned().collect::<Vec<_>>();
    if stage_node_ids != control_flow.stage_node_ids {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "driver-authored stage nodes differ from the imported control-flow contract".into(),
        ));
    }
    let stage_channel_ids = plan.graph.state.keys().cloned().collect::<Vec<_>>();
    if stage_channel_ids != control_flow.stage_channel_ids {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "driver-authored stage channels differ from the imported control-flow contract".into(),
        ));
    }
    if canonical_static_projection_digest(plan)? != control_flow.static_projection_digest {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "driver-authored stage differs from the imported static projection".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_control_flow_contract(
    control_flow: &NativeGraphControlContract,
) -> Result<(), NativeGraphLoweringError> {
    if ArtifactDigest::parse(control_flow.source_snapshot_digest.clone()).is_err()
        || ArtifactDigest::parse(control_flow.static_projection_digest.clone()).is_err()
        || ArtifactDigest::parse(control_flow.source_graph_digest.clone()).is_err()
    {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "control-flow contract has an invalid immutable projection identity".into(),
        ));
    }
    if !is_sorted_unique(&control_flow.stage_node_ids)
        || !is_sorted_unique(&control_flow.stage_channel_ids)
        || !is_sorted_unique(&control_flow.terminal_outputs)
    {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "control-flow contract has non-canonical stage or terminal identifiers".into(),
        ));
    }
    if control_flow
        .terminal_outputs
        .iter()
        .any(|terminal| !control_flow.stage_channel_ids.contains(terminal))
    {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "control-flow contract declares a terminal channel outside its stage".into(),
        ));
    }
    if !is_sorted_unique_by(&control_flow.branches, |branch| &branch.id)
        || !is_sorted_unique_by(&control_flow.joins, |join| &join.id)
        || !is_sorted_unique_by(&control_flow.loops, |loop_spec| &loop_spec.id)
    {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "control-flow contract has non-canonical control identifiers".into(),
        ));
    }
    for branch in &control_flow.branches {
        if branch.id.is_empty()
            || DeclaredDynamicControlName::parse(branch.id.clone()).is_err()
            || branch.selector_node.is_empty()
            || branch.selector_channel.is_empty()
            || branch.candidates.is_empty()
            || !is_sorted_unique_by(&branch.candidates, |candidate| &candidate.id)
            || branch.candidates.iter().any(|candidate| {
                candidate.id.is_empty()
                    || DeclaredDynamicControlName::parse(candidate.id.clone()).is_err()
                    || candidate.match_value.is_empty()
                    || candidate.edge.source.is_empty()
                    || candidate.edge.target.is_empty()
                    || candidate.nodes.is_empty()
                    || !is_sorted_unique(&candidate.nodes)
                    || !is_sorted_unique(&candidate.channels)
            })
        {
            return Err(NativeGraphLoweringError::InvalidClosure(
                "control-flow contract has an invalid branch candidate".into(),
            ));
        }
    }
    for join in &control_flow.joins {
        if join.id.is_empty()
            || DeclaredDynamicControlName::parse(join.id.clone()).is_err()
            || join.selector.is_empty()
            || join.output_channel.is_empty()
            || join.candidates.is_empty()
            || !is_sorted_unique(&join.candidates)
        {
            return Err(NativeGraphLoweringError::InvalidClosure(
                "control-flow contract has an invalid branch join".into(),
            ));
        }
    }
    for loop_spec in &control_flow.loops {
        if loop_spec.id.is_empty()
            || DeclaredDynamicControlName::parse(loop_spec.id.clone()).is_err()
            || loop_spec.selector_node.is_empty()
            || loop_spec.selector_channel.is_empty()
            || loop_spec.continue_match.is_empty()
            || loop_spec
                .retry_match
                .as_ref()
                .is_some_and(|retry| retry.is_empty() || retry == &loop_spec.continue_match)
            || loop_spec.members.is_empty()
            || !is_sorted_unique(&loop_spec.members)
            || loop_spec.entry.source.is_empty()
            || loop_spec.entry.target.is_empty()
            || loop_spec.backedge.source.is_empty()
            || loop_spec.backedge.target.is_empty()
            || loop_spec.exit.source.is_empty()
            || loop_spec.exit.target.is_empty()
        {
            return Err(NativeGraphLoweringError::InvalidClosure(
                "control-flow contract has an invalid bounded loop".into(),
            ));
        }
    }
    Ok(())
}

fn validate_control_bindings(
    control_flow: &NativeGraphControlContract,
    plan: &GraphTracePlan,
) -> Result<(), NativeGraphLoweringError> {
    let nodes = &plan.graph.nodes;
    let channels = &plan.graph.state;
    let has_edge = |expected: &NativeGraphControlEdge| {
        plan.graph
            .edges
            .iter()
            .any(|edge| edge.source == expected.source && edge.target == expected.target)
    };
    for branch in &control_flow.branches {
        let selector = nodes.get(&branch.selector_node).ok_or_else(|| {
            NativeGraphLoweringError::InvalidClosure(format!(
                "branch {:?} selects undeclared node {:?}",
                branch.id, branch.selector_node
            ))
        })?;
        if selector.output() != branch.selector_channel {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "branch {:?} selector channel {:?} does not match node output {:?}",
                branch.id,
                branch.selector_channel,
                selector.output()
            )));
        }
        for candidate in &branch.candidates {
            if !candidate.nodes.contains(&candidate.edge.target) {
                return Err(NativeGraphLoweringError::InvalidClosure(format!(
                    "branch {:?} candidate {:?} does not include its declared edge target {:?}",
                    branch.id, candidate.id, candidate.edge.target
                )));
            }
            if !has_edge(&candidate.edge)
                || candidate.edge.source != branch.selector_node
                || candidate.nodes.iter().any(|node| !nodes.contains_key(node))
                || candidate
                    .channels
                    .iter()
                    .any(|channel| !channels.contains_key(channel))
            {
                return Err(NativeGraphLoweringError::InvalidClosure(format!(
                    "branch {:?} candidate {:?} binds an undeclared path",
                    branch.id, candidate.id
                )));
            }
        }
    }
    for join in &control_flow.joins {
        let branch = control_flow
            .branches
            .iter()
            .find(|branch| branch.id == join.selector)
            .ok_or_else(|| {
                NativeGraphLoweringError::InvalidClosure(format!(
                    "join {:?} selects undeclared branch {:?}",
                    join.id, join.selector
                ))
            })?;
        if !channels.contains_key(&join.output_channel)
            || join
                .candidates
                .iter()
                .any(|candidate| !branch.candidates.iter().any(|entry| &entry.id == candidate))
        {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "join {:?} binds an undeclared candidate or output channel",
                join.id
            )));
        }
        let join_consumers = nodes
            .iter()
            .filter_map(|(node_id, node)| {
                node.read_channels()
                    .contains(&join.output_channel.as_str())
                    .then_some(node_id)
            })
            .collect::<Vec<_>>();
        let mut successors = HashMap::<String, Vec<String>>::new();
        for edge in &plan.graph.edges {
            successors
                .entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
        }
        for candidate_id in &join.candidates {
            let candidate = branch
                .candidates
                .iter()
                .find(|candidate| &candidate.id == candidate_id)
                .ok_or_else(|| {
                    NativeGraphLoweringError::InvalidClosure(format!(
                        "join {:?} lost declared candidate {:?}",
                        join.id, candidate_id
                    ))
                })?;
            if candidate.channels.len() != 1 {
                return Err(NativeGraphLoweringError::InvalidClosure(format!(
                    "join {:?} candidate {:?} must declare exactly one channel",
                    join.id, candidate.id
                )));
            }
            let channel = &candidate.channels[0];
            let producers = candidate
                .nodes
                .iter()
                .filter(|node_id| {
                    nodes
                        .get(*node_id)
                        .is_some_and(|node| node.output() == channel)
                })
                .collect::<Vec<_>>();
            if producers.is_empty() {
                return Err(NativeGraphLoweringError::InvalidClosure(format!(
                    "join {:?} candidate {:?} does not produce declared channel {:?}",
                    join.id, candidate.id, channel
                )));
            }
            if join_consumers.is_empty()
                || !producers.iter().any(|producer| {
                    let reachable = walk_graph(producer, &successors);
                    join_consumers
                        .iter()
                        .any(|consumer| reachable.contains(*consumer))
                })
            {
                return Err(NativeGraphLoweringError::InvalidClosure(format!(
                    "join {:?} candidate {:?} does not reach its declared join",
                    join.id, candidate.id
                )));
            }
        }
    }
    for loop_spec in &control_flow.loops {
        let selector = nodes.get(&loop_spec.selector_node).ok_or_else(|| {
            NativeGraphLoweringError::InvalidClosure(format!(
                "loop {:?} selects undeclared node {:?}",
                loop_spec.id, loop_spec.selector_node
            ))
        })?;
        if selector.output() != loop_spec.selector_channel
            || loop_spec.entry.source != loop_spec.selector_node
            || loop_spec.backedge.target != loop_spec.selector_node
            || loop_spec.exit.source != loop_spec.selector_node
            || !has_edge(&loop_spec.entry)
            || !has_edge(&loop_spec.backedge)
            || !has_edge(&loop_spec.exit)
            || !loop_spec.members.contains(&loop_spec.selector_node)
            || !loop_spec.members.contains(&loop_spec.entry.target)
            || !loop_spec.members.contains(&loop_spec.backedge.source)
            || loop_spec
                .members
                .iter()
                .any(|member| !nodes.contains_key(member))
        {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "loop {:?} binds undeclared selector or edge facts",
                loop_spec.id
            )));
        }
        let mut feedback_successors = HashMap::<String, Vec<String>>::new();
        let mut feedback_predecessors = HashMap::<String, Vec<String>>::new();
        for edge in plan.graph.edges.iter().filter(|edge| {
            edge.source != loop_spec.backedge.source || edge.target != loop_spec.backedge.target
        }) {
            feedback_successors
                .entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
            feedback_predecessors
                .entry(edge.target.clone())
                .or_default()
                .push(edge.source.clone());
        }
        let reaches_backedge = walk_graph(&loop_spec.entry.target, &feedback_successors);
        let reaches_entry = walk_graph(&loop_spec.backedge.source, &feedback_predecessors);
        let mut expected_members = reaches_backedge
            .intersection(&reaches_entry)
            .filter(|node| nodes.contains_key(*node))
            .cloned()
            .collect::<BTreeSet<_>>();
        expected_members.insert(loop_spec.selector_node.clone());
        let declared_members = loop_spec.members.iter().cloned().collect::<BTreeSet<_>>();
        let missing = expected_members
            .difference(&declared_members)
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "loop {:?} omits source feedback-path member(s): {}",
                loop_spec.id,
                missing.join(", ")
            )));
        }
        let unexpected = declared_members
            .difference(&expected_members)
            .cloned()
            .collect::<Vec<_>>();
        if !unexpected.is_empty() {
            return Err(NativeGraphLoweringError::InvalidClosure(format!(
                "loop {:?} declares member(s) outside its source feedback path: {}",
                loop_spec.id,
                unexpected.join(", ")
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_dynamic_native_graph_source(
    plan: &GraphTracePlan,
    control_flow: &NativeGraphControlContract,
) -> Result<(), NativeGraphLoweringError> {
    let mut acyclic = plan.clone();
    acyclic.graph.edges.retain(|edge| {
        !control_flow.loops.iter().any(|loop_spec| {
            edge.source == loop_spec.backedge.source && edge.target == loop_spec.backedge.target
        })
    });
    // The validation-only projection replaces each declared feedback edge with
    // its declared exit target. This proves complete source closure after the
    // cycle is removed without adding an executable edge or statically unrolling
    // a live model decision.
    acyclic
        .graph
        .edges
        .extend(control_flow.loops.iter().map(|loop_spec| StaticEdge {
            source: loop_spec.backedge.source.clone(),
            target: loop_spec.exit.target.clone(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        }));
    validate_native_graph_trace_plan(&acyclic)
}

pub(crate) fn initial_dynamic_stage(
    source: &GraphTracePlan,
) -> Result<GraphTracePlan, NativeGraphLoweringError> {
    let root_nodes = source
        .graph
        .edges
        .iter()
        .filter(|edge| edge.source == START_NODE_ID)
        .map(|edge| edge.target.as_str())
        .collect::<BTreeSet<_>>();
    if root_nodes.is_empty()
        || root_nodes
            .iter()
            .any(|node| !source.graph.nodes.contains_key(*node))
    {
        return Err(NativeGraphLoweringError::InvalidClosure(
            "dynamic NativeGraph source has no declared executable root stage".into(),
        ));
    }
    let graph = GraphRecord {
        version: source.graph.version.clone(),
        system: source.graph.system.clone(),
        state: source.graph.state.clone(),
        nodes: root_nodes
            .iter()
            .map(|node| {
                source
                    .graph
                    .nodes
                    .get(*node)
                    .cloned()
                    .map(|entry| ((*node).to_owned(), entry))
                    .ok_or_else(|| {
                        NativeGraphLoweringError::InvalidClosure(format!(
                            "dynamic root stage references unknown node {node:?}"
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        edges: root_nodes
            .iter()
            .flat_map(|node| {
                [
                    StaticEdge {
                        source: START_NODE_ID.into(),
                        target: (*node).to_owned(),
                        delay_after_predecessor_us: None,
                        min_start_delay_us: None,
                        delay_after_predecessor_start_us: None,
                        delay_after_predecessor_first_token_us: None,
                    },
                    StaticEdge {
                        source: (*node).to_owned(),
                        target: "END".into(),
                        delay_after_predecessor_us: None,
                        min_start_delay_us: None,
                        delay_after_predecessor_start_us: None,
                        delay_after_predecessor_first_token_us: None,
                    },
                ]
            })
            .collect(),
    };
    let stage = GraphTracePlan {
        graph,
        trace: source.trace.clone(),
        arrival_offset_ns: source.arrival_offset_ns,
    };
    validate_native_graph_trace_plan(&stage)?;
    Ok(stage)
}

pub(crate) fn canonical_static_projection_digest(
    plan: &GraphTracePlan,
) -> Result<String, NativeGraphLoweringError> {
    let projection = serde_json::to_vec(plan).map_err(|error| {
        NativeGraphLoweringError::InvalidClosure(format!(
            "serializing canonical static projection: {error}"
        ))
    })?;
    Ok(ArtifactDigest::from_bytes(&projection).as_str().to_owned())
}

pub(crate) fn canonical_control_digest(
    control_flow: &NativeGraphControlContract,
) -> Result<String, NativeGraphLoweringError> {
    let bytes = serde_json::to_vec(control_flow).map_err(|error| {
        NativeGraphLoweringError::InvalidClosure(format!(
            "serializing canonical NativeGraph control contract: {error}"
        ))
    })?;
    Ok(ArtifactDigest::from_bytes(&bytes).as_str().to_owned())
}

fn is_sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|window| window[0] < window[1])
}

fn is_sorted_unique_by<T>(values: &[T], key: impl Fn(&T) -> &String) -> bool {
    values
        .windows(2)
        .all(|window| key(&window[0]) < key(&window[1]))
}

fn walk_graph(start: &str, edges: &HashMap<String, Vec<String>>) -> HashSet<String> {
    let mut visited = HashSet::new();
    let mut pending = vec![start.to_owned()];
    while let Some(node) = pending.pop() {
        if !visited.insert(node.clone()) {
            continue;
        }
        if let Some(next) = edges.get(&node) {
            pending.extend(next.iter().cloned());
        }
    }
    visited
}

fn find_cycle(
    node_ids: &BTreeSet<String>,
    successors: &HashMap<String, Vec<String>>,
) -> Option<String> {
    fn visit(
        node: &str,
        node_ids: &BTreeSet<String>,
        successors: &HashMap<String, Vec<String>>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
    ) -> Option<String> {
        if !visiting.insert(node.to_owned()) {
            return Some(node.to_owned());
        }
        if let Some(next) = successors.get(node) {
            for target in next.iter().filter(|target| node_ids.contains(*target)) {
                if !visited.contains(target)
                    && let Some(cycle) = visit(target, node_ids, successors, visiting, visited)
                {
                    return Some(cycle);
                }
            }
        }
        visiting.remove(node);
        visited.insert(node.to_owned());
        None
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for node in node_ids {
        if !visited.contains(node)
            && let Some(cycle) = visit(node, node_ids, successors, &mut visiting, &mut visited)
        {
            return Some(cycle);
        }
    }
    None
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGraphSourceDto {
    schema_version: String,
    trace_id: String,
    stage_bound: NonZeroU32,
    channels: BTreeMap<String, NativeGraphChannelDto>,
    nodes: Vec<NativeGraphSourceNode>,
    edges: Vec<NativeGraphEdgeDto>,
    #[serde(default)]
    branches: Vec<ReservedNativeGraphBranch>,
    #[serde(default)]
    joins: Vec<ReservedNativeGraphJoin>,
    #[serde(default)]
    loops: Vec<ReservedNativeGraphLoop>,
    terminal_outputs: Vec<String>,
    #[serde(default)]
    initial_state: BTreeMap<String, Value>,
    #[serde(default)]
    arrival_offset_ns: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGraphChannelDto {
    #[serde(rename = "type")]
    channel_type: ChannelType,
    reducer: ReducerName,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum NativeGraphSourceNode {
    Model {
        id: String,
        binding: String,
        output: String,
        #[serde(default = "default_streaming")]
        streaming: bool,
        #[serde(default)]
        inputs: Vec<String>,
        #[serde(default)]
        max_tokens: Option<usize>,
    },
    Tool {
        id: String,
        adapter: String,
        operation: String,
        output: String,
        #[serde(default)]
        timeout_ns: Option<u64>,
    },
}

impl NativeGraphSourceNode {
    fn id(&self) -> &str {
        match self {
            Self::Model { id, .. } | Self::Tool { id, .. } => id,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGraphEdgeDto {
    source: String,
    target: String,
    #[serde(default)]
    delay_after_predecessor_us: Option<f64>,
    #[serde(default)]
    min_start_delay_us: Option<f64>,
    #[serde(default)]
    delay_after_predecessor_start_us: Option<f64>,
    #[serde(default)]
    delay_after_predecessor_first_token_us: Option<f64>,
}

fn default_streaming() -> bool {
    true
}

fn lower_input_requirements(
    inputs: &[String],
    initial_state: &BTreeMap<String, Value>,
    produced_channels: &BTreeSet<String>,
) -> Vec<ChannelRequirement> {
    inputs
        .iter()
        .filter(|channel| {
            !initial_state.contains_key(channel.as_str())
                || produced_channels.contains(channel.as_str())
        })
        .cloned()
        .map(|channel| ChannelRequirement {
            channel,
            count: Default::default(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_only_input_is_spliced_without_a_producer_wait() {
        let inputs = vec!["prompt".to_owned(), "reply".to_owned()];
        let initial_state = BTreeMap::from([
            ("prompt".to_owned(), Value::Null),
            ("reply".to_owned(), Value::Null),
        ]);
        let produced_channels = BTreeSet::from(["reply".to_owned()]);

        let requirements = lower_input_requirements(&inputs, &initial_state, &produced_channels);

        assert_eq!(requirements.len(), 1);
        assert_eq!(requirements[0].channel, "reply");
    }
}
