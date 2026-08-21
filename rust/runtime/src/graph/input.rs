// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical direct authored-workload compilation for Graph-IR.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

use crate::dataset::{
    ConversationBranchMode, ConversationContextMode, DispatchTiming, LoadConfig, PrerequisiteKind,
    SegmentPool, SegmentStore, TextTokenizer,
};
use bytes::Bytes;

use crate::graph::dag_source::{
    DagJsonlProgram, dag_jsonl_turn_token_counts, load_dag_jsonl_program,
};
use crate::graph::lowering::{
    CatalogBranch, CatalogConversation, CatalogPrerequisite, CatalogTurn, GraphCatalog,
    lower_catalog,
};
use crate::graph::model::{GraphTracePlan, GraphTraceProgram};
use crate::graph::validate::{find_graph_cycle, graph_cycle_message, validate_detailed};

/// Inputs supplied to one format-specific Graph-IR compiler.
pub struct GraphInputConfig {
    /// Resolved source, row bounds, fetcher, and format options.
    pub load: LoadConfig,
    /// Maximum complete root traces retained after full-program validation.
    pub root_limit: Option<usize>,
}

/// Static facts produced beside the executable trace plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInputMetadata {
    /// Stable authored format name.
    pub format: String,
    /// Authored root trace count.
    pub root_count: usize,
    /// Total nodes across root-expanded plans.
    pub node_count: usize,
    /// Deterministic non-fatal facts produced while lowering this input.
    pub warning_facts: Vec<GraphInputWarning>,
}

/// One structured non-fatal fact emitted by a graph-input adapter boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphInputWarning {
    /// Stable machine-readable warning code.
    pub code: String,
    /// Deterministic named warning context.
    pub context: BTreeMap<String, String>,
}

impl GraphInputWarning {
    /// Construct one warning fact from a stable code and named context.
    pub fn new(code: impl Into<String>, context: BTreeMap<String, String>) -> Self {
        Self {
            code: code.into(),
            context,
        }
    }
}

/// Canonical result of one direct graph-input pass.
pub struct GraphInputBundle {
    /// Complete, owned root-trace commands in authored order.
    pub programs: Vec<GraphTraceProgram>,
    /// Immutable content-addressed segment arena referenced by plan handles.
    pub segments: Arc<dyn SegmentStore>,
    /// Static load facts for reporting and validation.
    pub metadata: GraphInputMetadata,
}

/// Refuse lowered plans with cycles or unsafe timing before execution.
pub fn validate_lowered_bundle(bundle: GraphInputBundle) -> Result<GraphInputBundle, String> {
    validate_bundle_with_issues(
        bundle,
        &["graph-cycle", "non-finite-timing", "out-of-range-timing"],
    )
}

/// Refuse timing values that cannot safely cross an inspection serialization boundary.
///
/// Cycles intentionally remain inspectable so the local graph commands can report
/// their detailed structural finding.
pub(crate) fn validate_inspection_bundle(
    bundle: GraphInputBundle,
) -> Result<GraphInputBundle, String> {
    validate_bundle_with_issues(bundle, &["non-finite-timing", "out-of-range-timing"])
}

fn validate_bundle_with_issues(
    bundle: GraphInputBundle,
    refused_codes: &[&str],
) -> Result<GraphInputBundle, String> {
    for program in &bundle.programs {
        for plan in std::iter::once(&program.profiling).chain(program.warmup.as_ref()) {
            if let Some(issue) = validate_detailed(&plan.graph)
                .into_iter()
                .find(|issue| refused_codes.contains(&issue.code.as_str()))
            {
                return Err(issue.message);
            }
        }
    }
    Ok(bundle)
}

/// Parse, validate, intern, and lower one complete `dag_jsonl` source.
pub async fn compile_dag_jsonl_input(
    config: GraphInputConfig,
    tokenizer: &dyn TextTokenizer,
) -> Result<GraphInputBundle, GraphInputError> {
    for name in config.load.options.keys() {
        if name != "inter_turn_delay_cap_seconds" {
            return Err(GraphInputError(format!(
                "dag_jsonl Graph-IR input does not support loader option {name:?}"
            )));
        }
    }
    let delay_cap_ms = config
        .load
        .options
        .get("inter_turn_delay_cap_seconds")
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite() && *value >= 0.0)
                .map(|seconds| seconds * 1000.0)
                .ok_or_else(|| {
                    GraphInputError(
                        "inter_turn_delay_cap_seconds must be finite and non-negative".into(),
                    )
                })
        })
        .transpose()?;
    let program = load_dag_jsonl_program(&config.load)
        .await
        .map_err(|error| GraphInputError(error.to_string()))?;
    let mut catalog = catalog_from_program(program, delay_cap_ms, tokenizer)?;
    if config.root_limit == Some(0) {
        return Err(GraphInputError(
            "graph root_limit must be positive when configured".into(),
        ));
    }
    if let Some(limit) = config.root_limit {
        // Keep the complete child catalog but avoid compiling GraphRecords
        // for authored roots that cannot be selected by this run.
        catalog.roots.truncate(limit);
    }
    let lowered = lower_catalog(&catalog).map_err(|error| GraphInputError(error.to_string()))?;
    let programs = lowered
        .parsed
        .traces
        .iter()
        .map(|trace| {
            GraphTraceProgram::static_graph(GraphTracePlan {
                graph: lowered.parsed.resolve_trace_graph(trace).clone(),
                trace: trace.clone(),
                arrival_offset_ns: None,
            })
        })
        .collect::<Vec<_>>();
    let metadata = GraphInputMetadata {
        format: "dag_jsonl".to_string(),
        root_count: programs.len(),
        node_count: programs
            .iter()
            .map(|program| program.profiling.graph.llm_node_count())
            .sum(),
        warning_facts: Vec::new(),
    };
    Ok(GraphInputBundle {
        programs,
        segments: lowered.segments,
        metadata,
    })
}

fn catalog_from_program(
    program: DagJsonlProgram,
    delay_cap_ms: Option<f64>,
    tokenizer: &dyn TextTokenizer,
) -> Result<GraphCatalog, GraphInputError> {
    validate_program_topology(&program)?;
    let referenced = program
        .conversations
        .iter()
        .flat_map(|conversation| {
            conversation.pre_session_spawns.iter().cloned().chain(
                conversation.turns.iter().flat_map(|turn| {
                    turn.forks.iter().map(|fork| fork.child.clone()).chain(
                        turn.spawns
                            .iter()
                            .flat_map(|spawn| spawn.children.iter().cloned()),
                    )
                }),
            )
        })
        .collect::<HashSet<_>>();
    let roots = program
        .conversations
        .iter()
        .filter(|conversation| !referenced.contains(&conversation.session_id))
        .map(|conversation| conversation.session_id.clone())
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(GraphInputError(
            "dag_jsonl source contains no unreferenced root conversations".into(),
        ));
    }

    let mut pool = SegmentPool::new();
    let mut conversations = HashMap::with_capacity(program.conversations.len());
    for conversation in program.conversations {
        let id = conversation.session_id;
        let mut parent = None;
        let mut branches = Vec::new();
        let mut turns = Vec::with_capacity(conversation.turns.len());
        let mut pending = vec![Vec::<CatalogPrerequisite>::new(); conversation.turns.len()];
        for (turn_index, authored) in conversation.turns.into_iter().enumerate() {
            let (input_tokens, _) = dag_jsonl_turn_token_counts(
                &authored.messages,
                authored.tools.as_deref(),
                tokenizer,
            )
            .map_err(|error| GraphInputError(error.to_string()))?;
            let raw_messages = pool
                .intern_raw(parent, authored.messages.clone())
                .map_err(|error| GraphInputError(error.to_string()))?;
            parent = Some(raw_messages);
            let tools = intern_optional(&mut pool, &mut parent, authored.tools)?;
            let raw_system = intern_optional(&mut pool, &mut parent, authored.raw_system)?;
            let extra_body = intern_without_parent_update(&mut pool, parent, authored.extra)?;
            let extra_headers =
                intern_without_parent_update(&mut pool, parent, authored.extra_headers)?;
            let request_parameters =
                intern_without_parent_update(&mut pool, parent, authored.request_parameters)?;
            let mixed = !authored.forks.is_empty() && !authored.spawns.is_empty();
            let split_forks = authored.forks.iter().any(|fork| fork.background)
                && authored.forks.iter().any(|fork| !fork.background);
            let mut branch_ids = Vec::new();
            for (background, suffix) in [(false, "fork"), (true, "bg_fork")] {
                let children = authored
                    .forks
                    .iter()
                    .filter(|fork| fork.background == background)
                    .map(|fork| fork.child.clone())
                    .collect::<Vec<_>>();
                if children.is_empty() {
                    continue;
                }
                let branch_id =
                    branch_id(&id, turn_index, (mixed || split_forks).then_some(suffix));
                branch_ids.push(branch_id.clone());
                branches.push(CatalogBranch {
                    id: branch_id,
                    children,
                    mode: ConversationBranchMode::Fork,
                    dispatch_timing: DispatchTiming::Post,
                    background,
                });
            }
            let spawn_count = authored.spawns.len();
            for (group_index, spawn) in authored.spawns.into_iter().enumerate() {
                let suffix = (mixed || spawn_count > 1).then(|| {
                    if group_index == 0 {
                        "spawn".to_string()
                    } else {
                        format!("spawn{group_index}")
                    }
                });
                let branch_id = branch_id(&id, turn_index, suffix.as_deref());
                branch_ids.push(branch_id.clone());
                branches.push(CatalogBranch {
                    id: branch_id.clone(),
                    children: spawn.children,
                    mode: ConversationBranchMode::Spawn,
                    dispatch_timing: DispatchTiming::Post,
                    background: false,
                });
                let join_at = spawn.join_at.unwrap_or(turn_index + 1);
                if join_at < pending.len() {
                    pending[join_at].push(CatalogPrerequisite {
                        kind: PrerequisiteKind::SpawnJoin,
                        branch_id: Some(branch_id),
                        child_ids: Vec::new(),
                    });
                }
            }
            turns.push(CatalogTurn {
                messages: Vec::new(),
                raw_messages: Some(raw_messages),
                raw_payload: false,
                content: Vec::new(),
                role: None,
                model: authored.model,
                endpoint: authored.endpoint,
                streaming: authored.streaming,
                max_tokens: authored.max_tokens,
                input_tokens,
                timestamp_ms: None,
                delay_ms: Some(
                    delay_cap_ms.map_or(authored.delay_ms, |cap| authored.delay_ms.min(cap)),
                ),
                tools,
                raw_system,
                extra_body,
                extra_headers,
                request_parameters,
                branch_ids,
                prerequisites: std::mem::take(&mut pending[turn_index]),
            });
        }
        if !conversation.pre_session_spawns.is_empty() {
            let branch_id = format!("{id}:pre");
            turns[0].branch_ids.push(branch_id.clone());
            branches.push(CatalogBranch {
                id: branch_id,
                children: conversation.pre_session_spawns,
                mode: ConversationBranchMode::Spawn,
                dispatch_timing: DispatchTiming::Pre,
                background: false,
            });
        }
        conversations.insert(
            id.clone(),
            CatalogConversation {
                id,
                context_mode: ConversationContextMode::DeltasWithoutResponses,
                system: None,
                user_context: None,
                turns,
                branches,
            },
        );
    }
    Ok(GraphCatalog {
        conversations,
        roots,
        segments: Arc::new(pool.freeze()),
    })
}

fn intern_optional(
    pool: &mut SegmentPool,
    parent: &mut Option<crate::dataset::Handle>,
    wire: Option<Bytes>,
) -> Result<Option<crate::dataset::Handle>, GraphInputError> {
    let handle = wire
        .map(|wire| pool.intern_raw(*parent, wire))
        .transpose()
        .map_err(|error| GraphInputError(error.to_string()))?;
    if handle.is_some() {
        *parent = handle;
    }
    Ok(handle)
}

fn intern_without_parent_update(
    pool: &mut SegmentPool,
    parent: Option<crate::dataset::Handle>,
    wire: Option<Bytes>,
) -> Result<Option<crate::dataset::Handle>, GraphInputError> {
    wire.map(|wire| pool.intern_raw(parent, wire))
        .transpose()
        .map_err(|error| GraphInputError(error.to_string()))
}

fn validate_program_topology(program: &DagJsonlProgram) -> Result<(), GraphInputError> {
    let ids = program
        .conversations
        .iter()
        .map(|conversation| conversation.session_id.as_str())
        .collect::<HashSet<_>>();
    let mut edges = Vec::new();
    let mut fork_parent = HashMap::<String, String>::new();
    let mut pre_spawns = HashSet::<String>::new();
    for conversation in &program.conversations {
        let mut children = conversation.pre_session_spawns.clone();
        for child in &conversation.pre_session_spawns {
            pre_spawns.insert(child.clone());
        }
        for turn in &conversation.turns {
            for fork in &turn.forks {
                if let Some(previous) =
                    fork_parent.insert(fork.child.clone(), conversation.session_id.clone())
                {
                    return Err(GraphInputError(format!(
                        "DAG child {:?} has multiple fork parents {:?} and {:?}",
                        fork.child, previous, conversation.session_id
                    )));
                }
                children.push(fork.child.clone());
            }
            children.extend(
                turn.spawns
                    .iter()
                    .flat_map(|spawn| spawn.children.iter().cloned()),
            );
        }
        for child in &children {
            if !ids.contains(child.as_str()) {
                return Err(GraphInputError(format!(
                    "DAG session {:?} references unknown child {:?}",
                    conversation.session_id, child
                )));
            }
        }
        edges.extend(
            children
                .into_iter()
                .map(|child| (conversation.session_id.clone(), child)),
        );
    }
    if let Some(child) = pre_spawns
        .iter()
        .find(|child| fork_parent.contains_key(*child))
    {
        return Err(GraphInputError(format!(
            "DAG child {child:?} is both a pre-session spawn and a fork target"
        )));
    }
    let cycle_nodes = program
        .conversations
        .iter()
        .map(|conversation| conversation.session_id.clone())
        .collect();
    if let Some(cycle) = find_graph_cycle(&cycle_nodes, &edges) {
        return Err(GraphInputError(graph_cycle_message(&cycle)));
    }
    for conversation in &program.conversations {
        for (turn_index, turn) in conversation.turns.iter().enumerate() {
            if (turn_index > 0 || fork_parent.contains_key(&conversation.session_id))
                && messages_have_system(&turn.messages)?
            {
                return Err(GraphInputError(format!(
                    "DAG session {:?} turn {turn_index} contains a non-root system message",
                    conversation.session_id
                )));
            }
        }
    }
    Ok(())
}

fn messages_have_system(wire: &[u8]) -> Result<bool, GraphInputError> {
    let messages: serde_json::Value =
        serde_json::from_slice(wire).map_err(|error| GraphInputError(error.to_string()))?;
    Ok(messages.as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message.get("role").and_then(serde_json::Value::as_str) == Some("system")
        })
    }))
}

fn branch_id(session: &str, turn: usize, suffix: Option<&str>) -> String {
    suffix.map_or_else(
        || format!("{session}:{turn}"),
        |suffix| format!("{session}:{turn}:{suffix}"),
    )
}

/// Direct graph-input construction failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInputError(pub String);

impl Display for GraphInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for GraphInputError {}

#[cfg(test)]
mod tests {
    use crate::dataset::{DatasetSource, TiktokenTokenizer};
    use serde_json::json;

    use super::*;

    async fn load(value: serde_json::Value) -> Result<GraphInputBundle, GraphInputError> {
        compile_dag_jsonl_input(
            GraphInputConfig {
                load: LoadConfig::new(DatasetSource::Inline(value)),
                root_limit: None,
            },
            &TiktokenTokenizer::builtin(),
        )
        .await
    }

    #[tokio::test]
    async fn direct_compiler_builds_fork_spawn_and_join_without_dataset_composition() {
        let bundle = load(json!([
            {"session_id":"root","turns":[
                {"messages":[{"role":"user","content":"root"}],
                 "forks":[{"child":"fork","background":true}],
                 "spawns":[{"children":["spawn"],"join_at":1}]},
                {"messages":[{"role":"user","content":"joined"}]}
            ]},
            {"session_id":"fork","turns":[{"messages":[{"role":"user","content":"fork"}]}]},
            {"session_id":"spawn","turns":[{"messages":[{"role":"user","content":"spawn"}]}]}
        ]))
        .await
        .unwrap();
        assert_eq!(bundle.metadata.root_count, 1);
        assert_eq!(bundle.metadata.node_count, 4);
        let graph = &bundle.programs[0].profiling.graph;
        let root0 = graph
            .nodes
            .values()
            .find(|node| {
                node.metadata().and_then(|metadata| {
                    metadata
                        .get("conversation_id")
                        .and_then(serde_json::Value::as_str)
                }) == Some("root")
                    && node.metadata().and_then(|metadata| {
                        metadata
                            .get("turn_index")
                            .and_then(serde_json::Value::as_u64)
                    }) == Some(0)
            })
            .and_then(crate::graph::model::ExecutableGraphNode::as_llm)
            .unwrap();
        let fork = graph
            .nodes
            .values()
            .find(|node| {
                node.metadata().and_then(|metadata| {
                    metadata
                        .get("conversation_id")
                        .and_then(serde_json::Value::as_str)
                }) == Some("fork")
            })
            .and_then(crate::graph::model::ExecutableGraphNode::as_llm)
            .unwrap();
        let spawn = graph
            .nodes
            .values()
            .find(|node| {
                node.metadata().and_then(|metadata| {
                    metadata
                        .get("conversation_id")
                        .and_then(serde_json::Value::as_str)
                }) == Some("spawn")
            })
            .and_then(crate::graph::model::ExecutableGraphNode::as_llm)
            .unwrap();
        let joined = graph
            .nodes
            .values()
            .find(|node| {
                node.metadata().and_then(|metadata| {
                    metadata
                        .get("conversation_id")
                        .and_then(serde_json::Value::as_str)
                }) == Some("root")
                    && node.metadata().and_then(|metadata| {
                        metadata
                            .get("turn_index")
                            .and_then(serde_json::Value::as_u64)
                    }) == Some(1)
            })
            .and_then(crate::graph::model::ExecutableGraphNode::as_llm)
            .unwrap();
        assert!(fork.items.iter().any(
            |item| matches!(item, crate::graph::model::PromptItem::Splice { splice } if splice == &root0.output)
        ));
        assert!(
            !spawn
                .items
                .iter()
                .any(|item| matches!(item, crate::graph::model::PromptItem::Splice { .. }))
        );
        assert_eq!(joined.inputs[0].channel, spawn.output);
    }

    #[tokio::test]
    async fn topology_errors_fail_inside_the_direct_compiler() {
        assert!(
            load(json!([{
                "session_id":"root",
                "turns":[{"messages":[{"role":"user"}],"spawns":["missing"]}]
            }]))
            .await
            .is_err()
        );
        assert!(
            load(json!([
                {"session_id":"a","turns":[{"messages":[{"role":"user"}],"spawns":["b"]}]},
                {"session_id":"b","turns":[{"messages":[{"role":"user"}],"spawns":["a"]}]}
            ]))
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn root_limit_is_applied_only_after_complete_program_validation() {
        let bundle = compile_dag_jsonl_input(
            GraphInputConfig {
                load: LoadConfig::new(DatasetSource::Inline(json!([
                    {"session_id":"root","turns":[
                        {"messages":[{"role":"user","content":"root"}],"spawns":["child"]}
                    ]},
                    {"session_id":"child","turns":[
                        {"messages":[{"role":"user","content":"child"}]}
                    ]},
                    {"session_id":"other","turns":[
                        {"messages":[{"role":"user","content":"other"}]}
                    ]}
                ]))),
                root_limit: Some(1),
            },
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
        assert_eq!(bundle.metadata.root_count, 1);
        assert_eq!(bundle.metadata.node_count, 2);
    }
}
