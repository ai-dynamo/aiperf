// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authored conditional-graph compiler for the flat Graph-IR.
//!
//! This compiler ingests an authored graph whose edges may carry
//! model-independent conditional branches and whose nodes may be non-dispatching
//! *replay* nodes, and lowers it — per trace — into the flat `LlmNode` /
//! `StaticEdge` substrate the runtime executes. Branch keys resolve from
//! pre-execution data only (pinned `selected_branches`, per-trace distributions,
//! or static-seed `branch_weights`); the taken subgraph is pruned; recorded
//! replay outputs fold into `TraceRecord.initial_state`; and one validated
//! `GraphRecord` is emitted into `parsed.graphs[trace.id]`.
//! Completion-anchored edge delays and replay durations compose across folded
//! replay paths. Replay paths reject minimum-start, predecessor-start, and
//! predecessor-first-token anchors because the flat edge has no equivalent
//! event sequence; every authored timing value must be finite and fit signed
//! `i64` nanoseconds after applying its declared unit.
//!
//! No runtime node kind, edge kind, reducer, channel type, or reactive branch
//! machinery is introduced.

mod fold;
mod model;
mod resolve;

use std::fs;
use std::sync::Arc;

use crate::dataset::{DatasetSource, LoadConfig, SegmentPool, TextTokenizer};
use crate::graph::input::{GraphInputBundle, GraphInputConfig, GraphInputMetadata};
use crate::graph::model::{GraphTracePlan, GraphTraceProgram, TraceRecord};

pub use fold::{CompiledPrompts, FoldedTrace, compile_prompts, fold_replay_and_emit};
pub use model::{
    AuthoredChannelSpec, AuthoredConditionalEdge, AuthoredEdge, AuthoredGraph, AuthoredGraphDoc,
    AuthoredLlmNode, AuthoredNode, AuthoredReplayNode, AuthoredStaticEdge, AuthoredTrace,
    BranchTargets, ConditionalError, MessagePart, PromptGrammarItem, parse_authored_graph,
};
pub use resolve::{TakenEdge, TakenGraph, resolve_and_prune, resolve_branch_key};

/// Internal classified conditional-graph input error for engine fatal handling.
#[derive(Debug)]
pub(crate) enum ConditionalGraphInputError {
    Decode(serde_yaml::Error),
    Conditional(ConditionalError),
}

impl std::fmt::Display for ConditionalGraphInputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(error) => error.fmt(formatter),
            Self::Conditional(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ConditionalGraphInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Conditional(error) => Some(error),
        }
    }
}

impl From<ConditionalError> for ConditionalGraphInputError {
    fn from(error: ConditionalError) -> Self {
        Self::Conditional(error)
    }
}

impl ConditionalGraphInputError {
    fn into_compat(self) -> ConditionalError {
        match self {
            Self::Decode(error) => error.into(),
            Self::Conditional(error) => error,
        }
    }
}

/// Read one authored conditional-graph source as raw bytes (YAML or JSON).
fn load_source_bytes(load: &LoadConfig) -> Result<Vec<u8>, ConditionalError> {
    match &load.source {
        DatasetSource::Path(path) if path.is_file() => fs::read(path)
            .map_err(|error| ConditionalError::message(format!("{}: {error}", path.display()))),
        DatasetSource::Path(path) => Err(ConditionalError::message(format!(
            "conditional_graph source {} is not a readable file",
            path.display()
        ))),
        DatasetSource::Bytes(bytes) => Ok(bytes.to_vec()),
        DatasetSource::Inline(value) => serde_json::to_vec(value).map_err(ConditionalError::from),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => {
            Err(ConditionalError::message(
                "conditional_graph does not support URL or HuggingFace sources",
            ))
        }
    }
}

/// Parse, resolve, prune, fold, and lower one authored conditional-graph source
/// into per-trace flat Graph-IR plans plus one frozen segment store.
///
/// Every conditional edge is resolved from pre-execution data with
/// `workload_seed` (used only when a trace samples a weighted branch), the taken
/// subgraph is pruned, recorded replay outputs fold into each trace's
/// `initial_state`, and one validated flat `GraphRecord` is emitted per trace.
pub async fn compile_conditional_graph_input(
    config: GraphInputConfig,
    tokenizer: &dyn TextTokenizer,
    workload_seed: u64,
) -> Result<GraphInputBundle, ConditionalError> {
    compile_conditional_graph_input_classified(config, tokenizer, workload_seed)
        .await
        .map_err(ConditionalGraphInputError::into_compat)
}

pub(crate) async fn compile_conditional_graph_input_classified(
    config: GraphInputConfig,
    tokenizer: &dyn TextTokenizer,
    workload_seed: u64,
) -> Result<GraphInputBundle, ConditionalGraphInputError> {
    if config.root_limit == Some(0) {
        return Err(
            ConditionalError::message("graph root_limit must be positive when configured").into(),
        );
    }
    let bytes = load_source_bytes(&config.load)?;
    let raw = model::decode_authored_graph(&bytes).map_err(ConditionalGraphInputError::Decode)?;
    let doc = model::convert_authored_graph(raw)?;
    compile_decoded_conditional_graph_input(config, doc, tokenizer, workload_seed)
        .await
        .map_err(ConditionalGraphInputError::from)
}

async fn compile_decoded_conditional_graph_input(
    config: GraphInputConfig,
    doc: AuthoredGraphDoc,
    tokenizer: &dyn TextTokenizer,
    workload_seed: u64,
) -> Result<GraphInputBundle, ConditionalError> {
    if doc.traces.is_empty() {
        return Err(ConditionalError::message(
            "conditional_graph source declares no traces",
        ));
    }

    // Compile every authored prompt once into one shared segment pool.
    let mut pool = SegmentPool::new();
    let prompts = compile_prompts(&doc.graph, &mut pool, tokenizer)?;

    let selected = match config.root_limit {
        Some(limit) => &doc.traces[..limit.min(doc.traces.len())],
        None => &doc.traces[..],
    };

    let mut programs = Vec::with_capacity(selected.len());
    for trace in selected {
        let taken = resolve_and_prune(&doc.graph, trace, workload_seed)?;
        let folded = fold_replay_and_emit(&taken, trace, &doc.graph.state, &prompts)?;
        let arrival_offset_ns = trace
            .arrival_time
            .map(|seconds| (seconds * 1_000_000_000.0) as i64);
        programs.push(GraphTraceProgram::static_graph(GraphTracePlan {
            graph: folded.graph,
            trace: TraceRecord {
                id: trace.id.clone(),
                graph_ref: Some(trace.id.clone()),
                initial_state: folded.initial_state,
            },
            arrival_offset_ns,
        }));
    }

    let node_count = programs
        .iter()
        .map(|program| program.profiling.graph.llm_node_count())
        .sum();
    Ok(GraphInputBundle {
        programs,
        segments: Arc::new(pool.freeze()),
        metadata: GraphInputMetadata {
            format: "conditional_graph".to_string(),
            root_count: selected.len(),
            node_count,
            warning_facts: Vec::new(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetSource, TiktokenTokenizer};
    use std::collections::BTreeSet;
    use std::error::Error;

    // A shopping-agent diamond: START fans to `route` (shopping/non_shopping) and
    // `safety` (safe/unsafe). The shopping path chains an LLM planner through two
    // recorded replay nodes to a terminal summarizer; the unsafe path adds a
    // redirect. Three traces pin the three branch combinations.
    const FIXTURE: &str = r#"
graph:
  state:
    messages:      {type: messages, reducer: add_messages}
    intent:        {type: text}
    plan:          {type: text}
    raw_results:   {type: json}
    brand_cands:   {type: text}
    brand_mapping: {type: text}
    preprocessed:  {type: text}
    summary:       {type: text}
    safety:        {type: text}
    redirect:      {type: text}
  nodes:
    route:     {node_type: llm, prompt: [{role: system, content: "Classify."}, "@messages"], output: intent, streaming: false}
    plan:      {node_type: llm, prompt: [{role: system, content: "Plan."}], output: plan}
    tool_exec: {node_type: replay, outputs: [raw_results, brand_cands], duration_ms: 80}
    brandmap:  {node_type: llm, prompt: [{role: user, content: ["@brand_cands"]}], output: brand_mapping}
    preprocess: {node_type: replay, outputs: [preprocessed], duration_ms: 5}
    summarize: {node_type: llm, prompt: [{role: user, content: ["@preprocessed"]}], output: summary, terminal_for_user: true}
    safety:    {node_type: llm, prompt: [{role: system, content: "Safe?"}], output: safety, streaming: false}
    redirect:  {node_type: llm, prompt: [{role: system, content: "Redirect."}], output: redirect, terminal_for_user: true}
  edges:
    - {source: START, target: route}
    - {source: START, target: safety}
    - {source: route, branches: {shopping: plan, non_shopping: END}}
    - {source: plan, target: tool_exec}
    - {source: tool_exec, target: brandmap}
    - {source: brandmap, target: preprocess}
    - {source: preprocess, target: summarize}
    - {source: safety, branches: {safe: END, unsafe: redirect}}
traces:
  - id: t-shopping
    selected_branches: {route: shopping, safety: safe}
    initial_state: {messages: []}
    replay_outputs:
      tool_exec: {raw_results: [], brand_cands: "Nike, Adidas"}
      preprocess: {preprocessed: "Top hits"}
  - id: t-non-shopping
    selected_branches: {route: non_shopping, safety: safe}
    initial_state: {messages: []}
  - id: t-unsafe
    selected_branches: {route: shopping, safety: unsafe}
    initial_state: {messages: []}
    replay_outputs:
      tool_exec: {raw_results: [], brand_cands: "Nike, Adidas"}
      preprocess: {preprocessed: "Top hits"}
"#;

    async fn compile() -> GraphInputBundle {
        let config = GraphInputConfig {
            load: LoadConfig::new(DatasetSource::Bytes(FIXTURE.as_bytes().to_vec().into())),
            root_limit: None,
        };
        compile_conditional_graph_input(config, &TiktokenTokenizer::builtin(), 0)
            .await
            .unwrap()
    }

    fn config(source: &[u8]) -> GraphInputConfig {
        GraphInputConfig {
            load: LoadConfig::new(DatasetSource::Bytes(source.to_vec().into())),
            root_limit: None,
        }
    }

    #[tokio::test]
    async fn selected_conditional_cycle_is_rejected_by_execution_finalization() {
        let source = br#"
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
"#;
        let bundle =
            compile_conditional_graph_input(config(source), &TiktokenTokenizer::builtin(), 0)
                .await
                .expect("inspection lowering retains the selected cycle");
        let Err(error) = crate::graph::input::validate_lowered_bundle(bundle) else {
            panic!("selected cycle must not leave execution finalization");
        };
        assert!(error.to_string().contains("graph-cycle"));
    }

    #[tokio::test]
    async fn classified_error_compatibility_preserves_yaml_decode_source() {
        let error = compile_conditional_graph_input_classified(
            config(b"traces: ["),
            &TiktokenTokenizer::builtin(),
            0,
        )
        .await;
        let Err(error) = error else {
            panic!("malformed YAML must fail decoding");
        };

        assert!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<serde_yaml::Error>())
                .is_some()
        );
    }

    #[tokio::test]
    async fn classified_error_compatibility_keeps_semantic_failure_out_of_yaml_chain() {
        let error = compile_conditional_graph_input_classified(
            config(
                br#"
graph:
  state:
    result: {type: text, reducer: unsupported}
  nodes: {}
  edges: []
traces: []
"#,
            ),
            &TiktokenTokenizer::builtin(),
            0,
        )
        .await;
        let Err(error) = error else {
            panic!("unsupported reducer must fail conversion");
        };

        assert!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<serde_yaml::Error>())
                .is_none()
        );
    }

    fn nodes_of(bundle: &GraphInputBundle, trace_id: &str) -> BTreeSet<String> {
        bundle
            .programs
            .iter()
            .find(|program| program.profiling.trace.id == trace_id)
            .unwrap()
            .profiling
            .graph
            .nodes
            .keys()
            .cloned()
            .collect()
    }

    #[tokio::test]
    async fn static_channel_readiness_preserves_authored_inputs_through_lowering() {
        let bundle = compile_conditional_graph_input(
            config(
                br#"
graph:
  state:
    produced: {type: messages, reducer: add_messages}
    done: {type: messages, reducer: add_messages}
  nodes:
    producer:
      node_type: llm
      prompt: ["producer"]
      output: produced
    reader:
      node_type: llm
      prompt: ["reader"]
      output: done
      inputs: [{channel: produced, count: 1}]
  edges:
    - {source: START, target: producer}
    - {source: producer, target: reader}
traces:
  - id: authored-inputs
"#,
            ),
            &TiktokenTokenizer::builtin(),
            0,
        )
        .await
        .expect("authored channel inputs lower successfully");

        let node = bundle.programs[0]
            .profiling
            .graph
            .nodes
            .get("reader")
            .expect("reader node");
        assert_eq!(node.input_requirements().len(), 1);
        assert_eq!(node.input_requirements()[0].channel, "produced");
        assert_eq!(node.input_requirements()[0].count.as_int(), Some(1));
    }

    #[tokio::test]
    async fn replay_inputs_are_rejected_before_folding() {
        let result = compile_conditional_graph_input(
            config(
                br#"
graph:
  state:
    produced: {type: messages, reducer: add_messages}
  nodes:
    replay:
      node_type: replay
      outputs: [produced]
      inputs: [{channel: produced, count: 1}]
  edges:
    - {source: START, target: replay}
traces:
  - id: replay-inputs
    replay_outputs: {replay: {produced: []}}
"#,
            ),
            &TiktokenTokenizer::builtin(),
            0,
        )
        .await;
        let Err(error) = result else {
            panic!("replay inputs must not be silently dropped by folding")
        };

        assert!(error.to_string().contains("replay") && error.to_string().contains("inputs"));
    }

    #[tokio::test]
    async fn compiles_each_branch_to_its_pruned_flat_graph() {
        let bundle = compile().await;
        assert_eq!(bundle.metadata.format, "conditional_graph");
        assert_eq!(bundle.programs.len(), 3);

        // shopping: route/plan/brandmap/summarize + safety (tool_exec/preprocess folded out).
        assert_eq!(
            nodes_of(&bundle, "t-shopping"),
            ["brandmap", "plan", "route", "safety", "summarize"]
                .into_iter()
                .map(String::from)
                .collect()
        );
        // non_shopping: only route + safety fire.
        assert_eq!(
            nodes_of(&bundle, "t-non-shopping"),
            ["route", "safety"].into_iter().map(String::from).collect()
        );
        // unsafe: shopping path + redirect (dual terminal).
        assert_eq!(
            nodes_of(&bundle, "t-unsafe"),
            [
                "brandmap",
                "plan",
                "redirect",
                "route",
                "safety",
                "summarize"
            ]
            .into_iter()
            .map(String::from)
            .collect()
        );

        // Replay outputs fold into the shopping trace's seeded channel state.
        let shopping = bundle
            .programs
            .iter()
            .find(|program| program.profiling.trace.id == "t-shopping")
            .unwrap();
        assert!(
            shopping
                .profiling
                .trace
                .initial_state
                .contains_key("brand_cands")
        );
        assert!(
            shopping
                .profiling
                .trace
                .initial_state
                .contains_key("preprocessed")
        );
    }

    // Adapter registration + end-to-end routing of `conditional_graph` through
    // the resolver is proven by the product e2e (`rust/e2e-tests/tests/
    // test_conditional_graph.rs`), which drives the real `aiperf` binary. A unit
    // test of the resolver here would require compiling the `engine`-feature lib
    // test target, which is independently broken at this base commit.
}
