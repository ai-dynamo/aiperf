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
//!
//! No runtime node kind, edge kind, reducer, channel type, or reactive branch
//! machinery is introduced. See `specs/conditional-graph-lowering.md`.

mod fold;
mod model;
mod resolve;

use std::fs;
use std::sync::Arc;

use crate::dataset::{DatasetSource, LoadConfig, SegmentPool, TextTokenizer};
use crate::graph::input::{GraphInputBundle, GraphInputConfig, GraphInputMetadata};
use crate::graph::model::{GraphTracePlan, TraceRecord};

pub use fold::{CompiledPrompts, FoldedTrace, compile_prompts, fold_replay_and_emit};
pub use model::{
    AuthoredChannelSpec, AuthoredConditionalEdge, AuthoredEdge, AuthoredGraph, AuthoredGraphDoc,
    AuthoredLlmNode, AuthoredNode, AuthoredReplayNode, AuthoredStaticEdge, AuthoredTrace,
    BranchTargets, ConditionalError, MessagePart, PromptGrammarItem, parse_authored_graph,
};
pub use resolve::{TakenEdge, TakenGraph, resolve_and_prune, resolve_branch_key};

/// Read one authored conditional-graph source as raw bytes (YAML or JSON).
fn load_source_bytes(load: &LoadConfig) -> Result<Vec<u8>, ConditionalError> {
    match &load.source {
        DatasetSource::Path(path) if path.is_file() => {
            fs::read(path).map_err(|error| ConditionalError(format!("{}: {error}", path.display())))
        }
        DatasetSource::Path(path) => Err(ConditionalError(format!(
            "conditional_graph source {} is not a readable file",
            path.display()
        ))),
        DatasetSource::Bytes(bytes) => Ok(bytes.to_vec()),
        DatasetSource::Inline(value) => serde_json::to_vec(value).map_err(ConditionalError::from),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => Err(ConditionalError(
            "conditional_graph does not support URL or HuggingFace sources".into(),
        )),
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
    if config.root_limit == Some(0) {
        return Err(ConditionalError(
            "graph root_limit must be positive when configured".into(),
        ));
    }
    let bytes = load_source_bytes(&config.load)?;
    let doc = parse_authored_graph(&bytes)?;
    if doc.traces.is_empty() {
        return Err(ConditionalError(
            "conditional_graph source declares no traces".into(),
        ));
    }

    // Compile every authored prompt once into one shared segment pool.
    let mut pool = SegmentPool::new();
    let prompts = compile_prompts(&doc.graph, &mut pool, tokenizer)?;

    let selected = match config.root_limit {
        Some(limit) => &doc.traces[..limit.min(doc.traces.len())],
        None => &doc.traces[..],
    };

    let mut plans = Vec::with_capacity(selected.len());
    for trace in selected {
        let taken = resolve_and_prune(&doc.graph, trace, workload_seed)?;
        let folded = fold_replay_and_emit(&taken, trace, &doc.graph.state, &prompts)?;
        let arrival_offset_ns = trace
            .arrival_time
            .map(|seconds| (seconds * 1_000_000_000.0) as i64);
        plans.push(GraphTracePlan {
            graph: folded.graph,
            trace: TraceRecord {
                id: trace.id.clone(),
                graph_ref: Some(trace.id.clone()),
                initial_state: folded.initial_state,
            },
            arrival_offset_ns,
        });
    }

    let node_count = plans.iter().map(|plan| plan.graph.nodes.len()).sum();
    Ok(GraphInputBundle {
        plans,
        segments: Arc::new(pool.freeze()),
        metadata: GraphInputMetadata {
            format: "conditional_graph".to_string(),
            root_count: selected.len(),
            node_count,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetSource, TiktokenTokenizer};
    use std::collections::BTreeSet;

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

    fn nodes_of(bundle: &GraphInputBundle, trace_id: &str) -> BTreeSet<String> {
        bundle
            .plans
            .iter()
            .find(|plan| plan.trace.id == trace_id)
            .unwrap()
            .graph
            .nodes
            .keys()
            .cloned()
            .collect()
    }

    #[tokio::test]
    async fn compiles_each_branch_to_its_pruned_flat_graph() {
        let bundle = compile().await;
        assert_eq!(bundle.metadata.format, "conditional_graph");
        assert_eq!(bundle.plans.len(), 3);

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
            ["brandmap", "plan", "redirect", "route", "safety", "summarize"]
                .into_iter()
                .map(String::from)
                .collect()
        );

        // Replay outputs fold into the shopping trace's seeded channel state.
        let shopping = bundle
            .plans
            .iter()
            .find(|plan| plan.trace.id == "t-shopping")
            .unwrap();
        assert!(shopping.trace.initial_state.contains_key("brand_cands"));
        assert!(shopping.trace.initial_state.contains_key("preprocessed"));
    }

    // Adapter registration + end-to-end routing of `conditional_graph` through
    // the resolver is proven by the product e2e (`rust/e2e/tests/
    // test_conditional_graph.rs`), which drives the real `aiperf` binary. A unit
    // test of the resolver here would require compiling the `engine`-feature lib
    // test target, which is independently broken at this base commit.
}
