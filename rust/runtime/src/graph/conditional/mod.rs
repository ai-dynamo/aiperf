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

pub use model::{
    AuthoredChannelSpec, AuthoredConditionalEdge, AuthoredEdge, AuthoredGraph, AuthoredGraphDoc,
    AuthoredLlmNode, AuthoredNode, AuthoredReplayNode, AuthoredStaticEdge, AuthoredTrace,
    BranchTargets, ConditionalError, MessagePart, PromptGrammarItem, parse_authored_graph,
};
pub use fold::{CompiledPrompts, FoldedTrace, compile_prompts, fold_replay_and_emit};
pub use resolve::{TakenEdge, TakenGraph, resolve_and_prune, resolve_branch_key};
