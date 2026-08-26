// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared graph-workload scaffolding for the throughput benchmarks.
//!
//! Builds a multi-turn conversation as a content-addressed segment pool
//! (system + growing user turns, prefix-chained so the static prefix dedups
//! across every trace instance) and the per-node ISL map. The direct raw-HTTP
//! benchmark driver is the `graph-transport-bench`-gated `transport_bench`
//! module; this module remains neutral scaffolding shared with DynoSim.

use std::collections::HashMap;

use crate::dataset::TiktokenTokenizer;
use crate::graph::model::GraphRecord;
use crate::graph::segment::{InMemorySegmentStore, SegmentPool, intern_message};
use crate::graph::wire::OpenAiChatMessage as Msg;

/// Benchmark configuration.
pub struct BenchConfig {
    /// One or more server base URLs; workers round-robin across them so a
    /// high-concurrency run fans connections out under each server's accept
    /// backlog (and each server's own ephemeral-port space).
    pub base_urls: Vec<String>,
    pub model: String,
    /// Turns per conversation (nodes per trace DAG).
    pub turns: usize,
    /// Total trace instances (conversations) to run.
    pub instances: usize,
    /// OS worker threads (thread-per-core).
    pub workers: usize,
    /// Concurrent traces in flight per worker.
    pub concurrency: usize,
    /// Requested output tokens per node.
    pub max_tokens: usize,
    /// Global cap on concurrent in-flight requests across all workers/lanes.
    /// `None` = unbounded (limited only by trace-lane concurrency + fan-out).
    pub request_concurrency: Option<usize>,
    /// Global cap on concurrent requests in the prefill phase (dispatched but
    /// not yet first-token). `None` = unbounded.
    pub prefill_concurrency: Option<usize>,
    /// Optional wall-time bound in nanoseconds. When set, each worker stops
    /// pulling new trace instances once its clock passes `started + this`, so the
    /// run ends on `min(instances exhausted, duration elapsed)`. `None` = the run
    /// is bounded only by `instances`. Realized via the shared
    /// [`crate::timing::StopChecker`] duration condition (one per worker), so the
    /// online CLI and the graph path share one stop policy.
    pub max_duration_ns: Option<i64>,
}

/// A rough token estimate for a static segment's content (no tokenizer needed on
/// the hot path; segment content is fixed, so this is computed once at setup).
fn est_tokens(s: &str) -> usize {
    (s.len() / 4).max(1)
}

/// Build the segment pool + chain graph for a `turns`-turn conversation, and the
/// per-node ISL map. The static prefix (system + user turns) is content-addressed
/// and shared across every trace instance.
pub fn build_workload(turns: usize) -> (InMemorySegmentStore, GraphRecord, HashMap<String, usize>) {
    let tokenizer = TiktokenTokenizer::builtin();
    let mut pool = SegmentPool::new();
    let sys_text = "You are a helpful, concise assistant answering benchmark turns.";
    let sys = intern_message(&mut pool, &Msg::new("system", sys_text), None, &tokenizer)
        .expect("benchmark system message must intern");

    // Prefix-chained user-turn segments.
    let mut parent = sys;
    let mut user_segs = Vec::with_capacity(turns);
    let mut user_tokens = Vec::with_capacity(turns);
    for k in 0..turns {
        let text = format!("Turn {k}: please continue the conversation about topic {k}.");
        let tok = est_tokens(&text);
        let seg = intern_message(
            &mut pool,
            &Msg::new("user", &text),
            Some(parent),
            &tokenizer,
        )
        .expect("benchmark user message must intern");
        parent = seg;
        user_segs.push(seg);
        user_tokens.push(tok);
    }
    let sys_tok = est_tokens(sys_text);

    // Chain graph: n0 -> n1 -> ... ; node k gates on the prior node's channel and
    // splices every prior reply, so its prompt is the full growing conversation.
    let mut state = serde_json::Map::new();
    let mut nodes = serde_json::Map::new();
    let mut edges = vec![serde_json::json!({"edge_type":"static","source":"START","target":"n0"})];
    let mut isl_by_node = HashMap::new();

    for k in 0..turns {
        state.insert(
            format!("c{k}"),
            serde_json::json!({"type":"messages","reducer":"add_messages"}),
        );
        // items: sys, [u0, splice c0, u1, splice c1, ..., u_{k-1}, splice c_{k-1}], u_k
        let mut items = vec![serde_json::json!({"seg": sys})];
        let mut isl = sys_tok;
        for j in 0..k {
            items.push(serde_json::json!({"seg": user_segs[j]}));
            items.push(serde_json::json!({"splice": format!("c{j}")}));
            isl += user_tokens[j] + 1; // +1 for the spliced 1-token reply
        }
        items.push(serde_json::json!({"seg": user_segs[k]}));
        isl += user_tokens[k];
        isl_by_node.insert(format!("n{k}"), isl);

        let mut node = serde_json::json!({
            "node_type":"llm","prompt":[],"output":format!("c{k}"),"items":items
        });
        if k > 0 {
            node["inputs"] = serde_json::json!([{"channel": format!("c{}", k-1), "count": 1}]);
            edges.push(serde_json::json!({"edge_type":"static","source":format!("n{}",k-1),"target":format!("n{k}")}));
        }
        nodes.insert(format!("n{k}"), node);
    }
    edges.push(
        serde_json::json!({"edge_type":"static","source":format!("n{}",turns-1),"target":"END"}),
    );

    let graph: GraphRecord = serde_json::from_value(serde_json::json!({
        "state": state, "nodes": nodes, "edges": edges
    }))
    .expect("valid bench graph");
    (pool.freeze(), graph, isl_by_node)
}

/// Default server base URL when none is configured (co-located mock/frontend).
pub(crate) const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8000";

/// Resolve the worker server list: the configured base URLs, or a single
/// [`DEFAULT_BASE_URL`] when none were given.
pub(crate) fn resolve_servers(base_urls: &[String]) -> Vec<String> {
    if base_urls.is_empty() {
        vec![DEFAULT_BASE_URL.to_string()]
    } else {
        base_urls.to_vec()
    }
}
