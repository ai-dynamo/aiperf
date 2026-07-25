// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchable-semantics e2e: BOTH arms execute end-to-end on WEKA input.
//!
//! - **legacy** (`WekaSemantics::Legacy`): the byte-exact AgentX port reconstructs
//!   the trace into `ReconstructedConversation`s.
//! - **graph-ir** (`WekaSemantics::GraphIr`): the runtime's `graph::recorded` weka
//!   compiler lowers the trace into a `GraphInputBundle`.
//!
//! Both are actually *run* (not handed off) and produce output, proving the
//! switch is fully wired to two live, parallel semantics.

#![cfg(feature = "agentx")]

use std::collections::HashMap;

use aiperf_runtime::agentx::config::WekaConfig;
use aiperf_runtime::agentx::loader::{convert_trace_to_conversations, MainReconstructOptions};
use aiperf_runtime::agentx::switch::{run_graph_ir, WekaSemantics};
use aiperf_runtime::agentx::synth::TokenSynth;
use aiperf_runtime::agentx::trace::WekaTrace;
use aiperf_runtime::dataset::{DatasetSource, LoadConfig, TiktokenTokenizer};
use aiperf_runtime::graph::recorded::{PromptCorpus, RecordedTraceInputConfig};
use serde_json::json;

struct StubSynth;
impl TokenSynth for StubSynth {
    fn decode_block_tokens(&mut self, h: &[i64]) -> Vec<u32> {
        h.iter().flat_map(|&x| (0..16).map(move |i| x as u32 * 1000 + i)).collect()
    }
    fn sample_partial_tail_tokens(&mut self, n: usize, _s: &str) -> Vec<u32> {
        (0..n as u32).map(|i| 900_000 + i).collect()
    }
    fn decode_tokens_to_text(&self, t: &[u32]) -> String {
        t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
    }
}

#[tokio::test]
async fn both_switch_arms_execute_end_to_end() {
    // Resolve confirms the two semantics.
    assert_eq!(WekaSemantics::resolve(Some("legacy")).unwrap(), WekaSemantics::Legacy);
    assert_eq!(WekaSemantics::resolve(Some("graph-ir")).unwrap(), WekaSemantics::GraphIr);

    // ---- graph-ir arm: actually run the recorded-graph weka compiler. ----
    let gir_config = RecordedTraceInputConfig {
        load: LoadConfig::new(DatasetSource::Inline(json!({
            "id": "root",
            "models": ["m"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [
                {"t": 0, "type": "s", "model": "m", "in": 32, "out": 8,
                 "hash_ids": [1, 2], "api_time": 2, "ttft": 0.5},
                {"t": 0.5, "type": "subagent", "agent_id": "child",
                 "subagent_type": "x", "status": "completed", "models": ["m"],
                 "requests": [
                     {"t": 0.5, "type": "n", "model": "m", "in": 16,
                      "out": 4, "hash_ids": [9], "api_time": 0.5}
                 ]}
            ]
        }))),
        root_limit: None,
        max_context_length: None,
        max_osl: None,
        idle_gap_cap_seconds: Some(60.0),
        prompt_corpus: PromptCorpus::Sonnet,
        content_root_seed: 42,
    };
    let bundle = run_graph_ir(gir_config, &TiktokenTokenizer::builtin())
        .await
        .expect("graph-ir arm compiles the trace");
    // The graph-ir arm produced a real lowered graph (root + subagent child).
    assert!(!bundle.plans.is_empty(), "graph-ir produced plans");
    assert!(
        bundle.plans[0].graph.nodes.len() >= 2,
        "graph-ir lowered root + child nodes"
    );

    // ---- legacy arm: actually run the byte-exact AgentX reconstruction. ----
    let legacy_trace_json = json!({
        "id": "root",
        "models": ["m"],
        "block_size": 16,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {"t": 0.0, "type": "n", "model": "m", "in": 32, "out": 8, "hash_ids": [1, 2]},
            {"t": 1.0, "type": "n", "model": "m", "in": 48, "out": 8, "hash_ids": [1, 2, 3]}
        ]
    });
    let trace = WekaTrace::from_json_bytes(legacy_trace_json.to_string().as_bytes()).unwrap();
    let mut synth = StubSynth;
    let convs = convert_trace_to_conversations(
        "root",
        &trace,
        &mut synth,
        &HashMap::new(),
        &WekaConfig { split_flattened_agents: false, ..WekaConfig::default() },
        &MainReconstructOptions::default(),
    )
    .expect("legacy arm reconstructs the trace");
    // The legacy arm produced a real reconstructed conversation with turns.
    assert_eq!(convs.len(), 1);
    assert_eq!(convs[0].session_id, "root");
    assert_eq!(convs[0].turns.len(), 2);
    assert_eq!(convs[0].turns[0].source_kind, "weka_main");

    // Both semantics ran end-to-end from WEKA input to their respective outputs.
}
