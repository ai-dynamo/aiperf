// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public GraphTraceProgram serialization coverage.

use aiperf_runtime::graph::model::{ExecutableGraphNode, GraphTraceProgram, ParsedGraph};
use aiperf_runtime::graph::snapshot::chop_trie_at_tstar;
use serde_json::json;

#[test]
fn decodes_legacy_llm_nodes_and_tagged_tool_nodes() {
    let program: GraphTraceProgram = serde_json::from_value(json!({
        "profiling": {
            "graph": {
                "nodes": {
                    "request": {"output": "reply", "items": []},
                    "observation": {
                        "kind": "tool",
                        "output": "tool_result",
                        "commands": ["pwd"],
                        "timeout_ns": 100
                    }
                }
            },
            "trace": {"id": "legacy-compatible"}
        },
        "driver": {"kind": "static_graph", "data": {}}
    }))
    .unwrap();

    let graph = &program.profiling.graph;
    assert!(matches!(
        graph.nodes["request"],
        ExecutableGraphNode::Llm(_)
    ));
    assert!(matches!(
        graph.nodes["observation"],
        ExecutableGraphNode::Tool(_)
    ));
    assert_eq!(graph.llm_node_count(), 1);
    assert_eq!(graph.total_node_count(), 2);
    assert_eq!(graph.nodes["observation"].static_request_count(), 0);
    assert!(graph.nodes["observation"].read_channels().is_empty());
    assert_eq!(graph.nodes["observation"].output(), "tool_result");
}

#[test]
fn rejects_unknown_kinds_and_preserves_tools_in_static_transforms() {
    let error = serde_json::from_value::<GraphTraceProgram>(json!({
        "profiling": {
            "graph": {"nodes": {"unknown": {"kind": "barrier"}}},
            "trace": {"id": "unknown-kind"}
        },
        "driver": {"kind": "static_graph", "data": {}}
    }))
    .unwrap_err();
    assert!(error.to_string().contains("unknown graph node kind"));

    let program: GraphTraceProgram = serde_json::from_value(json!({
        "profiling": {
            "graph": {"nodes": {"tool": {
                "kind": "tool", "output": "observation", "commands": []
            }}},
            "trace": {"id": "tool-transform"}
        },
        "driver": {"kind": "static_graph", "data": {}}
    }))
    .unwrap();
    let parsed = ParsedGraph {
        graph: program.profiling.graph,
        ..Default::default()
    };
    let transformed = chop_trie_at_tstar(&parsed, 1.0);
    assert!(matches!(
        transformed.graph.nodes["tool"],
        ExecutableGraphNode::Tool(_)
    ));
}
