// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Static channel-topology derivations shared by the executor and store.

use crate::graph::model::GraphRecord;
use std::collections::BTreeMap;

/// Count the nodes that statically write each channel.
///
/// Every channel in `graph.state` is seeded to 0; channels a node writes that
/// are not declared in `state` are still counted.
pub fn producers_per_channel(graph: &GraphRecord) -> BTreeMap<String, i64> {
    let mut counts: BTreeMap<String, i64> = graph.state.keys().map(|c| (c.clone(), 0)).collect();
    for node in graph.nodes.values() {
        for ch in node.write_channels() {
            *counts.entry(ch.to_string()).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_declared_and_written() {
        let json = r#"{
            "state": {"a": {}, "b": {}},
            "nodes": {
                "n0": {"node_type":"llm","prompt":[],"output":"a"},
                "n1": {"node_type":"llm","prompt":[],"output":"a"},
                "n2": {"node_type":"llm","prompt":[],"output":"c"}
            },
            "edges": []
        }"#;
        let graph: GraphRecord = serde_json::from_str(json).unwrap();
        let counts = producers_per_channel(&graph);
        assert_eq!(counts.get("a"), Some(&2));
        assert_eq!(counts.get("b"), Some(&0));
        assert_eq!(counts.get("c"), Some(&1));
    }
}
