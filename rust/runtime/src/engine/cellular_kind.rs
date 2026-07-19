// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cellular execution classification for scheduled and graph runs.

use serde_json::Value;

/// Whether the run targets a graph program (`dag_jsonl` / `weka_trace` /
/// `dynamo_trace`), as opposed to a scheduled synthetic/linear dataset. Graph
/// programs partition cleanly by whole trace, so they take the concatenation merge
/// and bypass the scheduled request-budget guards.
pub(crate) fn is_graph_dataset(envelope: &Value) -> bool {
    envelope
        .pointer("/run/cfg/datasets")
        .and_then(Value::as_array)
        .is_some_and(|datasets| {
            datasets.iter().any(|dataset| {
                matches!(
                    dataset.get("format").and_then(Value::as_str),
                    Some("dag_jsonl" | "conditional_graph" | "weka_trace" | "dynamo_trace")
                )
            })
        })
}

/// The cellular execution path selected by the dataset format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellularRunKind {
    /// Synthetic/linear scheduled runs: request-bounded phases, pre-tiled global
    /// dispatch ordinals, byte-exact global-order merge.
    Scheduled,
    /// Graph programs (dag_jsonl/weka_trace/dynamo_trace): trace-partitioned,
    /// concatenation-merged.
    Graph,
}

impl CellularRunKind {
    /// A graph-format dataset selects the graph path; anything else is scheduled.
    pub fn detect(envelope: &Value) -> Self {
        if is_graph_dataset(envelope) {
            Self::Graph
        } else {
            Self::Scheduled
        }
    }

    /// The `workload` label for the terminal envelope.
    pub fn workload_label(&self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Graph => "graph",
        }
    }

    /// Whether the controller must reject a retain-path multi-turn run for this kind.
    ///
    /// A scheduled multi-turn conversation dispatches a variable number of turns, so
    /// its per-turn dispatch ordinal diverges from the sampler's per-conversation
    /// draw index that the retain (global-order) merge orders by — merging those in
    /// global dispatch order would silently reorder / re-sample the report, so the
    /// controller fails loud unless every cell shipped a folded store (exact-fold).
    /// A graph run partitions by whole trace and concatenation-merges regardless of
    /// turn count, so it is exempt.
    pub(crate) fn enforces_multiturn_retain_backstop(&self) -> bool {
        matches!(self, Self::Scheduled)
    }

    /// Whether the controller slices the per-cell SESSION (conversation) budget for
    /// this kind.
    ///
    /// A scheduled multi-turn run tiles its `sessions` budget across cells — cell `k`
    /// owns `owned_positions(total, k, C)` conversations, aligned with that cell's
    /// fixed owned-corpus conversation giver. A graph run gets the whole `sessions`
    /// budget WHOLE and partitions the trace at runtime (`PartitionedGraphTraceSource`
    /// over the session space), so slicing the budget here would double-partition.
    pub(crate) fn slices_session_budget(&self) -> bool {
        matches!(self, Self::Scheduled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_graph_and_scheduled_kinds() {
        let graph = json!({"run": {"cfg": {"datasets": [
            {"type": "file", "format": "dag_jsonl", "path": "/t.jsonl"}
        ]}}});
        let scheduled = json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}});
        assert_eq!(CellularRunKind::detect(&graph), CellularRunKind::Graph);
        assert_eq!(
            CellularRunKind::detect(&scheduled),
            CellularRunKind::Scheduled
        );
    }

    #[test]
    fn labels_and_backstop_track_kind() {
        assert_eq!(CellularRunKind::Scheduled.workload_label(), "scheduled");
        assert_eq!(CellularRunKind::Graph.workload_label(), "graph");
        assert!(CellularRunKind::Scheduled.enforces_multiturn_retain_backstop());
        assert!(!CellularRunKind::Graph.enforces_multiturn_retain_backstop());
        assert!(CellularRunKind::Scheduled.slices_session_budget());
        assert!(!CellularRunKind::Graph.slices_session_budget());
    }
}
