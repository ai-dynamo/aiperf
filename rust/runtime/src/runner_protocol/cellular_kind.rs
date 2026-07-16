// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one classification a cellular run is dispatched on: is it a scheduled
//! synthetic/linear run or a graph program?
//!
//! [`CellularRunKind`] is the single seam both sides of the cellular split name:
//! the controller ([`cellular_controller`](crate::runner_protocol::cellular_controller))
//! answers the three ways the two paths differ (phase validation, per-phase global
//! ordinal bases, record merge) through the `impl` block that lives beside those
//! controller-private helpers, while the cell and the frontend terminal envelope
//! read the pure, controller-independent facts defined here (detection from the
//! dataset format, the provenance `workload` label, and whether the retain-path
//! multi-turn backstop applies). Keeping the enum here — rather than private to the
//! controller — lets the cell name the kind explicitly instead of re-deriving
//! graph-ness ad hoc from the dataset format, and lets the frontend label a run's
//! provenance correctly.
//!
//! A future kind (e.g. a distinct gRPC-graph executor) is one variant plus the three
//! controller arms; transport (`http`/`grpc`) is orthogonal to the kind — both run
//! the same scheduled executor, so gRPC does NOT add a variant here.

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
                    Some("dag_jsonl" | "weka_trace" | "dynamo_trace")
                )
            })
        })
}

/// Which execution path a cellular run drives. The scheduled arrival-paced executor
/// and the graph trace executor differ in exactly three ways — how the phases are
/// validated, whether a per-phase global ordinal base applies, and how the cells'
/// records merge — answered by the `impl` block in
/// [`cellular_controller`](crate::runner_protocol::cellular_controller). The pure
/// facts every consumer needs (detection, provenance label, multi-turn backstop)
/// live here.
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

    /// The `workload` provenance label the frontend stamps on the terminal envelope.
    /// Distinct from the `transport` label (`http`/`grpc`), which is orthogonal.
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
    }
}
