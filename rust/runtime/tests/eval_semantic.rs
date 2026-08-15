// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    lower_semantic_graph, FidelityOutcome, PairedComparisonSpec, SemanticGraph, SemanticNode,
};

#[test]
fn unsupported_semantic_node_returns_typed_fidelity_refusal() {
    let graph = SemanticGraph::new(vec![SemanticNode::Barrier]).unwrap();

    let report = lower_semantic_graph(&graph).unwrap_err();

    assert_eq!(report.outcome(), FidelityOutcome::Unsupported);
}

#[test]
fn paired_report_rejects_changed_baseline_dimensions() {
    let baseline = PairedComparisonSpec::new("task", "model", 7, "policy", "image", 60).unwrap();
    let changed = PairedComparisonSpec::new("task", "model", 8, "policy", "image", 60).unwrap();

    assert!(baseline.compare_to(&changed).is_err());
}
