// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Executable semantic-lowering and paired-report contracts for Harbor trials.

use aiperf_runtime::eval::{
    ExecutableSemanticNode, FidelityError, FidelityOutcome, PairedComparisonError,
    PairedComparisonSpec, PairedMeasurements, SemanticGraph, SemanticNode, lower_semantic_graph,
};

#[test]
fn exact_lowering_preserves_each_executable_node_in_source_order() {
    let source = SemanticGraph::new(vec![SemanticNode::Llm, SemanticNode::Tool]).unwrap();

    let lowered = lower_semantic_graph(&source).unwrap();

    assert_eq!(lowered.outcome(), FidelityOutcome::Exact);
    assert_eq!(
        lowered.nodes(),
        &[ExecutableSemanticNode::Llm, ExecutableSemanticNode::Tool]
    );
}

#[test]
fn unsupported_node_refuses_at_its_source_position_without_a_partial_program() {
    let source = SemanticGraph::new(vec![SemanticNode::Llm, SemanticNode::Barrier]).unwrap();

    let error = lower_semantic_graph(&source).unwrap_err();

    assert_eq!(
        error,
        FidelityError::UnsupportedNode {
            index: 1,
            node: SemanticNode::Barrier,
        }
    );
    assert_eq!(error.outcome(), FidelityOutcome::Unsupported);
}

#[test]
fn paired_report_returns_independent_deltas_for_a_locked_baseline() {
    let baseline = PairedComparisonSpec::new("task", "model", 7, "policy", "image", 60).unwrap();
    let candidate = PairedComparisonSpec::new("task", "model", 7, "policy", "image", 60).unwrap();
    let baseline_measurements = PairedMeasurements::new(0.50, 4.0, 8.0, 5.0, 200, 3).unwrap();
    let candidate_measurements = PairedMeasurements::new(0.75, 5.5, 6.0, 4.0, 180, 5).unwrap();

    let report = baseline
        .compare_measurements(&candidate, baseline_measurements, candidate_measurements)
        .unwrap();

    assert_eq!(report.quality_delta(), 0.25);
    assert_eq!(report.cost_delta(), 1.5);
    assert_eq!(report.latency_seconds_delta(), -2.0);
    assert_eq!(report.critical_path_seconds_delta(), -1.0);
    assert_eq!(report.token_delta(), -20);
    assert_eq!(report.tool_call_delta(), 2);
}

#[test]
fn paired_report_rejects_changed_baseline_before_computing_deltas() {
    let baseline = PairedComparisonSpec::new("task", "model", 7, "policy", "image", 60).unwrap();
    let changed_seed =
        PairedComparisonSpec::new("task", "model", 8, "policy", "image", 60).unwrap();
    let measurements = PairedMeasurements::new(1.0, 0.0, 0.0, 0.0, 0, 0).unwrap();

    assert_eq!(
        baseline
            .compare_measurements(&changed_seed, measurements, measurements)
            .unwrap_err(),
        PairedComparisonError::ChangedBaseline
    );
}

#[test]
fn paired_measurements_reject_non_finite_system_metrics() {
    assert_eq!(
        PairedMeasurements::new(1.0, f64::NAN, 1.0, 1.0, 1, 1).unwrap_err(),
        PairedComparisonError::NonFiniteMeasurement("cost")
    );
}
