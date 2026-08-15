// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for recorded-agent replay across cellular partitions.

use std::collections::BTreeSet;

use aiperf_runtime::graph::supplement::{
    GraphCellPreflightBarrier, GraphCellSupplement, GraphSupplementError,
    PlannedReplayTraceInstance, ReplayBackendIdentity, TraceTerminalSupplement,
    merge_graph_cell_supplements,
};

fn trace(trace_id: &str, worker_id: usize) -> TraceTerminalSupplement {
    TraceTerminalSupplement::new(
        "run".into(),
        format!("trajectory-{trace_id}"),
        trace_id.into(),
        worker_id,
        "recorded_replay",
    )
}

#[test]
fn merge_rejects_missing_duplicate_or_unknown_backend_supplement() {
    let expected = BTreeSet::from([
        PlannedReplayTraceInstance::new(0, "trajectory-one", "one"),
        PlannedReplayTraceInstance::new(0, "trajectory-two", "two"),
    ]);
    let missing = GraphCellSupplement::new(0, vec![trace("one", 0)], BTreeSet::new());
    assert!(matches!(
        merge_graph_cell_supplements(&expected, [missing]),
        Err(GraphSupplementError::MissingTrace { .. })
    ));

    let duplicate = GraphCellSupplement::new(
        0,
        vec![trace("one", 0), trace("one", 0), trace("two", 1)],
        BTreeSet::new(),
    );
    assert!(matches!(
        merge_graph_cell_supplements(&expected, [duplicate]),
        Err(GraphSupplementError::DuplicateTrace { .. })
    ));

    let mut identities = BTreeSet::new();
    identities.insert(ReplayBackendIdentity::from_wire("remote:unknown"));
    let unknown = GraphCellSupplement::new(0, vec![trace("one", 0), trace("two", 1)], identities);
    assert!(matches!(
        merge_graph_cell_supplements(&expected, [unknown]),
        Err(GraphSupplementError::UnknownBackend { .. })
    ));
}

#[test]
fn merge_matches_controller_plan_without_runtime_run_identity() {
    let expected = BTreeSet::from([PlannedReplayTraceInstance::new(3, "trajectory-one", "one")]);
    let terminal = TraceTerminalSupplement::new(
        "runtime-origin-minted-after-start".into(),
        "trajectory-one".into(),
        "one".into(),
        4,
        "recorded_replay",
    )
    .with_planned_identity(PlannedReplayTraceInstance::new(3, "trajectory-one", "one"));
    let cell = GraphCellSupplement::new(3, vec![terminal], BTreeSet::new())
        .with_expected_traces(expected.clone());

    assert!(merge_graph_cell_supplements(&expected, [cell]).is_ok());
}

#[test]
fn aggregator_fold_preserves_original_cell_assignment() {
    let expected = BTreeSet::from([
        PlannedReplayTraceInstance::new(0, "trajectory-one", "one"),
        PlannedReplayTraceInstance::new(1, "trajectory-two", "two"),
    ]);
    let traces = vec![
        trace("one", 0).with_planned_identity(PlannedReplayTraceInstance::new(
            0,
            "trajectory-one",
            "one",
        )),
        trace("two", 0).with_planned_identity(PlannedReplayTraceInstance::new(
            1,
            "trajectory-two",
            "two",
        )),
    ];
    let aggregate =
        GraphCellSupplement::new(7, traces, BTreeSet::new()).with_expected_traces(expected.clone());

    assert!(merge_graph_cell_supplements(&expected, [aggregate]).is_ok());
}

#[tokio::test]
async fn cell_preflight_rejects_missing_docker_image_before_warmup() {
    let barrier = GraphCellPreflightBarrier::new(1);
    barrier.report(
        0,
        Err("Docker image preflight failed: image not found".into()),
    );

    assert!(matches!(
        barrier.await_all().await,
        Err(GraphSupplementError::FailedPreflight { cell_id: 0, .. })
    ));
}
