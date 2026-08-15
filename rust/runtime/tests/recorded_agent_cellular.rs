// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for recorded-agent replay across cellular partitions.

use std::collections::BTreeSet;

use aiperf_runtime::graph::supplement::{
    GraphCellPreflightBarrier, GraphCellSupplement, GraphSupplementError, ReplayBackendIdentity,
    ReplayTraceInstance, TraceTerminalSupplement, merge_graph_cell_supplements,
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
        ReplayTraceInstance::from(&trace("one", 0)),
        ReplayTraceInstance::from(&trace("two", 1)),
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
