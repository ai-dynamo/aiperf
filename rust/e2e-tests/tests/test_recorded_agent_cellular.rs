// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate cellular replay supplement contract coverage.

use std::collections::BTreeSet;

use aiperf_runtime::graph::replay::ToolCallMeasurement;
use aiperf_runtime::graph::supplement::{
    GraphCellSupplement, ReplayBackendIdentity, ReplayTraceInstance, TraceTerminalSupplement,
    merge_graph_cell_supplements,
};

fn trace(id: &str, worker_id: usize, backend: &str) -> TraceTerminalSupplement {
    let mut trace = TraceTerminalSupplement::new(
        "run".into(),
        format!("trajectory-{id}"),
        id.into(),
        worker_id,
        "recorded_replay",
    );
    trace.tools.push(ToolCallMeasurement::new(0.01, backend));
    trace
}

#[test]
fn cellular_replay_fold_is_stable_across_cell_arrival_order() {
    let first = trace("first", 0, "local");
    let second = trace("second", 0, "docker:replay-image");
    let expected = BTreeSet::from([
        ReplayTraceInstance::from(&first),
        ReplayTraceInstance::from(&second),
    ]);
    let cells = [
        GraphCellSupplement::new(
            1,
            vec![second],
            BTreeSet::from([ReplayBackendIdentity::from_wire("docker:replay-image")]),
        ),
        GraphCellSupplement::new(
            0,
            vec![first],
            BTreeSet::from([ReplayBackendIdentity::from_wire("local")]),
        ),
    ];

    let merged = merge_graph_cell_supplements(&expected, cells).expect("valid cellular fold");
    assert_eq!(
        merged
            .traces
            .iter()
            .map(|trace| trace.trace_id.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"],
    );
}
