// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for recorded-agent driver behavior after staged extension.

use std::collections::BTreeMap;

use aiperf_runtime::graph::driver::{
    RecordedReplayTraceProgramDriverFactory, TraceDriverContext, TraceDriverSpec, TraceIdentity,
    TraceProgramDriverFactory, WorkerIdentity,
};
use aiperf_runtime::graph::model::{GraphRecord, GraphTracePlan, GraphTraceProgram, TraceRecord};
use futures::executor::block_on;

#[test]
fn recorded_replay_retains_its_single_plan_contract_through_stage_defaults() {
    let program = GraphTraceProgram {
        profiling: GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: "recorded-trace".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        },
        warmup: None,
        environment: None,
        replay: None,
        driver: TraceDriverSpec::recorded_replay(),
    };
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "recorded-trace".into(),
    };
    let factory = RecordedReplayTraceProgramDriverFactory::default();
    let mut driver = factory
        .create(WorkerIdentity { worker_id: 7 }, &trace, &program.driver)
        .expect("recorded replay driver creates");
    let context = TraceDriverContext::metadata_only(&trace);

    block_on(async {
        assert!(
            driver
                .next_stage(&context)
                .await
                .expect("recorded replay keeps the default staged behavior")
                .is_none()
        );
        let supplement = driver
            .run(&program, &context)
            .await
            .expect("recorded replay's existing terminal execution remains available");
        assert_eq!(supplement.driver_kind, "recorded_replay");
        assert_eq!(supplement.worker_id, 7);
    });
}
