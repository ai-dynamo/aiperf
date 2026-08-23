// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Contract tests for the canonical recorded-agent scenario lock.
#![cfg(feature = "engine")]

use aiperf_runtime::agentx::scenario::{
    RecordedAgentScenarioInputs, apply_recorded_agent_scenario_locks, recorded_agent_default,
};
use aiperf_runtime::graph::recorded::agent_recording::CanonicalReplayFixture;

#[test]
fn canonical_scenario_rejects_noncanonical_task_order_and_sketch_metrics() {
    let fixture = CanonicalReplayFixture::load().expect("built-in fixture loads");
    let mut inputs = RecordedAgentScenarioInputs::canonical(&fixture);
    inputs.task_order.reverse();
    let error = apply_recorded_agent_scenario_locks(&inputs, &fixture)
        .expect_err("reordered canonical tasks are not comparable");
    assert!(
        error
            .violations
            .iter()
            .any(|violation| violation.flag == "dataset.task_order")
    );

    let mut inputs = RecordedAgentScenarioInputs::canonical(&fixture);
    inputs.sketch_metrics = true;
    let error = apply_recorded_agent_scenario_locks(&inputs, &fixture)
        .expect_err("sketch metrics cannot supply exact replay timing records");
    assert!(
        error
            .violations
            .iter()
            .any(|violation| violation.flag == "metrics.sketch")
    );
}

#[test]
fn unsafe_override_is_explicitly_noncomparable_but_virtual_tools_are_hard_failures() {
    let fixture = CanonicalReplayFixture::load().expect("built-in fixture loads");
    let mut inputs = RecordedAgentScenarioInputs::canonical(&fixture);
    inputs.unsafe_override = true;
    inputs.hardware_description = None;
    let outcome = apply_recorded_agent_scenario_locks(&inputs, &fixture)
        .expect("unsafe override permits bypassable conflict");
    assert_eq!(outcome.submission_valid, Some(false));
    assert!(
        outcome
            .submission_invalid_reasons
            .contains(&"unsafe_override".to_string())
    );
    assert!(
        outcome
            .submission_invalid_reasons
            .contains(&"non_comparable".to_string())
    );

    let mut inputs = RecordedAgentScenarioInputs::canonical(&fixture);
    inputs.virtual_clock = true;
    assert!(apply_recorded_agent_scenario_locks(&inputs, &fixture).is_err());
    assert_eq!(recorded_agent_default().name, "recorded-agent-default");
}
