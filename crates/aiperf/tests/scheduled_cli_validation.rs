// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI configuration gates for scheduled workloads.

use std::process::Command;

fn fails_with(arguments: &[&str], expected: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "invalid invocation unexpectedly passed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expected),
        "stderr did not contain {expected:?}: {stderr}"
    );
}

#[test]
fn user_centric_requires_users_multiturn_and_non_conflicting_rate() {
    fails_with(
        &["--user-centric-rate", "10", "--turns", "3"],
        "requires --num-users",
    );
    fails_with(
        &[
            "--user-centric-rate",
            "10",
            "--num-users",
            "2",
            "--turns",
            "1",
        ],
        "requires --turns >= 2",
    );
    fails_with(
        &[
            "--user-centric-rate",
            "10",
            "--request-rate",
            "10",
            "--num-users",
            "2",
            "--turns",
            "3",
        ],
        "conflicts with --request-rate",
    );
}

#[test]
fn fixed_schedule_requires_input_and_rejects_non_replay_controls() {
    fails_with(&["--fixed-schedule"], "requires --input-file");
    fails_with(
        &[
            "--fixed-schedule",
            "--input-file",
            "/does/not/matter",
            "--concurrency",
            "2",
        ],
        "pure open-loop replay",
    );
    fails_with(
        &["--fixed-schedule-start-offset-ms", "10"],
        "require --fixed-schedule",
    );
}

#[test]
fn ancillary_flags_reject_invalid_or_unowned_actuators_before_dispatch() {
    fails_with(
        &["--request-cancellation-delay", "0.5"],
        "requires --request-cancellation-rate",
    );
    fails_with(
        &["--request-cancellation-rate", "101"],
        "finite percentage in 0..=100",
    );
    fails_with(
        &["--request-rate-ramp-duration", "1"],
        "requires --request-rate",
    );
    fails_with(
        &["--prefill-concurrency-ramp-duration", "1"],
        "requires --prefill-concurrency",
    );
    fails_with(
        &[
            "--fixed-schedule",
            "--input-file",
            "/does/not/matter",
            "--concurrency-ramp-duration",
            "1",
        ],
        "authored timestamps and does not accept actuator ramps",
    );
    fails_with(
        &[
            "--user-centric-rate",
            "10",
            "--num-users",
            "1",
            "--turns",
            "2",
            "--requests",
            "2",
            "--request-rate-ramp-duration",
            "1",
        ],
        "schedule-authored and does not accept a request-rate ramp",
    );
    fails_with(
        &["--mode", "graph", "--request-cancellation-rate", "10"],
        "supported by online workloads, not --mode graph",
    );
    fails_with(
        &[
            "--duration",
            "3",
            "--concurrency",
            "2",
            "--concurrency-ramp-duration",
            "1",
            "--adaptive-scale",
            "--adaptive-control-min",
            "1",
            "--adaptive-control-max",
            "2",
            "--adaptive-sustain-duration",
            "1",
            "--adaptive-scale-sla",
            "request_latency:p95:le:100",
        ],
        "cannot own the same actuator",
    );
}
