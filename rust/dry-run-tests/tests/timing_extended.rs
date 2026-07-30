// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Dry-run ports for warmup, think-time, duration, and admission timing tests.

mod common;

use common::{
    assert_concurrency_limit_hit, assert_concurrency_limit_respected, assert_credits_balanced,
    assert_request_count, assert_session_credits_match, assert_turn_indices_sequential, run,
    verify_no_interleaving_within_session,
};

fn phase_count(run: &common::Run, phase: &str) -> usize {
    run.artifacts
        .jsonl()
        .iter()
        .filter(|record| record["metadata"]["benchmark_phase"] == phase)
        .count()
}

#[test]
fn warmup_request_count_is_isolated_from_profiling() {
    let run = run(&[
        "--request-count",
        "30",
        "--request-rate",
        "300",
        "--arrival-pattern",
        "constant",
        "--warmup-request-count",
        "20",
    ]);
    run.assert_success();
    assert_eq!(phase_count(&run, "warmup"), 20);
    assert_eq!(phase_count(&run, "profiling"), 30);
    assert_request_count(&run, 50, "warmup plus profiling records").expect("all terminal records");
    assert_credits_balanced(&run).expect("warmup and profiling credits complete");
}

#[test]
fn warmup_transitions_before_profiling_without_losing_requests() {
    let run = run(&[
        "--request-count",
        "20",
        "--request-rate",
        "300",
        "--arrival-pattern",
        "constant",
        "--warmup-request-count",
        "15",
    ]);
    run.assert_success();
    let records = run.artifacts.jsonl();
    let phases: Vec<_> = records
        .iter()
        .map(|record| record["metadata"]["benchmark_phase"].as_str().unwrap_or(""))
        .collect();
    let profiling_start = phases
        .iter()
        .position(|phase| *phase == "profiling")
        .expect("profiling records");
    assert!(
        phases[..profiling_start]
            .iter()
            .all(|phase| *phase == "warmup")
    );
    assert!(
        phases[profiling_start..]
            .iter()
            .all(|phase| *phase == "profiling")
    );
    assert_eq!(profiling_start, 15);
    assert_eq!(phases.len() - profiling_start, 20);
}

#[test]
fn session_turn_delay_preserves_each_conversation_lifecycle() {
    let run = run(&[
        "--num-sessions",
        "10",
        "--session-turns-mean",
        "3",
        "--session-turns-stddev",
        "0",
        "--session-turn-delay-mean",
        "100",
        "--request-rate",
        "150",
        "--arrival-pattern",
        "constant",
    ]);
    run.assert_success();
    assert_request_count(&run, 30, "think-time conversations").expect("all turns");
    assert_credits_balanced(&run).expect("balanced credits");
    assert_session_credits_match(&run, 3).expect("three turns per session");
    assert_turn_indices_sequential(&run).expect("turn indices");
    verify_no_interleaving_within_session(&run).expect("session issue order");
}

#[test]
fn prefill_cap_limits_prefill_without_serializing_requests() {
    let run = run(&[
        "--request-count",
        "20",
        "--concurrency",
        "10",
        "--prefill-concurrency",
        "2",
    ]);
    run.assert_success();
    assert_concurrency_limit_respected(&run, 2, true).expect("prefill cap respected");
    assert_concurrency_limit_hit(&run, 2, true).expect("prefill cap exercised");
    assert_concurrency_limit_respected(&run, 10, false).expect("total cap respected");
    assert_concurrency_limit_hit(&run, 4, false).expect("independent decode overlap");
}

#[test]
fn benchmark_duration_stops_admission_and_drains_issued_credits() {
    let run = run(&[
        "--num-sessions",
        "100",
        "--request-rate",
        "50",
        "--arrival-pattern",
        "constant",
        "--benchmark-duration",
        "0.2",
        "--benchmark-grace-period",
        "1.0",
    ]);
    run.assert_success();
    let count = run.artifacts.jsonl().len();
    assert!(
        (8..=12).contains(&count),
        "0.2 seconds at 50 QPS issued {count} requests"
    );
    assert_credits_balanced(&run).expect("duration cutoff drains issued credits");
}

#[test]
fn fixed_schedule_replays_authored_first_turns_and_turn_delays() {
    let files = tempfile::tempdir().expect("trace directory");
    let trace = files.path().join("schedule.jsonl");
    std::fs::write(
        &trace,
        concat!(
            "{\"session_id\":\"a\",\"timestamp\":0,\"input_length\":20}\n",
            "{\"session_id\":\"b\",\"timestamp\":20,\"input_length\":20}\n",
            "{\"session_id\":\"a\",\"delay\":30,\"input_length\":20}\n",
            "{\"session_id\":\"b\",\"delay\":30,\"input_length\":20}\n",
        ),
    )
    .expect("write schedule");
    let trace = trace.to_str().expect("UTF-8 trace path");
    let run = run(&[
        "--fixed-schedule",
        "--custom-dataset-type",
        "mooncake_trace",
        "--input-file",
        trace,
        "--concurrency",
        "2",
    ]);
    run.assert_success();
    assert_request_count(&run, 4, "fixed schedule").expect("all authored turns");
    assert_credits_balanced(&run).expect("balanced scheduled credits");
    assert_session_credits_match(&run, 2).expect("two turns per scheduled session");

    let rows = run.artifacts.jsonl();
    let first_issues: Vec<_> = rows
        .iter()
        .filter(|row| row["metadata"]["turn_index"] == 0)
        .map(|row| {
            row["metadata"]["request_start_ns"]
                .as_i64()
                .expect("start timestamp")
        })
        .collect();
    assert_eq!(first_issues, vec![0, 20_000_000]);
}
