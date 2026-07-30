// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the native timing helper harness.

mod common;

use common::*;
use serde_json::json;

fn result_with_records(records: Vec<serde_json::Value>) -> (tempfile::TempDir, RunResult) {
    let dir = tempfile::TempDir::new().expect("temp artifact directory");
    write_jsonl(dir.path(), "profile_export.jsonl", &records);
    let result = RunResult {
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        artifacts: ArtifactReader {
            dir: dir.path().to_owned(),
        },
    };
    (dir, result)
}

fn record(
    session: &str,
    turn: u64,
    issued: i64,
    start: i64,
    ack: i64,
    end: i64,
    worker: &str,
) -> serde_json::Value {
    json!({"metadata": {
        "x_correlation_id": session,
        "turn_index": turn,
        "credit_issued_ns": issued,
        "request_start_ns": start,
        "request_ack_ns": ack,
        "request_end_ns": end,
        "worker_id": worker,
    }})
}

#[test]
fn profile_export_helpers_validate_credit_sessions_and_concurrency() {
    let (_dir, result) = result_with_records(vec![
        record("a", 0, 0, 10, 20, 30, "worker-0"),
        record("b", 0, 1, 11, 21, 31, "worker-1"),
        record("a", 1, 31, 32, 40, 50, "worker-0"),
        record("b", 1, 32, 33, 41, 51, "worker-1"),
    ]);

    assert_request_count(&result, 4, "profile count").expect("request count");
    assert_credits_balanced(&result).expect("terminal credits");
    assert_session_credits_match(&result, 2).expect("turns per session");
    assert_turn_indices_sequential(&result).expect("sequential turns");
    verify_no_interleaving_within_session(&result).expect("no session overlap");
    verify_sessions_can_interleave(&result).expect("session interleaving");
    assert_concurrency_limit_respected(&result, 2, false).expect("total cap");
    assert_concurrency_limit_hit(&result, 2, false).expect("total cap hit");
    assert_concurrency_limit_respected(&result, 2, true).expect("prefill cap");
    assert_concurrency_limit_hit(&result, 2, true).expect("prefill cap hit");
    assert_fair_load_distribution(&result, 0.0).expect("perfect worker balance");
}

#[test]
fn timing_argument_builders_and_cap_predictions_match_the_python_harness() {
    let mut config = TimingTestConfig::new(12, 500.0);
    config.turns_per_session = 3;
    config.concurrency = Some(2);
    config.prefill_concurrency = Some(2);

    assert_eq!(config.expected_requests(), 36);
    assert_test_will_hit_concurrency_limit(&config, "").expect("total cap prediction");
    assert_test_will_hit_prefill_limit(&config, "").expect("prefill cap prediction");

    let args = build_timing_command(
        &config,
        TimingCommandOptions {
            arrival_pattern: Some("poisson"),
            random_seed: Some(DEFAULT_RANDOM_SEED),
            ..Default::default()
        },
    );
    assert!(args.contains("--request-rate 500"));
    assert!(args.contains("--arrival-pattern poisson"));
    assert!(args.contains("--random-seed 42"));

    let burst = build_burst_command(&config);
    assert!(burst.contains("--num-sessions 12"));
    assert!(!burst.contains("--request-rate"));
}
