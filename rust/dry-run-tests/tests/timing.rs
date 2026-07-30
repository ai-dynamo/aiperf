// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Deterministic dry-run ports of the Python timing component-integration suite.

mod common;

use std::collections::BTreeMap;

use common::{
    TimingTestConfig, assert_concurrency_limit_hit, assert_concurrency_limit_respected,
    assert_credits_balanced, assert_request_count, assert_session_credits_match,
    assert_test_will_hit_concurrency_limit, assert_test_will_hit_prefill_limit,
    assert_turn_indices_sequential, build_timing_command, run, run_timing,
    verify_no_interleaving_within_session, verify_sessions_can_interleave,
};
use serde_json::Value;

fn credit_issue_times(records: &[Value]) -> Vec<i64> {
    let mut times: Vec<_> = records
        .iter()
        .filter_map(|record| record["metadata"]["credit_issued_ns"].as_i64())
        .collect();
    times.sort_unstable();
    times
}

fn session_issue_times(records: &[Value]) -> BTreeMap<String, Vec<i64>> {
    let mut times = BTreeMap::<String, Vec<i64>>::new();
    for record in records {
        let Some(session) = record["metadata"]["x_correlation_id"].as_str() else {
            continue;
        };
        let Some(issued) = record["metadata"]["credit_issued_ns"].as_i64() else {
            continue;
        };
        times.entry(session.to_owned()).or_default().push(issued);
    }
    for values in times.values_mut() {
        values.sort_unstable();
    }
    times
}

#[test]
fn constant_rate_completes_with_exact_virtual_issue_intervals() {
    let config = TimingTestConfig::new(20, 100.0);
    let preview = build_timing_command(
        &config,
        common::TimingCommandOptions {
            arrival_pattern: Some("constant"),
            ..Default::default()
        },
    );
    assert!(preview.contains("--request-rate 100"));
    let run = run_timing(
        &config,
        common::TimingCommandOptions {
            arrival_pattern: Some("constant"),
            ..Default::default()
        },
    );
    run.assert_success();
    assert_request_count(&run, config.expected_requests() as usize, "constant rate")
        .expect("completed request count");
    assert_credits_balanced(&run).expect("balanced credits");

    let times = credit_issue_times(&run.artifacts.jsonl());
    assert_eq!(times.len(), 20);
    for gap in times.windows(2).map(|pair| pair[1] - pair[0]) {
        assert_eq!(gap, 10_000_000, "100 QPS must issue every 10ms in SimClock");
    }
}

#[test]
fn poisson_and_gamma_rate_runs_are_seeded_and_complete() {
    for (pattern, smoothness) in [("poisson", None), ("gamma", Some("2.0"))] {
        let mut args = vec![
            "--num-sessions",
            "40",
            "--request-rate",
            "100",
            "--arrival-pattern",
            pattern,
            "--random-seed",
            "42",
        ];
        if let Some(smoothness) = smoothness {
            args.extend(["--arrival-smoothness", smoothness]);
        }
        let run = run(&args);
        run.assert_success();
        assert_request_count(&run, 40, pattern).expect("completed request count");
        assert_credits_balanced(&run).expect("balanced credits");

        let times = credit_issue_times(&run.artifacts.jsonl());
        let gaps: Vec<_> = times.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(
            gaps.iter().any(|gap| *gap != gaps[0]),
            "{pattern} must not collapse to constant arrivals"
        );
    }
}

#[test]
fn rate_modes_preserve_multiturn_credit_lifecycle() {
    for pattern in ["constant", "poisson", "gamma"] {
        let mut args = vec![
            "--num-sessions",
            "12",
            "--request-rate",
            "75",
            "--arrival-pattern",
            pattern,
            "--session-turns-mean",
            "4",
            "--session-turns-stddev",
            "0",
            "--random-seed",
            "42",
        ];
        if pattern == "gamma" {
            args.extend(["--arrival-smoothness", "2.0"]);
        }
        let run = run(&args);
        run.assert_success();
        assert_request_count(&run, 48, pattern).expect("all turns complete");
        assert_credits_balanced(&run).expect("balanced credits");
        assert_session_credits_match(&run, 4).expect("four turns per session");
        assert_turn_indices_sequential(&run).expect("turn indices");
        verify_no_interleaving_within_session(&run).expect("sequential session credits");
        verify_sessions_can_interleave(&run).expect("sessions interleave");
    }
}

#[test]
fn concurrency_burst_enforces_and_exercises_total_and_prefill_caps() {
    let mut config = TimingTestConfig::new(30, 0.0);
    config.concurrency = Some(6);
    config.prefill_concurrency = Some(3);
    assert_test_will_hit_concurrency_limit(&config, "burst: ").expect("total cap design");
    assert_test_will_hit_prefill_limit(&config, "burst: ").expect("prefill cap design");

    let run = run(&[
        "--num-sessions",
        "30",
        "--concurrency",
        "6",
        "--prefill-concurrency",
        "3",
    ]);
    run.assert_success();
    assert_request_count(&run, 30, "burst").expect("completed request count");
    assert_concurrency_limit_respected(&run, 6, false).expect("total cap respected");
    assert_concurrency_limit_hit(&run, 6, false).expect("total cap hit");
    assert_concurrency_limit_respected(&run, 3, true).expect("prefill cap respected");
    assert_concurrency_limit_hit(&run, 3, true).expect("prefill cap hit");
}

#[test]
fn user_centric_rate_preserves_turn_order_and_per_user_pacing() {
    let run = run(&[
        "--num-users",
        "10",
        "--num-sessions",
        "10",
        "--user-centric-rate",
        "25",
        "--session-turns-mean",
        "4",
        "--session-turns-stddev",
        "0",
    ]);
    run.assert_success();
    assert_credits_balanced(&run).expect("balanced credits");
    assert_turn_indices_sequential(&run).expect("turn ordering");
    verify_no_interleaving_within_session(&run).expect("no user overlap");

    // Python's user-centric test expects `num_users / qps`: 10 / 25 = 400ms.
    for (session, times) in session_issue_times(&run.artifacts.jsonl()) {
        for gap in times.windows(2).map(|pair| pair[1] - pair[0]) {
            assert_eq!(gap, 400_000_000, "session {session} per-user issue gap");
        }
    }
}
