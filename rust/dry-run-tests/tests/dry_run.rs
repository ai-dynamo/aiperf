// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Standalone fast e2e coverage for the socket-free `aiperf --dry-run` path.
//!
//! Every test launches the product binary through the same subprocess and
//! artifact boundary used by the full e2e suite. No mock server, URL, network,
//! or wall-clock inference delay is involved.

mod common;
use common::{
    Artifacts, ISL, ITL_MS, OSL, TTFT_MS, assert_credits_balanced, assert_request_count,
    assert_turn_indices_sequential, run,
};

#[test]
fn records_have_exact_analytic_timing_without_a_url() {
    let run = run(&["--request-count", "6", "--concurrency", "2"]);
    run.assert_success();
    let records = run.artifacts.jsonl();
    assert_eq!(records.len(), 6);
    assert_request_count(&run, 6, "dry-run profile count").expect("shared timing request count");
    assert_credits_balanced(&run).expect("shared timing credit balance");
    assert_turn_indices_sequential(&run).expect("shared timing turn ordering");
    for record in records {
        assert_eq!(
            Artifacts::metric(&record, "input_sequence_length"),
            ISL as f64
        );
        assert_eq!(
            Artifacts::metric(&record, "output_sequence_length"),
            OSL as f64
        );
        assert_eq!(Artifacts::metric(&record, "time_to_first_token"), TTFT_MS);
        assert_eq!(Artifacts::metric(&record, "inter_token_latency"), ITL_MS);
        assert_eq!(
            Artifacts::metric(&record, "request_latency"),
            TTFT_MS + (OSL as f64 - 1.0) * ITL_MS
        );
        assert!(record["error"].is_null());
    }
}

#[test]
fn warmup_is_retained_but_excluded_from_profile_metrics() {
    let run = run(&[
        "--warmup-request-count",
        "3",
        "--request-count",
        "12",
        "--concurrency",
        "4",
        "--workers-max",
        "2",
    ]);
    run.assert_success();
    let records = run.artifacts.jsonl();
    assert_eq!(records.len(), 15);
    assert_eq!(
        records
            .iter()
            .filter(|record| record["metadata"]["benchmark_phase"] == "profiling")
            .count(),
        12
    );
    assert_eq!(run.artifacts.summary()["request_count"]["avg"], 12.0);
}

#[test]
fn raw_and_summary_exports_are_consistent() {
    let run = run(&["--request-count", "5", "--export-level", "raw"]);
    run.assert_success();
    assert_eq!(
        std::fs::read_to_string(run.artifacts.dir.join("profile_export_raw.jsonl"))
            .expect("read raw JSONL")
            .lines()
            .count(),
        5
    );
    assert!(
        run.artifacts
            .dir
            .join("profile_export_aiperf.csv")
            .is_file()
    );
    let report = run.artifacts.summary();
    assert_eq!(report["request_count"]["avg"], 5.0);
    assert_eq!(report["time_to_first_token"]["avg"], TTFT_MS);
    assert_eq!(report["request_latency"]["avg"], TTFT_MS + 6.0);
}

#[test]
fn steady_state_sweep_creates_the_standard_profile_tree() {
    let run = run(&[
        "--concurrency",
        "2,4",
        "--num-profile-runs",
        "2",
        "--request-count",
        "8",
        "--steady-state",
    ]);
    run.assert_success();
    for trial in ["trial_0001", "trial_0002"] {
        for concurrency in ["2", "4"] {
            let path = run
                .artifacts
                .dir
                .join("profile_runs")
                .join(trial)
                .join(format!("concurrency_{concurrency}"));
            assert!(path.join("profile_export_aiperf.json").is_file());
        }
    }
    assert!(run.artifacts.dir.join("aggregate").is_dir());
}

#[test]
fn sim_clock_makes_seeded_runs_byte_identical() {
    let a = run(&[
        "--request-count",
        "20",
        "--concurrency",
        "4",
        "--random-seed",
        "7",
    ]);
    let b = run(&[
        "--request-count",
        "20",
        "--concurrency",
        "4",
        "--random-seed",
        "7",
    ]);
    a.assert_success();
    b.assert_success();
    let report_a = std::fs::read(a.artifacts.dir.join("native-v2.json")).expect("read report A");
    let report_b = std::fs::read(b.artifacts.dir.join("native-v2.json")).expect("read report B");
    assert_eq!(report_a, report_b);
}
