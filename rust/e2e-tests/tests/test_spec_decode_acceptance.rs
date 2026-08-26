// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-profile coverage for canonical speculative-decode acceptance metrics.

mod common;
use common::*;

use serde_json::{Value, json};

const REQUESTS: usize = 2;
const SUMMARY_METRICS: &[(&str, f64)] = &[
    ("spec_decode_acceptance_length", 3.25),
    ("spec_decode_token_weighted_acceptance_length", 3.25),
    ("spec_decode_draft_acceptance_rate", 56.25),
    ("spec_decode_overall_draft_acceptance_rate", 56.25),
    ("spec_decode_accepted_per_verified", 0.65),
    ("spec_decode_steps", 8.0),
    ("spec_decode_accepted_draft_tokens", 18.0),
    ("spec_decode_draft_tokens", 32.0),
    ("total_spec_decode_steps", 16.0),
    ("total_accepted_draft_tokens", 36.0),
    ("total_draft_tokens", 64.0),
];
const RECORD_METRICS: &[(&str, f64)] = &[
    ("spec_decode_acceptance_length", 3.25),
    ("spec_decode_draft_acceptance_rate", 56.25),
    ("spec_decode_accepted_per_verified", 0.65),
    ("spec_decode_steps", 8.0),
    ("spec_decode_accepted_draft_tokens", 18.0),
    ("spec_decode_draft_tokens", 32.0),
];

async fn harness(is_enabled: bool) -> AIPerfHarness {
    let mut config = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        workers: 1,
        ..Default::default()
    };
    if is_enabled {
        config.spec_decode_acceptance = true;
    }
    AIPerfHarness::new_with(config).await
}

fn run_case(h: &AIPerfHarness) -> RunResult {
    run_endpoint_case(h, "chat", true, REQUESTS)
}

fn run_endpoint_case(
    h: &AIPerfHarness,
    endpoint_type: &str,
    is_streaming: bool,
    request_count: usize,
) -> RunResult {
    let streaming = if is_streaming { "--streaming" } else { "" };
    h.run(&format!(
        "--model test-model --url {} --endpoint-type {endpoint_type} {streaming} \
         --request-count {request_count} --concurrency 1 \
         --workers-max 1 --export-level raw --ui simple --tokenizer builtin",
        h.mock.url
    ))
}

fn console_text(result: &RunResult) -> String {
    let path = result
        .artifacts
        .find_file("**/profile_export_console.txt")
        .expect("profile_export_console.txt");
    std::fs::read_to_string(path).expect("read console artifact")
}

fn csv_text(result: &RunResult) -> String {
    let path = result
        .artifacts
        .find_file("**/profile_export_aiperf.csv")
        .expect("profile_export_aiperf.csv");
    std::fs::read_to_string(path).expect("read CSV artifact")
}

fn metric_average(summary: &Value, name: &str) -> f64 {
    summary[name]["avg"]
        .as_f64()
        .unwrap_or_else(|| panic!("missing numeric {name}.avg in aiperf.json"))
}

fn assert_per_record_stats(record: &Value) {
    let acceptance = &record["spec_decode_acceptance"];
    assert_eq!(acceptance["engine"], "vllm");
    assert_eq!(acceptance["mean_acceptance_length"], 3.25);
    assert_eq!(acceptance["draft_acceptance_rate"], 0.5625);
    assert_eq!(
        acceptance["acceptance_histogram"],
        json!({"0": 1, "1": 1, "2": 2, "3": 3, "4": 1})
    );
    assert_eq!(acceptance["num_accepted_draft_tokens"], 18);
    assert_eq!(acceptance["num_draft_tokens"], 32);
    assert_eq!(acceptance["num_spec_steps"], 8);
    assert_eq!(acceptance["num_spec_tokens"], 4);
    assert_eq!(
        acceptance["per_step_accepted"],
        json!([2, 3, 1, 4, 2, 0, 3, 3])
    );
    assert_eq!(
        acceptance["per_step_drafted"],
        json!([4, 4, 4, 4, 4, 4, 4, 4])
    );
    assert!(
        acceptance["completion_tokens"].as_u64().is_some(),
        "stream usage should reconcile completion_tokens"
    );
    for &(name, expected) in RECORD_METRICS {
        assert_eq!(record["metrics"][name]["value"].as_f64(), Some(expected));
    }
}

#[tokio::test]
async fn canonical_stats_flow_through_real_profile_console_and_artifacts() {
    let h = harness(true).await;
    let result = run_case(&h);
    assert!(
        result.success(),
        "aiperf profile failed:\nstdout:\n{}\nstderr:\n{}",
        result.stdout,
        result.stderr
    );

    let summary = result.artifacts.json();
    for &(name, expected) in SUMMARY_METRICS {
        assert_eq!(metric_average(&summary, name), expected, "metric {name}");
    }
    assert_eq!(
        summary["pooled_spec_decode_acceptance_histogram"],
        json!({"0": 2, "1": 2, "2": 4, "3": 6, "4": 2})
    );

    let console = console_text(&result);
    assert!(
        console.contains("NVIDIA AIPerf: Spec Decode"),
        "console artifact:\n{console}"
    );
    assert!(console.contains(
        "Accepted drafts per step (% of steps):  0: 12%   1: 12%   2: 25%   3: 38%   4: 12%"
    ));

    let csv = csv_text(&result);
    assert!(
        csv.contains("Acceptance Length (ratio),"),
        "CSV artifact:\n{csv}"
    );
    assert!(
        csv.contains("Overall Draft Acceptance Rate (%),56.25"),
        "CSV artifact:\n{csv}"
    );
    assert!(
        csv.contains("Total Spec Decode Steps,16.00"),
        "CSV artifact:\n{csv}"
    );
    assert!(!csv.contains("spec_decode_acceptance_length"));
    assert!(!csv.contains("total_spec_decode_steps"));

    let records = result.artifacts.jsonl();
    assert_eq!(records.len(), REQUESTS);
    for record in &records {
        assert_per_record_stats(record);
    }
}

#[tokio::test]
async fn root_metrics_flow_through_remaining_endpoint_and_stream_modes() {
    let h = harness(true).await;
    for (endpoint_type, is_streaming) in [
        ("chat", false),
        ("completions", false),
        ("completions", true),
    ] {
        let result = run_endpoint_case(&h, endpoint_type, is_streaming, 1);
        assert!(
            result.success(),
            "{endpoint_type} streaming={is_streaming} profile failed:\nstdout:\n{}\nstderr:\n{}",
            result.stdout,
            result.stderr
        );
        let records = result.artifacts.jsonl();
        assert_eq!(records.len(), 1);
        assert_per_record_stats(&records[0]);
    }
}

#[tokio::test]
async fn default_mock_suppresses_all_spec_decode_stats_and_outputs() {
    let h = harness(false).await;
    let result = run_case(&h);
    assert!(
        result.success(),
        "aiperf profile failed:\nstdout:\n{}\nstderr:\n{}",
        result.stdout,
        result.stderr
    );

    let summary = result.artifacts.json();
    for &(name, _) in SUMMARY_METRICS {
        assert!(
            summary.get(name).is_none(),
            "unexpected summary metric {name}"
        );
    }
    assert!(
        summary
            .get("pooled_spec_decode_acceptance_histogram")
            .is_none()
    );

    let console = console_text(&result);
    assert!(!console.contains("NVIDIA AIPerf: Spec Decode"));
    assert!(!console.contains("Accepted drafts per step"));
    let csv = csv_text(&result);
    assert!(!csv.contains("Spec Decode"));
    assert!(!csv.contains("spec_decode"));

    let records = result.artifacts.jsonl();
    assert_eq!(records.len(), REQUESTS);
    for record in &records {
        assert!(record.get("spec_decode_acceptance").is_none());
        for &(name, _) in RECORD_METRICS {
            assert!(
                record["metrics"].get(name).is_none(),
                "unexpected record metric {name}"
            );
        }
    }
}
