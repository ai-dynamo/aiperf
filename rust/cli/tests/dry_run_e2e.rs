// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! End-to-end proof for the `--dry-run` fake-transport path.
//!
//! `aiperf profile --dry-run` swaps the real HTTP leaf for a fake execution
//! backend that fabricates every request outcome from an analytic latency model
//! with **zero network** — no mock server, no sockets. The full pipeline
//! (scheduling → per-worker metrics → export plane) runs unchanged, so this test
//! verifies the raw per-record artifact (`profile_export.jsonl`) exactly, per the
//! repo's feature-complete bar: TTFT, ITL, request_latency, and OSL for every
//! record must equal the analytic config with no tolerance band (the fake sink
//! is itself the deterministic oracle).
//!
//! This test is `#[ignore]` only because synthetic prompt generation needs a
//! tokenizer (`Qwen/Qwen3-0.6B`), which is fetched from the Hugging Face hub on
//! first run and cached thereafter — the dry-run transport itself needs nothing
//! external. Run it with:
//!   cargo test -p aiperf-cli --test dry_run_e2e -- --ignored --nocapture

use std::path::PathBuf;
use std::process::Command;

/// Path to the built `aiperf` binary under test.
fn aiperf_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_aiperf"))
}

/// Analytic knobs asserted end to end.
const TTFT_MS: f64 = 10.0;
const ITL_MS: f64 = 2.0;
const OSL: f64 = 12.0;
const ISL: f64 = 20.0;
const REQUESTS: usize = 8;

/// Expected request latency for a zero-jitter analytic model:
/// `ttft + (osl - 1) * itl`.
fn expected_request_latency_ms() -> f64 {
    TTFT_MS + (OSL - 1.0) * ITL_MS
}

fn metric_value(record: &serde_json::Value, key: &str) -> f64 {
    record["metrics"][key]["value"]
        .as_f64()
        .unwrap_or_else(|| panic!("record missing numeric metric {key:?}: {record}"))
}

#[test]
#[ignore = "needs the Qwen/Qwen3-0.6B tokenizer (Hugging Face) for synthetic prompts; the dry-run transport itself needs no network"]
fn dry_run_fabricates_exact_per_record_timing_with_zero_network() {
    let out_dir = tempfile::tempdir().expect("tempdir");

    let output = Command::new(aiperf_bin())
        .args([
            "profile",
            "--model",
            "Qwen/Qwen3-0.6B",
            // Nothing listens here: the dry-run leaf opens no sockets.
            "--url",
            "127.0.0.1:9",
            "--endpoint-type",
            "chat",
            "--streaming",
            "--concurrency",
            "2",
            "--request-count",
            &REQUESTS.to_string(),
            "--synthetic-input-tokens-mean",
            &(ISL as u64).to_string(),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            &(OSL as u64).to_string(),
            "--output-tokens-stddev",
            "0",
            "--dry-run",
            "--dry-run-ttft-ms",
            &TTFT_MS.to_string(),
            "--dry-run-itl-ms",
            &ITL_MS.to_string(),
            "--artifact-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("spawn aiperf profile --dry-run");

    assert!(
        output.status.success(),
        "dry-run profile failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // The full export plane must have produced its normal artifact set.
    let native = out_dir.path().join("native-v2.json");
    let records_path = out_dir.path().join("profile_export.jsonl");
    assert!(native.exists(), "native-v2.json not written");
    assert!(records_path.exists(), "profile_export.jsonl not written");

    // Raw per-record verification: every record's fabricated timing must equal
    // the analytic config exactly (deterministic oracle, no tolerance band).
    let records = std::fs::read_to_string(&records_path).expect("read jsonl");
    let mut count = 0usize;
    for line in records.lines().filter(|line| !line.trim().is_empty()) {
        let record: serde_json::Value = serde_json::from_str(line).expect("record json");
        assert_eq!(metric_value(&record, "time_to_first_token"), TTFT_MS);
        assert_eq!(metric_value(&record, "inter_token_latency"), ITL_MS);
        assert_eq!(metric_value(&record, "output_sequence_length"), OSL);
        assert_eq!(
            metric_value(&record, "request_latency"),
            expected_request_latency_ms()
        );
        assert!(
            record["error"].is_null(),
            "dry-run record errored: {record}"
        );
        count += 1;
    }
    assert_eq!(count, REQUESTS, "expected {REQUESTS} fabricated records");

    // Summary aggregate (derived independently from the worker RecordIngest) must
    // agree exactly, proving the metrics + summary-export path end to end.
    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.path().join("profile_export_aiperf.json"))
            .expect("read summary"),
    )
    .expect("summary json");
    assert_eq!(
        summary["time_to_first_token"]["avg"].as_f64(),
        Some(TTFT_MS)
    );
    assert_eq!(summary["inter_token_latency"]["avg"].as_f64(), Some(ITL_MS));
    assert_eq!(
        summary["request_latency"]["avg"].as_f64(),
        Some(expected_request_latency_ms())
    );
    assert_eq!(summary["output_sequence_length"]["avg"].as_f64(), Some(OSL));
    assert_eq!(
        summary["request_count"]["avg"].as_f64(),
        Some(REQUESTS as f64)
    );
}
