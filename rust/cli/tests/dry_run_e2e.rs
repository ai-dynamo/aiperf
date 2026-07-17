// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! End-to-end coverage for the `--dry-run` transport.
//!
//! The analytic sink opens no sockets, and its TTFT, ITL, request latency, and
//! OSL are exact.
//!
//! This test is `#[ignore]` only because synthetic prompt generation needs a
//! tokenizer (`Qwen/Qwen3-0.6B`), which is fetched from the Hugging Face hub on
//! first run and cached thereafter — the dry-run transport itself needs nothing
//! external. Run it with:
//!   cargo test -p aiperf-cli --test dry_run_e2e -- --ignored --nocapture

use std::path::PathBuf;
use std::process::Command;

fn aiperf_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_aiperf"))
}

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

    let native = out_dir.path().join("native-v2.json");
    let records_path = out_dir.path().join("profile_export.jsonl");
    assert!(native.exists(), "native-v2.json not written");
    assert!(records_path.exists(), "profile_export.jsonl not written");

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

fn run_sim_dry_run(out: &std::path::Path, extra: &[&str]) -> std::process::Output {
    let mut args: Vec<String> = vec![
        "profile".into(),
        "--model".into(),
        "Qwen/Qwen3-0.6B".into(),
        "--url".into(),
        "127.0.0.1:9".into(),
        "--endpoint-type".into(),
        "chat".into(),
        "--streaming".into(),
        "--synthetic-input-tokens-mean".into(),
        "50".into(),
        "--synthetic-input-tokens-stddev".into(),
        "0".into(),
        "--output-tokens-mean".into(),
        (OSL as u64).to_string(),
        "--output-tokens-stddev".into(),
        "0".into(),
        "--dry-run".into(),
        "--dry-run-clock".into(),
        "sim".into(),
        "--dry-run-ttft-ms".into(),
        TTFT_MS.to_string(),
        "--dry-run-itl-ms".into(),
        ITL_MS.to_string(),
        "--artifact-dir".into(),
        out.to_str().unwrap().into(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    Command::new(aiperf_bin())
        .args(&args)
        .output()
        .expect("spawn aiperf profile --dry-run --dry-run-clock sim")
}

#[test]
#[ignore = "needs the Qwen/Qwen3-0.6B tokenizer (Hugging Face); the dry-run transport itself needs no network"]
fn sim_clock_dry_run_is_deterministic_and_exact() {
    let a = tempfile::tempdir().expect("tempdir");
    let b = tempfile::tempdir().expect("tempdir");
    let seeded = [
        "--concurrency",
        "4",
        "--request-count",
        "200",
        "--random-seed",
        "123",
    ];
    let out_a = run_sim_dry_run(a.path(), &seeded);
    let out_b = run_sim_dry_run(b.path(), &seeded);
    assert!(
        out_a.status.success(),
        "sim run A failed: {}",
        String::from_utf8_lossy(&out_a.stderr)
    );
    assert!(
        out_b.status.success(),
        "sim run B failed: {}",
        String::from_utf8_lossy(&out_b.stderr)
    );

    let report_a = std::fs::read(a.path().join("native-v2.json")).expect("report A");
    let report_b = std::fs::read(b.path().join("native-v2.json")).expect("report B");
    assert_eq!(
        report_a, report_b,
        "seeded sim runs must produce byte-identical native-v2.json"
    );

    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(a.path().join("profile_export_aiperf.json")).expect("summary"),
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
    assert_eq!(summary["request_count"]["avg"].as_f64(), Some(200.0));
}

#[test]
#[ignore = "needs the Qwen/Qwen3-0.6B tokenizer (Hugging Face); the dry-run transport itself needs no network"]
fn sim_clock_compresses_a_duration_bounded_run_to_instant() {
    // A 120-second duration-bounded run must complete in far less than 120s of
    // wall time under the virtual clock (arrival pacing + duration bound run on
    // SimClock). Allow generous headroom for tokenizer/dataset startup.
    let out_dir = tempfile::tempdir().expect("tempdir");
    let start = std::time::Instant::now();
    let output = run_sim_dry_run(
        out_dir.path(),
        &["--request-rate", "10", "--benchmark-duration", "120"],
    );
    let wall = start.elapsed();
    assert!(
        output.status.success(),
        "sim duration run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        wall < std::time::Duration::from_secs(60),
        "virtual-clock 120s run took {wall:?} of wall time — the SimClock is not governing pacing",
    );
    // The run reports ~120s of *virtual* benchmark duration despite finishing fast.
    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.path().join("profile_export_aiperf.json"))
            .expect("summary"),
    )
    .expect("summary json");
    let virtual_duration = summary["benchmark_duration"]["avg"]
        .as_f64()
        .expect("benchmark_duration");
    assert!(
        (100.0..=130.0).contains(&virtual_duration),
        "expected ~120s of virtual benchmark_duration, got {virtual_duration}",
    );
}
