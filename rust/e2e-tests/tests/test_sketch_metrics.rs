// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::Value;

// `--sketch-metrics` (and `AIPERF_METRICS_SKETCH`) swaps the per-record value
// vectors for bounded-memory t-digests. Counts and extrema stay exact, percentiles
// become estimates, and per-record outputs are no longer available.
//
// Latency percentiles are not compared across the two runs: these are two separate
// live runs, so their latencies differ for reasons that have nothing to do with the
// sketch. What is comparable is the pinned-length distributions, where the exact and
// estimated summaries must agree value for value.

const REQUEST_COUNT: u32 = 12;
const ISL: u32 = 16;
const OSL: u32 = 4;

fn workload(h: &AIPerfHarness, extra: &str) -> String {
    format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --synthetic-input-tokens-mean {ISL} --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean {OSL} --output-tokens-stddev 0 \
         --request-count {REQUEST_COUNT} --concurrency 2 --random-seed 7 \
         --export-level raw {extra} --ui none",
        h.mock.url
    )
}

/// Fields of a metric summary that must survive the sketch unchanged for a
/// distribution with zero spread.
fn summary_fields(metric: &Value, label: &str) -> Vec<(String, f64)> {
    let object = metric
        .as_object()
        .unwrap_or_else(|| panic!("{label} is not a summary object: {metric}"));
    let mut fields: Vec<(String, f64)> = object
        .iter()
        .filter_map(|(key, value)| value.as_f64().map(|v| (key.clone(), v)))
        .collect();
    fields.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        !fields.is_empty(),
        "{label} has no numeric fields: {metric}"
    );
    fields
}

/// With input and output lengths pinned, every retained value is identical, so the
/// t-digest summary has no room to approximate: it must match the exact summary.
/// Request counts must match too — those are exact under either mode.
#[tokio::test]
async fn sketch_metrics_match_exact_summaries_for_pinned_distributions() {
    let exact = {
        let h = AIPerfHarness::new().await;
        let r = h.run(&workload(&h, ""));
        assert!(r.success(), "exact run failed: {}", r.stderr);
        r.artifacts.json()
    };
    let sketch = {
        let h = AIPerfHarness::new().await;
        let r = h.run(&workload(&h, "--sketch-metrics"));
        assert!(r.success(), "sketch run failed: {}", r.stderr);
        r.artifacts.json()
    };

    for metric in [
        "input_sequence_length",
        "output_sequence_length",
        "request_count",
    ] {
        assert_eq!(
            summary_fields(&exact[metric], &format!("exact {metric}")),
            summary_fields(&sketch[metric], &format!("sketch {metric}")),
            "{metric} must be identical under --sketch-metrics; \
             exact: {:#}, sketch: {:#}",
            exact[metric],
            sketch[metric]
        );
    }

    // Latency percentiles are estimates, but they must still be present, finite,
    // monotonic, and bracketed by the exact extrema the mode promises to keep.
    let latency = &sketch["request_latency"];
    let value = |key: &str| {
        latency[key]
            .as_f64()
            .unwrap_or_else(|| panic!("sketch request_latency.{key} missing: {latency:#}"))
    };
    assert_eq!(
        value("count"),
        f64::from(REQUEST_COUNT),
        "sketch must retain an exact count: {latency:#}"
    );
    let mut previous = value("min");
    for key in ["p1", "p25", "p50", "p90", "p99", "max"] {
        let current = value(key);
        assert!(
            current.is_finite() && current >= previous,
            "sketch request_latency is not monotonic at {key}: {latency:#}"
        );
        previous = current;
    }
}

/// Sketch mode trades per-record retention for bounded memory, so the per-record
/// exports are unavailable even when `--export-level raw` asks for them. The
/// control run proves the same workload does produce them without the flag.
#[tokio::test]
async fn sketch_metrics_suppresses_per_record_exports() {
    let h = AIPerfHarness::new().await;
    let control = h.run(&workload(&h, ""));
    assert!(control.success(), "control run failed: {}", control.stderr);
    assert_eq!(
        control.artifacts.raw_records().len(),
        REQUEST_COUNT as usize,
        "the control run must write per-record output"
    );

    let h = AIPerfHarness::new().await;
    let sketched = h.run(&workload(&h, "--sketch-metrics"));
    assert!(sketched.success(), "sketch run failed: {}", sketched.stderr);
    assert!(
        sketched
            .artifacts
            .find_file("**/*profile_export_raw.jsonl")
            .is_none(),
        "sketch mode must not write profile_export_raw.jsonl"
    );
    assert!(
        sketched
            .artifacts
            .find_file("**/*profile_export.jsonl")
            .is_none(),
        "sketch mode must not write profile_export.jsonl"
    );
    // The summary is still produced: this is bounded retention, not a disabled run.
    assert_eq!(
        sketched.artifacts.json()["request_count"]["sum"].as_f64(),
        Some(f64::from(REQUEST_COUNT))
    );
}

/// `AIPERF_METRICS_SKETCH` is documented as the environment equivalent of the flag
/// and must select the same mode.
#[tokio::test]
async fn sketch_metrics_env_var_matches_the_flag() {
    let h = AIPerfHarness::new().await;
    let r = h.run_env(&workload(&h, ""), &[("AIPERF_METRICS_SKETCH", "1")]);

    assert!(r.success(), "env-selected sketch run failed: {}", r.stderr);
    assert!(
        r.artifacts
            .find_file("**/*profile_export_raw.jsonl")
            .is_none(),
        "AIPERF_METRICS_SKETCH=1 must select sketch mode, but per-record output was written"
    );
    assert_eq!(
        r.artifacts.json()["request_count"]["sum"].as_f64(),
        Some(f64::from(REQUEST_COUNT))
    );
}
