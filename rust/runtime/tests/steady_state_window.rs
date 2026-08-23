// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product-level check that closed-loop steady-state windowing excludes the
//! ramp-up and drain of a concurrency-target run and summarizes only the
//! saturated interval.
//!
//! Rather than asserting on synthetic event lists (covered by the unit tests in
//! `metrics_core::steady_state`), this drives the real [`MetricsAccumulator`]:
//! it ingests a deterministic ramp/steady/drain record set, detects the window,
//! and inspects the per-record attribution and the steady summary produced by
//! the ordinary accumulator export path.
#![cfg(feature = "engine")]

use aiperf_runtime::metrics_core::window::Phase;
use aiperf_runtime::metrics_core::{
    ExportContext, MetricTag, MetricsAccumulator, MetricsConfig, RecordIngest, SteadyStateConfig,
    steady_state_summary,
};

const SECOND_NS: i64 = 1_000_000_000;

/// Builds a profiling record spanning `[start_s, end_s)` seconds with a single
/// output token so request-latency is exactly the request span.
fn record(correlation: &str, start_s: i64, end_s: i64) -> RecordIngest {
    let start_ns = start_s * SECOND_NS;
    let end_ns = end_s * SECOND_NS;
    let mut ingest = RecordIngest::minimal(start_ns, end_ns, Phase::Profiling);
    ingest.correlation_id = correlation.to_string();
    ingest.request_index = None;
    ingest.first_token_ns = Some(start_ns + 100_000_000);
    ingest.first_output_token_ns = ingest.first_token_ns;
    ingest.token_arrival_ns = vec![start_ns + 100_000_000, start_ns + 200_000_000];
    ingest.tokens.input = Some(50);
    ingest.tokens.output = Some(2);
    ingest
}

/// Ingests a concurrency-target run: sparse ramp, saturated plateau, sparse drain.
fn ramp_steady_drain_accumulator() -> MetricsAccumulator {
    // target concurrency 5 -> steady threshold ceil(0.8 * 5) = 4.
    let records = [
        // Ramp: at most two concurrent, 5s each.
        record("ramp-a", 0, 5),
        record("ramp-b", 2, 7),
        // Steady plateau: six 10s requests staggered by 1s. Concurrency reaches
        // 4 at t=13s and last falls below 4 at t=22s.
        record("steady-0", 10, 20),
        record("steady-1", 11, 21),
        record("steady-2", 12, 22),
        record("steady-3", 13, 23),
        record("steady-4", 14, 24),
        record("steady-5", 15, 25),
        // Drain: two concurrent, 5s each, well after the plateau.
        record("drain-a", 40, 45),
        record("drain-b", 42, 47),
    ];

    let mut accumulator = MetricsAccumulator::with_config(MetricsConfig {
        steady_state: SteadyStateConfig {
            enabled: true,
            fraction: 0.8,
            hybrid_latency: false,
        },
        ..MetricsConfig::default()
    });
    for ingest in &records {
        accumulator.process_record(ingest);
    }
    accumulator
}

#[test]
fn steady_window_excludes_ramp_and_drain_records() {
    let accumulator = ramp_steady_drain_accumulator();
    let outcome = steady_state_summary(&accumulator, &accumulator_config(), 5)
        .expect("steady window must be detected for a saturated concurrency run");

    // Window bounds land on the first threshold entry and last drop below it.
    assert_eq!(outcome.window.start_ns, 13 * SECOND_NS);
    assert_eq!(outcome.window.end_ns, 22 * SECOND_NS);
    assert_eq!(outcome.window.threshold, 4);
    assert_eq!(outcome.window.peak_concurrency, 6);

    // Half-open [start, end) attribution keeps only steady-3/4/5 (starts 13,14,15).
    let mask = accumulator.query_time_range(outcome.window.start_ns, outcome.window.end_ns);
    assert_eq!(mask.iter().filter(|selected| **selected).count(), 3);
}

#[test]
fn steady_summary_differs_from_whole_run_summary() {
    let accumulator = ramp_steady_drain_accumulator();
    let outcome = steady_state_summary(&accumulator, &accumulator_config(), 5).unwrap();

    let whole = accumulator.summarize();
    let whole_latency = whole
        .finite_value(MetricTag::RequestLatency)
        .expect("whole-run request latency");
    let steady_latency = outcome
        .summary
        .finite_value(MetricTag::RequestLatency)
        .expect("steady-window request latency");

    // Whole-run blends the fast 5s ramp/drain requests with the 10s plateau
    // requests; the steady window sees only the 10s plateau requests.
    assert!(
        (whole_latency - 8_000.0).abs() < 1.0,
        "whole-run mean latency ~8000ms, got {whole_latency}"
    );
    assert!(
        (steady_latency - 10_000.0).abs() < 1.0,
        "steady mean latency ~10000ms, got {steady_latency}"
    );
    assert!(
        steady_latency > whole_latency,
        "steady summary must differ from the whole-run summary"
    );

    // The steady summary is exactly the ordinary export over the window range.
    let direct = accumulator.export_results(&ExportContext::time_range(
        outcome.window.start_ns,
        outcome.window.end_ns,
    ));
    assert_eq!(outcome.summary, direct);
}

#[test]
fn steady_state_is_gated_off_by_default() {
    let mut accumulator = MetricsAccumulator::new();
    for ingest in [record("r0", 0, 5), record("r1", 1, 6)] {
        accumulator.process_record(&ingest);
    }
    // Feature disabled by default and no target -> no steady outcome.
    assert!(steady_state_summary(&accumulator, &SteadyStateConfig::default(), 5).is_none());
    let enabled = SteadyStateConfig {
        enabled: true,
        fraction: 0.8,
        hybrid_latency: false,
    };
    assert!(steady_state_summary(&accumulator, &enabled, 0).is_none());
}

/// The steady-state configuration used by the fixture accumulator.
fn accumulator_config() -> SteadyStateConfig {
    SteadyStateConfig {
        enabled: true,
        fraction: 0.8,
        hybrid_latency: false,
    }
}
