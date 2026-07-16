// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live end-to-end proof that the `rust_parity` RNG backend reproduces the Rust
//! `aiperf_runtime::rng` seeded Poisson arrival schedule inside a real `aiperf profile` run.
//!
//! The Python timing engine draws Poisson inter-arrivals from
//! `rng.derive("timing.request.poisson_interval").expovariate(rate)`
//! (`src/aiperf/timing/intervals.py`). With `AIPERF_RNG_BACKEND=rust_parity` that derive
//! resolves to the pure-Python byte-exact port of `aiperf_runtime::rng`, so the interval sequence
//! must equal the one this test computes directly from the Rust `RandomGenerator` — the
//! same `timing.request.poisson_interval` stream, same root seed.
//!
//! The test runs the benchmark **live** against the mock server twice through the legacy
//! Python service mesh (`AIPERF_RUNTIME_ENGINE=python`): once with the `rust_parity`
//! backend and once with `legacy`. It extracts each run's observed per-request arrival
//! schedule (`start_ns` from `profile_export_raw.jsonl`) and compares the inter-arrival
//! gaps against the Rust reference. Because measured dispatch has real (bounded, non-
//! accumulating) loopback jitter, the assertion is on mean per-gap error, not bytes:
//! `rust_parity` tracks the Rust reference to within jitter while `legacy` — a different
//! PRNG and SHA-256 derivation — produces an unrelated exponential stream that diverges
//! by ~the mean interval. Both engine schedules and the Rust reference are also written
//! out as JSONL for inspection.

mod common;

use std::io::Write;

use aiperf_runtime::rng::{RandomGenerator, RngRoot, namespace};
use common::*;

/// Global seed for the run (also aiperf's default, set explicitly for clarity).
const SEED: u64 = 42;
/// Target average request rate (requests/second). Mean interval = 1/RATE = 20 ms.
const RATE: f64 = 50.0;
/// Requests per run — enough that a divergent PRNG separates clearly from jitter.
const COUNT: usize = 150;

/// Rust reference: the exact Poisson inter-arrival sequence (seconds) the Python
/// engine must reproduce under `rust_parity`. Mirrors `intervals.py`
/// `PoissonIntervalGenerator`: derive the `timing.request.poisson_interval` seed off the
/// root, then draw `expovariate(rate)` repeatedly.
fn rust_reference_intervals(seed: u64, rate: f64, count: usize) -> Vec<f64> {
    let child = RngRoot::new(Some(seed))
        .derive_seed(namespace::TIMING_REQUEST_POISSON_INTERVAL)
        .expect("seeded root yields a poisson-interval seed");
    let mut rng = RandomGenerator::from_seed(Some(child));
    (0..count)
        .map(|_| rng.expovariate(rate).expect("rate > 0"))
        .collect()
}

/// Observed per-request scheduled-dispatch offsets (seconds, relative to the first) from a
/// completed run, ordered by dispatch time. Uses `metadata.credit_issued_ns` — the moment
/// the timing manager issued the credit after sleeping the Poisson interval, i.e. the
/// arrival the seeded interval generator scheduled (minimal added jitter vs the later
/// `request_start_ns` worker pickup).
fn observed_arrival_offsets(result: &RunResult) -> Vec<f64> {
    let mut issued: Vec<i64> = result
        .artifacts
        .jsonl()
        .iter()
        .filter_map(|record| {
            record
                .get("metadata")
                .and_then(|m| m.get("credit_issued_ns"))
                .and_then(|v| v.as_i64())
        })
        .collect();
    assert!(
        issued.len() >= COUNT,
        "expected >= {COUNT} records with metadata.credit_issued_ns, got {} (exit={}, stderr tail: {})",
        issued.len(),
        result.exit_code,
        result
            .stderr
            .lines()
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .join(" | ")
    );
    issued.sort_unstable();
    issued.truncate(COUNT);
    let origin = issued[0] as f64;
    issued.iter().map(|s| (*s as f64 - origin) / 1e9).collect()
}

/// Consecutive inter-arrival gaps (seconds) from arrival offsets.
fn gaps(offsets: &[f64]) -> Vec<f64> {
    offsets.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Mean absolute error between two equal-length gap sequences.
fn mean_abs_error(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    assert!(n > 0, "empty gap sequence");
    (0..n).map(|i| (a[i] - b[i]).abs()).sum::<f64>() / n as f64
}

/// Write a schedule as JSONL (one object per interval) for inspection.
fn write_schedule_jsonl(path: &std::path::Path, label: &str, gaps: &[f64]) {
    let mut f = std::fs::File::create(path).expect("create schedule jsonl");
    let mut cumulative = 0.0_f64;
    for (i, gap) in gaps.iter().enumerate() {
        cumulative += *gap;
        writeln!(
            f,
            "{{\"source\":\"{label}\",\"i\":{i},\"interval_s\":{gap:.9},\"cumulative_s\":{cumulative:.9}}}"
        )
        .expect("write schedule line");
    }
}

/// Run a live Poisson benchmark through the Python engine with the given RNG backend and
/// return its observed inter-arrival gaps.
fn run_python_poisson(harness: &AIPerfHarness, backend: &str) -> Vec<f64> {
    let args = format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --request-rate {RATE} --arrival-pattern poisson \
         --request-count {COUNT} --random-seed {SEED} \
         --synthetic-input-tokens-mean 32 --output-tokens-mean 4 \
         --workers-max 4 --ui simple",
        harness.mock.url
    );
    let result = harness.run_env(
        &args,
        &[
            ("AIPERF_RUNTIME_ENGINE", "python"),
            ("AIPERF_RNG_BACKEND", backend),
        ],
    );
    assert_eq!(
        result.exit_code,
        0,
        "python/{backend} run failed; stderr tail: {}",
        result
            .stderr
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join(" | ")
    );
    gaps(&observed_arrival_offsets(&result))
}

/// The `rust_parity` backend makes the live Python Poisson schedule reproduce the Rust
/// `aiperf_runtime::rng` reference within loopback jitter, while `legacy` does not.
#[tokio::test]
async fn test_seeded_poisson_schedule_parity() {
    let mut cfg = MockServerConfig::default();
    cfg.no_tokenizer = true;
    // Small, deterministic latency so dispatch is prompt and gaps reflect the schedule.
    cfg.ttft = 2.0;
    cfg.itl = 1.0;
    let harness = AIPerfHarness::new_with(cfg).await;

    // Rust reference interval sequence (drop the first draw: it is consumed before the
    // first dispatch, so observed gaps start at the second interval).
    let reference = rust_reference_intervals(SEED, RATE, COUNT);
    let reference_gaps: Vec<f64> = reference[1..].to_vec();

    let parity_gaps = run_python_poisson(&harness, "rust_parity");
    let legacy_gaps = run_python_poisson(&harness, "legacy");

    let n = reference_gaps
        .len()
        .min(parity_gaps.len())
        .min(legacy_gaps.len());
    let reference_gaps = &reference_gaps[..n];
    let parity_err = mean_abs_error(&parity_gaps[..n], reference_gaps);
    let legacy_err = mean_abs_error(&legacy_gaps[..n], reference_gaps);
    let mean_interval = reference_gaps.iter().sum::<f64>() / n as f64;

    // Persist all three schedules as JSONL artifacts for inspection.
    let out_dir = std::env::temp_dir().join("aiperf_poisson_parity");
    std::fs::create_dir_all(&out_dir).expect("create schedule output dir");
    write_schedule_jsonl(
        &out_dir.join("poisson_rust_reference.jsonl"),
        "rust",
        reference_gaps,
    );
    write_schedule_jsonl(
        &out_dir.join("poisson_python_parity.jsonl"),
        "python-parity",
        &parity_gaps[..n],
    );
    write_schedule_jsonl(
        &out_dir.join("poisson_python_legacy.jsonl"),
        "python-legacy",
        &legacy_gaps[..n],
    );

    eprintln!(
        "seeded-poisson parity: n={n} mean_interval={mean_interval:.6}s \
         parity_mean_err={parity_err:.6}s legacy_mean_err={legacy_err:.6}s\n  \
         schedules written under {}",
        out_dir.display()
    );

    // rust_parity must track the Rust reference to within loopback jitter — a small
    // fraction of the mean interval.
    assert!(
        parity_err < mean_interval * 0.15,
        "rust_parity schedule should match the Rust aiperf_runtime::rng reference within jitter, \
         but mean per-gap error {parity_err:.6}s >= 15% of mean interval {mean_interval:.6}s"
    );
    // legacy is a different PRNG + derivation: its exponential draws are unrelated, so its
    // gaps differ from the reference by ~the mean interval. It must be far worse.
    assert!(
        legacy_err > parity_err * 5.0,
        "legacy schedule (different PRNG) should diverge from the Rust reference far more \
         than rust_parity: legacy_err={legacy_err:.6}s parity_err={parity_err:.6}s"
    );
}
