// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;

use std::io::Write;

use aiperf_runtime::rng::{RandomGenerator, RngRoot, namespace};
use common::*;

const SEED: u64 = 42;
const RATE: f64 = 50.0;
const COUNT: usize = 150;

fn rust_reference_intervals(seed: u64, rate: f64, count: usize) -> Vec<f64> {
    let child = RngRoot::new(Some(seed))
        .derive_seed(namespace::TIMING_REQUEST_POISSON_INTERVAL)
        .expect("seeded root yields a poisson-interval seed");
    let mut rng = RandomGenerator::from_seed(Some(child));
    (0..count)
        .map(|_| rng.expovariate(rate).expect("rate > 0"))
        .collect()
}

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
    // Credit issue time is the closest observed point to scheduled arrival.
    let origin = issued[0] as f64;
    issued.iter().map(|s| (*s as f64 - origin) / 1e9).collect()
}

fn gaps(offsets: &[f64]) -> Vec<f64> {
    offsets.windows(2).map(|w| w[1] - w[0]).collect()
}

fn mean_abs_error(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    assert!(n > 0, "empty gap sequence");
    (0..n).map(|i| (a[i] - b[i]).abs()).sum::<f64>() / n as f64
}

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

#[tokio::test]
async fn test_seeded_poisson_schedule_parity() {
    let mut cfg = MockServerConfig::default();
    cfg.no_tokenizer = true;
    cfg.ttft = 2.0;
    cfg.itl = 1.0;
    let harness = AIPerfHarness::new_with(cfg).await;

    let reference = rust_reference_intervals(SEED, RATE, COUNT);
    // The first draw precedes the first dispatch, so observed gaps start at draw two.
    let reference_gaps: Vec<f64> = reference[1..].to_vec();

    let parity_gaps = run_python_poisson(&harness, "rust_parity");
    let comparison_gaps = run_python_poisson(&harness, "legacy");

    let n = reference_gaps
        .len()
        .min(parity_gaps.len())
        .min(comparison_gaps.len());
    let reference_gaps = &reference_gaps[..n];
    let parity_err = mean_abs_error(&parity_gaps[..n], reference_gaps);
    let comparison_err = mean_abs_error(&comparison_gaps[..n], reference_gaps);
    let mean_interval = reference_gaps.iter().sum::<f64>() / n as f64;

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
        &out_dir.join("poisson_python_comparison.jsonl"),
        "python-comparison",
        &comparison_gaps[..n],
    );

    eprintln!(
        "seeded-poisson parity: n={n} mean_interval={mean_interval:.6}s \
         parity_mean_err={parity_err:.6}s comparison_mean_err={comparison_err:.6}s\n  \
         schedules written under {}",
        out_dir.display()
    );

    // Allow bounded loopback jitter while requiring schedule-level agreement.
    assert!(
        parity_err < mean_interval * 0.15,
        "rust_parity schedule should match the Rust aiperf_runtime::rng reference within jitter, \
         but mean per-gap error {parity_err:.6}s >= 15% of mean interval {mean_interval:.6}s"
    );
    assert!(
        comparison_err > parity_err * 5.0,
        "comparison schedule should diverge from the Rust reference far more than \
         rust_parity: comparison_err={comparison_err:.6}s parity_err={parity_err:.6}s"
    );
}
