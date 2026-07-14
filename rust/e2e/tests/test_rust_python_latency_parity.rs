// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-vs-Python engine latency parity.
//!
//! Boots the mock server at several deterministic, non-zero latencies and runs
//! the same benchmark back-to-back through the native Rust engine (default) and
//! the legacy Python service mesh (`AIPERF_RUNTIME_ENGINE=python`), asserting the
//! two engines measure the same TTFT, inter-token latency, and request latency
//! within a small tolerance.
//!
//! Both runs pass `--use-server-token-count` so each engine derives the
//! output-sequence-length from the server's `usage.completion_tokens` rather
//! than by re-tokenizing the response text. That keeps the ITL denominator
//! (`(last - first) / (osl - 1)`) identical across engines — otherwise Python's
//! client-side re-tokenization yields a different OSL and ITL drifts a few
//! percent purely from token counting, not timing.
//!
//! This is the regression guard for the multi-worker TTFT bug: each HTTP worker
//! sink was constructed with a placeholder run origin of `0`, so its token
//! timestamps were anchored to the RealClock anchor instead of the run origin
//! shared by the observer. TTFT/ITL were then offset by the setup duration —
//! but only when `--workers-max > 1`. The runs below use `--workers-max 4` on
//! purpose so a regression re-appears here.

mod common;

use common::*;

/// Concurrency for both engines. Latency is analytic (scheduler disabled), so
/// per-request latency is independent of concurrency; this just keeps the run
/// short.
const CONCURRENCY: u32 = 8;
/// Requests per run — enough to average out loopback scheduling jitter.
const REQUEST_COUNT: u32 = 64;
/// More than one worker so the multi-worker execution path (the one that
/// regressed) is exercised.
const WORKERS_MAX: u32 = 4;
/// Deterministic short output; keeps runs fast while still streaming.
const OUTPUT_TOKENS: u32 = 8;
/// Allowed relative difference between the two engines' measured latency.
const TOLERANCE: f64 = 0.03;

/// `json()[metric]["avg"]` as f64, panicking with context when absent so a
/// failed/short run surfaces clearly instead of silently reading `0.0`.
fn metric_avg(result: &RunResult, metric: &str) -> f64 {
    result
        .artifacts
        .json()
        .get(metric)
        .and_then(|value| value.get("avg"))
        .and_then(|value| value.as_f64())
        .unwrap_or_else(|| panic!("metric {metric:?} avg missing from artifacts"))
}

/// Rust and Python engines must measure the same TTFT and request latency
/// (within [`TOLERANCE`]) for a mock server configured at several fixed
/// latencies.
///
/// `gpt-4` is used deliberately: the mock emits plain `content` for it (reasoning
/// models stream `reasoning_content`, which the two engines count differently).
#[tokio::test]
async fn test_rust_python_latency_parity() {
    // (configured TTFT ms, configured ITL ms) sweep.
    for (ttft_ms, itl_ms) in [(150.0_f64, 15.0_f64), (250.0_f64, 25.0_f64)] {
        let mut cfg = MockServerConfig::default();
        cfg.no_tokenizer = true;
        cfg.ttft = ttft_ms;
        cfg.itl = itl_ms;
        let harness = AIPerfHarness::new_with(cfg).await;

        let args = format!(
            "--model gpt-4 --url {} --endpoint-type chat --streaming \
             --concurrency {CONCURRENCY} --request-count {REQUEST_COUNT} \
             --workers-max {WORKERS_MAX} --synthetic-input-tokens-mean 64 \
             --output-tokens-mean {OUTPUT_TOKENS} --use-server-token-count \
             --ui simple",
            harness.mock.url
        );

        // Rust engine (default). Read metrics before the Python run overwrites
        // the artifact directory.
        let rust = harness.run(&args);
        assert!(
            rust.success(),
            "rust run failed (ttft={ttft_ms} itl={itl_ms}):\n{}",
            rust.stderr
        );
        let rust_ttft = metric_avg(&rust, "time_to_first_token");
        let rust_itl = metric_avg(&rust, "inter_token_latency");
        let rust_latency = metric_avg(&rust, "request_latency");

        // Python legacy mesh, same mock, back-to-back.
        let python = harness.run_env(&args, &[("AIPERF_RUNTIME_ENGINE", "python")]);
        assert!(
            python.success(),
            "python run failed (ttft={ttft_ms} itl={itl_ms}):\n{}",
            python.stderr
        );
        let python_ttft = metric_avg(&python, "time_to_first_token");
        let python_itl = metric_avg(&python, "inter_token_latency");
        let python_latency = metric_avg(&python, "request_latency");

        // Ground-truth sanity: the Rust engine must measure near the configured
        // TTFT. This alone would have caught the multi-worker regression (which
        // reported ~4.5x the configured TTFT), independent of the Python engine.
        let rust_ttft_error = (rust_ttft - ttft_ms).abs() / ttft_ms;
        assert!(
            rust_ttft_error <= 0.15,
            "rust TTFT {rust_ttft:.2}ms is not near configured {ttft_ms}ms \
             ({:.1}% off)",
            rust_ttft_error * 100.0
        );

        let ttft_diff = (rust_ttft - python_ttft).abs() / python_ttft;
        assert!(
            ttft_diff <= TOLERANCE,
            "TTFT parity exceeded {:.0}% at ttft={ttft_ms} itl={itl_ms}: \
             rust={rust_ttft:.2}ms python={python_ttft:.2}ms ({:.1}%)",
            TOLERANCE * 100.0,
            ttft_diff * 100.0
        );

        let itl_diff = (rust_itl - python_itl).abs() / python_itl;
        assert!(
            itl_diff <= TOLERANCE,
            "ITL parity exceeded {:.0}% at ttft={ttft_ms} itl={itl_ms}: \
             rust={rust_itl:.2}ms python={python_itl:.2}ms ({:.1}%)",
            TOLERANCE * 100.0,
            itl_diff * 100.0
        );

        let latency_diff = (rust_latency - python_latency).abs() / python_latency;
        assert!(
            latency_diff <= TOLERANCE,
            "request-latency parity exceeded {:.0}% at ttft={ttft_ms} itl={itl_ms}: \
             rust={rust_latency:.2}ms python={python_latency:.2}ms ({:.1}%)",
            TOLERANCE * 100.0,
            latency_diff * 100.0
        );
    }
}
