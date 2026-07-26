// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;

use common::*;

const CONCURRENCY: u32 = 8;
const REQUEST_COUNT: u32 = 64;
const WORKERS_MAX: u32 = 4;
const OUTPUT_TOKENS: u32 = 8;
// Loopback scheduling permits a 3% relative engine difference.
const TOLERANCE: f64 = 0.03;

fn metric_avg(result: &RunResult, metric: &str) -> f64 {
    result
        .artifacts
        .json()
        .get(metric)
        .and_then(|value| value.get("avg"))
        .and_then(|value| value.as_f64())
        .unwrap_or_else(|| panic!("metric {metric:?} avg missing from artifacts"))
}

#[tokio::test]
async fn test_rust_python_latency_parity() {
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

        let rust = harness.run(&args);
        assert!(
            rust.success(),
            "rust run failed (ttft={ttft_ms} itl={itl_ms}):\n{}",
            rust.stderr
        );
        let rust_ttft = metric_avg(&rust, "time_to_first_token");
        let rust_itl = metric_avg(&rust, "inter_token_latency");
        let rust_latency = metric_avg(&rust, "request_latency");

        // Run back-to-back against the same mock to limit environmental drift.
        let python = harness.run_env(&args, &[("AIPERF_RUNTIME_ENGINE", "python")]);
        assert!(
            python.success(),
            "python run failed (ttft={ttft_ms} itl={itl_ms}):\n{}",
            python.stderr
        );
        let python_ttft = metric_avg(&python, "time_to_first_token");
        let python_itl = metric_avg(&python, "inter_token_latency");
        let python_latency = metric_avg(&python, "request_latency");

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
