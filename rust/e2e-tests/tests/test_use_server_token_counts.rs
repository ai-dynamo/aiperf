// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

fn metric(json: &serde_json::Value, metric: &str, key: &str) -> f64 {
    json[metric][key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing numeric {metric}.{key} in aiperf.json"))
}

async fn run_server_token_counts_case(streaming: bool, extra_inputs: &str) {
    let h = AIPerfHarness::new_with(MockServerConfig {
        fast: true,
        workers: 1,
        ..Default::default()
    })
    .await;

    let streaming_flag = if streaming { "--streaming" } else { "" };

    let r = h.run(&format!(
        "--model openai/gpt-oss-120b \
            --url {} \
            --endpoint-type chat \
            {streaming_flag} \
            {extra_inputs} \
            --use-server-token-count \
            --request-count {DEFAULT_REQUEST_COUNT} \
            --concurrency {DEFAULT_CONCURRENCY} \
            --ui simple",
        h.mock.url
    ));

    assert!(r.success(), "aiperf profile failed: {}", r.stderr);

    let json = r.artifacts.json();

    for key in ["avg", "min", "max", "p50", "p75", "p90", "p95", "p99"] {
        let isl = metric(&json, "input_sequence_length", key);
        let prompt = metric(&json, "usage_prompt_tokens", key);
        assert!(
            (isl - prompt).abs() < 1e-6,
            "input_sequence_length.{key} ({isl}) should match usage_prompt_tokens.{key} ({prompt})"
        );

        let otc = metric(&json, "output_token_count", key);
        let completion = metric(&json, "usage_completion_tokens", key);
        let reasoning = metric(&json, "usage_reasoning_tokens", key);
        assert!(
            (otc - (completion - reasoning)).abs() < 1e-6,
            "output_token_count.{key} ({otc}) should match usage_completion - usage_reasoning ({})",
            completion - reasoning
        );

        let rtc = metric(&json, "reasoning_token_count", key);
        assert!(
            (rtc - reasoning).abs() < 1e-6,
            "reasoning_token_count.{key} ({rtc}) should match usage_reasoning_tokens.{key} ({reasoning})"
        );
    }

    assert!(
        json.get("usage_prompt_tokens_diff_pct").is_none(),
        "usage_prompt_tokens_diff_pct should not be present"
    );
    assert!(
        json.get("usage_completion_tokens_diff_pct").is_none(),
        "usage_completion_tokens_diff_pct should not be present"
    );
    assert!(
        json.get("usage_reasoning_tokens_diff_pct").is_none(),
        "usage_reasoning_tokens_diff_pct should not be present"
    );
    assert!(
        json.get("usage_discrepancy_count").is_none(),
        "usage_discrepancy_count should not be present"
    );
}

#[tokio::test]
async fn test_server_token_counts_match_primary_metrics_non_streaming() {
    run_server_token_counts_case(false, "").await;
}

#[tokio::test]
async fn test_server_token_counts_match_primary_metrics_streaming() {
    run_server_token_counts_case(
        true,
        r#"--extra-inputs '{"stream_options": {"include_usage": true}}'"#,
    )
    .await;
}
