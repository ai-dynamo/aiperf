// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;

const NETWORK_LATENCY_GLOB: &str = "**/*network_latency.jsonl";
const ADJUSTED_KEYS: [&str; 2] = ["network_adjusted_time_to_first_token", "network_rtt"];

const REQUEST_COUNT: u32 = 10;
const CONCURRENCY: u32 = 2;
const WORKERS_MAX: u32 = 1;
const UI: &str = "simple";

const MODEL: &str = "Qwen/Qwen2.5-32B-Instruct";

fn read_network_lines(r: &RunResult) -> Option<Vec<serde_json::Value>> {
    let path = r.artifacts.find_file(NETWORK_LATENCY_GLOB)?;
    let text = std::fs::read_to_string(&path).expect("read network latency jsonl");
    Some(
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse network latency line"))
            .collect(),
    )
}

#[tokio::test]
async fn test_calibration_writes_jsonl_and_adjusted_metrics() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {MODEL} --url {} --endpoint-type chat --streaming \
         --network-latency-automatic --network-latency-ping-interval 0.05 \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);

    let lines = read_network_lines(&r).expect("network latency JSONL artifact was not written");
    assert!(!lines.is_empty(), "network latency JSONL artifact is empty");
    for sample in &lines {
        assert!(sample.get("success").is_some(), "sample missing `success`");
        assert!(
            !sample["target_port"].is_null(),
            "sample target_port is null"
        );
    }

    let export = r.artifacts.json();
    for key in ADJUSTED_KEYS {
        assert!(export.get(key).is_some(), "{key} missing from JSON export");
    }
}

#[tokio::test]
async fn test_baseline_without_flag_has_no_network_artifacts() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {MODEL} --url {} --endpoint-type chat --streaming \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);

    assert!(
        r.artifacts.find_file(NETWORK_LATENCY_GLOB).is_none(),
        "unexpected network latency JSONL artifact"
    );

    let export = r.artifacts.json();
    if let Some(obj) = export.as_object() {
        assert!(
            !obj.keys().any(|k| k.starts_with("network_adjusted_")),
            "unexpected network_adjusted_* keys in export"
        );
        assert!(
            !obj.contains_key("network_rtt"),
            "unexpected network_rtt key in export"
        );
    }
}

#[tokio::test]
async fn test_rtt_override_adjusts_without_probing() {
    let cfg = MockServerConfig {
        ttft: 80.0,
        itl: 10.0,
        workers: 8,
        no_tokenizer: true,
        ..MockServerConfig::default()
    };
    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run(&format!(
        "--model {MODEL} --url {} --endpoint-type chat --streaming \
         --network-latency-mean 5.0 \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);

    let export = r.artifacts.json();
    for key in ADJUSTED_KEYS {
        assert!(
            export.get(key).is_some(),
            "{key} missing from JSON export with override"
        );
    }

    // Fixed RTT shifts latency averages without changing dispersion.
    let rtt_avg = export["network_rtt"]["avg"]
        .as_f64()
        .expect("network_rtt.avg");
    assert!((rtt_avg - 5.0).abs() <= 1e-3, "network_rtt.avg={rtt_avg}");

    for (raw_tag, adjusted_tag) in [
        (
            "time_to_first_token",
            "network_adjusted_time_to_first_token",
        ),
        ("request_latency", "network_adjusted_request_latency"),
    ] {
        let raw_avg = export[raw_tag]["avg"].as_f64().expect("raw avg");
        let adj_avg = export[adjusted_tag]["avg"].as_f64().expect("adjusted avg");
        assert!(
            ((raw_avg - adj_avg) - 5.0).abs() <= 1e-2,
            "{raw_tag}: raw_avg={raw_avg} adj_avg={adj_avg}"
        );
        let raw_std = export[raw_tag]["std"].as_f64().expect("raw std");
        let adj_std = export[adjusted_tag]["std"].as_f64().expect("adjusted std");
        assert!(
            (raw_std - adj_std).abs() <= 1e-6,
            "{raw_tag}: raw_std={raw_std} adj_std={adj_std}"
        );
    }

    assert!(
        r.artifacts.find_file(NETWORK_LATENCY_GLOB).is_none(),
        "unexpected network latency JSONL artifact with override"
    );
}
