// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use serde_json::Value;

const LEGACY_NVIDIA_FIELDS: &[&str] = &[
    "gpu_power_usage",
    "energy_consumption",
    "gpu_utilization",
    "mem_utilization",
    "gpu_memory_used",
    "gpu_temperature",
    "decoder_utilization",
    "encoder_utilization",
    "jpg_utilization",
    "sm_utilization",
    "xid_errors",
    "power_violation",
];

fn has_gpu_telemetry(json: &Value) -> bool {
    json.get("telemetry_data")
        .and_then(|t| t.get("endpoints"))
        .and_then(|e| e.as_object())
        .map(|m| !m.is_empty())
        .unwrap_or(false)
}

#[tokio::test]
async fn test_native_gpu_platform_propagates_through_raw_profile_artifacts() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new_with(MockServerConfig {
        fast: true,
        no_tokenizer: true,
        workers: 8,
        random_seed: Some(17),
        dcgm_seed: Some(23),
        ..MockServerConfig::default()
    })
    .await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --gpu-telemetry {dcgm} --request-count 8 --concurrency 2 \
         --workers-max 1 --random-seed 7 --tokenizer builtin \
         --synthetic-input-tokens-mean 32 --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean 8 --output-tokens-stddev 0 \
         --export-level raw --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "profile failed: {}", r.stderr);
    assert_eq!(r.artifacts.raw_records().len(), 8);

    let telemetry_path = r
        .artifacts
        .find_file("**/*gpu_telemetry*.jsonl")
        .expect("native telemetry JSONL");
    let telemetry_text = std::fs::read_to_string(telemetry_path).expect("read telemetry JSONL");
    let telemetry_rows = telemetry_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("valid telemetry JSONL row"))
        .collect::<Vec<_>>();
    assert!(!telemetry_rows.is_empty(), "expected telemetry JSONL rows");
    for row in &telemetry_rows {
        assert_eq!(row["platform"], "nvidia");
        let metrics = row["telemetry_data"]
            .as_object()
            .expect("telemetry_data object");
        assert!(metrics.contains_key("nvidia_power_usage"));
        for legacy in LEGACY_NVIDIA_FIELDS {
            assert!(
                !metrics.contains_key(*legacy),
                "telemetry JSONL emitted legacy NVIDIA field {legacy}"
            );
        }
    }

    let summary = r.artifacts.json();
    let endpoints = summary
        .pointer("/telemetry_data/endpoints")
        .and_then(Value::as_object)
        .expect("telemetry summary endpoints");
    assert!(
        !endpoints.is_empty(),
        "expected telemetry summary endpoints"
    );
    for endpoint in endpoints.values() {
        for gpu in endpoint["gpus"].as_object().expect("summary gpus").values() {
            assert_eq!(gpu["platform"], "nvidia");
            let metrics = gpu["metrics"].as_object().expect("summary metrics");
            assert!(metrics.contains_key("nvidia_power_usage"));
            for legacy in LEGACY_NVIDIA_FIELDS {
                assert!(
                    !metrics.contains_key(*legacy),
                    "telemetry summary emitted legacy NVIDIA field {legacy}"
                );
            }
        }
    }
}

#[tokio::test]
async fn test_gpu_telemetry() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --endpoint-type chat --gpu-telemetry {dcgm} --streaming \
         --request-count 100 --concurrency 2 --workers-max 2 --ui dashboard",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 100);

    let json = r.artifacts.json();
    assert!(
        has_gpu_telemetry(&json),
        "GPU telemetry should be collected"
    );

    let endpoints = json["telemetry_data"]["endpoints"]
        .as_object()
        .expect("telemetry_data.endpoints must be an object");
    assert!(!endpoints.is_empty());

    let counter_metrics = [
        "nvidia_energy_consumption",
        "nvidia_xid_errors",
        "nvidia_power_violation",
    ];

    for (_dcgm_url, endpoint) in endpoints {
        let gpus = endpoint["gpus"]
            .as_object()
            .expect("endpoint.gpus must be an object");
        assert!(!gpus.is_empty());

        for gpu_data in gpus.values() {
            let metrics = gpu_data["metrics"]
                .as_object()
                .expect("gpu.metrics must be an object");
            assert!(!metrics.is_empty());

            for (metric_name, metric_value) in metrics {
                assert!(!metric_value.is_null());
                assert!(!metric_value["avg"].is_null());
                assert!(!metric_value["unit"].is_null());
                // Counter summaries omit min/max.
                if !counter_metrics.contains(&metric_name.as_str()) {
                    assert!(!metric_value["min"].is_null());
                    assert!(!metric_value["max"].is_null());
                }
            }
        }
    }
}

#[tokio::test]
async fn test_gpu_telemetry_export() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --endpoint-type chat --gpu-telemetry {dcgm} --streaming \
         --request-count 50 --concurrency 2 --workers-max 2",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 50);
    assert!(has_gpu_telemetry(&r.artifacts.json()));

    let export = r
        .artifacts
        .find_file("**/gpu_telemetry_export.jsonl")
        .expect("GPU telemetry export file should exist");
    let content = std::fs::read_to_string(&export).expect("read export file");
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        !lines.is_empty(),
        "Export file should contain telemetry records"
    );

    let mut gpu_uuids = std::collections::HashSet::new();

    for line in &lines {
        let record: Value = serde_json::from_str(line).expect("valid telemetry record JSON");

        assert!(record["timestamp_ns"].as_i64().expect("timestamp_ns") > 0);
        assert!(!record["dcgm_url"].is_null());
        assert!(record["gpu_index"].as_i64().expect("gpu_index") >= 0);
        assert!(!record["gpu_uuid"].is_null());
        assert!(!record["gpu_model_name"].is_null());
        assert!(!record["telemetry_data"].is_null());

        gpu_uuids.insert(record["gpu_uuid"].as_str().unwrap_or_default().to_string());
    }

    // Asynchronous scrapes need not be timestamp-ordered.
    assert!(
        gpu_uuids.len() >= 2,
        "Should have records from at least two GPUs"
    );
}

#[tokio::test]
async fn test_gpu_telemetry_export_with_custom_prefix() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --endpoint-type chat --gpu-telemetry {dcgm} --streaming \
         --request-count 25 --concurrency 1 --workers-max 1 \
         --profile-export-prefix custom_test",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);

    if let Some(export) = r.artifacts.find_file("**/custom_test_gpu_telemetry.jsonl") {
        let content = std::fs::read_to_string(&export).expect("read export file");
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(
            !lines.is_empty(),
            "Export file should contain telemetry records"
        );

        let first: Value = serde_json::from_str(lines[0]).expect("valid first record JSON");
        assert!(first["timestamp_ns"].as_i64().expect("timestamp_ns") > 0);
        assert!(!first["dcgm_url"].is_null());
    }
}

#[tokio::test]
async fn test_gpu_telemetry_disabled() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --endpoint-type chat --streaming \
         --request-count 25 --concurrency 1 --workers-max 1 --no-gpu-telemetry",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 25);

    assert!(
        !has_gpu_telemetry(&r.artifacts.json()),
        "GPU telemetry should not be collected"
    );

    assert!(
        r.artifacts.find_file("**/*gpu_telemetry*.jsonl").is_none(),
        "Unexpected GPU telemetry files present"
    );
}
