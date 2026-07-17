// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::PathBuf;

use serde_json::Value;
use tempfile::TempDir;

// The faker emits eight of the twelve default telemetry metrics.
const DCGM_FAKER_DEFAULT_METRIC_COUNT: usize = 8;

const FREQ_MEGAHERTZ: &str = "MHz";
const TEMP_CELSIUS: &str = "°C";
const GENERIC_PERCENT: &str = "%";

fn skip_platform() -> bool {
    cfg!(target_os = "macos") || cfg!(target_os = "windows")
}

fn custom_gpu_metrics_csv(dir: &TempDir) -> PathBuf {
    let csv_content = "# Custom GPU Metrics Test File
# Format: DCGM_FIELD, metric_type, help_message

# Custom clock metrics (DCGMFaker returns these)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)

# Custom temperature metrics (DCGMFaker returns this)
DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (in °C)

# This is already a default metric (maps to mem_utilization), included to test deduplication
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory copy utilization (in %)
";
    write_text(dir.path(), "custom_gpu_metrics.csv", csv_content)
}

fn custom_gpu_metrics_csv_with_defaults(dir: &TempDir) -> PathBuf {
    let csv_content = "# Mix of default and custom metrics
# This should deduplicate the default metrics

# Default metrics (should be skipped to avoid duplicates)
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)
DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W)

# Custom metrics (should be added - DCGMFaker returns these)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)
";
    write_text(dir.path(), "custom_gpu_metrics.csv", csv_content)
}

fn custom_gpu_metrics_csv_invalid(dir: &TempDir) -> PathBuf {
    let csv_content = "# CSV with invalid entries for error handling tests

# Invalid entries (should be skipped)
INVALID_FIELD, gauge, Invalid field name
DCGM_FI_DEV_GPU_UTIL, invalid_type, Invalid metric type

# Valid entries (should be processed)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
";
    write_text(dir.path(), "custom_gpu_metrics.csv", csv_content)
}

fn endpoints(json: &Value) -> Value {
    json.get("telemetry_data")
        .and_then(|t| t.get("endpoints"))
        .cloned()
        .unwrap_or(Value::Null)
}

fn has_gpu_telemetry(json: &Value) -> bool {
    endpoints(json)
        .as_object()
        .map(|m| !m.is_empty())
        .unwrap_or(false)
}

fn run_with_csv(h: &AIPerfHarness, csv_path: &PathBuf) -> RunResult {
    let dcgm = h.mock.dcgm_urls().join(" ");
    h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --tokenizer gpt2 --endpoint-type chat \
         --gpu-telemetry {} {} \
         --benchmark-duration 2 --concurrency 2 --workers-max 2",
        h.mock.url,
        csv_path.display(),
        dcgm
    ))
}

#[tokio::test]
async fn test_custom_metrics_csv_loading_basic() {
    if skip_platform() {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dir = TempDir::new().unwrap();
    let csv_path = custom_gpu_metrics_csv(&dir);
    let r = run_with_csv(&h, &csv_path);

    assert!(r.artifacts.request_count() > 0.0, "exit={}", r.exit_code);
    let json = r.artifacts.json();
    assert!(has_gpu_telemetry(&json));

    let eps = endpoints(&json);
    let eps = eps.as_object().expect("endpoints object");
    assert!(!eps.is_empty());

    for (_dcgm_url, endpoint_data) in eps {
        let gpus = endpoint_data.get("gpus").and_then(|g| g.as_object());
        let gpus = gpus.expect("gpus object");
        assert!(!gpus.is_empty());

        for (_gpu_id, gpu_data) in gpus {
            let metrics = gpu_data
                .get("metrics")
                .and_then(|m| m.as_object())
                .expect("metrics object");

            let expected_min_metrics = DCGM_FAKER_DEFAULT_METRIC_COUNT + 3;
            assert!(
                metrics.len() >= expected_min_metrics,
                "Expected at least {expected_min_metrics} metrics, got {}",
                metrics.len()
            );

            for metric_name in ["sm_clock", "mem_clock", "memory_temp"] {
                assert!(
                    metrics.contains_key(metric_name),
                    "Missing {metric_name}. Available metrics: {:?}",
                    metrics.keys().collect::<Vec<_>>()
                );
            }

            for (metric_name, metric_value) in metrics {
                assert!(
                    !metric_value.is_null(),
                    "Metric {metric_name} has None value"
                );
                assert!(
                    !metric_value
                        .get("unit")
                        .map(|u| u.is_null())
                        .unwrap_or(true),
                    "Metric {metric_name} has None unit"
                );
            }

            let unit = |name: &str| metrics[name]["unit"].as_str().unwrap_or("");
            assert_eq!(
                unit("sm_clock"),
                FREQ_MEGAHERTZ,
                "sm_clock unit is {}, expected {FREQ_MEGAHERTZ}",
                unit("sm_clock")
            );
            assert_eq!(
                unit("mem_clock"),
                FREQ_MEGAHERTZ,
                "mem_clock unit is {}, expected {FREQ_MEGAHERTZ}",
                unit("mem_clock")
            );
            assert_eq!(
                unit("memory_temp"),
                TEMP_CELSIUS,
                "memory_temp unit is {}, expected {TEMP_CELSIUS}",
                unit("memory_temp")
            );
            assert_eq!(
                unit("mem_utilization"),
                GENERIC_PERCENT,
                "mem_utilization unit is {}, expected {GENERIC_PERCENT}",
                unit("mem_utilization")
            );
        }
    }
}

#[tokio::test]
async fn test_custom_metrics_deduplication() {
    if skip_platform() {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dir = TempDir::new().unwrap();
    let csv_path = custom_gpu_metrics_csv_with_defaults(&dir);
    let r = run_with_csv(&h, &csv_path);

    let json = r.artifacts.json();
    assert!(has_gpu_telemetry(&json));

    let eps = endpoints(&json);
    let eps = eps.as_object().expect("endpoints object");

    for (_dcgm_url, endpoint_data) in eps {
        let gpus = endpoint_data
            .get("gpus")
            .and_then(|g| g.as_object())
            .expect("gpus object");
        for (_gpu_id, gpu_data) in gpus {
            let metrics = gpu_data
                .get("metrics")
                .and_then(|m| m.as_object())
                .expect("metrics object");

            assert!(metrics.contains_key("gpu_utilization"));
            assert!(metrics.contains_key("gpu_power_usage"));

            assert!(metrics.contains_key("sm_clock"));
            assert!(metrics.contains_key("mem_clock"));

            let expected_min_metrics = DCGM_FAKER_DEFAULT_METRIC_COUNT + 2;
            assert!(metrics.len() >= expected_min_metrics);
        }
    }
}

#[tokio::test]
async fn test_invalid_csv_fallback_to_defaults() {
    if skip_platform() {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dir = TempDir::new().unwrap();
    let csv_path = custom_gpu_metrics_csv_invalid(&dir);
    let r = run_with_csv(&h, &csv_path);

    assert!(r.artifacts.request_count() > 0.0);
    let json = r.artifacts.json();
    assert!(has_gpu_telemetry(&json));

    let eps = endpoints(&json);
    let eps = eps.as_object().expect("endpoints object");

    for (_dcgm_url, endpoint_data) in eps {
        let gpus = endpoint_data
            .get("gpus")
            .and_then(|g| g.as_object())
            .expect("gpus object");
        for (_gpu_id, gpu_data) in gpus {
            let metrics = gpu_data
                .get("metrics")
                .and_then(|m| m.as_object())
                .expect("metrics object");

            assert!(metrics.contains_key("sm_clock"));

            let expected_min_metrics = DCGM_FAKER_DEFAULT_METRIC_COUNT + 1;
            assert!(metrics.len() >= expected_min_metrics);
        }
    }
}

#[tokio::test]
async fn test_nonexistent_csv_file_error() {
    if skip_platform() {
        return;
    }
    let dir = TempDir::new().unwrap();
    let nonexistent_csv = dir.path().join("nonexistent_custom_gpu_metrics.csv");

    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --tokenizer gpt2 --endpoint-type chat \
         --gpu-telemetry {} {} \
         --request-count 10 --concurrency 2",
        h.mock.url,
        nonexistent_csv.display(),
        dcgm
    ));

    assert_ne!(r.exit_code, 0);
}
