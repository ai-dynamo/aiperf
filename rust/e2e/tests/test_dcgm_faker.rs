// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;

use aiperf_mock_server::dcgm::{DcgmFaker, lookup_gpu};
use aiperf_runtime::gpu_telemetry::{
    DcgmPrometheusDecoder, GpuTelemetryDecoder, GpuTelemetryRecord,
};

const GPU_CONFIGS: &[&str] = &[
    "rtx6000", "a100", "h100", "h100-sxm", "h200", "b200", "gb200",
];

#[test]
fn all_gpu_configs_have_a_parametrized_test() {
    for name in GPU_CONFIGS {
        assert!(lookup_gpu(name).is_some(), "missing gpu config: {name}");
    }
}

fn decode(text: &str) -> Vec<GpuTelemetryRecord> {
    let scrape = DcgmPrometheusDecoder::new()
        .decode("http://fake", 0, text)
        .expect("faker output should decode");
    scrape.records
}

fn faker_output_parsed_by_real_telemetry_collector(gpu_name: &str) {
    let cfg = lookup_gpu(gpu_name).expect("known gpu config");
    let faker = DcgmFaker::new(gpu_name, 2, Some(42), "testnode").unwrap();
    let metrics_text = faker.generate();

    let records = decode(&metrics_text);

    assert_eq!(records.len(), 2);

    let mut gpu_indices: Vec<i32> = records.iter().map(|r| r.metadata.gpu_index).collect();
    gpu_indices.sort_unstable();
    assert_eq!(gpu_indices, vec![0, 1]);

    for record in &records {
        assert_eq!(record.metadata.gpu_model_name, cfg.model);
        assert_eq!(record.metadata.hostname.as_deref(), Some("testnode"));
        assert!(record.metadata.gpu_uuid.starts_with("GPU-"));
        assert!(record.metadata.pci_bus_id.is_some());
        assert!(record.metadata.device.is_some());

        let util = record.metrics["gpu_utilization"];
        let power = record.metrics["gpu_power_usage"];
        let temp = record.metrics["gpu_temperature"];

        assert!(record.metrics.contains_key("energy_consumption"));
        assert!(record.metrics.contains_key("gpu_memory_used"));
        assert!(record.metrics.contains_key("xid_errors"));
        assert!(record.metrics.contains_key("power_violation"));

        assert!((0.0..=100.0).contains(&util));
        assert!(power > 0.0 && power <= cfg.max_power_w as f64);
        assert!(temp > 0.0 && temp <= 100.0);
    }
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_rtx6000() {
    faker_output_parsed_by_real_telemetry_collector("rtx6000");
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_a100() {
    faker_output_parsed_by_real_telemetry_collector("a100");
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_h100() {
    faker_output_parsed_by_real_telemetry_collector("h100");
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_h100_sxm() {
    faker_output_parsed_by_real_telemetry_collector("h100-sxm");
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_h200() {
    faker_output_parsed_by_real_telemetry_collector("h200");
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_b200() {
    faker_output_parsed_by_real_telemetry_collector("b200");
}

#[tokio::test]
async fn test_faker_output_parsed_by_real_telemetry_collector_gb200() {
    faker_output_parsed_by_real_telemetry_collector("gb200");
}

#[tokio::test]
async fn test_load_affects_telemetry_records() {
    let faker = DcgmFaker::new("b200", 1, Some(42), "localhost").unwrap();

    faker.set_load(0.1);
    let low_records = decode(&faker.generate());
    let low = &low_records[0].metrics;

    faker.set_load(0.9);
    let high_records = decode(&faker.generate());
    let high = &high_records[0].metrics;

    assert!(high["gpu_power_usage"] > low["gpu_power_usage"]);
    assert!(high["gpu_temperature"] > low["gpu_temperature"]);
    assert!(high["gpu_utilization"] > low["gpu_utilization"]);
    assert!(high["gpu_memory_used"] > low["gpu_memory_used"]);
}
