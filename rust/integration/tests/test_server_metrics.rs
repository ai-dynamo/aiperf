// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Tests for server metrics collection and reporting.
//
// Ported from `tests/integration/test_server_metrics.py`. These tests verify the
// full end-to-end flow of server metrics collection, including scraping from
// multiple mock server endpoints and validating the exported data
// (JSON, JSONL, CSV, Parquet).

use serde_json::Value;

// ============================================================================
// Local helpers replicating the Python AIPerfCLI result convenience accessors.
// ============================================================================

/// Load the server-metrics JSON export as a `Value` (`Null` when absent).
fn server_metrics_json(r: &RunResult) -> Value {
    r.artifacts.server_metrics_json()
}

/// Assert the server-metrics JSON export exists and carries a non-empty summary
/// (at least one successful endpoint) and at least one metric.
fn assert_server_metrics_valid(r: &RunResult) {
    let j = server_metrics_json(r);
    assert!(!j.is_null(), "server metrics JSON export should exist");
    let successful = j["summary"]["endpoints_successful"].as_array();
    assert!(
        successful.map_or(false, |a| !a.is_empty()),
        "should have at least one successful endpoint"
    );
    assert!(
        j["metrics"].as_object().map_or(false, |m| !m.is_empty()),
        "should have at least one metric"
    );
}

/// True when the metric name is present in the JSON export's `metrics` map.
fn has_server_metric(r: &RunResult, name: &str) -> bool {
    server_metrics_json(r)["metrics"].get(name).is_some()
}

/// Return the metric `Value` for `name`, if present.
fn get_server_metric(r: &RunResult, name: &str) -> Value {
    server_metrics_json(r)["metrics"]
        .get(name)
        .cloned()
        .unwrap_or(Value::Null)
}

/// The `summary.endpoints_successful` list as owned strings.
fn endpoints_successful(r: &RunResult) -> Vec<String> {
    server_metrics_json(r)["summary"]["endpoints_successful"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// True when a file matching the glob exists under the artifact dir.
fn has_file(r: &RunResult, glob: &str) -> bool {
    r.artifacts.find_file(glob).is_some()
}

// ============================================================================
// Basic Server Metrics Tests
// ============================================================================

/// Server metrics are auto-collected from base_url/metrics without --server-metrics.
///
/// When no --server-metrics flag is provided, AIPerf should automatically scrape
/// server metrics from the inference endpoint's base URL + /metrics.
#[tokio::test]
async fn test_server_metrics_auto_collected() {
    // Isolated mock server with workers=1 to avoid Prometheus metrics issues.
    let mut cfg = aiperf_mock_rs::config::MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    cfg.no_tokenizer = true;
    let h = AIPerfHarness::new_with(cfg).await;

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --ui simple",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, 50);

    // Server metrics should be auto-collected from default /metrics endpoint.
    assert_server_metrics_valid(&r);

    // Verify we collected AIPerf mock server metrics (default endpoint).
    assert!(has_server_metric(&r, "aiperf_mock_requests"));
    assert!(has_server_metric(&r, "aiperf_mock_request_latency_seconds"));
    assert!(has_server_metric(&r, "aiperf_mock_time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "aiperf_mock_tokens_streamed"));

    // Verify the auto-collected endpoint is correct.
    let expected_endpoint = format!("http://127.0.0.1:{}/metrics", h.mock.port);
    let successful = endpoints_successful(&r);
    assert!(
        successful.contains(&expected_endpoint),
        "Expected {expected_endpoint} in successful endpoints: {successful:?}"
    );
}

// ============================================================================
// Multiple Endpoints Tests
// ============================================================================

/// Server metrics collection from multiple endpoints (vLLM + SGLang).
#[tokio::test]
async fn test_server_metrics_multiple_endpoints_vllm_sglang() {
    let h = AIPerfHarness::new().await;
    let urls = h.mock.server_metrics_urls();
    let vllm_url = &urls["vllm"];
    let sglang_url = &urls["sglang"];

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --server-metrics {vllm_url} {sglang_url}",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, 50);
    assert_server_metrics_valid(&r);

    // Default /metrics endpoint is always auto-collected (default + vllm + sglang).
    assert!(endpoints_successful(&r).len() >= 2);

    // Verify vLLM metrics.
    assert!(has_server_metric(&r, "vllm:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "vllm:time_to_first_token_seconds"));

    // Verify SGLang metrics.
    assert!(has_server_metric(&r, "sglang:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "sglang:time_to_first_token_seconds"));
}

// ============================================================================
// Ultimate Full Stack Test
// ============================================================================

/// Ultimate test: Server metrics from ALL available mock endpoints.
///
/// Collects metrics from vLLM, SGLang, TensorRT-LLM, Dynamo frontend, Dynamo
/// prefill, and Dynamo decode endpoints simultaneously.
#[tokio::test]
async fn test_server_metrics_all_endpoints() {
    // Isolated mock server with workers=1 to avoid Prometheus metrics issues.
    let mut cfg = aiperf_mock_rs::config::MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    cfg.no_tokenizer = true;
    let h = AIPerfHarness::new_with(cfg).await;

    let m = h.mock.server_metrics_urls();
    let all_urls = [
        &m["vllm"],
        &m["sglang"],
        &m["trtllm"],
        &m["dynamo_frontend"],
        &m["dynamo_prefill"],
        &m["dynamo_decode"],
    ]
    .iter()
    .map(|s| s.as_str())
    .collect::<Vec<_>>()
    .join(" ");

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 100 \
         --concurrency 4 \
         --workers-max 2 \
         --server-metrics {all_urls}",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, 100);
    assert_server_metrics_valid(&r);

    // Verify all 6+ endpoints were successful (default + 6 explicit).
    let successful = endpoints_successful(&r);
    assert!(
        successful.len() >= 6,
        "Expected at least 6 successful endpoints, got {}: {successful:?}",
        successful.len()
    );

    // Verify vLLM metrics.
    assert!(has_server_metric(&r, "vllm:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "vllm:time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "vllm:inter_token_latency_seconds"));
    assert!(has_server_metric(&r, "vllm:kv_cache_usage_perc"));

    // Verify SGLang metrics.
    assert!(has_server_metric(&r, "sglang:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "sglang:time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "sglang:gen_throughput"));

    // Verify TRT-LLM metrics.
    assert!(has_server_metric(&r, "trtllm:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "trtllm:time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "trtllm:time_per_output_token_seconds"));

    // Verify Dynamo frontend metrics. `dynamo_frontend_request_duration_seconds`
    // intentionally not asserted (histogram emits no rows until first .observe()).
    assert!(has_server_metric(&r, "dynamo_frontend_time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "dynamo_frontend_inter_token_latency_seconds"));

    // Verify Dynamo component metrics.
    assert!(has_server_metric(&r, "dynamo_component_request_duration_seconds"));
    assert!(has_server_metric(&r, "dynamo_component_requests"));
}

// ============================================================================
// Export File Validation Tests
// ============================================================================

/// Test server metrics export files (JSON, JSONL, CSV, Parquet) are valid.
#[tokio::test]
async fn test_server_metrics_export_files() {
    let h = AIPerfHarness::new().await;
    let m = h.mock.server_metrics_urls();
    let urls = format!("{} {}", m["vllm"], m["sglang"]);

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --server-metrics-formats json csv jsonl parquet \
         --server-metrics {urls}",
        h.mock.url
    ));

    // Verify all export files exist.
    assert!(has_file(&r, "**/*server_metrics_export.json"));
    assert!(has_file(&r, "**/*server_metrics_export.jsonl"));
    assert!(has_file(&r, "**/*server_metrics_export.csv"));
    assert!(has_file(&r, "**/*server_metrics_export.parquet"));

    // Verify JSON export structure.
    let j = server_metrics_json(&r);
    assert!(!j.is_null());
    assert!(!j["summary"].is_null());
    // At least 2 endpoints (vllm + sglang), possibly more with auto-collected default.
    assert!(endpoints_successful(&r).len() >= 2);
    assert!(j["metrics"].as_object().map_or(false, |mm| !mm.is_empty()));

    // Verify JSONL records structure.
    let records = r.artifacts.server_metrics_jsonl();
    assert!(!records.is_empty());

    // Check records have expected structure.
    for record in &records {
        assert!(record["endpoint_url"].is_string());
        assert!(record["timestamp_ns"].as_i64().unwrap_or(0) > 0);
        assert!(record["endpoint_latency_ns"].as_i64().unwrap_or(-1) >= 0);
        assert!(record["metrics"].as_object().map_or(false, |mm| !mm.is_empty()));
    }

    // Verify CSV content.
    let csv_path = r
        .artifacts
        .find_file("**/*server_metrics_export.csv")
        .expect("csv export exists");
    let csv = std::fs::read_to_string(csv_path).unwrap();
    let csv_lines: Vec<&str> = csv.trim().split('\n').collect();
    assert!(csv_lines.len() > 1); // Header + data rows.
}

/// Config-v2 honors CLI --server-metrics-formats jsonl override.
#[tokio::test]
async fn test_config_file_cli_server_metrics_formats_generates_jsonl() {
    let h = AIPerfHarness::new().await;
    let m = h.mock.server_metrics_urls();
    let vllm_url = &m["vllm"];

    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("server_metrics_config_v2.yaml");
    let yaml = format!(
        "schemaVersion: \"2.0\"\n\
         \n\
         benchmark:\n\
        \x20 model: nvidia/llama-3.1-nemotron-70b-instruct\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: 20\n\
        \x20   prompts:\n\
        \x20     isl: 128\n\
        \x20     osl: 64\n\
        \x20 phases:\n\
        \x20   type: concurrency\n\
        \x20   requests: 20\n\
        \x20   concurrency: 2\n\
        \x20 server_metrics:\n\
        \x20   enabled: true\n\
        \x20   urls:\n\
        \x20     - {vllm}\n\
        \x20   formats:\n\
        \x20     - json\n\
        \x20     - csv\n",
        url = h.mock.url,
        vllm = vllm_url
    );
    std::fs::write(&cfg_file, yaml).unwrap();

    let r = h.run(&format!(
        "--config {} --server-metrics-formats json csv jsonl",
        cfg_file.display()
    ));

    assert!(!server_metrics_json(&r).is_null());
    assert!(has_file(&r, "**/*server_metrics_export.csv"));
    let records = r.artifacts.server_metrics_jsonl();
    assert!(!records.is_empty());

    let jsonl_file = r
        .artifacts
        .find_file("**/server_metrics_export.jsonl")
        .expect("jsonl export exists");
    let jsonl_content = std::fs::read_to_string(jsonl_file).unwrap();
    let jsonl_lines: Vec<&str> = jsonl_content.lines().filter(|l| !l.is_empty()).collect();
    assert!(jsonl_lines.len() >= records.len());
}

/// Test JSONL records contain expected metrics with valid data.
#[tokio::test]
async fn test_server_metrics_jsonl_records() {
    let h = AIPerfHarness::new().await;
    let m = h.mock.server_metrics_urls();
    let vllm_url = &m["vllm"];

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --server-metrics-formats jsonl \
         --server-metrics {vllm_url}",
        h.mock.url
    ));

    let records = r.artifacts.server_metrics_jsonl();
    assert!(!records.is_empty());

    let mut endpoints_seen = std::collections::HashSet::new();
    let mut timestamps: Vec<i64> = Vec::new();

    for record in &records {
        if let Some(u) = record["endpoint_url"].as_str() {
            endpoints_seen.insert(u.to_string());
        }
        timestamps.push(record["timestamp_ns"].as_i64().unwrap_or(0));

        // Verify record has metrics.
        assert!(record["metrics"].as_object().map_or(false, |mm| !mm.is_empty()));

        // Check for expected vLLM metrics in at least some records.
        if let Some(samples) = record["metrics"].get("vllm:kv_cache_usage_perc") {
            let arr = samples.as_array().expect("samples is array");
            assert!(!arr.is_empty());
            assert!(!arr[0]["value"].is_null());
        }
    }

    // Timestamps generally increasing; multiple endpoints may interleave.
    assert!(!timestamps.is_empty(), "Should have timestamp records");
    assert!(
        timestamps.iter().min().copied().unwrap_or(0) > 0,
        "Timestamps should be positive"
    );

    // Captured data from at least the expected endpoint(s).
    assert!(endpoints_seen.len() >= 1);
}

/// Test histogram metrics are properly captured and exported.
#[tokio::test]
async fn test_server_metrics_histogram_data() {
    // Isolated mock server with workers=1 to avoid Prometheus metrics issues.
    let mut cfg = aiperf_mock_rs::config::MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    cfg.no_tokenizer = true;
    let h = AIPerfHarness::new_with(cfg).await;
    let m = h.mock.server_metrics_urls();
    let vllm_url = &m["vllm"];

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --server-metrics {vllm_url}",
        h.mock.url
    ));
    assert_server_metrics_valid(&r);

    // Get histogram metric from JSON export.
    let ttft_metric = get_server_metric(&r, "vllm:time_to_first_token_seconds");
    assert!(!ttft_metric.is_null());
    assert_eq!(ttft_metric["type"].as_str(), Some("histogram"));
    let series = ttft_metric["series"].as_array().expect("series array");
    assert!(!series.is_empty());

    // Verify histogram stats are computed.
    let first_series = &series[0];
    assert!(!first_series["stats"].is_null());
    let count = first_series["stats"]["count"].as_f64();
    assert!(count.is_some());
    assert!(count.unwrap() > 0.0);

    // Verify JSONL records have histogram data.
    for record in r.artifacts.server_metrics_jsonl() {
        if let Some(samples) = record["metrics"].get("vllm:time_to_first_token_seconds") {
            let arr = samples.as_array().expect("samples array");
            assert!(!arr.is_empty());
            // Histogram samples should have a buckets dict.
            assert!(arr[0]["buckets"].is_object());
        }
    }
}

// ============================================================================
// Non-Streaming Tests
// ============================================================================

/// Server metrics collection works with non-streaming requests.
#[tokio::test]
async fn test_server_metrics_non_streaming() {
    let h = AIPerfHarness::new().await;
    let m = h.mock.server_metrics_urls();
    let vllm_url = &m["vllm"];

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --server-metrics {vllm_url}",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, 50);
    assert_server_metrics_valid(&r);

    // Verify metrics are collected even for non-streaming.
    assert!(has_server_metric(&r, "vllm:e2e_request_latency_seconds"));
}

// ============================================================================
// Custom Prefix Tests
// ============================================================================

/// Test server metrics export with custom filename prefix.
#[tokio::test]
async fn test_server_metrics_custom_prefix() {
    let h = AIPerfHarness::new().await;
    let m = h.mock.server_metrics_urls();
    let vllm_url = &m["vllm"];

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 25 \
         --concurrency 1 \
         --workers-max 1 \
         --server-metrics {vllm_url} \
         --profile-export-prefix custom_test",
        h.mock.url
    ));

    // Verify custom prefix files exist and are non-empty.
    if let Some(json_file) = r.artifacts.find_file("**/custom_test_server_metrics.json") {
        let content = std::fs::read_to_string(json_file).unwrap();
        assert!(!content.is_empty());
    }

    if let Some(jsonl_file) = r.artifacts.find_file("**/custom_test_server_metrics.jsonl") {
        let content = std::fs::read_to_string(jsonl_file).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert!(!lines.is_empty());
        // Validate first record.
        let first: Value = serde_json::from_str(lines[0]).expect("valid jsonl record");
        assert!(first["timestamp_ns"].as_i64().unwrap_or(0) > 0);
    }
}

// ============================================================================
// Server Metrics Disabled Tests
// ============================================================================

/// Server metrics collection is disabled with --no-server-metrics flag.
#[tokio::test]
async fn test_server_metrics_disabled() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 25 \
         --concurrency 1 \
         --workers-max 1 \
         --no-server-metrics",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, 25);

    // Server metrics should NOT be collected when disabled.
    assert!(
        !has_file(&r, "**/*server_metrics*.json"),
        "JSON export should not exist"
    );
    assert!(
        !has_file(&r, "**/*server_metrics*.jsonl"),
        "JSONL export should not exist"
    );
    assert!(
        !has_file(&r, "**/*server_metrics*.csv"),
        "CSV export should not exist"
    );
}

/// Test Parquet export with raw time-series data and delta calculations.
///
/// The detailed dataframe-level validation (column schema, delta monotonicity,
/// histogram bucket normalization) requires a Parquet reader to parse the file
/// into a tabular form, mirroring the Python test's `pyarrow`/`pandas` usage.
#[tokio::test]
#[ignore] // requires: parquet reader (pyarrow/pandas parity)
async fn test_server_metrics_parquet_export() {
    // Isolated mock server with workers=1 to avoid Prometheus metrics issues.
    let mut cfg = aiperf_mock_rs::config::MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    cfg.no_tokenizer = true;
    let h = AIPerfHarness::new_with(cfg).await;
    let m = h.mock.server_metrics_urls();
    let urls = format!("{} {}", m["vllm"], m["sglang"]);

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct \
         --url {} \
         --endpoint-type chat \
         --streaming \
         --request-count 50 \
         --concurrency 2 \
         --workers-max 2 \
         --server-metrics {urls} \
         --server-metrics-formats parquet \
         --ui simple",
        h.mock.url
    ));

    // Verify Parquet file exists (the rest of the tabular assertions require a
    // Parquet reader and are covered by the Python integration suite).
    assert!(
        has_file(&r, "**/*server_metrics_export.parquet"),
        "Parquet file should exist"
    );
}
