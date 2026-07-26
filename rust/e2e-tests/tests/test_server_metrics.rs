// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::Value;

fn server_metrics_json(r: &RunResult) -> Value {
    r.artifacts.server_metrics_json()
}

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

fn has_server_metric(r: &RunResult, name: &str) -> bool {
    server_metrics_json(r)["metrics"].get(name).is_some()
}

fn get_server_metric(r: &RunResult, name: &str) -> Value {
    server_metrics_json(r)["metrics"]
        .get(name)
        .cloned()
        .unwrap_or(Value::Null)
}

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

fn has_file(r: &RunResult, glob: &str) -> bool {
    r.artifacts.find_file(glob).is_some()
}

#[tokio::test]
async fn test_server_metrics_auto_collected() {
    let mut cfg = aiperf_mock_server::config::MockServerConfig::default();
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

    assert_server_metrics_valid(&r);

    assert!(has_server_metric(&r, "aiperf_mock_requests"));
    assert!(has_server_metric(&r, "aiperf_mock_request_latency_seconds"));
    assert!(has_server_metric(
        &r,
        "aiperf_mock_time_to_first_token_seconds"
    ));
    assert!(has_server_metric(&r, "aiperf_mock_tokens_streamed"));

    let expected_endpoint = format!("http://127.0.0.1:{}/metrics", h.mock.port);
    let successful = endpoints_successful(&r);
    assert!(
        successful.contains(&expected_endpoint),
        "Expected {expected_endpoint} in successful endpoints: {successful:?}"
    );
}

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

    assert!(endpoints_successful(&r).len() >= 2);

    assert!(has_server_metric(&r, "vllm:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "vllm:time_to_first_token_seconds"));

    assert!(has_server_metric(&r, "sglang:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "sglang:time_to_first_token_seconds"));
}

#[tokio::test]
async fn test_server_metrics_all_endpoints() {
    let mut cfg = aiperf_mock_server::config::MockServerConfig::default();
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

    let successful = endpoints_successful(&r);
    assert!(
        successful.len() >= 6,
        "Expected at least 6 successful endpoints, got {}: {successful:?}",
        successful.len()
    );

    assert!(has_server_metric(&r, "vllm:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "vllm:time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "vllm:inter_token_latency_seconds"));
    assert!(has_server_metric(&r, "vllm:kv_cache_usage_perc"));

    assert!(has_server_metric(&r, "sglang:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "sglang:time_to_first_token_seconds"));
    assert!(has_server_metric(&r, "sglang:gen_throughput"));

    assert!(has_server_metric(&r, "trtllm:e2e_request_latency_seconds"));
    assert!(has_server_metric(&r, "trtllm:time_to_first_token_seconds"));
    assert!(has_server_metric(
        &r,
        "trtllm:time_per_output_token_seconds"
    ));

    assert!(has_server_metric(
        &r,
        "dynamo_frontend_time_to_first_token_seconds"
    ));
    assert!(has_server_metric(
        &r,
        "dynamo_frontend_inter_token_latency_seconds"
    ));

    assert!(has_server_metric(
        &r,
        "dynamo_component_request_duration_seconds"
    ));
    assert!(has_server_metric(&r, "dynamo_component_requests"));
}

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

    assert!(has_file(&r, "**/*server_metrics_export.json"));
    assert!(has_file(&r, "**/*server_metrics_export.jsonl"));
    assert!(has_file(&r, "**/*server_metrics_export.csv"));
    assert!(has_file(&r, "**/*server_metrics_export.parquet"));

    let j = server_metrics_json(&r);
    assert!(!j.is_null());
    assert!(!j["summary"].is_null());
    assert!(endpoints_successful(&r).len() >= 2);
    assert!(j["metrics"].as_object().map_or(false, |mm| !mm.is_empty()));

    let records = r.artifacts.server_metrics_jsonl();
    assert!(!records.is_empty());

    for record in &records {
        assert!(record["endpoint_url"].is_string());
        assert!(record["timestamp_ns"].as_i64().unwrap_or(0) > 0);
        assert!(record["endpoint_latency_ns"].as_i64().unwrap_or(-1) >= 0);
        assert!(
            record["metrics"]
                .as_object()
                .map_or(false, |mm| !mm.is_empty())
        );
    }

    let csv_path = r
        .artifacts
        .find_file("**/*server_metrics_export.csv")
        .expect("csv export exists");
    let csv = std::fs::read_to_string(csv_path).unwrap();
    let csv_lines: Vec<&str> = csv.trim().split('\n').collect();
    assert!(csv_lines.len() > 1);
}

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

        assert!(
            record["metrics"]
                .as_object()
                .map_or(false, |mm| !mm.is_empty())
        );

        if let Some(samples) = record["metrics"].get("vllm:kv_cache_usage_perc") {
            let arr = samples.as_array().expect("samples is array");
            assert!(!arr.is_empty());
            assert!(!arr[0]["value"].is_null());
        }
    }

    assert!(!timestamps.is_empty(), "Should have timestamp records");
    assert!(
        timestamps.iter().min().copied().unwrap_or(0) > 0,
        "Timestamps should be positive"
    );

    assert!(endpoints_seen.len() >= 1);
}

#[tokio::test]
async fn test_server_metrics_histogram_data() {
    let mut cfg = aiperf_mock_server::config::MockServerConfig::default();
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

    let ttft_metric = get_server_metric(&r, "vllm:time_to_first_token_seconds");
    assert!(!ttft_metric.is_null());
    assert_eq!(ttft_metric["type"].as_str(), Some("histogram"));
    let series = ttft_metric["series"].as_array().expect("series array");
    assert!(!series.is_empty());

    let first_series = &series[0];
    assert!(!first_series["stats"].is_null());
    let count = first_series["stats"]["count"].as_f64();
    assert!(count.is_some());
    assert!(count.unwrap() > 0.0);

    for record in r.artifacts.server_metrics_jsonl() {
        if let Some(samples) = record["metrics"].get("vllm:time_to_first_token_seconds") {
            let arr = samples.as_array().expect("samples array");
            assert!(!arr.is_empty());
            assert!(arr[0]["buckets"].is_object());
        }
    }
}

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

    assert!(has_server_metric(&r, "vllm:e2e_request_latency_seconds"));
}

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

    if let Some(json_file) = r.artifacts.find_file("**/custom_test_server_metrics.json") {
        let content = std::fs::read_to_string(json_file).unwrap();
        assert!(!content.is_empty());
    }

    if let Some(jsonl_file) = r.artifacts.find_file("**/custom_test_server_metrics.jsonl") {
        let content = std::fs::read_to_string(jsonl_file).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert!(!lines.is_empty());
        let first: Value = serde_json::from_str(lines[0]).expect("valid jsonl record");
        assert!(first["timestamp_ns"].as_i64().unwrap_or(0) > 0);
    }
}

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

#[tokio::test]
#[ignore] // requires: parquet reader (pyarrow/pandas parity)
async fn test_server_metrics_parquet_export() {
    let mut cfg = aiperf_mock_server::config::MockServerConfig::default();
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

    assert!(
        has_file(&r, "**/*server_metrics_export.parquet"),
        "Parquet file should exist"
    );
}
