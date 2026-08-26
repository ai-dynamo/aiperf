// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end worker-local metric accumulation coverage.
//!
//! Latency distribution fields are wall-clock (`RealClock`) and jitter run to
//! run, so only deterministic count and token metrics are compared.

use std::io::Write;
use std::net::SocketAddr;
use std::process::{Command, Output, Stdio};

use axum::{Router, http::header, response::IntoResponse, routing::post};
use serde_json::{Value, json};

/// Deterministic streaming response: five content tokens then authoritative
/// usage (`prompt_tokens = 4`, `completion_tokens = 5`).
async fn chat() -> impl IntoResponse {
    let body = concat!(
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"c\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"d\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"e\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"x\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

fn benchmark_run(source: Value) -> Value {
    let mut endpoint = source["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    json!({
        "benchmark_id": source["identity"]["benchmark_id"],
        "artifact_dir": source["artifact_target"],
        "random_seed": source["identity"]["random_seed"],
        "cfg": {
            "models": source["resources"]["models"],
            "endpoint": endpoint,
            "datasets": [source["workload"]["config"]["dataset"]],
            "phases": source["workload"]["config"]["phases"],
            "tokenizer": source["workload"]["config"]["tokenizer"],
            "transport": {"type": source["transport"]["type"]},
            "runtime": {"workers": source["workload"]["config"]["worker_count"]}
        }
    })
}

fn run_child(mut request: Value) -> Output {
    request["run"] = benchmark_run(request["run"].take());
    let bytes = serde_json::to_vec(&request["run"]).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child.wait_with_output().unwrap()
}

/// Run the fixed scenario at `worker_count` and return the native-v2 report.
fn run_scenario(address: SocketAddr, worker_count: u64) -> Value {
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_target = artifacts.path().join("run");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "identity": {
                "benchmark_id": "worker-local-accumulation-parity",
                "random_seed": 7
            },
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": [format!("http://{address}")],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": worker_count,
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 5.0}
                    }
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "warmup",
                    "exclude_from_results": true,
                    "requests": 4,
                    "concurrency": 4
                }, {
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 12,
                    "concurrency": 4
                }]
            }}
        }
    });
    let output = run_child(request);
    assert!(
        output.status.success(),
        "worker_count={worker_count} stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], true, "{terminal}");
    serde_json::from_slice(&std::fs::read(artifact_target.join("native-v2.json")).unwrap()).unwrap()
}

/// Count/token metric identities whose counts/values are deterministic given the
/// fixed mock, so their series counts and distribution stats must be
/// byte-identical across worker counts. The wall-clock-derived `rate`
/// (throughput = count / duration) is intentionally *not* compared: four-way
/// parallel dispatch makes the four-worker run finish faster, so a higher `rate`
/// is the expected win, not a regression.
const DETERMINISTIC_METRICS: &[&str] = &[
    "request_count",
    "error_request_count",
    "total_isl",
    "total_output_tokens",
    "total_usage_prompt_tokens",
    "total_usage_completion_tokens",
];

/// Serialize a metric's `series` after dropping the wall-clock `rate` field from
/// every entry's stats, so only the deterministic count/token surface is compared.
fn canonical_series(report: &Value, block: &str, metric: &str) -> String {
    let mut series = report[block][metric]["series"].clone();
    if let Some(entries) = series.as_array_mut() {
        for entry in entries {
            if let Some(stats) = entry.get_mut("stats").and_then(Value::as_object_mut) {
                stats.remove("rate");
            }
        }
    }
    serde_json::to_string(&series).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn worker_local_accumulation_is_byte_identical_across_worker_counts() {
    let app = Router::new().route("/v1/chat/completions", post(chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let single = {
        tokio::task::spawn_blocking(move || run_scenario(address, 1))
            .await
            .unwrap()
    };
    let split = {
        tokio::task::spawn_blocking(move || run_scenario(address, 4))
            .await
            .unwrap()
    };

    // Sanity: the scenario actually ran twelve profiling requests of five tokens
    // each — enough to spread across four workers and exercise token replay.
    assert_eq!(
        single["metrics"]["request_count"]["series"][0]["stats"]["total"], 12.0,
        "single-worker report: {single}"
    );
    assert_eq!(
        split["metrics"]["request_count"]["series"][0]["stats"]["total"], 12.0,
        "four-worker report: {split}"
    );
    assert_eq!(
        split["metrics"]["total_output_tokens"]["series"][0]["stats"]["value"], 60.0,
        "twelve requests * five output tokens: {split}"
    );

    // The run-level metadata block is worker-count independent.
    assert_eq!(
        single["run"]["endpoint_profiles"],
        split["run"]["endpoint_profiles"]
    );
    assert_eq!(single["run"]["mode"], split["run"]["mode"]);

    // Every deterministic count/token series must be byte-identical between the
    // coordinator-accumulated (workers=1) and worker-local-split (workers=4) runs.
    for metric in DETERMINISTIC_METRICS {
        assert_eq!(
            canonical_series(&single, "metrics", metric),
            canonical_series(&split, "metrics", metric),
            "metric {metric} diverged between worker_count=1 and worker_count=4:\n\
             single={}\nsplit={}",
            single["metrics"][metric]["series"],
            split["metrics"][metric]["series"],
        );
    }

    // The warmup block (excluded from results) must likewise match.
    for metric in DETERMINISTIC_METRICS {
        assert_eq!(
            canonical_series(&single, "warmup_metrics", metric),
            canonical_series(&split, "warmup_metrics", metric),
            "warmup metric {metric} diverged between worker counts"
        );
    }

    server.abort();
}
