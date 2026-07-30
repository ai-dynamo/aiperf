// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use serde_json::Value;

fn gpu_metric_names(json: &Value) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let Some(endpoints) = json
        .get("telemetry_data")
        .and_then(|t| t.get("endpoints"))
        .and_then(|e| e.as_object())
    else {
        return names;
    };
    for endpoint in endpoints.values() {
        let Some(gpus) = endpoint.get("gpus").and_then(|g| g.as_object()) else {
            continue;
        };
        for gpu in gpus.values() {
            if let Some(metrics) = gpu.get("metrics").and_then(|m| m.as_object()) {
                for name in metrics.keys() {
                    names.insert(name.clone());
                }
            }
        }
    }
    names
}

fn first_gpu_metric_avg(json: &Value, metric: &str) -> Option<f64> {
    let endpoints = json.get("telemetry_data")?.get("endpoints")?.as_object()?;
    for endpoint in endpoints.values() {
        let gpus = endpoint.get("gpus").and_then(|g| g.as_object())?;
        for gpu in gpus.values() {
            if let Some(v) = gpu
                .get("metrics")
                .and_then(|m| m.get(metric))
                .and_then(|m| m.get("avg"))
                .and_then(Value::as_f64)
            {
                return Some(v);
            }
        }
    }
    None
}

fn server_metrics_json(r: &RunResult) -> Value {
    r.artifacts.server_metrics_json()
}

fn has_server_metric(j: &Value, name: &str) -> bool {
    j["metrics"].get(name).is_some()
}

fn server_metric_value(j: &Value, name: &str) -> Option<f64> {
    let metric = j["metrics"].get(name)?;
    // Exports use either a flat aggregate or per-series statistics.
    if let Some(v) = metric.get("avg").and_then(Value::as_f64) {
        return Some(v);
    }
    if let Some(series) = metric.get("series").and_then(Value::as_array) {
        for s in series {
            for key in ["avg", "value", "max"] {
                if let Some(v) = s
                    .get("stats")
                    .and_then(|st| st.get(key))
                    .and_then(Value::as_f64)
                {
                    return Some(v);
                }
                if let Some(v) = s.get(key).and_then(Value::as_f64) {
                    return Some(v);
                }
            }
        }
    }
    // Unknown metric shapes fall back to the first finite scalar.
    fn any_f64(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
            Value::Array(a) => a.iter().find_map(any_f64),
            Value::Object(o) => o.values().find_map(any_f64),
            _ => None,
        }
    }
    any_f64(metric)
}

#[tokio::test]
async fn test_dcgm_and_vllm_telemetry_fills() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let vllm_url = h.mock.server_metrics_urls()["vllm"].clone();

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --endpoint-type chat --streaming \
         --gpu-telemetry {dcgm} \
         --server-metrics {vllm_url} \
         --request-count 100 --concurrency 4 --workers-max 2 --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 100);

    let json = r.artifacts.json();
    let names = gpu_metric_names(&json);
    assert!(
        !names.is_empty(),
        "GPU telemetry summary should carry per-GPU metrics"
    );
    // Canonical emitted names are `nvidia_`-prefixed. The unprefixed forms in
    // `LEGACY_NVIDIA_METRIC_ALIASES` (`gpu_telemetry/fields.rs:13`) are accepted
    // only when *reading* telemetry and are deliberately never re-emitted, so
    // asserting on them would pass only against an ingest-side regression.
    for expected in [
        "nvidia_encoder_utilization",
        "nvidia_decoder_utilization",
        "nvidia_sm_utilization",
    ] {
        assert!(
            names.contains(expected),
            "GPU metric `{expected}` missing from telemetry summary; present: {names:?}"
        );
    }

    // DCGM reports SM activity as a ratio; the runtime exports percent.
    let sm = first_gpu_metric_avg(&json, "nvidia_sm_utilization")
        .expect("nvidia_sm_utilization avg present");
    assert!(
        (0.0..=100.0).contains(&sm),
        "nvidia_sm_utilization out of [0,100]: {sm}"
    );
    for metric in ["nvidia_encoder_utilization", "nvidia_decoder_utilization"] {
        let v =
            first_gpu_metric_avg(&json, metric).unwrap_or_else(|| panic!("{metric} avg present"));
        assert!((0.0..=100.0).contains(&v), "{metric} out of [0,100]: {v}");
    }

    let sm_json = server_metrics_json(&r);
    assert!(!sm_json.is_null(), "server_metrics.json should exist");

    assert!(
        has_server_metric(&sm_json, "vllm:external_prefix_cache_hits"),
        "raw vllm:external_prefix_cache_hits missing"
    );
    assert!(
        has_server_metric(&sm_json, "vllm:external_prefix_cache_queries"),
        "raw vllm:external_prefix_cache_queries missing"
    );
    assert!(
        has_server_metric(&sm_json, "vllm:cpu_cache_usage_perc"),
        "raw vllm:cpu_cache_usage_perc missing"
    );

    assert!(
        has_server_metric(&sm_json, "external_prefix_cache_hit_rate"),
        "derived external_prefix_cache_hit_rate missing"
    );
    let ext_rate = server_metric_value(&sm_json, "external_prefix_cache_hit_rate")
        .expect("external_prefix_cache_hit_rate value");
    assert!(
        ext_rate > 0.0 && ext_rate <= 100.0,
        "external_prefix_cache_hit_rate should be a nonzero percent: {ext_rate}"
    );

    assert!(
        has_server_metric(&sm_json, "cpu_kv_cache_usage_pct"),
        "derived cpu_kv_cache_usage_pct missing"
    );

    // Counter deltas across the profiling window must remain positive.
    assert!(
        has_server_metric(&sm_json, "num_preemptions"),
        "derived num_preemptions missing"
    );
    let preempt = server_metric_value(&sm_json, "num_preemptions").expect("num_preemptions value");
    assert!(
        preempt > 0.0,
        "num_preemptions delta should be nonzero: {preempt}"
    );
}

#[tokio::test]
async fn test_sglang_counter_fills() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let sglang_url = h.mock.server_metrics_urls()["sglang"].clone();

    let r = h.run(&format!(
        "--model nvidia/llama-3.1-nemotron-70b-instruct --url {} \
         --endpoint-type chat --streaming \
         --no-gpu-telemetry \
         --server-metrics {sglang_url} \
         --request-count 100 --concurrency 4 --workers-max 2 --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 100);

    let sm_json = server_metrics_json(&r);
    assert!(!sm_json.is_null(), "server_metrics.json should exist");

    for raw in [
        "sglang:cached_tokens",
        "sglang:prompt_tokens",
        "sglang:generation_tokens",
        "sglang:num_retracted_reqs",
    ] {
        assert!(
            has_server_metric(&sm_json, raw),
            "raw {raw} missing from server metrics"
        );
    }

    assert!(
        has_server_metric(&sm_json, "num_preemptions"),
        "derived num_preemptions (sglang fallback) missing"
    );
    let preempt = server_metric_value(&sm_json, "num_preemptions").expect("num_preemptions value");
    assert!(
        preempt > 0.0,
        "sglang num_preemptions delta should be nonzero: {preempt}"
    );

    assert!(
        has_server_metric(&sm_json, "prefix_cache_hit_rate"),
        "derived prefix_cache_hit_rate (sglang fallback) missing"
    );
    let hit_rate = server_metric_value(&sm_json, "prefix_cache_hit_rate")
        .expect("prefix_cache_hit_rate value");
    assert!(
        hit_rate > 0.0 && hit_rate <= 100.0,
        "prefix_cache_hit_rate should be a nonzero percent: {hit_rate}"
    );
}
