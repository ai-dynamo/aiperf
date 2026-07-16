// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end validation of the mock-server "telemetry fills": DCGM gauges and
//! vLLM/SGLang server-metric counters/gauges the runner decodes but the mock
//! previously never emitted.
//!
//! These runs drive the real `aiperf profile` frontend against the in-process
//! mock so the native runner actually SCRAPES the mock's `/dcgm{1,2}/metrics`
//! (via `--gpu-telemetry`) and `/vllm/metrics` | `/sglang/metrics` (via
//! `--server-metrics`) endpoints, then decodes them through
//! `aiperf::gpu_telemetry` and `aiperf::server_metrics`. We assert the newly
//! filled fields land in the exported artifacts:
//!
//!   * GPU telemetry summary (`profile_export_aiperf.json` -> `telemetry_data`)
//!     carries `encoder_utilization` / `decoder_utilization` / `sm_utilization`
//!     (runner GPU field decoders: `rust/aiperf/src/gpu_telemetry/fields.rs`
//!     lines 116 / 124 / 132).
//!   * Server-metrics export (`server_metrics.json`) carries the raw scraped
//!     names AND the atlas-derived rows (`rust/aiperf/src/server_metrics/
//!     atlas.rs`): `external_prefix_cache_hit_rate` (atlas.rs:114/118),
//!     `cpu_kv_cache_usage_pct` (atlas.rs:154), and a nonzero `num_preemptions`
//!     (atlas.rs:207), plus the SGLang counter fallbacks (atlas.rs:70/223/233).

mod common;
use common::*;

use serde_json::Value;

/// Extract the set of per-GPU metric names present anywhere in the run's
/// `telemetry_data` endpoints. Returns an empty set when telemetry is absent.
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

/// Average value of a GPU metric across all endpoints/GPUs (first hit wins for a
/// representative value); `None` when the metric is absent.
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

/// Load the server-metrics JSON export as a `Value` (`Null` when absent).
fn server_metrics_json(r: &RunResult) -> Value {
    r.artifacts.server_metrics_json()
}

/// True when `name` is a key in the server-metrics `metrics` map.
fn has_server_metric(j: &Value, name: &str) -> bool {
    j["metrics"].get(name).is_some()
}

/// Representative scalar for a server metric: prefer `avg`, else the raw value,
/// searching the metric's series/stats shape. `None` when absent.
fn server_metric_value(j: &Value, name: &str) -> Option<f64> {
    let metric = j["metrics"].get(name)?;
    // The export shapes a metric as { unit, series: [ { stats: {...} } ] } or a
    // flatter { avg } depending on gauge/counter. Probe the common locations.
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
    // Fall back to any finite f64 anywhere in the subtree.
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

/// DCGM encode/decode/SM-activity fills flow through the runner GPU decoder into
/// the exported telemetry summary.
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

    // ---- DCGM fills (fields.rs 116/124/132) ---------------------------------
    let json = r.artifacts.json();
    let names = gpu_metric_names(&json);
    assert!(
        !names.is_empty(),
        "GPU telemetry summary should carry per-GPU metrics"
    );
    for expected in [
        "encoder_utilization",
        "decoder_utilization",
        "sm_utilization",
    ] {
        assert!(
            names.contains(expected),
            "GPU metric `{expected}` missing from telemetry summary; present: {names:?}"
        );
    }

    // SM_ACTIVE is a DCGM 0..1 ratio the runner scales x100 -> percent in [0,100].
    let sm = first_gpu_metric_avg(&json, "sm_utilization").expect("sm_utilization avg present");
    assert!(
        (0.0..=100.0).contains(&sm),
        "sm_utilization out of [0,100]: {sm}"
    );
    // Encoder/decoder utilization are percents in [0,100].
    for metric in ["encoder_utilization", "decoder_utilization"] {
        let v =
            first_gpu_metric_avg(&json, metric).unwrap_or_else(|| panic!("{metric} avg present"));
        assert!((0.0..=100.0).contains(&v), "{metric} out of [0,100]: {v}");
    }

    // ---- vLLM server-metric fills -------------------------------------------
    let sm_json = server_metrics_json(&r);
    assert!(!sm_json.is_null(), "server_metrics.json should exist");

    // Raw scraped names newly emitted by the mock (prom.rs VllmMetrics).
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

    // Atlas-derived rows (atlas.rs) that only resolve now that the mock emits
    // the underlying counters/gauge.
    assert!(
        has_server_metric(&sm_json, "external_prefix_cache_hit_rate"),
        "derived external_prefix_cache_hit_rate missing (atlas.rs:114/118)"
    );
    let ext_rate = server_metric_value(&sm_json, "external_prefix_cache_hit_rate")
        .expect("external_prefix_cache_hit_rate value");
    assert!(
        ext_rate > 0.0 && ext_rate <= 100.0,
        "external_prefix_cache_hit_rate should be a nonzero percent: {ext_rate}"
    );

    assert!(
        has_server_metric(&sm_json, "cpu_kv_cache_usage_pct"),
        "derived cpu_kv_cache_usage_pct missing (atlas.rs:154)"
    );

    // NUM_PREEMPTIONS is now actually incremented (metrics.rs), so the
    // reset-clamped phase-boundary delta must be strictly positive.
    assert!(
        has_server_metric(&sm_json, "num_preemptions"),
        "derived num_preemptions missing (atlas.rs:207)"
    );
    let preempt = server_metric_value(&sm_json, "num_preemptions").expect("num_preemptions value");
    assert!(
        preempt > 0.0,
        "num_preemptions delta should be nonzero: {preempt}"
    );
}

/// SGLang counter fills drive the runner's SGLang-fallback atlas derivations.
#[tokio::test]
async fn test_sglang_counter_fills() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let sglang_url = h.mock.server_metrics_urls()["sglang"].clone();

    // Point server-metrics ONLY at the SGLang endpoint so the atlas resolves the
    // SGLang fallbacks (vLLM would otherwise take precedence for shared rows).
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

    // Raw SGLang counters newly emitted by the mock (prom.rs SglangMetrics).
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

    // SGLang-fallback derivations (atlas.rs). num_preemptions now derives from
    // sglang:num_retracted_reqs (atlas.rs:207) and must be nonzero;
    // prefix_cache_hit_rate derives from sglang:cached_tokens/prompt_tokens
    // (atlas.rs:70).
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
