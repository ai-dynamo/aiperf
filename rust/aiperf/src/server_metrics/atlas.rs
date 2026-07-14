// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral derived inference-server metric atlas.
//!
//! The vLLM-first, SGLang-fallback mapping and its type guards are defined here.
//! The native view replaces Python's reconstructed counter windows with the
//! exact phase-boundary deltas required by the telemetry design addendum.

use std::collections::BTreeMap;

use crate::metrics_core::Unit;

/// Typed lookup seam consumed by server-specific metric atlases.
pub trait ServerMetricView {
    /// Reset-clamped phase-boundary delta summed across matching counter series.
    fn counter_delta(&self, metric_name: &str) -> Option<f64>;

    /// Counter delta divided by the authoritative phase duration.
    fn counter_rate(&self, metric_name: &str) -> Option<f64>;

    /// Maximum latest value across matching gauge series and endpoints.
    fn gauge_latest_max(&self, metric_name: &str) -> Option<f64>;

    /// Maximum per-endpoint ratio of two latest gauges.
    fn max_endpoint_gauge_ratio(&self, numerator_name: &str, denominator_name: &str)
    -> Option<f64>;
}

/// One finite scalar emitted by a server metric atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedServerMetric {
    /// Derived scalar value.
    pub value: f64,
    /// Native report unit.
    pub unit: Unit,
    /// Stable human-readable derivation description.
    pub description: &'static str,
}

/// Object-safe policy for deriving stable metrics from backend-specific names.
pub trait ServerMetricAtlas {
    /// Derive every metric supported by the supplied typed view.
    fn derive(&self, view: &dyn ServerMetricView) -> BTreeMap<String, DerivedServerMetric>;
}

/// Native vLLM/SGLang metric-name and fallback policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct VllmSglangMetricAtlas;

impl ServerMetricAtlas for VllmSglangMetricAtlas {
    fn derive(&self, view: &dyn ServerMetricView) -> BTreeMap<String, DerivedServerMetric> {
        let mut output = BTreeMap::new();
        add_prefix_cache_metrics(&mut output, view);
        add_external_prefix_cache_metric(&mut output, view);
        add_cache_usage_metrics(&mut output, view);
        add_queue_depth_metrics(&mut output, view);
        add_preemption_metric(&mut output, view);
        add_token_throughput_metrics(&mut output, view);
        output
    }
}

fn add_prefix_cache_metrics(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    view: &dyn ServerMetricView,
) {
    let pairs = [
        ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
        ("sglang:cached_tokens", "sglang:prompt_tokens"),
    ];
    for (hits_name, queries_name) in pairs {
        let Some(hits) = view.counter_delta(hits_name) else {
            continue;
        };
        let Some(queries) = view
            .counter_delta(queries_name)
            .filter(|value| *value > 0.0)
        else {
            continue;
        };
        insert(
            output,
            "prefix_cache_hit_rate",
            100.0 * hits.min(queries) / queries,
            Unit::Percent,
            "Server prefix-cache hit rate over the phase boundary.",
        );
        insert(
            output,
            "unique_input_tokens_srv",
            (queries - hits).max(0.0),
            Unit::Token,
            "Server-observed input tokens not served by the prefix cache.",
        );
        return;
    }

    if let Some(value) = view.gauge_latest_max("sglang:cache_hit_rate") {
        insert(
            output,
            "prefix_cache_hit_rate",
            to_percent(value),
            Unit::Percent,
            "Latest SGLang per-batch prefix-cache hit rate.",
        );
    }
}

fn add_external_prefix_cache_metric(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    view: &dyn ServerMetricView,
) {
    let Some(hits) = view.counter_delta("vllm:external_prefix_cache_hits") else {
        return;
    };
    let Some(queries) = view
        .counter_delta("vllm:external_prefix_cache_queries")
        .filter(|value| *value > 0.0)
    else {
        return;
    };
    insert(
        output,
        "external_prefix_cache_hit_rate",
        100.0 * hits.min(queries) / queries,
        Unit::Percent,
        "Server external prefix-cache hit rate over the phase boundary.",
    );
}

fn add_cache_usage_metrics(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    view: &dyn ServerMetricView,
) {
    if let Some(value) = first_gauge(
        view,
        &[
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "sglang:token_usage",
        ],
    ) {
        insert(
            output,
            "kv_cache_usage_pct",
            to_percent(value),
            Unit::Percent,
            "Latest maximum device KV-cache usage.",
        );
    }

    let cpu_usage = view
        .gauge_latest_max("vllm:cpu_cache_usage_perc")
        .or_else(|| {
            view.max_endpoint_gauge_ratio(
                "sglang:hicache_host_used_tokens",
                "sglang:hicache_host_total_tokens",
            )
        });
    if let Some(value) = cpu_usage {
        insert(
            output,
            "cpu_kv_cache_usage_pct",
            to_percent(value),
            Unit::Percent,
            "Latest maximum host KV-cache usage.",
        );
    }
}

fn add_queue_depth_metrics(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    view: &dyn ServerMetricView,
) {
    if let Some(value) = first_gauge(
        view,
        &["vllm:num_requests_running", "sglang:num_running_reqs"],
    ) {
        insert(
            output,
            "num_running",
            value,
            Unit::Request,
            "Latest maximum server running-request count.",
        );
    }
    if let Some(value) = first_gauge(
        view,
        &["vllm:num_requests_waiting", "sglang:num_queue_reqs"],
    ) {
        insert(
            output,
            "num_waiting",
            value,
            Unit::Request,
            "Latest maximum server waiting-request count.",
        );
    }
}

fn add_preemption_metric(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    view: &dyn ServerMetricView,
) {
    if let Some(value) =
        first_counter_delta(view, &["vllm:num_preemptions", "sglang:num_retracted_reqs"])
    {
        insert(
            output,
            "num_preemptions",
            value,
            Unit::Count,
            "Reset-clamped server preemption count over the phase.",
        );
    }
}

fn add_token_throughput_metrics(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    view: &dyn ServerMetricView,
) {
    if let Some(value) = first_counter_rate(view, &["vllm:prompt_tokens", "sglang:prompt_tokens"]) {
        insert(
            output,
            "input_token_throughput_srv",
            value,
            Unit::TokensPerSecond,
            "Server-observed input token throughput over the phase.",
        );
    }
    if let Some(value) = first_counter_rate(
        view,
        &["vllm:generation_tokens", "sglang:generation_tokens"],
    ) {
        insert(
            output,
            "output_token_throughput_srv",
            value,
            Unit::TokensPerSecond,
            "Server-observed output token throughput over the phase.",
        );
    }
}

fn first_gauge(view: &dyn ServerMetricView, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| view.gauge_latest_max(name))
}

fn first_counter_delta(view: &dyn ServerMetricView, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| view.counter_delta(name))
}

fn first_counter_rate(view: &dyn ServerMetricView, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| view.counter_rate(name))
}

fn to_percent(value: f64) -> f64 {
    if value <= 1.0 { value * 100.0 } else { value }
}

fn insert(
    output: &mut BTreeMap<String, DerivedServerMetric>,
    name: &str,
    value: f64,
    unit: Unit,
    description: &'static str,
) {
    if value.is_finite() {
        output.insert(
            name.to_string(),
            DerivedServerMetric {
                value,
                unit,
                description,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MapView {
        counters: BTreeMap<String, f64>,
        rates: BTreeMap<String, f64>,
        gauges: BTreeMap<String, f64>,
        ratios: BTreeMap<(String, String), f64>,
    }

    impl ServerMetricView for MapView {
        fn counter_delta(&self, metric_name: &str) -> Option<f64> {
            self.counters.get(metric_name).copied()
        }

        fn counter_rate(&self, metric_name: &str) -> Option<f64> {
            self.rates.get(metric_name).copied()
        }

        fn gauge_latest_max(&self, metric_name: &str) -> Option<f64> {
            self.gauges.get(metric_name).copied()
        }

        fn max_endpoint_gauge_ratio(
            &self,
            numerator_name: &str,
            denominator_name: &str,
        ) -> Option<f64> {
            self.ratios
                .get(&(numerator_name.to_string(), denominator_name.to_string()))
                .copied()
        }
    }

    #[test]
    fn vllm_metrics_take_precedence_and_hit_rates_are_capped() {
        let view = MapView {
            counters: BTreeMap::from([
                ("vllm:prefix_cache_hits".to_string(), 110.0),
                ("vllm:prefix_cache_queries".to_string(), 100.0),
                ("sglang:cached_tokens".to_string(), 1.0),
                ("sglang:prompt_tokens".to_string(), 10.0),
                ("vllm:num_preemptions".to_string(), 2.0),
            ]),
            rates: BTreeMap::from([
                ("vllm:prompt_tokens".to_string(), 500.0),
                ("vllm:generation_tokens".to_string(), 250.0),
            ]),
            gauges: BTreeMap::from([
                ("vllm:kv_cache_usage_perc".to_string(), 0.75),
                ("vllm:num_requests_running".to_string(), 4.0),
            ]),
            ..MapView::default()
        };

        let metrics = VllmSglangMetricAtlas.derive(&view);

        assert_eq!(metrics["prefix_cache_hit_rate"].value, 100.0);
        assert_eq!(metrics["unique_input_tokens_srv"].value, 0.0);
        assert_eq!(metrics["kv_cache_usage_pct"].value, 75.0);
        assert_eq!(metrics["num_running"].value, 4.0);
        assert_eq!(metrics["num_preemptions"].value, 2.0);
        assert_eq!(metrics["input_token_throughput_srv"].value, 500.0);
        assert_eq!(metrics["output_token_throughput_srv"].value, 250.0);
    }

    #[test]
    fn sglang_counter_and_within_endpoint_ratio_fallbacks_are_preserved() {
        let view = MapView {
            counters: BTreeMap::from([
                ("sglang:cached_tokens".to_string(), 30.0),
                ("sglang:prompt_tokens".to_string(), 100.0),
                ("sglang:num_retracted_reqs".to_string(), 3.0),
            ]),
            gauges: BTreeMap::from([
                ("sglang:token_usage".to_string(), 42.0),
                ("sglang:num_queue_reqs".to_string(), 7.0),
            ]),
            ratios: BTreeMap::from([(
                (
                    "sglang:hicache_host_used_tokens".to_string(),
                    "sglang:hicache_host_total_tokens".to_string(),
                ),
                0.25,
            )]),
            ..MapView::default()
        };

        let metrics = VllmSglangMetricAtlas.derive(&view);

        assert_eq!(metrics["prefix_cache_hit_rate"].value, 30.0);
        assert_eq!(metrics["unique_input_tokens_srv"].value, 70.0);
        assert_eq!(metrics["kv_cache_usage_pct"].value, 42.0);
        assert_eq!(metrics["cpu_kv_cache_usage_pct"].value, 25.0);
        assert_eq!(metrics["num_waiting"].value, 7.0);
        assert_eq!(metrics["num_preemptions"].value, 3.0);
    }

    #[test]
    fn per_batch_gauge_is_only_the_last_cache_hit_fallback() {
        let view = MapView {
            gauges: BTreeMap::from([("sglang:cache_hit_rate".to_string(), 0.8)]),
            ..MapView::default()
        };

        let metrics = VllmSglangMetricAtlas.derive(&view);

        assert_eq!(metrics["prefix_cache_hit_rate"].value, 80.0);
        assert!(!metrics.contains_key("unique_input_tokens_srv"));
    }
}
