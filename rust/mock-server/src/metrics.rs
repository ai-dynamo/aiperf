// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-level metric recording functions.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use dashmap::DashMap;
use prometheus::Histogram;
use prometheus::core::{AtomicI64 as PromAtomicI64, AtomicU64, GenericCounter, GenericGauge};

use crate::models::Usage;
use crate::prom::AllMetrics;
use crate::throughput::Throughput;

/// Pre-resolved metric handles for one (endpoint, model) pair. Hot-path streaming
/// hits `record_ttft` / `record_itl` / `record_streamed_token` once per token —
/// caching these child refs avoids a label HashMap lookup inside each prometheus
/// MetricVec on every call.
pub struct LabeledMetrics {
    pub tokens_streamed: GenericCounter<AtomicU64>,
    pub prompt_tokens: GenericCounter<AtomicU64>,
    pub completion_tokens: GenericCounter<AtomicU64>,
    pub tokens_per_request_prompt: Histogram,
    pub tokens_per_request_completion: Histogram,
    pub ttft_by_endpoint: Histogram,
    pub itl_by_endpoint: Histogram,
    pub df_ttft: Histogram,
    pub df_itl: Histogram,
    pub df_request_duration: Histogram,
    pub df_requests: GenericCounter<AtomicU64>,
    pub df_input_seq_tokens: GenericCounter<AtomicU64>,
    pub df_output_tokens: GenericCounter<AtomicU64>,
    pub df_output_seq_tokens: GenericCounter<AtomicU64>,
    pub df_inflight: GenericGauge<PromAtomicI64>,
    pub df_queued: GenericGauge<PromAtomicI64>,
    pub dp_request_duration: Histogram,
    pub dp_requests: GenericCounter<AtomicU64>,
    pub dp_inflight: GenericGauge<PromAtomicI64>,
    pub dd_request_duration: Histogram,
    pub dd_requests: GenericCounter<AtomicU64>,
    pub dd_inflight: GenericGauge<PromAtomicI64>,
    pub requests_total_200: GenericCounter<AtomicU64>,
    pub requests_total_500: GenericCounter<AtomicU64>,
    pub requests_in_progress: GenericGauge<PromAtomicI64>,
    pub request_latency: Histogram,
    pub requests_by_model: GenericCounter<AtomicU64>,
    pub request_bytes: GenericCounter<AtomicU64>,
    pub response_bytes: GenericCounter<AtomicU64>,
    pub streaming_requests: GenericCounter<AtomicU64>,
}

/// Pre-resolved metric handles keyed by `model` alone (no `endpoint` label),
/// for the request-lifecycle recorders (`record_llm_inflight_start`/`_end`,
/// `record_dynamo_success`) that used to call `.with_label_values()` fresh on
/// every single request/token instead of going through a cache like
/// `LabeledMetrics` — profiling showed those uncached lookups as one of the
/// hottest costs in the whole request path (prometheus's `MetricVec` does a
/// label-key hash + map lookup per call).
pub struct ModelMetrics {
    pub df_inflight: GenericGauge<PromAtomicI64>,
    pub df_queued: GenericGauge<PromAtomicI64>,
    pub df_request_duration: Histogram,
    pub df_requests: GenericCounter<AtomicU64>,
    pub df_input_seq_tokens: GenericCounter<AtomicU64>,
    pub df_output_tokens: GenericCounter<AtomicU64>,
    pub df_output_seq_tokens: GenericCounter<AtomicU64>,
    pub dp_request_duration: Histogram,
    pub dp_requests: GenericCounter<AtomicU64>,
    pub dp_inflight: GenericGauge<PromAtomicI64>,
    pub dd_request_duration: Histogram,
    pub dd_requests: GenericCounter<AtomicU64>,
    pub dd_inflight: GenericGauge<PromAtomicI64>,
}

pub struct MetricRecorder {
    pub metrics: AllMetrics,
    pub throughput: Arc<Throughput>,
    // When false (`--no-metrics`), every recording method is a no-op and
    // `labeled()` returns a shared throwaway handle without touching the
    // DashMap — the request hot path skips all per-request metric work.
    enabled: bool,
    disabled_handle: Arc<LabeledMetrics>,
    inflight_count: AtomicI64,
    total_kv_blocks: i64,
    // Lock-free membership set: the hot path only ever *reads* it (the model is
    // already present after the first request), so a `DashMap` read beats
    // taking a global `Mutex` on every request just to find the model present.
    initialized_models: DashMap<String, ()>,
    labeled_cache: DashMap<(String, String), Arc<LabeledMetrics>>,
    model_cache: DashMap<String, Arc<ModelMetrics>>,
}

#[derive(Debug, Clone)]
pub struct LLMLatencyInfo {
    pub e2e: std::time::Duration,
    pub prefill: std::time::Duration,
    pub decode: std::time::Duration,
}

impl MetricRecorder {
    pub fn new() -> Self {
        Self::with_enabled(true)
    }

    /// Construct a recorder, optionally with all hot-path recording disabled
    /// (`--no-metrics`). The metric families are still created so the exposition
    /// endpoints respond, but no per-request updates occur.
    pub fn with_enabled(enabled: bool) -> Self {
        let metrics = AllMetrics::new();
        let disabled_handle = Arc::new(Self::build_labeled(&metrics, "__disabled__", "__disabled__"));
        Self {
            metrics,
            throughput: Arc::new(Throughput::new()),
            enabled,
            disabled_handle,
            inflight_count: AtomicI64::new(0),
            total_kv_blocks: 1024,
            initialized_models: DashMap::new(),
            labeled_cache: DashMap::with_capacity_and_shard_amount(256, 32),
            model_cache: DashMap::with_capacity_and_shard_amount(256, 32),
        }
    }

    /// Whether hot-path metric recording is active.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Cache per-request handles so token loops avoid label-map lookups.
    pub fn labeled(&self, endpoint: &str, model: &str) -> Arc<LabeledMetrics> {
        // Disabled: hand back the shared throwaway handle. Every `*_fast`
        // recorder short-circuits on `!enabled`, so its counters are never
        // touched — no DashMap access or key allocation on the hot path.
        if !self.enabled {
            return self.disabled_handle.clone();
        }
        let key = (endpoint.to_string(), model.to_string());
        if let Some(hit) = self.labeled_cache.get(&key) {
            return hit.clone();
        }
        let built = Self::build_labeled(&self.metrics, endpoint, model);
        let arc = Arc::new(built);
        self.labeled_cache.entry(key).or_insert(arc).clone()
    }

    /// Resolve every labeled child handle for one (endpoint, model) pair once.
    fn build_labeled(m: &AllMetrics, endpoint: &str, model: &str) -> LabeledMetrics {
        LabeledMetrics {
            tokens_streamed: m
                .aiperf
                .TOKENS_STREAMED_TOTAL
                .with_label_values(&[endpoint, model]),
            prompt_tokens: m
                .aiperf
                .PROMPT_TOKENS_TOTAL
                .with_label_values(&[endpoint, model]),
            completion_tokens: m
                .aiperf
                .COMPLETION_TOKENS_TOTAL
                .with_label_values(&[endpoint, model]),
            tokens_per_request_prompt: m
                .aiperf
                .TOKENS_PER_REQUEST
                .with_label_values(&[endpoint, "prompt"]),
            tokens_per_request_completion: m
                .aiperf
                .TOKENS_PER_REQUEST
                .with_label_values(&[endpoint, "completion"]),
            ttft_by_endpoint: m
                .aiperf
                .TIME_TO_FIRST_TOKEN_SECONDS
                .with_label_values(&[endpoint]),
            itl_by_endpoint: m
                .aiperf
                .INTER_TOKEN_LATENCY_SECONDS
                .with_label_values(&[endpoint]),
            df_ttft: m
                .dynamo_frontend
                .TIME_TO_FIRST_TOKEN_SECONDS
                .with_label_values(&[model]),
            df_itl: m
                .dynamo_frontend
                .INTER_TOKEN_LATENCY_SECONDS
                .with_label_values(&[model]),
            df_request_duration: m
                .dynamo_frontend
                .REQUEST_DURATION_SECONDS
                .with_label_values(&[model]),
            df_requests: m.dynamo_frontend.REQUESTS.with_label_values(&[model]),
            df_input_seq_tokens: m
                .dynamo_frontend
                .INPUT_SEQUENCE_TOKENS
                .with_label_values(&[model]),
            df_output_tokens: m.dynamo_frontend.OUTPUT_TOKENS.with_label_values(&[model]),
            df_output_seq_tokens: m
                .dynamo_frontend
                .OUTPUT_SEQUENCE_TOKENS
                .with_label_values(&[model]),
            df_inflight: m
                .dynamo_frontend
                .INFLIGHT_REQUESTS
                .with_label_values(&[model]),
            df_queued: m
                .dynamo_frontend
                .QUEUED_REQUESTS
                .with_label_values(&[model]),
            dp_request_duration: m
                .dynamo_prefill
                .REQUEST_DURATION_SECONDS
                .with_label_values(&["generate", model]),
            dp_requests: m
                .dynamo_prefill
                .REQUESTS
                .with_label_values(&["generate", model]),
            dp_inflight: m
                .dynamo_prefill
                .INFLIGHT_REQUESTS
                .with_label_values(&["generate", model]),
            dd_request_duration: m
                .dynamo_decode
                .REQUEST_DURATION_SECONDS
                .with_label_values(&["generate", model]),
            dd_requests: m
                .dynamo_decode
                .REQUESTS
                .with_label_values(&["generate", model]),
            dd_inflight: m
                .dynamo_decode
                .INFLIGHT_REQUESTS
                .with_label_values(&["generate", model]),
            requests_total_200: m
                .aiperf
                .REQUESTS_TOTAL
                .with_label_values(&[endpoint, "POST", "200"]),
            requests_total_500: m
                .aiperf
                .REQUESTS_TOTAL
                .with_label_values(&[endpoint, "POST", "500"]),
            requests_in_progress: m.aiperf.REQUESTS_IN_PROGRESS.with_label_values(&[endpoint]),
            request_latency: m
                .aiperf
                .REQUEST_LATENCY_SECONDS
                .with_label_values(&[endpoint]),
            requests_by_model: m
                .aiperf
                .REQUESTS_BY_MODEL
                .with_label_values(&[model, endpoint]),
            request_bytes: m.aiperf.REQUEST_BYTES_TOTAL.with_label_values(&[endpoint]),
            response_bytes: m.aiperf.RESPONSE_BYTES_TOTAL.with_label_values(&[endpoint]),
            streaming_requests: m
                .aiperf
                .STREAMING_REQUESTS_TOTAL
                .with_label_values(&[endpoint, model]),
        }
    }

    /// Cache per-request handles for the model-only-keyed metrics used by
    /// `record_llm_inflight_start`/`_end` and `record_dynamo_success`.
    fn model_metrics(&self, model: &str) -> Arc<ModelMetrics> {
        if let Some(hit) = self.model_cache.get(model) {
            return hit.clone();
        }
        let m = &self.metrics;
        let built = ModelMetrics {
            df_inflight: m
                .dynamo_frontend
                .INFLIGHT_REQUESTS
                .with_label_values(&[model]),
            df_queued: m
                .dynamo_frontend
                .QUEUED_REQUESTS
                .with_label_values(&[model]),
            df_request_duration: m
                .dynamo_frontend
                .REQUEST_DURATION_SECONDS
                .with_label_values(&[model]),
            df_requests: m.dynamo_frontend.REQUESTS.with_label_values(&[model]),
            df_input_seq_tokens: m
                .dynamo_frontend
                .INPUT_SEQUENCE_TOKENS
                .with_label_values(&[model]),
            df_output_tokens: m.dynamo_frontend.OUTPUT_TOKENS.with_label_values(&[model]),
            df_output_seq_tokens: m
                .dynamo_frontend
                .OUTPUT_SEQUENCE_TOKENS
                .with_label_values(&[model]),
            dp_request_duration: m
                .dynamo_prefill
                .REQUEST_DURATION_SECONDS
                .with_label_values(&["generate", model]),
            dp_requests: m
                .dynamo_prefill
                .REQUESTS
                .with_label_values(&["generate", model]),
            dp_inflight: m
                .dynamo_prefill
                .INFLIGHT_REQUESTS
                .with_label_values(&["generate", model]),
            dd_request_duration: m
                .dynamo_decode
                .REQUEST_DURATION_SECONDS
                .with_label_values(&["generate", model]),
            dd_requests: m
                .dynamo_decode
                .REQUESTS
                .with_label_values(&["generate", model]),
            dd_inflight: m
                .dynamo_decode
                .INFLIGHT_REQUESTS
                .with_label_values(&["generate", model]),
        };
        let arc = Arc::new(built);
        self.model_cache
            .entry(model.to_string())
            .or_insert(arc)
            .clone()
    }

    pub fn record_request_start(&self, endpoint: &str, model: &str) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .REQUESTS_IN_PROGRESS
            .with_label_values(&[endpoint])
            .inc();
        self.metrics
            .aiperf
            .REQUESTS_BY_MODEL
            .with_label_values(&[model, endpoint])
            .inc();
    }

    pub fn record_request_end(&self, endpoint: &str) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .REQUESTS_IN_PROGRESS
            .with_label_values(&[endpoint])
            .dec();
    }

    pub fn record_error(&self, endpoint: &str, error_type: &str) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .REQUESTS_TOTAL
            .with_label_values(&[endpoint, "POST", "500"])
            .inc();
        self.metrics
            .aiperf
            .ERRORS_TOTAL
            .with_label_values(&[endpoint, error_type])
            .inc();
    }

    pub fn record_basic_success(&self, endpoint: &str, latency_secs: f64) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .REQUESTS_TOTAL
            .with_label_values(&[endpoint, "POST", "200"])
            .inc();
        self.metrics
            .aiperf
            .REQUEST_LATENCY_SECONDS
            .with_label_values(&[endpoint])
            .observe(latency_secs);
    }

    pub fn record_request_bytes(&self, endpoint: &str, req_bytes: u64, resp_bytes: u64) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .REQUEST_BYTES_TOTAL
            .with_label_values(&[endpoint])
            .inc_by(req_bytes);
        self.metrics
            .aiperf
            .RESPONSE_BYTES_TOTAL
            .with_label_values(&[endpoint])
            .inc_by(resp_bytes);
    }

    pub fn record_token_metrics(&self, endpoint: &str, model: &str, usage: &Usage) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .PROMPT_TOKENS_TOTAL
            .with_label_values(&[endpoint, model])
            .inc_by(usage.prompt_tokens as u64);
        self.metrics
            .aiperf
            .COMPLETION_TOKENS_TOTAL
            .with_label_values(&[endpoint, model])
            .inc_by(usage.completion_tokens as u64);
        self.metrics
            .aiperf
            .TOKENS_PER_REQUEST
            .with_label_values(&[endpoint, "prompt"])
            .observe(usage.prompt_tokens as f64);
        self.metrics
            .aiperf
            .TOKENS_PER_REQUEST
            .with_label_values(&[endpoint, "completion"])
            .observe(usage.completion_tokens as f64);
        self.throughput
            .record_tokens(usage.completion_tokens as u64);
    }

    /// Record TTFT without resolving labels.
    pub fn record_ttft_fast(&self, labeled: &LabeledMetrics, ttft_secs: f64) {
        if !self.enabled {
            return;
        }
        labeled.ttft_by_endpoint.observe(ttft_secs);
        self.metrics
            .vllm
            .TIME_TO_FIRST_TOKEN_SECONDS
            .observe(ttft_secs);
        self.metrics
            .sglang
            .TIME_TO_FIRST_TOKEN_SECONDS
            .observe(ttft_secs);
        self.metrics
            .trtllm
            .TIME_TO_FIRST_TOKEN_SECONDS
            .observe(ttft_secs);
        labeled.df_ttft.observe(ttft_secs);
    }

    /// Record ITL without resolving labels.
    pub fn record_itl_fast(&self, labeled: &LabeledMetrics, itl_secs: f64) {
        if !self.enabled {
            return;
        }
        labeled.itl_by_endpoint.observe(itl_secs);
        self.metrics
            .vllm
            .INTER_TOKEN_LATENCY_SECONDS
            .observe(itl_secs);
        self.metrics
            .trtllm
            .TIME_PER_OUTPUT_TOKEN_SECONDS
            .observe(itl_secs);
        labeled.df_itl.observe(itl_secs);
    }

    #[inline]
    pub fn record_streamed_token_fast(&self, labeled: &LabeledMetrics) {
        if !self.enabled {
            return;
        }
        labeled.tokens_streamed.inc();
        self.throughput.record_tokens(1);
    }

    /// Record a pre-rendered batch with one counter update.
    #[inline]
    pub fn record_streamed_tokens_fast(&self, labeled: &LabeledMetrics, count: u64) {
        if !self.enabled {
            return;
        }
        labeled.tokens_streamed.inc_by(count);
        self.throughput.record_tokens(count);
    }

    /// Record the zero-latency observations represented by a pre-rendered batch.
    #[inline]
    pub fn record_zero_ttft_and_itls(&self, labeled: &LabeledMetrics, itl_count: usize) {
        if !self.enabled {
            return;
        }
        labeled.ttft_by_endpoint.observe(0.0);
        self.metrics.vllm.TIME_TO_FIRST_TOKEN_SECONDS.observe(0.0);
        self.metrics.sglang.TIME_TO_FIRST_TOKEN_SECONDS.observe(0.0);
        self.metrics.trtllm.TIME_TO_FIRST_TOKEN_SECONDS.observe(0.0);
        labeled.df_ttft.observe(0.0);
        for _ in 0..itl_count {
            labeled.itl_by_endpoint.observe(0.0);
            self.metrics.vllm.INTER_TOKEN_LATENCY_SECONDS.observe(0.0);
            self.metrics
                .trtllm
                .TIME_PER_OUTPUT_TOKEN_SECONDS
                .observe(0.0);
            labeled.df_itl.observe(0.0);
        }
    }

    pub fn record_ttft(&self, endpoint: &str, model: &str, ttft_secs: f64) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .TIME_TO_FIRST_TOKEN_SECONDS
            .with_label_values(&[endpoint])
            .observe(ttft_secs);
        self.metrics
            .vllm
            .TIME_TO_FIRST_TOKEN_SECONDS
            .observe(ttft_secs);
        self.metrics
            .sglang
            .TIME_TO_FIRST_TOKEN_SECONDS
            .observe(ttft_secs);
        self.metrics
            .trtllm
            .TIME_TO_FIRST_TOKEN_SECONDS
            .observe(ttft_secs);
        self.metrics
            .dynamo_frontend
            .TIME_TO_FIRST_TOKEN_SECONDS
            .with_label_values(&[model])
            .observe(ttft_secs);
    }

    pub fn record_itl(&self, endpoint: &str, model: &str, itl_secs: f64) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .INTER_TOKEN_LATENCY_SECONDS
            .with_label_values(&[endpoint])
            .observe(itl_secs);
        self.metrics
            .vllm
            .INTER_TOKEN_LATENCY_SECONDS
            .observe(itl_secs);
        self.metrics
            .trtllm
            .TIME_PER_OUTPUT_TOKEN_SECONDS
            .observe(itl_secs);
        self.metrics
            .dynamo_frontend
            .INTER_TOKEN_LATENCY_SECONDS
            .with_label_values(&[model])
            .observe(itl_secs);
    }

    pub fn record_streamed_token(&self, endpoint: &str, model: &str) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .TOKENS_STREAMED_TOTAL
            .with_label_values(&[endpoint, model])
            .inc();
        self.throughput.record_tokens(1);
    }

    fn update_kv_cache_gauges(&self, _model: &str) {
        let inflight = self.inflight_count.load(Ordering::Relaxed).max(0);
        let active = (inflight * 10).min(self.total_kv_blocks);
        let usage = if self.total_kv_blocks > 0 {
            active as f64 / self.total_kv_blocks as f64
        } else {
            0.0
        };
        self.metrics.vllm.KV_CACHE_USAGE.set(usage);
        // A fixed host-cache fraction keeps telemetry deterministic and bounded.
        self.metrics.vllm.CPU_CACHE_USAGE.set(usage * 0.5);
        self.metrics.sglang.TOKEN_USAGE.set(usage);
        self.metrics.sglang.CACHE_HIT_RATE.set(0.3);
        self.metrics
            .dynamo_prefill
            .KVSTATS_ACTIVE_BLOCKS
            .set(active);
        self.metrics
            .dynamo_prefill
            .KVSTATS_TOTAL_BLOCKS
            .set(self.total_kv_blocks);
        self.metrics
            .dynamo_prefill
            .KVSTATS_GPU_CACHE_USAGE_PERCENT
            .set(usage);
        self.metrics.dynamo_decode.KVSTATS_ACTIVE_BLOCKS.set(active);
        self.metrics
            .dynamo_decode
            .KVSTATS_TOTAL_BLOCKS
            .set(self.total_kv_blocks);
        self.metrics
            .dynamo_decode
            .KVSTATS_GPU_CACHE_USAGE_PERCENT
            .set(usage);
    }

    /// Nonnegative concurrency input for the analytic latency model.
    pub fn inflight_count(&self) -> i64 {
        self.inflight_count.load(Ordering::Relaxed).max(0)
    }

    pub fn record_llm_inflight_start(&self, model: &str) {
        if !self.enabled {
            return;
        }
        self.inflight_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.vllm.NUM_REQUESTS_RUNNING.inc();
        self.metrics.vllm.NUM_REQUESTS_WAITING.set(0);
        self.metrics.sglang.NUM_RUNNING_REQS.inc();
        self.metrics.sglang.NUM_QUEUE_REQS.set(0);
        let mm = self.model_metrics(model);
        mm.df_inflight.inc();
        mm.df_queued.set(0);
        mm.dp_inflight.inc();
        mm.dd_inflight.inc();
        self.update_kv_cache_gauges(model);
    }

    pub fn record_llm_inflight_end(&self, model: &str) {
        if !self.enabled {
            return;
        }
        let prev = self.inflight_count.fetch_sub(1, Ordering::Relaxed);
        if prev <= 0 {
            self.inflight_count.store(0, Ordering::Relaxed);
        }
        self.metrics.vllm.NUM_REQUESTS_RUNNING.dec();
        self.metrics.sglang.NUM_RUNNING_REQS.dec();
        let mm = self.model_metrics(model);
        mm.df_inflight.dec();
        mm.dp_inflight.dec();
        mm.dd_inflight.dec();
        self.update_kv_cache_gauges(model);
    }

    pub fn record_llm_backend_success(&self, latency_secs: f64, usage: &Usage) {
        if !self.enabled {
            return;
        }
        let p = usage.prompt_tokens as u64;
        let c = usage.completion_tokens as u64;
        let t = usage.total_tokens as u64;

        self.metrics
            .vllm
            .E2E_REQUEST_LATENCY_SECONDS
            .observe(latency_secs);
        self.metrics.vllm.PROMPT_TOKENS.inc_by(p);
        self.metrics.vllm.GENERATION_TOKENS.inc_by(c);
        self.metrics.vllm.REQUEST_SUCCESS.inc();
        self.metrics.vllm.ITERATION_TOKENS_TOTAL.observe(t as f64);
        self.metrics.vllm.REQUEST_QUEUE_TIME_SECONDS.observe(0.0);
        self.metrics.vllm.PREFIX_CACHE_QUERIES.inc_by(p);
        self.metrics
            .vllm
            .PREFIX_CACHE_HITS
            .inc_by((p as f64 * 0.3) as u64);
        // A smaller external-cache fraction exercises AIPerf's hit-rate derivation.
        self.metrics.vllm.EXTERNAL_PREFIX_CACHE_QUERIES.inc_by(p);
        self.metrics
            .vllm
            .EXTERNAL_PREFIX_CACHE_HITS
            .inc_by((p as f64 * 0.15) as u64);
        // One bump per request gives interval scrapes deterministic nonzero deltas.
        self.metrics.vllm.NUM_PREEMPTIONS.inc();

        self.metrics
            .sglang
            .E2E_REQUEST_LATENCY_SECONDS
            .observe(latency_secs);
        self.metrics.sglang.QUEUE_TIME_SECONDS.observe(0.0);
        if latency_secs > 0.0 {
            self.metrics
                .sglang
                .GEN_THROUGHPUT
                .set(c as f64 / latency_secs);
        }
        self.metrics.sglang.NUM_USED_TOKENS.add(t as i64);
        // These counters exercise SGLang-specific telemetry derivations.
        self.metrics.sglang.PROMPT_TOKENS.inc_by(p);
        self.metrics.sglang.GENERATION_TOKENS.inc_by(c);
        self.metrics
            .sglang
            .CACHED_TOKENS
            .inc_by((p as f64 * 0.3) as u64);
        self.metrics.sglang.NUM_RETRACTED_REQS.inc();

        self.metrics
            .trtllm
            .E2E_REQUEST_LATENCY_SECONDS
            .observe(latency_secs);
        self.metrics.trtllm.REQUEST_SUCCESS.inc();
        self.metrics.trtllm.REQUEST_QUEUE_TIME_SECONDS.observe(0.0);
    }

    pub fn record_dynamo_success(
        &self,
        model: &str,
        latency_secs: f64,
        usage: &Usage,
        info: &LLMLatencyInfo,
    ) {
        if !self.enabled {
            return;
        }
        let p = usage.prompt_tokens as u64;
        let c = usage.completion_tokens as u64;

        let mm = self.model_metrics(model);
        mm.df_request_duration.observe(latency_secs);
        mm.df_requests.inc();
        mm.df_input_seq_tokens.inc_by(p);
        mm.df_output_tokens.inc_by(c);
        mm.df_output_seq_tokens.inc_by(c);

        mm.dp_request_duration.observe(info.prefill.as_secs_f64());
        mm.dp_requests.inc();
        mm.dd_request_duration.observe(info.decode.as_secs_f64());
        mm.dd_requests.inc();
    }

    pub fn record_llm_success(
        &self,
        endpoint: &str,
        model: &str,
        latency_secs: f64,
        usage: &Usage,
        info: &LLMLatencyInfo,
    ) {
        if !self.enabled {
            return;
        }
        self.record_token_metrics(endpoint, model, usage);
        self.record_basic_success(endpoint, latency_secs);
        self.record_llm_backend_success(latency_secs, usage);
        self.record_dynamo_success(model, latency_secs, usage, info);
    }

    /// Admit a non-streaming request using pre-resolved `LabeledMetrics` handles.
    ///
    /// Behaviourally identical to `record_request_start` + `record_llm_inflight_start`
    /// but every labeled metric is reached through the cached child handle in
    /// `labeled`, so no `MetricVec` label hash/lookup (nor the separate
    /// `model_metrics` DashMap lookup) happens on the request hot path. Bracket
    /// the request with `complete_fast` after the response is built.
    pub fn admit_fast(&self, l: &LabeledMetrics) {
        if !self.enabled {
            return;
        }
        // request_start
        l.requests_in_progress.inc();
        l.requests_by_model.inc();
        // llm_inflight_start
        self.inflight_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.vllm.NUM_REQUESTS_RUNNING.inc();
        self.metrics.vllm.NUM_REQUESTS_WAITING.set(0);
        self.metrics.sglang.NUM_RUNNING_REQS.inc();
        self.metrics.sglang.NUM_QUEUE_REQS.set(0);
        l.df_inflight.inc();
        l.df_queued.set(0);
        l.dp_inflight.inc();
        l.dd_inflight.inc();
        self.update_kv_cache_gauges("");
    }

    /// Retire a non-streaming request admitted with `admit_fast`, recording bytes,
    /// token/latency/backend/dynamo success, inflight-end and request-end through
    /// the cached `labeled` handles. Mirrors the exact sequence of
    /// `record_request_bytes` + `record_llm_success` + `record_llm_inflight_end`
    /// + `record_request_end`, minus every per-call label lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn complete_fast(
        &self,
        l: &LabeledMetrics,
        latency_secs: f64,
        usage: &Usage,
        info: &LLMLatencyInfo,
        req_bytes: u64,
        resp_bytes: u64,
    ) {
        if !self.enabled {
            return;
        }
        let p = usage.prompt_tokens as u64;
        let c = usage.completion_tokens as u64;

        // request_bytes
        l.request_bytes.inc_by(req_bytes);
        l.response_bytes.inc_by(resp_bytes);

        // token_metrics
        l.prompt_tokens.inc_by(p);
        l.completion_tokens.inc_by(c);
        l.tokens_per_request_prompt.observe(usage.prompt_tokens as f64);
        l.tokens_per_request_completion
            .observe(usage.completion_tokens as f64);
        self.throughput.record_tokens(c);

        // basic_success
        l.requests_total_200.inc();
        l.request_latency.observe(latency_secs);

        // backend_success (global, non-labeled metrics — nothing to cache)
        self.record_llm_backend_success(latency_secs, usage);

        // dynamo_success via cached handles
        l.df_request_duration.observe(latency_secs);
        l.df_requests.inc();
        l.df_input_seq_tokens.inc_by(p);
        l.df_output_tokens.inc_by(c);
        l.df_output_seq_tokens.inc_by(c);
        l.dp_request_duration.observe(info.prefill.as_secs_f64());
        l.dp_requests.inc();
        l.dd_request_duration.observe(info.decode.as_secs_f64());
        l.dd_requests.inc();

        // llm_inflight_end
        let prev = self.inflight_count.fetch_sub(1, Ordering::Relaxed);
        if prev <= 0 {
            self.inflight_count.store(0, Ordering::Relaxed);
        }
        self.metrics.vllm.NUM_REQUESTS_RUNNING.dec();
        self.metrics.sglang.NUM_RUNNING_REQS.dec();
        l.df_inflight.dec();
        l.dp_inflight.dec();
        l.dd_inflight.dec();
        self.update_kv_cache_gauges("");

        // request_end
        l.requests_in_progress.dec();
    }

    pub fn record_embedding_success(
        &self,
        endpoint: &str,
        model: &str,
        prompt_tokens: usize,
        num_embeddings: usize,
        latency_secs: f64,
    ) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .PROMPT_TOKENS_TOTAL
            .with_label_values(&[endpoint, model])
            .inc_by(prompt_tokens as u64);
        self.metrics
            .aiperf
            .EMBEDDINGS_GENERATED_TOTAL
            .with_label_values(&[model])
            .inc_by(num_embeddings as u64);
        self.record_basic_success(endpoint, latency_secs);
    }

    pub fn record_ranking_success(
        &self,
        endpoint: &str,
        model: &str,
        prompt_tokens: usize,
        num_passages: usize,
        latency_secs: f64,
    ) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .PROMPT_TOKENS_TOTAL
            .with_label_values(&[endpoint, model])
            .inc_by(prompt_tokens as u64);
        self.metrics
            .aiperf
            .RANKINGS_GENERATED_TOTAL
            .with_label_values(&[endpoint])
            .inc();
        self.metrics
            .aiperf
            .PASSAGES_RANKED_TOTAL
            .with_label_values(&[endpoint])
            .inc_by(num_passages as u64);
        self.record_basic_success(endpoint, latency_secs);
    }

    pub fn record_image_retrieval_success(
        &self,
        endpoint: &str,
        num_images: usize,
        latency_secs: f64,
    ) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .IMAGES_PROCESSED_TOTAL
            .with_label_values(&[endpoint])
            .inc_by(num_images as u64);
        self.record_basic_success(endpoint, latency_secs);
    }

    pub fn record_content_bytes_fetched(&self, endpoint: &str, bytes: u64) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .CONTENT_BYTES_FETCHED_TOTAL
            .with_label_values(&[endpoint])
            .inc_by(bytes);
    }

    pub fn record_tgi_success(&self, endpoint: &str, usage: &Usage, latency_secs: f64) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .PROMPT_TOKENS_TOTAL
            .with_label_values(&[endpoint, "tgi"])
            .inc_by(usage.prompt_tokens as u64);
        self.metrics
            .aiperf
            .COMPLETION_TOKENS_TOTAL
            .with_label_values(&[endpoint, "tgi"])
            .inc_by(usage.completion_tokens as u64);
        self.record_basic_success(endpoint, latency_secs);
    }

    pub fn record_streaming_start(&self, endpoint: &str, model: &str) {
        if !self.enabled {
            return;
        }
        self.metrics
            .aiperf
            .STREAMING_REQUESTS_TOTAL
            .with_label_values(&[endpoint, model])
            .inc();
    }

    /// Return observed models in deterministic order.
    pub fn seen_models(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .initialized_models
            .iter()
            .map(|e| e.key().clone())
            .collect();
        v.sort();
        v
    }

    pub fn init_model_config(&self, model: &str) {
        if !self.enabled {
            return;
        }
        // Fast path: an already-seen model is the overwhelming common case, and
        // a `DashMap` read here avoids a global lock on every request.
        if self.initialized_models.contains_key(model) {
            return;
        }
        if self.initialized_models.insert(model.to_string(), ()).is_some() {
            return;
        }
        self.metrics
            .dynamo_frontend
            .MODEL_CONTEXT_LENGTH
            .with_label_values(&[model])
            .set(8192.0);
        self.metrics
            .dynamo_frontend
            .MODEL_KV_CACHE_BLOCK_SIZE
            .with_label_values(&[model])
            .set(16.0);
        self.metrics
            .dynamo_frontend
            .MODEL_TOTAL_KV_BLOCKS
            .with_label_values(&[model])
            .set(self.total_kv_blocks as f64);
        // Touch the label so it appears in the exposition even if never incremented.
        self.metrics
            .dynamo_frontend
            .DISCONNECTED_CLIENTS
            .with_label_values(&[model])
            .inc_by(0);
    }
}

impl Default for MetricRecorder {
    fn default() -> Self {
        Self::new()
    }
}
