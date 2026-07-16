// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metric registries - one per exposed endpoint.

use prometheus::{
    Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec,
    IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};

#[allow(non_snake_case)]
pub struct AIPerfMockMetrics {
    pub registry: Registry,

    pub REQUESTS_TOTAL: IntCounterVec,
    pub REQUESTS_IN_PROGRESS: IntGaugeVec,
    pub REQUEST_LATENCY_SECONDS: HistogramVec,
    pub PROMPT_TOKENS_TOTAL: IntCounterVec,
    pub COMPLETION_TOKENS_TOTAL: IntCounterVec,
    pub TOKENS_PER_REQUEST: HistogramVec,
    pub STREAMING_REQUESTS_TOTAL: IntCounterVec,
    pub TOKENS_STREAMED_TOTAL: IntCounterVec,
    pub TIME_TO_FIRST_TOKEN_SECONDS: HistogramVec,
    pub INTER_TOKEN_LATENCY_SECONDS: HistogramVec,
    pub ERRORS_TOTAL: IntCounterVec,
    pub REQUESTS_BY_MODEL: IntCounterVec,
    pub EMBEDDINGS_GENERATED_TOTAL: IntCounterVec,
    pub RANKINGS_GENERATED_TOTAL: IntCounterVec,
    pub PASSAGES_RANKED_TOTAL: IntCounterVec,
    pub IMAGES_PROCESSED_TOTAL: IntCounterVec,
    pub SERVER_UPTIME_SECONDS: Gauge,
    pub REQUEST_BYTES_TOTAL: IntCounterVec,
    pub RESPONSE_BYTES_TOTAL: IntCounterVec,
}

fn b(buckets: &[f64]) -> Vec<f64> {
    buckets.to_vec()
}

impl AIPerfMockMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let request_latency_buckets = vec![
            0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05,
            0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ];
        let ttft_buckets = vec![
            0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025,
            0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5,
        ];
        let itl_buckets = vec![
            0.000001, 0.0000025, 0.000005, 0.0000075, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
            0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
        ];

        let m = Self {
            REQUESTS_TOTAL: IntCounterVec::new(
                Opts::new("aiperf_mock_requests_total", "Total number of requests"),
                &["endpoint", "method", "status"],
            )
            .unwrap(),
            REQUESTS_IN_PROGRESS: IntGaugeVec::new(
                Opts::new(
                    "aiperf_mock_requests_in_progress",
                    "Number of requests currently being processed",
                ),
                &["endpoint"],
            )
            .unwrap(),
            REQUEST_LATENCY_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "aiperf_mock_request_latency_seconds",
                    "Request latency in seconds",
                )
                .buckets(b(&request_latency_buckets)),
                &["endpoint"],
            )
            .unwrap(),
            PROMPT_TOKENS_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_prompt_tokens_total",
                    "Total number of prompt/input tokens processed",
                ),
                &["endpoint", "model"],
            )
            .unwrap(),
            COMPLETION_TOKENS_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_completion_tokens_total",
                    "Total number of completion/output tokens generated",
                ),
                &["endpoint", "model"],
            )
            .unwrap(),
            TOKENS_PER_REQUEST: HistogramVec::new(
                HistogramOpts::new("aiperf_mock_tokens_per_request", "Tokens per request").buckets(
                    vec![
                        1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
                        25000.0, 50000.0,
                    ],
                ),
                &["endpoint", "token_type"],
            )
            .unwrap(),
            STREAMING_REQUESTS_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_streaming_requests_total",
                    "Total number of streaming requests",
                ),
                &["endpoint", "model"],
            )
            .unwrap(),
            TOKENS_STREAMED_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_tokens_streamed_total",
                    "Total number of tokens streamed",
                ),
                &["endpoint", "model"],
            )
            .unwrap(),
            TIME_TO_FIRST_TOKEN_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "aiperf_mock_time_to_first_token_seconds",
                    "Time to first token in seconds",
                )
                .buckets(b(&ttft_buckets)),
                &["endpoint"],
            )
            .unwrap(),
            INTER_TOKEN_LATENCY_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "aiperf_mock_inter_token_latency_seconds",
                    "Inter-token latency in seconds",
                )
                .buckets(b(&itl_buckets)),
                &["endpoint"],
            )
            .unwrap(),
            ERRORS_TOTAL: IntCounterVec::new(
                Opts::new("aiperf_mock_errors_total", "Total number of errors"),
                &["endpoint", "error_type"],
            )
            .unwrap(),
            REQUESTS_BY_MODEL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_requests_by_model_total",
                    "Total requests by model",
                ),
                &["model", "endpoint"],
            )
            .unwrap(),
            EMBEDDINGS_GENERATED_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_embeddings_generated_total",
                    "Total number of embeddings generated",
                ),
                &["model"],
            )
            .unwrap(),
            RANKINGS_GENERATED_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_rankings_generated_total",
                    "Total number of rankings generated",
                ),
                &["endpoint"],
            )
            .unwrap(),
            PASSAGES_RANKED_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_passages_ranked_total",
                    "Total number of passages ranked",
                ),
                &["endpoint"],
            )
            .unwrap(),
            IMAGES_PROCESSED_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_images_processed_total",
                    "Total number of images processed by image retrieval",
                ),
                &["endpoint"],
            )
            .unwrap(),
            SERVER_UPTIME_SECONDS: Gauge::new(
                "aiperf_mock_uptime_seconds",
                "Server uptime in seconds (updated on metrics scrape)",
            )
            .unwrap(),
            REQUEST_BYTES_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_request_bytes_total",
                    "Total number of bytes received in requests",
                ),
                &["endpoint"],
            )
            .unwrap(),
            RESPONSE_BYTES_TOTAL: IntCounterVec::new(
                Opts::new(
                    "aiperf_mock_response_bytes_total",
                    "Total number of bytes sent in responses",
                ),
                &["endpoint"],
            )
            .unwrap(),
            registry,
        };
        m.registry
            .register(Box::new(m.REQUESTS_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUESTS_IN_PROGRESS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUEST_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.PROMPT_TOKENS_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.COMPLETION_TOKENS_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TOKENS_PER_REQUEST.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.STREAMING_REQUESTS_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TOKENS_STREAMED_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TIME_TO_FIRST_TOKEN_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.INTER_TOKEN_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.ERRORS_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUESTS_BY_MODEL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.EMBEDDINGS_GENERATED_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.RANKINGS_GENERATED_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.PASSAGES_RANKED_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.IMAGES_PROCESSED_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.SERVER_UPTIME_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUEST_BYTES_TOTAL.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.RESPONSE_BYTES_TOTAL.clone()))
            .unwrap();
        m
    }
}

#[allow(non_snake_case)]
pub struct VllmMetrics {
    pub registry: Registry,
    pub E2E_REQUEST_LATENCY_SECONDS: Histogram,
    pub TIME_TO_FIRST_TOKEN_SECONDS: Histogram,
    pub INTER_TOKEN_LATENCY_SECONDS: Histogram,
    pub PROMPT_TOKENS: IntCounter,
    pub GENERATION_TOKENS: IntCounter,
    pub REQUEST_QUEUE_TIME_SECONDS: Histogram,
    pub REQUEST_SUCCESS: IntCounter,
    pub NUM_REQUESTS_RUNNING: IntGauge,
    pub NUM_REQUESTS_WAITING: IntGauge,
    pub KV_CACHE_USAGE: Gauge,
    pub NUM_PREEMPTIONS: IntCounter,
    pub PREFIX_CACHE_HITS: IntCounter,
    pub PREFIX_CACHE_QUERIES: IntCounter,
    pub EXTERNAL_PREFIX_CACHE_HITS: IntCounter,
    pub EXTERNAL_PREFIX_CACHE_QUERIES: IntCounter,
    pub CPU_CACHE_USAGE: Gauge,
    pub ITERATION_TOKENS_TOTAL: Histogram,
}

impl VllmMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let vllm_latency = vec![
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
        ];
        let vllm_itl = vec![
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0,
        ];
        let vllm_iter = vec![
            1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
            16384.0,
        ];

        let m = Self {
            E2E_REQUEST_LATENCY_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "vllm:e2e_request_latency_seconds",
                    "Histogram of e2e request latency in seconds.",
                )
                .buckets(vllm_latency.clone()),
            )
            .unwrap(),
            TIME_TO_FIRST_TOKEN_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "vllm:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(vllm_latency.clone()),
            )
            .unwrap(),
            INTER_TOKEN_LATENCY_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "vllm:inter_token_latency_seconds",
                    "Histogram of inter-token latency in seconds.",
                )
                .buckets(vllm_itl),
            )
            .unwrap(),
            PROMPT_TOKENS: IntCounter::with_opts(Opts::new(
                "vllm:prompt_tokens",
                "Number of prefill tokens processed.",
            ))
            .unwrap(),
            GENERATION_TOKENS: IntCounter::with_opts(Opts::new(
                "vllm:generation_tokens",
                "Number of generation tokens processed.",
            ))
            .unwrap(),
            REQUEST_QUEUE_TIME_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "vllm:request_queue_time_seconds",
                    "Histogram of time spent in WAITING phase for request.",
                )
                .buckets(vllm_latency.clone()),
            )
            .unwrap(),
            REQUEST_SUCCESS: IntCounter::with_opts(Opts::new(
                "vllm:request_success",
                "Count of successfully processed requests.",
            ))
            .unwrap(),
            NUM_REQUESTS_RUNNING: IntGauge::with_opts(Opts::new(
                "vllm:num_requests_running",
                "Number of requests in model execution batches.",
            ))
            .unwrap(),
            NUM_REQUESTS_WAITING: IntGauge::with_opts(Opts::new(
                "vllm:num_requests_waiting",
                "Number of requests waiting to be processed.",
            ))
            .unwrap(),
            KV_CACHE_USAGE: Gauge::with_opts(Opts::new(
                "vllm:kv_cache_usage_perc",
                "KV-cache usage. 1 means 100 percent usage.",
            ))
            .unwrap(),
            NUM_PREEMPTIONS: IntCounter::with_opts(Opts::new(
                "vllm:num_preemptions",
                "Cumulative number of preemption from the engine.",
            ))
            .unwrap(),
            PREFIX_CACHE_HITS: IntCounter::with_opts(Opts::new(
                "vllm:prefix_cache_hits",
                "Prefix cache hits, in terms of number of cached tokens.",
            ))
            .unwrap(),
            PREFIX_CACHE_QUERIES: IntCounter::with_opts(Opts::new(
                "vllm:prefix_cache_queries",
                "Prefix cache queries, in terms of number of queried tokens.",
            ))
            .unwrap(),
            EXTERNAL_PREFIX_CACHE_HITS: IntCounter::with_opts(Opts::new(
                "vllm:external_prefix_cache_hits",
                "External prefix cache hits, in terms of number of cached tokens.",
            ))
            .unwrap(),
            EXTERNAL_PREFIX_CACHE_QUERIES: IntCounter::with_opts(Opts::new(
                "vllm:external_prefix_cache_queries",
                "External prefix cache queries, in terms of number of queried tokens.",
            ))
            .unwrap(),
            CPU_CACHE_USAGE: Gauge::with_opts(Opts::new(
                "vllm:cpu_cache_usage_perc",
                "CPU KV-cache usage. 1 means 100 percent usage.",
            ))
            .unwrap(),
            ITERATION_TOKENS_TOTAL: Histogram::with_opts(
                HistogramOpts::new(
                    "vllm:iteration_tokens_total",
                    "Histogram of number of tokens per engine_step.",
                )
                .buckets(vllm_iter),
            )
            .unwrap(),
            registry,
        };
        m.registry
            .register(Box::new(m.E2E_REQUEST_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TIME_TO_FIRST_TOKEN_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.INTER_TOKEN_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.PROMPT_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.GENERATION_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUEST_QUEUE_TIME_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUEST_SUCCESS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_REQUESTS_RUNNING.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_REQUESTS_WAITING.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.KV_CACHE_USAGE.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_PREEMPTIONS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.PREFIX_CACHE_HITS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.PREFIX_CACHE_QUERIES.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.EXTERNAL_PREFIX_CACHE_HITS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.EXTERNAL_PREFIX_CACHE_QUERIES.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.CPU_CACHE_USAGE.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.ITERATION_TOKENS_TOTAL.clone()))
            .unwrap();
        m
    }
}

#[allow(non_snake_case)]
pub struct SglangMetrics {
    pub registry: Registry,
    pub GEN_THROUGHPUT: Gauge,
    pub NUM_QUEUE_REQS: IntGauge,
    pub NUM_RUNNING_REQS: IntGauge,
    pub CACHE_HIT_RATE: Gauge,
    pub NUM_USED_TOKENS: IntGauge,
    pub TOKEN_USAGE: Gauge,
    pub CACHED_TOKENS: IntCounter,
    pub PROMPT_TOKENS: IntCounter,
    pub GENERATION_TOKENS: IntCounter,
    pub NUM_RETRACTED_REQS: IntCounter,
    pub QUEUE_TIME_SECONDS: Histogram,
    pub E2E_REQUEST_LATENCY_SECONDS: Histogram,
    pub TIME_TO_FIRST_TOKEN_SECONDS: Histogram,
}

impl SglangMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let expo = vec![
            0.001, 0.00162, 0.002624, 0.004252, 0.006887, 0.011158, 0.018075, 0.029282, 0.047437,
            0.076848, 0.124494, 0.201681, 0.326723, 0.529292, 0.857453, 1.389073, 2.250299,
            3.645484, 5.905685, 9.567209, 15.498879, 25.108184, 40.675258, 65.893919, 106.748148,
        ];
        let sglang_ttft = vec![
            0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
        ];

        let m = Self {
            GEN_THROUGHPUT: Gauge::with_opts(Opts::new(
                "sglang:gen_throughput",
                "The generation throughput (token/s).",
            ))
            .unwrap(),
            NUM_QUEUE_REQS: IntGauge::with_opts(Opts::new(
                "sglang:num_queue_reqs",
                "The number of requests in the waiting queue.",
            ))
            .unwrap(),
            NUM_RUNNING_REQS: IntGauge::with_opts(Opts::new(
                "sglang:num_running_reqs",
                "The number of running requests.",
            ))
            .unwrap(),
            CACHE_HIT_RATE: Gauge::with_opts(Opts::new(
                "sglang:cache_hit_rate",
                "The prefix cache hit rate.",
            ))
            .unwrap(),
            NUM_USED_TOKENS: IntGauge::with_opts(Opts::new(
                "sglang:num_used_tokens",
                "The number of used tokens.",
            ))
            .unwrap(),
            TOKEN_USAGE: Gauge::with_opts(Opts::new("sglang:token_usage", "The token usage."))
                .unwrap(),
            CACHED_TOKENS: IntCounter::with_opts(Opts::new(
                "sglang:cached_tokens",
                "The number of cached prefix tokens (prefix cache hits).",
            ))
            .unwrap(),
            PROMPT_TOKENS: IntCounter::with_opts(Opts::new(
                "sglang:prompt_tokens",
                "The number of prompt (prefill) tokens processed.",
            ))
            .unwrap(),
            GENERATION_TOKENS: IntCounter::with_opts(Opts::new(
                "sglang:generation_tokens",
                "The number of generation tokens processed.",
            ))
            .unwrap(),
            NUM_RETRACTED_REQS: IntCounter::with_opts(Opts::new(
                "sglang:num_retracted_reqs",
                "The number of retracted (preempted) requests.",
            ))
            .unwrap(),
            QUEUE_TIME_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "sglang:queue_time_seconds",
                    "Histogram of queueing time in seconds.",
                )
                .buckets(expo.clone()),
            )
            .unwrap(),
            E2E_REQUEST_LATENCY_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "sglang:e2e_request_latency_seconds",
                    "Histogram of end to end request latency in seconds.",
                )
                .buckets(expo),
            )
            .unwrap(),
            TIME_TO_FIRST_TOKEN_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "sglang:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(sglang_ttft),
            )
            .unwrap(),
            registry,
        };
        m.registry
            .register(Box::new(m.GEN_THROUGHPUT.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_QUEUE_REQS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_RUNNING_REQS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.CACHE_HIT_RATE.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_USED_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TOKEN_USAGE.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.CACHED_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.PROMPT_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.GENERATION_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.NUM_RETRACTED_REQS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.QUEUE_TIME_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.E2E_REQUEST_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TIME_TO_FIRST_TOKEN_SECONDS.clone()))
            .unwrap();
        m
    }
}

#[allow(non_snake_case)]
pub struct TrtllmMetrics {
    pub registry: Registry,
    pub E2E_REQUEST_LATENCY_SECONDS: Histogram,
    pub TIME_TO_FIRST_TOKEN_SECONDS: Histogram,
    pub TIME_PER_OUTPUT_TOKEN_SECONDS: Histogram,
    pub REQUEST_QUEUE_TIME_SECONDS: Histogram,
    pub REQUEST_SUCCESS: IntCounter,
}

impl TrtllmMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let trtllm_latency = vec![
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
        ];
        let trtllm_ttft = vec![
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
        ];
        let trtllm_tpot = vec![
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0,
        ];
        let m = Self {
            E2E_REQUEST_LATENCY_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "trtllm:e2e_request_latency_seconds",
                    "Histogram of end to end request latency in seconds.",
                )
                .buckets(trtllm_latency.clone()),
            )
            .unwrap(),
            TIME_TO_FIRST_TOKEN_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "trtllm:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(trtllm_ttft),
            )
            .unwrap(),
            TIME_PER_OUTPUT_TOKEN_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "trtllm:time_per_output_token_seconds",
                    "Histogram of time per output token in seconds.",
                )
                .buckets(trtllm_tpot),
            )
            .unwrap(),
            REQUEST_QUEUE_TIME_SECONDS: Histogram::with_opts(
                HistogramOpts::new(
                    "trtllm:request_queue_time_seconds",
                    "Histogram of time spent in WAITING phase for request.",
                )
                .buckets(trtllm_latency),
            )
            .unwrap(),
            REQUEST_SUCCESS: IntCounter::with_opts(Opts::new(
                "trtllm:request_success",
                "Count of successfully processed requests.",
            ))
            .unwrap(),
            registry,
        };
        m.registry
            .register(Box::new(m.E2E_REQUEST_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TIME_TO_FIRST_TOKEN_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TIME_PER_OUTPUT_TOKEN_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUEST_QUEUE_TIME_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.REQUEST_SUCCESS.clone()))
            .unwrap();
        m
    }
}

#[allow(non_snake_case)]
pub struct DynamoFrontendMetrics {
    pub registry: Registry,
    pub REQUEST_DURATION_SECONDS: HistogramVec,
    pub TIME_TO_FIRST_TOKEN_SECONDS: HistogramVec,
    pub INTER_TOKEN_LATENCY_SECONDS: HistogramVec,
    pub REQUESTS: IntCounterVec,
    pub INPUT_SEQUENCE_TOKENS: IntCounterVec,
    pub OUTPUT_SEQUENCE_TOKENS: IntCounterVec,
    pub OUTPUT_TOKENS: IntCounterVec,
    pub QUEUED_REQUESTS: IntGaugeVec,
    pub INFLIGHT_REQUESTS: IntGaugeVec,
    pub DISCONNECTED_CLIENTS: IntCounterVec,
    pub MODEL_CONTEXT_LENGTH: GaugeVec,
    pub MODEL_KV_CACHE_BLOCK_SIZE: GaugeVec,
    pub MODEL_TOTAL_KV_BLOCKS: GaugeVec,
}

impl DynamoFrontendMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let duration = vec![
            0.0, 1.9, 3.4, 6.3, 12.0, 22.0, 40.0, 75.0, 140.0, 260.0, 480.0, 900.0,
        ];
        let ttft = vec![
            0.0, 0.0022, 0.0047, 0.01, 0.022, 0.047, 0.1, 0.22, 0.47, 1.0, 2.2, 4.7, 10.0, 22.0,
            48.0, 100.0, 220.0, 480.0,
        ];
        let itl = vec![
            0.0, 0.0019, 0.0035, 0.0067, 0.013, 0.024, 0.045, 0.084, 0.16, 0.3, 0.56, 1.1, 2.0,
        ];
        let m = Self {
            REQUEST_DURATION_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_frontend_request_duration_seconds",
                    "Duration of LLM requests",
                )
                .buckets(duration),
                &["model"],
            )
            .unwrap(),
            TIME_TO_FIRST_TOKEN_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_frontend_time_to_first_token_seconds",
                    "Time to first token in seconds",
                )
                .buckets(ttft),
                &["model"],
            )
            .unwrap(),
            INTER_TOKEN_LATENCY_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_frontend_inter_token_latency_seconds",
                    "Inter-token latency in seconds",
                )
                .buckets(itl),
                &["model"],
            )
            .unwrap(),
            REQUESTS: IntCounterVec::new(
                Opts::new("dynamo_frontend_requests", "Total number of requests"),
                &["model"],
            )
            .unwrap(),
            INPUT_SEQUENCE_TOKENS: IntCounterVec::new(
                Opts::new(
                    "dynamo_frontend_input_sequence_tokens",
                    "Total input sequence tokens",
                ),
                &["model"],
            )
            .unwrap(),
            OUTPUT_SEQUENCE_TOKENS: IntCounterVec::new(
                Opts::new(
                    "dynamo_frontend_output_sequence_tokens",
                    "Total output sequence tokens",
                ),
                &["model"],
            )
            .unwrap(),
            OUTPUT_TOKENS: IntCounterVec::new(
                Opts::new("dynamo_frontend_output_tokens", "Total output tokens"),
                &["model"],
            )
            .unwrap(),
            QUEUED_REQUESTS: IntGaugeVec::new(
                Opts::new(
                    "dynamo_frontend_queued_requests",
                    "Number of requests in the queue",
                ),
                &["model"],
            )
            .unwrap(),
            INFLIGHT_REQUESTS: IntGaugeVec::new(
                Opts::new(
                    "dynamo_frontend_inflight_requests",
                    "Number of requests currently being processed",
                ),
                &["model"],
            )
            .unwrap(),
            DISCONNECTED_CLIENTS: IntCounterVec::new(
                Opts::new(
                    "dynamo_frontend_disconnected_clients",
                    "Total number of disconnected clients",
                ),
                &["model"],
            )
            .unwrap(),
            MODEL_CONTEXT_LENGTH: GaugeVec::new(
                Opts::new(
                    "dynamo_frontend_model_context_length",
                    "Maximum context length in tokens for a worker serving the model",
                ),
                &["model"],
            )
            .unwrap(),
            MODEL_KV_CACHE_BLOCK_SIZE: GaugeVec::new(
                Opts::new(
                    "dynamo_frontend_model_kv_cache_block_size",
                    "KV cache block size",
                ),
                &["model"],
            )
            .unwrap(),
            MODEL_TOTAL_KV_BLOCKS: GaugeVec::new(
                Opts::new(
                    "dynamo_frontend_model_total_kv_blocks",
                    "Total number of KV cache blocks",
                ),
                &["model"],
            )
            .unwrap(),
            registry,
        };
        m.registry
            .register(Box::new(m.REQUEST_DURATION_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.TIME_TO_FIRST_TOKEN_SECONDS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.INTER_TOKEN_LATENCY_SECONDS.clone()))
            .unwrap();
        m.registry.register(Box::new(m.REQUESTS.clone())).unwrap();
        m.registry
            .register(Box::new(m.INPUT_SEQUENCE_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.OUTPUT_SEQUENCE_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.OUTPUT_TOKENS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.QUEUED_REQUESTS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.INFLIGHT_REQUESTS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.DISCONNECTED_CLIENTS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.MODEL_CONTEXT_LENGTH.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.MODEL_KV_CACHE_BLOCK_SIZE.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.MODEL_TOTAL_KV_BLOCKS.clone()))
            .unwrap();
        m
    }
}

#[allow(non_snake_case)]
pub struct DynamoComponentMetrics {
    pub registry: Registry,
    pub REQUEST_DURATION_SECONDS: HistogramVec,
    pub REQUESTS: IntCounterVec,
    pub INFLIGHT_REQUESTS: IntGaugeVec,
    pub KVSTATS_ACTIVE_BLOCKS: IntGauge,
    pub KVSTATS_TOTAL_BLOCKS: IntGauge,
    pub KVSTATS_GPU_CACHE_USAGE_PERCENT: Gauge,
}

impl DynamoComponentMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let buckets = vec![
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ];
        let m = Self {
            REQUEST_DURATION_SECONDS: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_component_request_duration_seconds",
                    "Time spent processing requests by work handler",
                )
                .buckets(buckets),
                &["dynamo_endpoint", "model"],
            )
            .unwrap(),
            REQUESTS: IntCounterVec::new(
                Opts::new(
                    "dynamo_component_requests",
                    "Total number of requests processed by work handler",
                ),
                &["dynamo_endpoint", "model"],
            )
            .unwrap(),
            INFLIGHT_REQUESTS: IntGaugeVec::new(
                Opts::new(
                    "dynamo_component_inflight_requests",
                    "Number of requests currently being processed by work handler",
                ),
                &["dynamo_endpoint", "model"],
            )
            .unwrap(),
            KVSTATS_ACTIVE_BLOCKS: IntGauge::with_opts(Opts::new(
                "dynamo_component_kvstats_active_blocks",
                "Number of active KV cache blocks currently in use",
            ))
            .unwrap(),
            KVSTATS_TOTAL_BLOCKS: IntGauge::with_opts(Opts::new(
                "dynamo_component_kvstats_total_blocks",
                "Total number of KV cache blocks available",
            ))
            .unwrap(),
            KVSTATS_GPU_CACHE_USAGE_PERCENT: Gauge::with_opts(Opts::new(
                "dynamo_component_kvstats_gpu_cache_usage_percent",
                "GPU cache usage as a percentage (0.0-1.0)",
            ))
            .unwrap(),
            registry,
        };
        m.registry
            .register(Box::new(m.REQUEST_DURATION_SECONDS.clone()))
            .unwrap();
        m.registry.register(Box::new(m.REQUESTS.clone())).unwrap();
        m.registry
            .register(Box::new(m.INFLIGHT_REQUESTS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.KVSTATS_ACTIVE_BLOCKS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.KVSTATS_TOTAL_BLOCKS.clone()))
            .unwrap();
        m.registry
            .register(Box::new(m.KVSTATS_GPU_CACHE_USAGE_PERCENT.clone()))
            .unwrap();
        m
    }
}

pub struct AllMetrics {
    pub aiperf: AIPerfMockMetrics,
    pub vllm: VllmMetrics,
    pub sglang: SglangMetrics,
    pub trtllm: TrtllmMetrics,
    pub dynamo_frontend: DynamoFrontendMetrics,
    pub dynamo_prefill: DynamoComponentMetrics,
    pub dynamo_decode: DynamoComponentMetrics,
}

impl AllMetrics {
    pub fn new() -> Self {
        Self {
            aiperf: AIPerfMockMetrics::new(),
            vllm: VllmMetrics::new(),
            sglang: SglangMetrics::new(),
            trtllm: TrtllmMetrics::new(),
            dynamo_frontend: DynamoFrontendMetrics::new(),
            dynamo_prefill: DynamoComponentMetrics::new(),
            dynamo_decode: DynamoComponentMetrics::new(),
        }
    }
}

impl Default for AllMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AIPerfMockMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for VllmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SglangMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TrtllmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DynamoFrontendMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DynamoComponentMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub fn encode(registry: &Registry) -> Vec<u8> {
    let encoder = TextEncoder::new();
    let mut buf = Vec::new();
    encoder
        .encode(&registry.gather(), &mut buf)
        .expect("prometheus encode cannot fail");
    buf
}

/// Escape a Prometheus label value (backslash, double-quote, newline).
fn escape_label(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Append the live accuracy tally to an already-encoded exposition body. These
/// metric names are not registered in the `prometheus` `Registry` (the tally is
/// a set of plain atomics read at scrape time), so appending them as exposition
/// text is valid — each name carries its own single `# HELP`/`# TYPE` block.
pub fn append_accuracy_metrics(buf: &mut Vec<u8>, snap: &crate::accuracy::AccuracyLiveSnapshot) {
    use std::fmt::Write as _;
    let mut s = String::new();
    let mut scalar = |name: &str, kind: &str, help: &str, value: String| {
        let _ = writeln!(s, "# HELP aiperf_mock_{name} {help}");
        let _ = writeln!(s, "# TYPE aiperf_mock_{name} {kind}");
        let _ = writeln!(s, "aiperf_mock_{name} {value}");
    };
    scalar(
        "accuracy_matched_total",
        "counter",
        "Requests matched to a dataset prompt and answered.",
        snap.matched.to_string(),
    );
    scalar(
        "accuracy_correct_total",
        "counter",
        "Matched requests answered correctly.",
        snap.correct.to_string(),
    );
    scalar(
        "accuracy_incorrect_total",
        "counter",
        "Matched requests answered incorrectly.",
        snap.incorrect.to_string(),
    );
    scalar(
        "accuracy_unmatched_total",
        "counter",
        "Accuracy-enabled requests whose prompt matched no dataset row.",
        snap.unmatched.to_string(),
    );
    scalar(
        "accuracy_adversarial_total",
        "counter",
        "Answered responses rendered as an adversarial parser-choke shape.",
        snap.adversarial.to_string(),
    );
    scalar(
        "accuracy_cot_total",
        "counter",
        "Answered responses rendered as chain-of-thought.",
        snap.cot.to_string(),
    );
    scalar(
        "accuracy_ratio",
        "gauge",
        "Live correct/matched accuracy for the run.",
        format!("{:.6}", snap.accuracy),
    );

    if !snap.tasks.is_empty() {
        let _ = writeln!(
            s,
            "# HELP aiperf_mock_accuracy_task_matched_total Per-task matched requests."
        );
        let _ = writeln!(s, "# TYPE aiperf_mock_accuracy_task_matched_total counter");
        for (task, t) in &snap.tasks {
            let _ = writeln!(
                s,
                "aiperf_mock_accuracy_task_matched_total{{task=\"{}\"}} {}",
                escape_label(task),
                t.matched
            );
        }
        let _ = writeln!(
            s,
            "# HELP aiperf_mock_accuracy_task_correct_total Per-task correct answers."
        );
        let _ = writeln!(s, "# TYPE aiperf_mock_accuracy_task_correct_total counter");
        for (task, t) in &snap.tasks {
            let _ = writeln!(
                s,
                "aiperf_mock_accuracy_task_correct_total{{task=\"{}\"}} {}",
                escape_label(task),
                t.correct
            );
        }
        let _ = writeln!(
            s,
            "# HELP aiperf_mock_accuracy_task_ratio Per-task live correct/matched accuracy."
        );
        let _ = writeln!(s, "# TYPE aiperf_mock_accuracy_task_ratio gauge");
        for (task, t) in &snap.tasks {
            let _ = writeln!(
                s,
                "aiperf_mock_accuracy_task_ratio{{task=\"{}\"}} {:.6}",
                escape_label(task),
                t.accuracy
            );
        }
    }
    buf.extend_from_slice(s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_metrics_build() {
        let m = AllMetrics::new();
        m.aiperf
            .REQUESTS_TOTAL
            .with_label_values(&["/x", "POST", "200"])
            .inc();
        m.vllm.REQUEST_SUCCESS.inc();
        m.sglang.GEN_THROUGHPUT.set(42.0);
        m.trtllm.REQUEST_SUCCESS.inc();
        m.dynamo_frontend
            .REQUESTS
            .with_label_values(&["model"])
            .inc();
        let body = encode(&m.aiperf.registry);
        assert!(!body.is_empty());
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("aiperf_mock_requests_total"));
    }

    #[test]
    fn encode_vllm_emits_kv_cache() {
        let m = VllmMetrics::new();
        m.KV_CACHE_USAGE.set(0.5);
        let body = encode(&m.registry);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("vllm:kv_cache_usage_perc"));
        assert!(text.contains("0.5"));
    }
}
