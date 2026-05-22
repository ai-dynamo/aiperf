---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Server Metrics Reference
---
# AIPerf Server Metrics Reference

Comprehensive reference for server metrics collected during AIPerf benchmark runs from NVIDIA Dynamo, vLLM, SGLang, TensorRT-LLM, and Triton Inference Server endpoints.

## Table of Contents

1. [Quick Reference: Common Questions](#quick-reference-common-questions)
2. [Backend Comparison Matrix](#backend-comparison-matrix)
3. [Metric Interpretation Guide](#metric-interpretation-guide)
4. [Detailed Metric Definitions](#detailed-metric-definitions)
   - [Dynamo Frontend](#dynamo-frontend)
   - [Dynamo Component](#dynamo-component)
   - [vLLM](#vllm)
   - [SGLang](#sglang)
   - [TensorRT-LLM](#tensorrt-llm)
   - [Triton Inference Server](#triton-inference-server)
   - [KVBM (KV Block Manager)](#kvbm-kv-block-manager)
5. [Appendix](#appendix)

---

## Quick Reference: Common Questions

### "What is my throughput?"

| Metric | Field | Description |
|--------|-------|-------------|
| `dynamo_frontend_requests` | `stats.rate` | Requests per second |
| `dynamo_frontend_output_tokens` | `stats.rate` | Output tokens per second |
| `vllm:prompt_tokens` | `stats.rate` | Input tokens per second (vLLM) |
| `vllm:generation_tokens` | `stats.rate` | Generation throughput (vLLM) |
| `sglang:prompt_tokens` | `stats.rate` | Prefill throughput (SGLang) |
| `sglang:generation_tokens` | `stats.rate` | Generation throughput (SGLang) |
| `sglang:gen_throughput` | `stats.avg` | Real-time generation throughput (SGLang) |
| `nv_inference_request_success` | `stats.rate` | Successful requests per second (Triton) |
| `nv_inference_count` | `stats.rate` | Inferences per second (Triton) |

### "What is my latency?"

| Metric | Field | Description |
|--------|-------|-------------|
| `dynamo_frontend_request_duration_seconds` | `stats.p99_estimate` | End-to-end p99 latency |
| `dynamo_frontend_request_duration_seconds` | `stats.avg` | Average request latency |
| `dynamo_frontend_time_to_first_token_seconds` | `stats.p99_estimate` | Time to first token (TTFT) p99 |
| `dynamo_frontend_inter_token_latency_seconds` | `stats.p99_estimate` | Inter-token latency (ITL) p99 |
| `vllm:time_to_first_token_seconds` | `stats.p99_estimate` | TTFT p99 (vLLM) |
| `sglang:time_to_first_token_seconds` | `stats.p99_estimate` | TTFT p99 (SGLang) |
| `sglang:e2e_request_latency_seconds` | `stats.p99_estimate` | End-to-end p99 latency (SGLang) |
| `sglang:inter_token_latency_seconds` | `stats.p99_estimate` | ITL p99 (SGLang) |
| `sglang:queue_time_seconds` | `stats.p99_estimate` | Queue time p99 (SGLang) |
| `trtllm_time_to_first_token_seconds` | `stats.p99_estimate` | TTFT p99 (TensorRT-LLM) |
| `nv_inference_request_duration_us` | `stats.rate` | End-to-end request time accumulation rate (Triton, microseconds/s) |
| `nv_inference_first_response_histogram_ms` | `stats.p99_estimate` | First-response latency p99 when Triton histogram latencies are enabled |

### "Am I hitting capacity limits?"

| Metric | Field | Threshold | Meaning |
|--------|-------|-----------|---------|
| `vllm:kv_cache_usage_perc` | `stats.max` | >0.9 | KV cache near full capacity |
| `vllm:num_preemptions` | `stats.total` | >0 | Memory pressure causing preemptions |
| `vllm:num_requests_waiting` | `stats.avg` | Growing | Queue building up |
| `dynamo_frontend_queued_requests` | `stats.avg` | High | Requests awaiting first token |
| `sglang:token_usage` | `stats.max` | >0.9 | High memory utilization (SGLang) |
| `sglang:num_queue_reqs` | `stats.avg` | Growing | Saturation (SGLang) |
| `trtllm_request_queue_time_seconds` | `stats.avg` | High | Saturation (TensorRT-LLM) |
| `nv_inference_pending_request_count` | `stats.max` | Growing | Triton backend queue saturation |
| `nv_gpu_memory_used_bytes` | `stats.max` | Near total | Triton GPU memory pressure |

### "What does my workload look like?"

| Metric | Field | Description |
|--------|-------|-------------|
| `dynamo_frontend_input_sequence_tokens` | `stats.avg` | Average prompt length |
| `dynamo_frontend_input_sequence_tokens` | `stats.p99_estimate` | Longest prompts (p99) |
| `dynamo_frontend_output_sequence_tokens` | `stats.avg` | Average response length |
| `dynamo_frontend_output_sequence_tokens` | `stats.p99_estimate` | Longest responses (p99) |
| `nv_inference_count` / `nv_inference_exec_count` | `stats.total` | Triton average batch size (`inference_count / exec_count`) |

### "Where is time being spent?"

**vLLM latency breakdown:**
```
Total latency = Queue + Prefill + Decode
vllm:e2e_request_latency_seconds ≈
    vllm:request_queue_time_seconds +
    vllm:request_prefill_time_seconds +
    vllm:request_decode_time_seconds
```

| Phase | Metric | What it means |
|-------|--------|---------------|
| Queue | `vllm:request_queue_time_seconds` | Waiting for GPU resources |
| Prefill | `vllm:request_prefill_time_seconds` | Processing input tokens |
| Decode | `vllm:request_decode_time_seconds` | Generating output tokens |

**SGLang latency breakdown** (via `sglang:per_stage_req_latency_seconds` with `stage` label):

| Stage Label | What it means |
|-------------|---------------|
| `request_process` | Unified-mode request processing before queue entry |
| `prefill_bootstrap` | Prefill bootstrap queue time in disaggregated prefill mode |
| `prefill_forward` | Prefill forward pass execution |
| `chunked_prefill` | Additional chunked-prefill forward slices |
| `prefill_transfer_kv_cache` | KV cache transfer from prefill to decode worker |
| `decode_prepare` | Decode preallocation preparation |
| `decode_bootstrap` | Decode bootstrap/transfer setup |
| `decode_waiting` | Waiting before decode forward execution |
| `decode_transferred` | Decode-side transferred request processing before queue entry |
| `fake_output` | Fake-output/prebuilt decode stage |

**TensorRT-LLM latency breakdown:**

| Phase | Metric | What it means |
|-------|--------|---------------|
| Queue | `trtllm_request_queue_time_seconds` | Waiting for GPU resources |
| TTFT | `trtllm_time_to_first_token_seconds` | Time to first output token |
| Total | `trtllm_e2e_request_latency_seconds` | Complete request duration |

---

## Backend Comparison Matrix

Key equivalent metrics across backends:

| Capability | Dynamo Frontend | vLLM | SGLang | TensorRT-LLM | Triton |
|------------|----------------|------|--------|--------------|--------|
| **End-to-end latency** | `dynamo_frontend_request_duration_seconds` | `vllm:e2e_request_latency_seconds` | `sglang:e2e_request_latency_seconds` | `trtllm_e2e_request_latency_seconds` | `nv_inference_request_duration_us` |
| **TTFT / first response** | `dynamo_frontend_time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` | `sglang:time_to_first_token_seconds` | `trtllm_time_to_first_token_seconds` | `nv_inference_first_response_histogram_ms` |
| **ITL** | `dynamo_frontend_inter_token_latency_seconds` | `vllm:inter_token_latency_seconds` | `sglang:inter_token_latency_seconds` | `trtllm_time_per_output_token_seconds` | — |
| **Queue time** | — | `vllm:request_queue_time_seconds` | `sglang:queue_time_seconds` | `trtllm_request_queue_time_seconds` | `nv_inference_queue_duration_us` |
| **KV/cache usage** | `dynamo_component_gpu_cache_usage_percent` | `vllm:kv_cache_usage_perc` | `sglang:token_usage` | `trtllm_kv_cache_utilization` | response cache `nv_cache_*` |
| **Requests running** | `dynamo_frontend_inflight_requests` | `vllm:num_requests_running` | `sglang:num_running_reqs` | `trtllm_num_requests_running` | — |
| **Requests queued** | `dynamo_frontend_queued_requests` | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | `trtllm_num_requests_waiting` | `nv_inference_pending_request_count` |
| **Successful requests** | `dynamo_frontend_requests` | `vllm:request_success` | `sglang:num_requests` | `trtllm_request_success` | `nv_inference_request_success` |
| **Prompt tokens** | `dynamo_frontend_input_sequence_tokens` | `vllm:request_prompt_tokens` | `sglang:prompt_tokens_histogram` | `trtllm_prompt_tokens` | — |
| **Generation tokens** | `dynamo_frontend_output_sequence_tokens` | `vllm:request_generation_tokens` | `sglang:generation_tokens_histogram` | `trtllm_generation_tokens` | — |

**Key insight:** Dynamo metrics measure at the HTTP/routing layer (user-facing), while backend metrics measure inside the inference engine (debugging). Use both for complete visibility.

---

## Metric Interpretation Guide

### Metric Types

**Counter** (cumulative, monotonically increasing):
- `stats.total` = Total change during benchmark
- `stats.rate` = Rate of change (per second)
- Example: `vllm:prompt_tokens` with `stats.rate` = prefill throughput
- AIPerf stores Prometheus counter family names without the exposition sample's trailing `_total` suffix, so upstream `*_total` counter samples usually appear as `*` in AIPerf exports.

**Gauge** (point-in-time snapshot):
- `stats.avg` = Typical value
- `stats.max` = Peak value
- `stats.min` = Minimum value
- `stats.p50`, `stats.p90`, `stats.p99` = Percentile values
- Example: `vllm:num_requests_waiting` with `stats.max` = worst-case queue depth

**Histogram** (distribution):
- `stats.total` = Total count of observations
- `stats.sum` = Sum of all observed values
- `stats.avg` = Mean (sum/count)
- `stats.p50_estimate`, `stats.p90_estimate`, `stats.p95_estimate`, `stats.p99_estimate` = Estimated percentiles from buckets
- Example: `vllm:e2e_request_latency_seconds` with `stats.p99_estimate` = tail latency

**Info** (static labels):
- Only `stats.avg` is meaningful (value is typically 1.0)
- Labels contain the actual configuration data
- Example: `vllm:cache_config_info` exposes cache settings as labels

### Understanding Percentiles

Histogram percentiles are *estimated* from bucket boundaries, not exact values. Accuracy depends on bucket granularity. See [Histogram Buckets](#histogram-buckets) for bucket definitions.

### Multiple Endpoints

When scraping multiple server instances, each series includes an `endpoint_url` label to identify the source.

---

## Detailed Metric Definitions

### Dynamo Frontend

The Dynamo frontend is the HTTP entry point that receives client requests and routes them to backend workers. These metrics provide user-facing visibility into request processing.

#### Request Flow

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_frontend_requests_started` | counter | requests | `endpoint`, `model`, `request_type` | Requests accepted by the frontend handler. |
| `dynamo_frontend_requests` | counter | requests | `endpoint`, `error_type`, `model`, `request_type`, `status` | Completed LLM requests. Use `stats.total` for count during benchmark, `stats.rate` for throughput (req/s). |
| `dynamo_frontend_active_requests` | gauge | requests | `model` | Requests currently being handled by the frontend, from HTTP handler entry to response completion. |
| `dynamo_frontend_inflight_requests` | gauge | requests | `model` | Engine-bound requests currently being processed. |
| `dynamo_frontend_queued_requests` | gauge | requests | `model` | HTTP-processing queue: requests from handler start until first token generation. |
| `dynamo_frontend_disconnected_clients` | gauge | clients | — | Client connections that disconnected. |

**Label values:**
- `endpoint`: `completions`, `chat_completions`, `embeddings`, `images`, `videos`, `audios`, `responses`, `anthropic_messages`, `tensor`
- `request_type`: `stream`, `unary`
- `status`: `success`, `error`
- `error_type`: empty string for success, or `validation`, `not_found`, `overload`, `cancelled`, `response_timeout`, `internal`, `not_implemented`

#### Latency

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_frontend_request_duration_seconds` | histogram | seconds | `model` | **End-to-end request latency** from HTTP handler entry to response completion. Key metric for SLA compliance. Use `stats.p99_estimate` for tail latency. |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | seconds | `model` | **Time to first token (TTFT)** - latency until first token is generated. Critical for perceived responsiveness. |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | seconds | `model` | **Inter-token latency (ITL)** - time between consecutive tokens. Lower is better for streaming UX. |

**Histogram buckets:**
- `dynamo_frontend_request_duration_seconds`: `0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 130.0, 260.0, 510.0, +Inf`
- `dynamo_frontend_time_to_first_token_seconds`: `0.0, 0.0022, 0.0047, 0.01, 0.022, 0.047, 0.1, 0.22, 0.47, 1.0, 2.2, 4.7, 10.0, 22.0, 48.0, 100.0, 220.0, 480.0, +Inf`
- `dynamo_frontend_inter_token_latency_seconds`: `0.0, 0.0019, 0.0035, 0.0067, 0.013, 0.024, 0.045, 0.084, 0.16, 0.3, 0.56, 1.1, 2.0, +Inf`

#### Tokens

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_frontend_output_tokens` | counter | tokens | `model` | Total output tokens generated. `stats.rate` = output token throughput (tokens/s). |
| `dynamo_frontend_cached_tokens` | histogram | tokens | `model` | Cached tokens (prefix cache hits) per request. |
| `dynamo_frontend_tokenizer_latency_ms` | histogram | milliseconds | `operation` | Tokenizer latency. `operation`: `tokenize`, `detokenize`. |
| `dynamo_frontend_input_sequence_tokens` | histogram | tokens | `model` | **Input sequence length distribution**. `stats.avg` = mean prompt length, `stats.p99_estimate` = longest prompts. |
| `dynamo_frontend_output_sequence_tokens` | histogram | tokens | `model` | **Output sequence length distribution**. `stats.avg` = mean response length. |

**Histogram buckets:**
- `dynamo_frontend_cached_tokens`: Same as `dynamo_frontend_input_sequence_tokens`
- `dynamo_frontend_tokenizer_latency_ms`: `0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, +Inf`
- `dynamo_frontend_input_sequence_tokens`: `0.0, 100.0, 210.0, 430.0, 870.0, 1800.0, 3600.0, 7400.0, 15000.0, 31000.0, 63000.0, 130000.0, +Inf`
- `dynamo_frontend_output_sequence_tokens`: `0.0, 100.0, 210.0, 430.0, 880.0, 1800.0, 3700.0, 7600.0, 16000.0, 32000.0, +Inf`

#### Model Configuration (Static Gauges)

These are constant values that don't change during the benchmark. Only `stats.avg` is meaningful.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_frontend_model_context_length` | gauge | `model` | Maximum context window size in tokens. |
| `dynamo_frontend_model_kv_cache_block_size` | gauge | `model` | KV cache block size in tokens. |
| `dynamo_frontend_model_max_num_batched_tokens` | gauge | `model` | Maximum tokens that can be batched together. |
| `dynamo_frontend_model_max_num_seqs` | gauge | `model` | Maximum concurrent sequences per worker. |
| `dynamo_frontend_model_total_kv_blocks` | gauge | `model` | Total KV cache blocks available per worker. |
| `dynamo_frontend_model_migration_limit` | gauge | `model` | Maximum request migrations allowed for the model. |
| `dynamo_frontend_model_migration` | counter | `migration_type`, `model` | Request migrations due to worker unavailability. `migration_type`: `new_request`, `ongoing_request`. |
| `dynamo_frontend_model_migration_max_seq_len_exceeded` | counter | `model` | Migrations disabled because the sequence length exceeded the configured limit. |
| `dynamo_frontend_model_cancellation` | counter | `endpoint`, `model`, `request_type` | Request cancellations. |
| `dynamo_frontend_model_rejection` | counter | `endpoint`, `model` | Requests rejected due to resource exhaustion. |

#### Frontend Pipeline, Routing, and Worker Load

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_frontend_stage_requests` | gauge | requests | `phase`, `stage` | Requests currently in a frontend pipeline stage. `stage`: `preprocess`, `route`, `dispatch`; `phase`: empty string, `prefill`, `decode`, or `aggregated`. |
| `dynamo_frontend_stage_duration_seconds` | histogram | seconds | `stage` | Pipeline stage duration. |
| `dynamo_frontend_tokenize_seconds` | histogram | seconds | — | Tokenization time in the preprocessor. |
| `dynamo_frontend_template_seconds` | histogram | seconds | — | Chat-template application time in the preprocessor. |
| `dynamo_frontend_detokenize_total_us` | counter | microseconds | — | Cumulative detokenization time. |
| `dynamo_frontend_detokenize_token_count` | counter | tokens | — | Tokens detokenized. |
| `dynamo_frontend_worker_active_decode_blocks` | gauge | blocks | `dp_rank`, `worker_id`, `worker_type` | Active KV-cache decode blocks per worker. |
| `dynamo_frontend_worker_active_prefill_tokens` | gauge | tokens | `dp_rank`, `worker_id`, `worker_type` | Active prefill tokens queued per worker. |
| `dynamo_frontend_worker_last_time_to_first_token_seconds` | gauge | seconds | `dp_rank`, `worker_id`, `worker_type` | Last observed TTFT for a worker. |
| `dynamo_frontend_worker_last_input_sequence_tokens` | gauge | tokens | `dp_rank`, `worker_id`, `worker_type` | Input-token count from the same request as the last observed worker TTFT. |
| `dynamo_frontend_worker_last_inter_token_latency_seconds` | gauge | seconds | `dp_rank`, `worker_id`, `worker_type` | Last observed ITL for a worker. |
| `dynamo_frontend_router_queue_pending_requests` | gauge | requests | `worker_type` | Requests pending in the router scheduler queue. |
| `dynamo_frontend_router_queue_pending_isl_tokens` | gauge | tokens | `worker_type` | Sum of input-sequence tokens for pending router scheduler requests. |

**Histogram buckets:**
- `dynamo_frontend_stage_duration_seconds`: `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, +Inf`
- `dynamo_frontend_tokenize_seconds`: `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, +Inf`
- `dynamo_frontend_template_seconds`: `0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, +Inf`

#### Tokio Runtime and Event Loop Metrics

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_tokio_global_queue_depth` | gauge | tasks | — | Tokio runtime global queue depth. |
| `dynamo_tokio_budget_forced_yield` | counter | yields | — | Tasks forced to yield after exhausting Tokio's cooperative budget. |
| `dynamo_tokio_blocking_threads` | gauge | threads | — | Threads in Tokio's blocking pool. |
| `dynamo_tokio_blocking_idle_threads` | gauge | threads | — | Idle threads in Tokio's blocking pool. |
| `dynamo_tokio_blocking_queue_depth` | gauge | tasks | — | Blocking-pool queue depth. |
| `dynamo_tokio_alive_tasks` | gauge | tasks | — | Alive Tokio tasks. |
| `dynamo_tokio_worker_mean_poll_time_ns` | gauge | nanoseconds | `worker` | Worker mean poll time. |
| `dynamo_tokio_worker_busy_ratio` | gauge | ratio | `worker` | Worker busy ratio. |
| `dynamo_tokio_worker_park_count` | counter | parks | `worker` | Worker park count. |
| `dynamo_tokio_worker_local_queue_depth` | gauge | tasks | `worker` | Worker local queue depth. |
| `dynamo_tokio_worker_steal_count` | counter | steals | `worker` | Worker steal count. |
| `dynamo_tokio_worker_overflow_count` | counter | overflows | `worker` | Worker local-queue overflow count. |
| `dynamo_frontend_event_loop_delay_seconds` | histogram | seconds | — | Event-loop delay canary. |
| `dynamo_frontend_event_loop_stall` | counter | stalls | — | Event-loop stalls over the configured threshold. |

#### Router Request and Overhead Metrics

Router request metrics are component-scoped and therefore also carry `dynamo_namespace`, `dynamo_component`, optional `dynamo_endpoint`, `worker_id`, and `router_id` labels.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_router_requests` | counter | requests | hierarchy labels + `router_id` | Requests processed by the router. |
| `dynamo_component_router_time_to_first_token_seconds` | histogram | seconds | hierarchy labels + `router_id` | Time to first token observed at the router. |
| `dynamo_component_router_inter_token_latency_seconds` | histogram | seconds | hierarchy labels + `router_id` | Average inter-token latency observed at the router. |
| `dynamo_component_router_input_sequence_tokens` | histogram | tokens | hierarchy labels + `router_id` | Input sequence length observed at the router. |
| `dynamo_component_router_output_sequence_tokens` | histogram | tokens | hierarchy labels + `router_id` | Output sequence length observed at the router. |
| `dynamo_component_router_kv_hit_rate` | histogram | ratio | hierarchy labels + `router_id` | Predicted KV cache hit rate at routing time. |
| `dynamo_component_router_kv_transfer_estimated_latency_seconds` | histogram | seconds | hierarchy labels + `router_id` | Upper-bound estimate of KV transfer latency in disaggregated serving. |
| `dynamo_component_router_shared_cache_hit_rate` | histogram | ratio | hierarchy labels + `router_id` | Fraction of request blocks found in shared KV cache. |
| `dynamo_component_router_shared_cache_beyond_blocks` | histogram | blocks | hierarchy labels + `router_id` | Shared cache blocks beyond device overlap for the selected worker. |
| `dynamo_component_router_remote_indexer_query_failures` | counter | errors | hierarchy labels + `router_id` | Remote indexer overlap queries that failed. |
| `dynamo_component_router_remote_indexer_write_failures` | counter | errors | hierarchy labels + `router_id` | Remote indexer routing-decision writes that failed. |
| `dynamo_router_overhead_block_hashing_ms` | histogram | milliseconds | `router_id` | Time spent computing block hashes. |
| `dynamo_router_overhead_indexer_find_matches_ms` | histogram | milliseconds | `router_id` | Time spent in indexer `find_matches`. |
| `dynamo_router_overhead_seq_hashing_ms` | histogram | milliseconds | `router_id` | Time spent computing sequence hashes. |
| `dynamo_router_overhead_scheduling_ms` | histogram | milliseconds | `router_id` | Time spent in scheduler worker selection. |
| `dynamo_router_overhead_total_ms` | histogram | milliseconds | `router_id` | Total routing overhead per request. |
| `dynamo_router_overhead_shared_cache_query_ms` | histogram | milliseconds | `router_id` | Time spent querying shared KV cache. |
| `dynamo_router_shared_cache_errors` | counter | errors | `router_id` | Shared cache query errors. |

**Histogram buckets:**
- `dynamo_component_router_time_to_first_token_seconds`: Same as `dynamo_frontend_time_to_first_token_seconds`
- `dynamo_component_router_inter_token_latency_seconds`: Same as `dynamo_frontend_inter_token_latency_seconds`
- `dynamo_component_router_input_sequence_tokens`: Same as `dynamo_frontend_input_sequence_tokens`
- `dynamo_component_router_output_sequence_tokens`: Same as `dynamo_frontend_output_sequence_tokens`
- `dynamo_component_router_kv_hit_rate`: `0.0, 0.05, 0.1, ... 1.0, +Inf`
- `dynamo_component_router_kv_transfer_estimated_latency_seconds`: `0.0, 0.0019, 0.0037, 0.0072, 0.014, 0.027, 0.052, 0.1, 0.19, 0.37, 0.72, 1.4, 2.7, 5.2, 10.0, +Inf`
- `dynamo_component_router_shared_cache_hit_rate`: `0.0, 0.05, 0.1, ... 1.0, +Inf`
- `dynamo_component_router_shared_cache_beyond_blocks`: `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, +Inf`
- `dynamo_router_overhead_block_hashing_ms`: exponential `0.001 * 2^n`, 15 buckets
- `dynamo_router_overhead_indexer_find_matches_ms`: exponential `0.01 * 3^n`, 17 buckets
- `dynamo_router_overhead_seq_hashing_ms`: exponential `0.001 * 2^n`, 15 buckets
- `dynamo_router_overhead_scheduling_ms`: exponential `0.01 * 3^n`, 17 buckets
- `dynamo_router_overhead_total_ms`: exponential `0.01 * 3^n`, 17 buckets
- `dynamo_router_overhead_shared_cache_query_ms`: exponential `0.01 * 3^n`, 17 buckets

#### KV Publisher Metrics

These component-scoped metrics track Dynamo's KV-event publisher and relay path.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_kv_publisher_engines_dropped_events` | counter | events | hierarchy labels | Raw KV events dropped by engines before reaching the publisher, detected through event ID gaps. |
| `dynamo_component_kv_publisher_zmq_events` | counter | events | hierarchy labels + `stage`, `event_type` | ZMQ KV events seen by the relay. |
| `dynamo_component_kv_publisher_zmq_filtered_events` | counter | events | hierarchy labels + `event_type`, `reason` | ZMQ KV events filtered before conversion. |
| `dynamo_component_kv_publisher_zmq_conversion_issues` | counter | events | hierarchy labels + `event_type`, `reason` | ZMQ KV events dropped due to conversion issues. |
| `dynamo_component_kv_publisher_zmq_suspicious_events` | counter | events | hierarchy labels + `event_type`, `reason` | Suspicious ZMQ KV events that were forwarded. |

---

### Dynamo Component

Dynamo component metrics come from worker, router, and backend processes. Metrics created through Dynamo's hierarchy usually carry `dynamo_namespace`, `dynamo_component`, optional `dynamo_endpoint`, and `worker_id` labels; endpoint handlers may also add engine labels such as `model`.

#### Work Handler Request Processing

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_requests` | counter | requests | hierarchy labels plus engine labels | Requests processed by the work handler. Compare across workers to check load balancing. |
| `dynamo_component_inflight_requests` | gauge | requests | hierarchy labels plus engine labels | Requests currently being processed by the work handler. |
| `dynamo_component_errors` | counter | errors | hierarchy labels plus engine labels, `error_type` | Work-handler errors. `error_type`: `deserialization`, `invalid_message`, `response_stream`, `generate`, `publish_response`, `publish_final`. |
| `dynamo_component_cancellation` | counter | requests | hierarchy labels plus engine labels | Requests cancelled by the work handler. |
| `dynamo_component_request_duration_seconds` | histogram | seconds | hierarchy labels plus engine labels | Worker-level request processing time. Compare to frontend duration to measure routing overhead. |

**Histogram buckets:**
- `dynamo_component_request_duration_seconds`: `0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, +Inf`

#### Work Handler Data Transfer, Queue, and Pool Saturation

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_request_bytes` | counter | bytes | hierarchy labels plus engine labels | Total bytes received in requests. `stats.rate` = inbound bandwidth. |
| `dynamo_component_response_bytes` | counter | bytes | hierarchy labels plus engine labels | Total bytes sent in responses. `stats.rate` = outbound bandwidth. |
| `dynamo_work_handler_network_transit_seconds` | histogram | seconds | — | Frontend-to-backend network transit time. |
| `dynamo_work_handler_time_to_first_response_seconds` | histogram | seconds | — | Backend processing time from payload handling to first response. |
| `dynamo_work_handler_queue_depth` | gauge | requests | — | Items in the bounded work queue awaiting dispatcher pickup. |
| `dynamo_work_handler_queue_capacity` | gauge | requests | — | Configured capacity of the bounded work queue. |
| `dynamo_work_handler_enqueue_rejected` | counter | requests | — | Times enqueuing failed because the dispatcher channel was closed. |
| `dynamo_work_handler_permit_wait_seconds` | histogram | seconds | — | Time spent waiting for a worker-pool permit. |
| `dynamo_work_handler_pool_active_tasks` | gauge | tasks | — | Active worker-pool tasks holding permits. |
| `dynamo_work_handler_pool_capacity` | gauge | tasks | — | Configured worker-pool capacity. |

**Histogram buckets:**
- `dynamo_work_handler_network_transit_seconds`: `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, +Inf`
- `dynamo_work_handler_time_to_first_response_seconds`: `0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, +Inf`
- `dynamo_work_handler_permit_wait_seconds`: `0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, +Inf`

#### Backend KV Cache and Model Info

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_total_blocks` | gauge | blocks | `dynamo_component`, `dp_rank`, `model` | Total KV cache blocks available on a worker. |
| `dynamo_component_gpu_cache_usage_percent` | gauge | ratio | `dynamo_component`, `dp_rank`, `model` | GPU cache utilization (0.0-1.0). High values (>0.9) indicate capacity pressure. |
| `dynamo_component_model_load_time_seconds` | gauge | seconds | `dynamo_component`, `model` | Model load time. |
| `dynamo_component_embedding_cache_hits` | counter | hits | `dynamo_component`, `model` | Multimodal embedding-cache hits. |
| `dynamo_component_embedding_cache_misses` | counter | misses | `dynamo_component`, `model` | Multimodal embedding-cache misses. |
| `dynamo_component_embedding_cache_evictions` | counter | evictions | `dynamo_component`, `model` | Multimodal embedding-cache evictions. |
| `dynamo_component_embedding_cache_utilization` | gauge | ratio | `dynamo_component`, `model` | Multimodal embedding-cache memory utilization (0.0-1.0). |
| `dynamo_component_embedding_cache_current_bytes` | gauge | bytes | `dynamo_component`, `model` | Current multimodal embedding-cache memory usage. |
| `dynamo_component_embedding_cache_entries` | gauge | entries | `dynamo_component`, `model` | Current number of multimodal embedding-cache entries. |

#### Transport and NATS Messaging

Dynamo's current in-code NATS metric is a transport error counter. Older `dynamo_component_nats_client_*` and `dynamo_component_nats_service_*` families were not verified in current upstream code and are not documented as current.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_transport_nats_errors` | counter | errors | `error_type` | NATS request errors. Current `error_type` value: `request_failed`. |
| `dynamo_transport_tcp_bytes_sent` | counter | bytes | — | Bytes sent by the TCP request client. |
| `dynamo_transport_tcp_bytes_received` | counter | bytes | — | Bytes received by the TCP request client. |
| `dynamo_transport_tcp_errors` | counter | errors | — | TCP request send failures or timeouts. |
| `dynamo_request_plane_queue_seconds` | histogram | seconds | — | Time from `generate()` entry to `send_request()`. |
| `dynamo_request_plane_send_seconds` | histogram | seconds | — | Time for `send_request()` to complete. |
| `dynamo_request_plane_roundtrip_ttft_seconds` | histogram | seconds | — | Time from `send_request()` to first response item. |
| `dynamo_request_plane_inflight_requests` | gauge | requests | — | Currently in-flight requests at `AddressedPushRouter`. |

**Histogram buckets:**
- `dynamo_request_plane_queue_seconds`: `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, +Inf`
- `dynamo_request_plane_send_seconds`: `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, +Inf`
- `dynamo_request_plane_roundtrip_ttft_seconds`: `0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, +Inf`

---

### vLLM

vLLM is a high-performance inference engine. These metrics provide deep visibility into model execution, cache usage, and request processing phases. Current vLLM v1 Prometheus metrics use `model_name` and `engine` labels unless noted otherwise.

#### Cache & Memory

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:kv_cache_usage_perc` | gauge | ratio | `model_name`, `engine` | **KV cache utilization** (0.0-1.0). Key capacity indicator. Values near 1.0 cause performance degradation. Monitor `stats.max`. |
| `vllm:prefix_cache_hits` | counter | tokens | `model_name`, `engine` | Prefix cache hits, in terms of number of cached tokens. |
| `vllm:prefix_cache_queries` | counter | tokens | `model_name`, `engine` | Prefix cache queries, in terms of number of queried tokens. `hits/queries` = hit rate. |
| `vllm:external_prefix_cache_hits` | counter | tokens | `model_name`, `engine` | External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens. |
| `vllm:external_prefix_cache_queries` | counter | tokens | `model_name`, `engine` | External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens. |
| `vllm:prompt_tokens_cached` | counter | tokens | `model_name`, `engine` | Cached prompt tokens (local + external). |
| `vllm:mm_cache_hits` | counter | items | `model_name`, `engine` | Multi-modal cache hits, in terms of number of cached items. |
| `vllm:mm_cache_queries` | counter | items | `model_name`, `engine` | Multi-modal cache queries, in terms of number of queried items. |
| `vllm:num_preemptions` | counter | preemptions | `model_name`, `engine` | Cumulative number of preemptions from the engine. Non-zero indicates capacity pressure. |
| `vllm:corrupted_requests` | counter | requests | `model_name`, `engine` | Requests with NaNs in logits. Only emitted when `VLLM_COMPUTE_NANS_IN_LOGITS` is enabled. |

#### Queue & Engine State

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:num_requests_running` | gauge | requests | `model_name`, `engine` | Requests currently in model execution batches. Indicates batch size. |
| `vllm:num_requests_waiting` | gauge | requests | `model_name`, `engine` | Requests queued waiting for execution. High values indicate saturation. |
| `vllm:num_requests_waiting_by_reason` | gauge | requests | `model_name`, `engine`, `reason` | Waiting requests split by reason. `capacity` means waiting for scheduling capacity; `deferred` means deferred by transient constraints such as LoRA budget, KV transfer, or blocked status. |
| `vllm:engine_sleep_state` | gauge | — | `model_name`, `engine`, `sleep_state` | Engine sleep state. `sleep_state` values are `awake`, `weights_offloaded`, and `discard_all`; the active state is reported as 1. |

#### Token Throughput

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:prompt_tokens` | counter | tokens | `model_name`, `engine` | Number of prefill tokens processed. `stats.rate` = prefill throughput. |
| `vllm:prompt_tokens_by_source` | counter | tokens | `model_name`, `engine`, `source` | Number of prompt tokens by source. `source` values are `local_compute`, `local_cache_hit`, and `external_kv_transfer`. |
| `vllm:generation_tokens` | counter | tokens | `model_name`, `engine` | Number of generation tokens processed. `stats.rate` = decode throughput. |
| `vllm:request_success` | counter | requests | `model_name`, `engine`, `finished_reason` | Successfully completed requests. |

**Common `finished_reason` values:** `stop`, `length`, `abort`, `error`, `repetition`

#### Request-Level Latency Breakdown

These histograms show where time is spent for each request. Together they decompose the end-to-end latency.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:e2e_request_latency_seconds` | histogram | seconds | `model_name`, `engine` | Histogram of e2e request latency in seconds. |
| `vllm:request_queue_time_seconds` | histogram | seconds | `model_name`, `engine` | Histogram of time spent in **WAITING** phase for request. |
| `vllm:request_prefill_time_seconds` | histogram | seconds | `model_name`, `engine` | Histogram of time spent in **PREFILL** phase for request. |
| `vllm:request_decode_time_seconds` | histogram | seconds | `model_name`, `engine` | Histogram of time spent in **DECODE** phase for request. |
| `vllm:request_inference_time_seconds` | histogram | seconds | `model_name`, `engine` | Histogram of time spent in **RUNNING** phase for request. |

**Histogram buckets:**
- `vllm:e2e_request_latency_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `vllm:request_queue_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `vllm:request_prefill_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `vllm:request_decode_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `vllm:request_inference_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`

#### Token-Level Latency

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:time_to_first_token_seconds` | histogram | seconds | `model_name`, `engine` | **TTFT** - histogram of time to first token in seconds. |
| `vllm:inter_token_latency_seconds` | histogram | seconds | `model_name`, `engine` | **ITL** - histogram of inter-token latency in seconds. |
| `vllm:request_time_per_output_token_seconds` | histogram | seconds | `model_name`, `engine` | Histogram of time_per_output_token_seconds per request. |

**Histogram buckets:**
- `vllm:time_to_first_token_seconds`: `0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf`
- `vllm:inter_token_latency_seconds`: `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf`
- `vllm:request_time_per_output_token_seconds`: `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf`

#### Request Parameters

These histograms show the distribution of request parameters processed by vLLM.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:request_prompt_tokens` | histogram | tokens | `model_name`, `engine` | Number of prefill tokens processed per request. Bucket maximum is derived from the configured model length. |
| `vllm:request_generation_tokens` | histogram | tokens | `model_name`, `engine` | Number of generation tokens processed per request. Bucket maximum is derived from the configured model length. |
| `vllm:request_max_num_generation_tokens` | histogram | tokens | `model_name`, `engine` | Histogram of maximum number of requested generation tokens. |
| `vllm:request_params_max_tokens` | histogram | tokens | `model_name`, `engine` | Histogram of the `max_tokens` request parameter. |
| `vllm:request_params_n` | histogram | — | `model_name`, `engine` | Histogram of the `n` request parameter. |
| `vllm:iteration_tokens_total` | histogram | tokens | `model_name`, `engine` | Histogram of number of tokens per engine step. |
| `vllm:request_prefill_kv_computed_tokens` | histogram | tokens | `model_name`, `engine` | Histogram of new KV tokens computed during prefill, excluding cached tokens. |

**Histogram buckets:**
- `vllm:request_prompt_tokens`: `1, 2, 5, 10, 20, 50, ... up to max_model_len, +Inf`
- `vllm:request_generation_tokens`: `1, 2, 5, 10, 20, 50, ... up to max_model_len, +Inf`
- `vllm:request_max_num_generation_tokens`: `1, 2, 5, 10, 20, 50, ... up to max_model_len, +Inf`
- `vllm:request_params_max_tokens`: `1, 2, 5, 10, 20, 50, ... up to max_model_len, +Inf`
- `vllm:request_params_n`: `1, 2, 5, 10, 20, +Inf`
- `vllm:iteration_tokens_total`: `1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, +Inf`
- `vllm:request_prefill_kv_computed_tokens`: `1, 2, 5, 10, 20, 50, ... up to max_model_len, +Inf`

#### Speculative Decoding

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:spec_decode_num_drafts` | counter | drafts | `model_name`, `engine` | Number of spec decoding drafts. |
| `vllm:spec_decode_num_draft_tokens` | counter | tokens | `model_name`, `engine` | Number of draft tokens. |
| `vllm:spec_decode_num_accepted_tokens` | counter | tokens | `model_name`, `engine` | Number of accepted tokens. |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | counter | tokens | `model_name`, `engine`, `position` | Accepted tokens per draft position. |

#### Optional KV and Performance Metrics

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:kv_block_lifetime_seconds` | histogram | seconds | `model_name`, `engine` | KV cache block lifetime from allocation to eviction. Only emitted when KV cache metrics are enabled. |
| `vllm:kv_block_idle_before_evict_seconds` | histogram | seconds | `model_name`, `engine` | Idle time before KV cache block eviction. Only emitted when KV cache metrics are enabled. |
| `vllm:kv_block_reuse_gap_seconds` | histogram | seconds | `model_name`, `engine` | Time gaps between consecutive KV cache block accesses. Only emitted when KV cache metrics are enabled. |
| `vllm:kv_offload_size` | histogram | bytes | `model_name`, `engine`, `transfer_type` | KV offload transfer size, in bytes. |
| `vllm:kv_offload_total_bytes` | counter | bytes | `model_name`, `engine`, `transfer_type` | Number of bytes offloaded by KV connector. |
| `vllm:kv_offload_total_time` | counter | seconds | `model_name`, `engine`, `transfer_type` | Total time measured by all KV offloading operations. |
| `vllm:estimated_flops_per_gpu_total` | counter | operations | `model_name`, `engine` | Estimated number of floating point operations per GPU for Model Flops Utilization calculations. Available via `--enable-mfu-metrics`. |
| `vllm:estimated_read_bytes_per_gpu_total` | counter | bytes | `model_name`, `engine` | Estimated number of bytes read from memory per GPU for Model Flops Utilization calculations. Available via `--enable-mfu-metrics`. |
| `vllm:estimated_write_bytes_per_gpu_total` | counter | bytes | `model_name`, `engine` | Estimated number of bytes written to memory per GPU for Model Flops Utilization calculations. Available via `--enable-mfu-metrics`. |

**Histogram buckets:**
- `vllm:kv_block_lifetime_seconds`: `0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, +Inf`
- `vllm:kv_block_idle_before_evict_seconds`: same as above
- `vllm:kv_block_reuse_gap_seconds`: same as above
- `vllm:kv_offload_size`: `1000000, 5000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 150000000, 200000000, +Inf`

#### Configuration Info

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:cache_config_info` | gauge | `engine`, cache config labels such as `block_size`, `cache_dtype`, `enable_prefix_caching`, `gpu_memory_utilization`, `num_gpu_blocks`, etc. | Static cache configuration. Exposed as a gauge with value 1.0. |
| `vllm:lora_requests_info` | gauge | `max_lora`, `waiting_lora_adapters`, `running_lora_adapters` | Running stats on LoRA requests. Only emitted when LoRA is configured. |

**Common cache config labels:**
- `block_size`: KV cache block size in tokens (e.g., `16`)
- `cache_dtype`: Cache data type (e.g., `auto`)
- `enable_prefix_caching`: Whether prefix caching is enabled (`True`/`False`)
- `gpu_memory_utilization`: GPU memory utilization target (e.g., `0.9`)
- `num_gpu_blocks`: Total GPU blocks allocated (e.g., `71671`)

---

### SGLang

SGLang is a fast inference engine with RadixAttention for efficient prefix caching. These metrics provide visibility into SGLang's scheduling, execution, token accounting, disaggregated inference, speculative decoding, and optional cache features.

Unless noted otherwise, scheduler metrics use labels `model_name`, `engine_type`, `tp_rank`, `pp_rank`, and `moe_ep_rank`. `dp_rank` is added when data parallel rank is present, `priority` is added when priority scheduling is enabled, and user-configured `extra_metric_labels` may add more labels.

#### Throughput, Tokens & Requests

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:gen_throughput` | gauge | tokens/s | scheduler labels | Generation throughput in tokens per second. |
| `sglang:realtime_tokens` | counter | tokens | scheduler labels + `mode` | Tokens processed on each log interval. `mode`: `prefill_compute`, `prefill_cache`, `decode`. |
| `sglang:dp_cooperation_realtime_tokens` | counter | tokens | scheduler labels + `mode`, `num_prefill_ranks` | Token counts with DP cooperation labels. |
| `sglang:prompt_tokens` | counter | tokens | `model_name`, `engine_type` | Number of prefill tokens processed. |
| `sglang:generation_tokens` | counter | tokens | `model_name`, `engine_type` | Number of generation tokens processed. |
| `sglang:cached_tokens` | counter | tokens | `model_name`, `engine_type`, `cache_source` | Cached prompt tokens split by source. `cache_source` values include `device`, `host`, `storage_<backend>`, and `total`. |
| `sglang:prompt_tokens_histogram` | histogram | tokens | `model_name`, `engine_type` | Prompt token length distribution. Buckets can be overridden by server args. |
| `sglang:uncached_prompt_tokens_histogram` | histogram | tokens | `model_name`, `engine_type` | Uncached prompt token length distribution. |
| `sglang:generation_tokens_histogram` | histogram | tokens | `model_name`, `engine_type` | Generation token length distribution. Buckets can be overridden by server args. |
| `sglang:num_requests` | counter | requests | `model_name`, `engine_type` | Number of requests processed. |
| `sglang:num_aborted_requests` | counter | requests | `model_name`, `engine_type` | Number of requests aborted. |
| `sglang:num_so_requests` | counter | requests | `model_name`, `engine_type` | Number of structured-output requests processed. |
| `sglang:get_loads_duration_seconds` | histogram | seconds | `model_name`, `engine_type` | Time spent serving `/v1/loads` requests. |

**Histogram buckets:**
- `sglang:prompt_tokens_histogram`: `100, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 35000, 40000, 60000, 80000, 100000, 200000, 300000, 400000, 600000, 800000, 1000000, 1100000, +Inf`
- `sglang:uncached_prompt_tokens_histogram`: Same as `sglang:prompt_tokens_histogram`
- `sglang:generation_tokens_histogram`: Same as `sglang:prompt_tokens_histogram` by default
- `sglang:get_loads_duration_seconds`: `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, +Inf`

#### Queue, Cache & Memory State

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:num_running_reqs` | gauge | requests | scheduler labels | Requests currently executing in the batch. With priority scheduling, totals use `priority=""` and per-priority series use `priority="<int>"`. |
| `sglang:num_queue_reqs` | gauge | requests | scheduler labels | Requests in the waiting queue. High values indicate saturation. |
| `sglang:num_grammar_queue_reqs` | gauge | requests | scheduler labels | Requests waiting for grammar processing. |
| `sglang:num_used_tokens` | gauge | tokens | scheduler labels | Number of used tokens; for hybrid-SWA models this is the max of full-attention and SWA pools, and it does not include the Mamba pool. |
| `sglang:decode_sum_seq_lens` | gauge | tokens | scheduler labels | Sum of all sequence lengths in decode. |
| `sglang:cache_hit_rate` | gauge | ratio | scheduler labels | Prefix cache hit rate. Higher = better prompt reuse via RadixAttention. |
| `sglang:token_usage` | gauge | ratio | scheduler labels | Bottleneck token usage ratio across full, SWA, and Mamba pools. |
| `sglang:full_token_usage` | gauge | ratio | scheduler labels | Full-attention KV cache pool usage ratio. |
| `sglang:swa_token_usage` | gauge | ratio | scheduler labels | Sliding-window attention token pool usage ratio. |
| `sglang:mamba_usage` | gauge | ratio | scheduler labels | Mamba SSM state pool usage ratio. |
| `sglang:kv_available_tokens` | gauge | tokens | scheduler labels | Free token slots in the KV cache pool. |
| `sglang:kv_evictable_tokens` | gauge | tokens | scheduler labels | Evictable radix-cached token slots in the KV cache pool. |
| `sglang:kv_used_tokens` | gauge | tokens | scheduler labels | Actively used token slots in the KV cache pool. |
| `sglang:swa_available_tokens` | gauge | tokens | scheduler labels | Free token slots in the SWA pool. |
| `sglang:swa_evictable_tokens` | gauge | tokens | scheduler labels | Evictable radix-cached token slots in the SWA pool. |
| `sglang:swa_used_tokens` | gauge | tokens | scheduler labels | Actively used token slots in the SWA pool. |
| `sglang:mamba_available_tokens` | gauge | tokens | scheduler labels | Free state slots in the Mamba SSM pool. |
| `sglang:mamba_evictable_tokens` | gauge | tokens | scheduler labels | Evictable radix-cached state slots in the Mamba SSM pool. |
| `sglang:mamba_used_tokens` | gauge | tokens | scheduler labels | Actively used state slots in the Mamba SSM pool. |
| `sglang:num_retracted_reqs` | gauge | requests | scheduler labels | Current number of retracted requests. |
| `sglang:num_retracted_requests` | counter | requests | scheduler labels | Total retracted requests. |
| `sglang:num_retracted_input_tokens` | counter | tokens | scheduler labels | Total retracted input tokens. |
| `sglang:num_retracted_output_tokens` | counter | tokens | scheduler labels | Total retracted output tokens. |
| `sglang:num_paused_reqs` | gauge | requests | scheduler labels | Requests paused by async weight sync. |

#### Request Latency Breakdown

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:time_to_first_token_seconds` | histogram | seconds | `model_name`, `engine_type` | Time to first token. Buckets can be overridden by server args. |
| `sglang:inter_token_latency_seconds` | histogram | seconds | `model_name`, `engine_type` | Inter-token latency. Buckets can be overridden by server args. |
| `sglang:e2e_request_latency_seconds` | histogram | seconds | `model_name`, `engine_type` | End-to-end request latency. Buckets can be overridden by server args. |
| `sglang:queue_time_seconds` | histogram | seconds | scheduler labels | Time spent in the waiting queue before execution starts. |
| `sglang:per_stage_req_latency_seconds` | histogram | seconds | scheduler labels + `stage` | Per-stage latency breakdown. `stage` label identifies the phase. |

**Histogram buckets:**
- `sglang:time_to_first_token_seconds`: `0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, +Inf`
- `sglang:inter_token_latency_seconds`: `0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000, +Inf`
- `sglang:e2e_request_latency_seconds`: `0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 1200, 1800, 2400, +Inf`
- `sglang:queue_time_seconds`: `0.0, 0.001, 0.005, 0.010, 0.050, 0.100, 0.200, 0.500, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, +Inf`
- `sglang:per_stage_req_latency_seconds`: *(see below)*

**Histogram buckets for `sglang:per_stage_req_latency_seconds`:**
```
0.001, 0.0016, 0.0026, 0.0043, 0.0069, 0.0112, 0.0181, 0.0293, 0.0474, 0.0768, 0.1245, 0.2017, 0.3267, 0.5293, 0.8575, 1.3891, 2.2503, 3.6455, 5.9057, 9.5672, 15.4989, 25.1082, 40.6753, 65.8939, 106.7481, 172.9319, 280.1497, 453.8426, 735.2250, 1191.0646, +Inf
```

**Observed stage labels for `sglang:per_stage_req_latency_seconds`:**

| Stage | Description |
|-------|-------------|
| `request_process` | Unified-mode request processing before queue entry |
| `prefill_bootstrap` | Prefill bootstrap queue time in disaggregated prefill mode |
| `prefill_forward` | Time executing prefill forward pass |
| `chunked_prefill` | Time executing a chunked-prefill slice |
| `prefill_transfer_kv_cache` | Time transferring KV cache from prefill to decode worker |
| `decode_prepare` | Decode preallocation preparation time |
| `decode_bootstrap` | Decode bootstrap/transfer setup time |
| `decode_waiting` | Time waiting before decode forward execution |
| `decode_transferred` | Decode-side transferred request processing before queue entry |
| `fake_output` | Fake-output/prebuilt decode stage |

#### Disaggregated Inference Queues and KV Transfer

For disaggregated prefill/decode deployments where prefill and decode run on separate instances.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:num_prefill_bootstrap_queue_reqs` | gauge | requests | scheduler labels | Requests in the prefill bootstrap queue. |
| `sglang:num_prefill_inflight_queue_reqs` | gauge | requests | scheduler labels | Requests in the prefill inflight queue. |
| `sglang:num_decode_prealloc_queue_reqs` | gauge | requests | scheduler labels | Requests in the decode preallocation queue. |
| `sglang:num_decode_transfer_queue_reqs` | gauge | requests | scheduler labels | Requests in the decode transfer queue. |
| `sglang:pending_prealloc_token_usage` | gauge | ratio | scheduler labels | Token usage for pending preallocated tokens. |
| `sglang:kv_transfer_latency_ms` | histogram | milliseconds | scheduler labels | KV cache transfer latency. |
| `sglang:kv_transfer_speed_gb_s` | histogram | GB/s | scheduler labels | KV cache transfer throughput. |
| `sglang:kv_transfer_total_mb` | histogram | megabytes | scheduler labels | KV cache transfer size. |
| `sglang:kv_transfer_alloc_ms` | histogram | milliseconds | scheduler labels | Time waiting for KV cache allocation. |
| `sglang:kv_transfer_bootstrap_ms` | histogram | milliseconds | scheduler labels | KV transfer bootstrap time. |
| `sglang:num_bootstrap_failed_reqs` | counter | requests | scheduler labels | Number of bootstrap-failed requests. |
| `sglang:num_transfer_failed_reqs` | counter | requests | scheduler labels | Number of transfer-failed requests. |
| `sglang:num_prefill_retries` | counter | requests | scheduler labels | Total number of prefill retries. |

**Histogram buckets:**
- `sglang:kv_transfer_latency_ms`: `1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, +Inf`
- `sglang:kv_transfer_speed_gb_s`: `0.1, 0.5, 1, 5, 10, 25, 50, 100, 200, 400, +Inf`
- `sglang:kv_transfer_total_mb`: `1, 5, 10, 50, 100, 500, 1000, 5000, 10000, +Inf`
- `sglang:kv_transfer_alloc_ms`: `1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, +Inf`
- `sglang:kv_transfer_bootstrap_ms`: `1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, +Inf`

#### Speculative Decoding

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:spec_accept_rate` | gauge | ratio | scheduler labels | Speculative acceptance rate (`accepted drafts / proposed drafts` in batch). |
| `sglang:spec_accept_length` | gauge | tokens | scheduler labels | Mean acceptance length of speculative decoding (accepted drafts plus bonus token per forward). |
| `sglang:spec_verify_calls` | counter | calls | `model_name`, `engine_type` | Number of speculative decoding verification calls. |

#### Execution, CUDA Graph, and Estimated Performance

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:utilization` | gauge | ratio | scheduler labels | Scheduler utilization. |
| `sglang:fwd_occupancy` | gauge | percent | scheduler labels | Forward-pass GPU occupancy percentage. |
| `sglang:new_token_ratio` | gauge | ratio | scheduler labels | New-token ratio from the scheduler policy. |
| `sglang:is_cuda_graph` | gauge | — | scheduler labels | Whether the batch is using CUDA graph (1=yes, 0=no). |
| `sglang:cuda_graph_passes` | counter | passes | scheduler labels + `mode` | Forward passes categorized by graph use. `mode`: `decode_cuda_graph`, `decode_none`, `prefill_cuda_graph`, `prefill_none`. |
| `sglang:num_unique_running_routing_keys` | gauge | keys | scheduler labels | Unique routing keys present in the running batch. |
| `sglang:routing_key_running_req_count` | histogram | requests | scheduler labels | Distribution of routing keys by running request count. |
| `sglang:routing_key_all_req_count` | histogram | requests | scheduler labels | Distribution of routing keys by running plus waiting request count. |
| `sglang:forward_execution_seconds` | counter | seconds | scheduler labels + `category` | Total GPU-busy time executing model forward passes. |
| `sglang:dp_cooperation_forward_execution_seconds` | counter | seconds | scheduler labels + `category`, `num_prefill_ranks` | Forward execution time with DP cooperation labels. |
| `sglang:estimated_flops_per_gpu` | counter | FLOPs | scheduler labels | Estimated floating-point operations per GPU; requires `--enable-mfu-metrics`. |
| `sglang:estimated_read_bytes_per_gpu` | counter | bytes | scheduler labels | Estimated bytes read from memory per GPU; requires `--enable-mfu-metrics`. |
| `sglang:estimated_write_bytes_per_gpu` | counter | bytes | scheduler labels | Estimated bytes written to memory per GPU; requires `--enable-mfu-metrics`. |

#### Optional Feature Metrics

These metric families are emitted only when the corresponding feature is enabled.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:lora_pool_slots_used` | gauge | slots | scheduler labels | LoRA adapter slots currently occupied in GPU memory. |
| `sglang:lora_pool_slots_total` | gauge | slots | scheduler labels | Total LoRA adapter slots available. |
| `sglang:lora_pool_utilization` | gauge | ratio | scheduler labels | LoRA pool utilization ratio. |
| `sglang:hicache_host_used_tokens` | gauge | tokens | scheduler labels | Tokens currently used in the host KV cache. |
| `sglang:hicache_host_total_tokens` | gauge | tokens | scheduler labels | Total host KV-cache capacity in tokens. |
| `sglang:num_streaming_sessions` | gauge | sessions | scheduler labels | Number of streaming sessions. |
| `sglang:streaming_session_held_tokens` | gauge | tokens | scheduler labels | KV tokens held by streaming session slots. |
| `sglang:grammar_compilation_time_seconds` | histogram | seconds | scheduler labels | Grammar compilation time for structured-output requests. |
| `sglang:num_grammar_cache_hit` | counter | requests | scheduler labels | Grammar cache hits. |
| `sglang:num_grammar_aborted` | counter | requests | scheduler labels | Grammar-aborted requests. |
| `sglang:num_grammar_timeout` | counter | requests | scheduler labels | Grammar timeouts. |
| `sglang:num_grammar_total` | counter | requests | scheduler labels | Total grammar requests. |
| `sglang:grammar_schema_count` | histogram | schemas | scheduler labels | Number of grammar schemas. |
| `sglang:grammar_ebnf_size` | histogram | bytes | scheduler labels | Grammar EBNF size. |
| `sglang:grammar_tree_traversal_time_avg` | histogram | seconds | scheduler labels | Average grammar tree traversal time. |
| `sglang:grammar_tree_traversal_time_max` | histogram | seconds | scheduler labels | Maximum grammar tree traversal time. |
| `sglang:prefill_delayer_wait_forward_passes` | histogram | passes | scheduler labels | Forward passes spent waiting in the prefill delayer. |
| `sglang:prefill_delayer_wait_seconds` | histogram | seconds | scheduler labels | Time spent waiting in the prefill delayer. |
| `sglang:prefill_delayer_outcomes` | counter | outcomes | scheduler labels + `input_estimation`, `output_allow`, `output_reason`, `actual_execution` | Prefill-delayer scheduling outcomes. |
| `sglang:eplb_gpu_physical_count` | histogram | GPUs | scheduler labels + `layer` | Physical GPU count distribution for expert-parallel load balancing. |
| `sglang:prefetched_tokens` | counter | tokens | scheduler labels | Prompt tokens prefetched from storage. |
| `sglang:backuped_tokens` | counter | tokens | scheduler labels | Tokens backed up to storage. |
| `sglang:prefetch_pgs` | histogram | pages | scheduler labels | Prefetch pages per batch. |
| `sglang:backup_pgs` | histogram | pages | scheduler labels | Backup pages per batch. |
| `sglang:prefetch_bandwidth` | histogram | GB/s | scheduler labels | Prefetch bandwidth. |
| `sglang:backup_bandwidth` | histogram | GB/s | scheduler labels | Backup bandwidth. |
| `sglang:eviction_duration_seconds` | histogram | seconds | scheduler labels | Time to evict memory from GPU to CPU. |
| `sglang:evicted_tokens` | counter | tokens | scheduler labels | Tokens evicted from GPU to CPU. |
| `sglang:load_back_duration_seconds` | histogram | seconds | scheduler labels | Time to load memory back from CPU to GPU. |
| `sglang:load_back_tokens` | counter | tokens | scheduler labels | Tokens loaded back from CPU to GPU. |

#### System Configuration

These are constant gauges emitted once at startup.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:max_total_num_tokens` | gauge | tokens | scheduler labels | Maximum total tokens in the KV cache pool. |
| `sglang:max_running_requests_under_SLO` | gauge | requests | scheduler labels | Maximum running requests under SLO, when configured. |
| `sglang:engine_startup_time` | gauge | seconds | scheduler labels | Engine startup time. |
| `sglang:engine_load_weights_time` | gauge | seconds | scheduler labels | Time to load model weights. |
| `sglang:page_size` | gauge | tokens | scheduler labels | KV cache page size in tokens. |
| `sglang:num_pages` | gauge | pages | scheduler labels | Number of KV cache pages. |
| `sglang:context_len` | gauge | tokens | scheduler labels | Maximum context length. |
| `sglang:startup_available_gpu_memory_gb` | gauge | GB | scheduler labels | Available GPU memory at startup. |

**Common label values:**
- `engine_type`: `unified`, `prefill`, or `decode`
- `model_name`: Model identifier (e.g., `Qwen/Qwen3-0.6B`)
- `tp_rank`: Tensor parallel rank (e.g., `0`, `1`, ...)
- `pp_rank`: Pipeline parallel rank (e.g., `0`, `1`, ...)
- `moe_ep_rank`: MoE expert-parallel rank
- `dp_rank`: Data-parallel rank when present
- `priority`: empty string for totals, or a priority value for per-priority queue gauges

---

### TensorRT-LLM

TensorRT-LLM (trtllm) is NVIDIA's high-performance inference engine optimized for NVIDIA GPUs. These metrics cover request latency, token accounting, queue/load state, KV cache behavior, memory usage, and optional speculative decoding stats. Dynamo-TRTLLM does not rename the engine's native `trtllm_` metrics, but it can emit additional Python-side metrics with the same `trtllm_` prefix so they pass the same prefix filters.

> [!IMPORTANT]
> **TRT-LLM exposes Prometheus at a non-standard path.** By default `trtllm-serve` serves an iteration-stats JSON array at `/metrics` (not Prometheus exposition format). The metrics below are only available when the server is launched with `return_perf_metrics: true` in `extra_llm_api_options.yaml`, which mounts the proper Prometheus exposition at `/prometheus/metrics`. Iteration-derived metrics additionally require iteration stats to be enabled (`enable_iter_perf_stats: true` for the PyTorch backend; TensorRT backend iteration stats are enabled by default). AIPerf detects the JSON response on `/metrics`, probes the alt path automatically, and swaps the collector's URL on success — see [Compatibility & auto-disable](server-metrics.md#compatibility--auto-disable).

AIPerf records Prometheus family names as exposed by the server, with Prometheus counter samples grouped under the counter family name without the sample's trailing `_total` suffix. For example, upstream `trtllm_request_success_total` samples appear under `trtllm_request_success` in AIPerf outputs.

#### Request Latency

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm_e2e_request_latency_seconds` | histogram | seconds | `engine_type`, `model_name` | End-to-end request latency in seconds. |
| `trtllm_request_queue_time_seconds` | histogram | seconds | `engine_type`, `model_name` | Time spent in the waiting phase before scheduling. |
| `trtllm_time_to_first_token_seconds` | histogram | seconds | `engine_type`, `model_name` | Time to first token in seconds. |
| `trtllm_time_per_output_token_seconds` | histogram | seconds | `engine_type`, `model_name` | Time per output token in seconds. |
| `trtllm_request_prefill_time_seconds` | histogram | seconds | `engine_type`, `model_name` | Prefill/context phase duration (`first_token_time - first_scheduled_time`). |
| `trtllm_request_decode_time_seconds` | histogram | seconds | `engine_type`, `model_name` | Decode/generation phase duration (`last_token_time - first_token_time`). |
| `trtllm_request_inference_time_seconds` | histogram | seconds | `engine_type`, `model_name` | Total inference duration (`last_token_time - first_scheduled_time`). |

**Histogram buckets:**
- `trtllm_e2e_request_latency_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `trtllm_request_queue_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `trtllm_time_to_first_token_seconds`: `0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf`
- `trtllm_time_per_output_token_seconds`: `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf`
- `trtllm_request_prefill_time_seconds`: `0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf`
- `trtllm_request_decode_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`
- `trtllm_request_inference_time_seconds`: `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf`

#### Request Completion and Tokens

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm_request_success` | counter | requests | `engine_type`, `finished_reason`, `model_name` | Successfully completed requests. |
| `trtllm_prompt_tokens` | counter | tokens | `engine_type`, `model_name` | Cumulative number of prompt/input tokens processed. |
| `trtllm_generation_tokens` | counter | tokens | `engine_type`, `model_name` | Cumulative number of generation/output tokens produced. |

**Common label values:**
- `engine_type`: `pytorch`, `_autodeploy`, or `unknown` from the configured backend (not always `trtllm`).
- `model_name`: Model identifier (e.g., `Qwen/Qwen3-0.6B`).
- `finished_reason`: `stop`, `length`, `timeout`, or `cancelled`. Upstream code does not emit `error` as a `finished_reason` value for `trtllm_request_success`.

#### Queue, Batch, and Memory State

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm_num_requests_running` | gauge | requests | `engine_type`, `model_name` | Number of active requests. |
| `trtllm_num_requests_waiting` | gauge | requests | `engine_type`, `model_name` | Number of queued requests. |
| `trtllm_num_requests_completed` | counter | requests | `engine_type`, `model_name` | Total completed requests reported by iteration stats. |
| `trtllm_max_num_active_requests` | gauge | requests | `engine_type`, `model_name` | Maximum number of active requests. |
| `trtllm_iteration_latency_seconds` | gauge | seconds | `engine_type`, `model_name` | Iteration latency converted from milliseconds to seconds. |
| `trtllm_gpu_memory_usage_bytes` | gauge | bytes | `engine_type`, `model_name` | GPU memory usage in bytes. |
| `trtllm_cpu_memory_usage_bytes` | gauge | bytes | `engine_type`, `model_name` | CPU memory usage in bytes. |
| `trtllm_pinned_memory_usage_bytes` | gauge | bytes | `engine_type`, `model_name` | Pinned memory usage in bytes. |
| `trtllm_max_batch_size_static` | gauge | requests | `engine_type`, `model_name` | Static maximum batch size. |
| `trtllm_max_batch_size_runtime` | gauge | requests | `engine_type`, `model_name` | Runtime maximum batch size. |
| `trtllm_max_num_tokens_runtime` | gauge | tokens | `engine_type`, `model_name` | Runtime maximum number of tokens. |
| `trtllm_num_context_requests` | gauge | requests | `engine_type`, `model_name` | Number of context/prefill requests. |
| `trtllm_num_generation_requests` | gauge | requests | `engine_type`, `model_name` | Number of generation/decode requests. |
| `trtllm_num_paused_requests` | gauge | requests | `engine_type`, `model_name` | Number of paused requests. |
| `trtllm_num_scheduled_requests` | gauge | requests | `engine_type`, `model_name` | Number of scheduled requests. |
| `trtllm_total_context_tokens` | gauge | tokens | `engine_type`, `model_name` | Total context tokens in the current iteration stats. |
| `trtllm_avg_decoded_tokens_per_iter` | gauge | tokens | `engine_type`, `model_name` | Average decoded tokens per iteration. |

#### KV Cache Metrics

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm_kv_cache_hit_rate` | gauge | ratio | `engine_type`, `model_name` | KV cache hit rate. |
| `trtllm_kv_cache_utilization` | gauge | ratio | `engine_type`, `model_name` | Used KV cache blocks divided by max KV cache blocks. |
| `trtllm_kv_cache_host_utilization` | gauge | ratio | `engine_type`, `model_name` | Secondary/host KV cache utilization. |
| `trtllm_kv_cache_iter_reuse_rate` | gauge | ratio | `engine_type`, `model_name` | Per-iteration KV cache block reuse rate. |
| `trtllm_kv_cache_reused_blocks` | counter | blocks | `engine_type`, `model_name` | Cumulative reused KV cache blocks. |
| `trtllm_kv_cache_missed_blocks` | counter | blocks | `engine_type`, `model_name` | Cumulative missed KV cache blocks. |
| `trtllm_kv_cache_iter_reused_blocks` | counter | blocks | `engine_type`, `model_name` | Total reused KV cache blocks per iteration stats. |
| `trtllm_kv_cache_iter_full_reused_blocks` | counter | blocks | `engine_type`, `model_name` | Total fully reused KV cache blocks. |
| `trtllm_kv_cache_iter_partial_reused_blocks` | counter | blocks | `engine_type`, `model_name` | Total partially reused KV cache blocks. |
| `trtllm_kv_cache_iter_missed_blocks` | counter | blocks | `engine_type`, `model_name` | Total missed KV cache blocks in context phase. |
| `trtllm_kv_cache_gen_alloc_blocks` | counter | blocks | `engine_type`, `model_name` | Blocks allocated during generation phase. |
| `trtllm_kv_cache_onboard_bytes` | counter | bytes | `engine_type`, `model_name` | Bytes transferred from host to GPU. |
| `trtllm_kv_cache_offload_bytes` | counter | bytes | `engine_type`, `model_name` | Bytes transferred from GPU to host. |
| `trtllm_kv_cache_intra_device_copy_bytes` | counter | bytes | `engine_type`, `model_name` | Bytes copied within GPU. |
| `trtllm_kv_cache_max_blocks` | gauge | blocks | `engine_type`, `model_name` | Maximum number of KV cache blocks. |
| `trtllm_kv_cache_free_blocks` | gauge | blocks | `engine_type`, `model_name` | Number of free KV cache blocks. |
| `trtllm_kv_cache_used_blocks` | gauge | blocks | `engine_type`, `model_name` | Number of used KV cache blocks. |
| `trtllm_kv_cache_tokens_per_block` | gauge | tokens | `engine_type`, `model_name` | Number of tokens per KV cache block. |

#### Speculative Decoding and Config Info

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm_spec_decode_num_draft_tokens` | counter | tokens | `engine_type`, `model_name` | Total draft tokens in speculative decoding. |
| `trtllm_spec_decode_num_accepted_tokens` | counter | tokens | `engine_type`, `model_name` | Total accepted tokens in speculative decoding. |
| `trtllm_spec_decode_acceptance_length` | gauge | tokens | `engine_type`, `model_name` | Acceptance length in speculative decoding. |
| `trtllm_spec_decode_draft_overhead` | gauge | ratio | `engine_type`, `model_name` | Draft overhead in speculative decoding. |
| `trtllm_model_config_info` | gauge | — | `engine_type`, `model_name`, `model`, `served_model_name`, `dtype`, `quantization`, `max_model_len`, `gpu_type` | Static model configuration as labels, value `1`. |
| `trtllm_parallel_config_info` | gauge | — | `engine_type`, `model_name`, `tensor_parallel_size`, `pipeline_parallel_size`, `context_parallel_size`, `gpu_count`, `expert_parallel_size` | Static parallelism configuration as labels, value `1`. |
| `trtllm_speculative_config_info` | gauge | — | `engine_type`, `model_name`, `spec_enabled`, `spec_method`, `spec_num_tokens`, `spec_draft_model` | Static speculative-decoding configuration as labels, value `1`; emitted only when speculative config exists. |
| `trtllm_kv_cache_config_info` | gauge | — | `engine_type`, `model_name`, `page_size`, `enable_block_reuse`, `enable_partial_reuse`, `free_gpu_memory_fraction`, `cache_dtype` | Static KV cache configuration as labels, value `1`; emitted only when KV cache config exists. |

#### Dynamo-TRTLLM Additional Metrics

These are emitted by Dynamo's TRT-LLM worker integration in addition to the engine-native TensorRT-LLM metrics above. They intentionally use the `trtllm_` prefix.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm_num_aborted_requests` | counter | requests | Dynamo-TRTLLM labels such as `model_name`, `disaggregation_mode`, `engine_type` | Aborted or cancelled requests. |
| `trtllm_request_type_image` | counter | requests | Dynamo-TRTLLM labels | Requests containing image or multimodal content. |
| `trtllm_request_type_structured_output` | counter | requests | Dynamo-TRTLLM labels | Requests using guided or structured decoding. |
| `trtllm_kv_transfer_success` | counter | transfers | Dynamo-TRTLLM labels | Successful KV cache transfers. |
| `trtllm_kv_transfer_latency_seconds` | histogram | seconds | Dynamo-TRTLLM labels | KV cache transfer latency per request. |
| `trtllm_kv_transfer_bytes` | histogram | bytes | Dynamo-TRTLLM labels | KV cache transfer size per request. |
| `trtllm_kv_transfer_speed_gb_s` | histogram | GB/s | Dynamo-TRTLLM labels | KV cache transfer speed per request. |

---

### Triton Inference Server

Triton Inference Server exposes Prometheus text metrics on a dedicated metrics service, by default `http://localhost:8002/metrics`. The endpoint is enabled unless `tritonserver --allow-metrics=false` is set; `--allow-gpu-metrics=false` and `--allow-cpu-metrics=false` disable only those metric groups. Use `--metrics-port`, `--metrics-address`, and `--metrics-interval-ms` to change where interval metrics are served and how often they refresh.

#### Request Counts and Queue State

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `nv_inference_request_success` | counter | requests | `model`, `version` | Successful inference requests received by Triton. Each request counts as one, even when batched. |
| `nv_inference_request_failure` | counter | requests | `model`, `reason`, `version` | Failed inference requests. `reason` values include `REJECTED`, `CANCELED`, `BACKEND`, and `OTHER`. |
| `nv_inference_count` | counter | inferences | `model`, `version` | Inferences performed; a batch of `n` counts as `n` inferences and cached requests are excluded. |
| `nv_inference_exec_count` | counter | executions | `model`, `version` | Backend batch executions. `nv_inference_count / nv_inference_exec_count` approximates average batch size. |
| `nv_inference_pending_request_count` | gauge | requests | `model`, `version` | Requests received by Triton core but not yet executing in a backend. Use as Triton's queue-depth signal. |

#### Latency Counters and Optional Histograms

By default, Triton exposes cumulative latency counters in microseconds. AIPerf reports `stats.total` for the benchmark-window increase and `stats.rate` as microseconds accumulated per second. Optional histogram and summary latency families are controlled with `--metrics-config`; AIPerf exports histograms but skips Prometheus summary metrics. Model-level metrics use `model` and `version` labels, and can also include `model_namespace`, model tag labels prefixed with `_`, and `gpu_uuid` when configured by Triton.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `nv_inference_request_duration_us` | counter | microseconds | `model`, `version` | Cumulative end-to-end request handling time, including cached requests. |
| `nv_inference_queue_duration_us` | counter | microseconds | `model`, `version` | Cumulative time requests spent waiting in Triton's scheduling queue. |
| `nv_inference_compute_input_duration_us` | counter | microseconds | `model`, `version` | Cumulative backend input-processing time, excluding cached requests. |
| `nv_inference_compute_infer_duration_us` | counter | microseconds | `model`, `version` | Cumulative backend model execution time, excluding cached requests. |
| `nv_inference_compute_output_duration_us` | counter | microseconds | `model`, `version` | Cumulative backend output-processing time, excluding cached requests. |
| `nv_inference_first_response_histogram_ms` | histogram | milliseconds | `model`, `version` | Optional first-response latency histogram. Enable with `--metrics-config histogram_latencies=true`; default buckets are `100, 500, 2000, 5000, +Inf` unless overridden per model. |

#### GPU, CPU, Pinned Memory, and Response Cache

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `nv_gpu_power_usage` | gauge | watts | `gpu_uuid` | Instantaneous GPU power. |
| `nv_gpu_power_limit` | gauge | watts | `gpu_uuid` | GPU power limit. |
| `nv_energy_consumption` | counter | joules | `gpu_uuid` | GPU energy consumption since Triton started. |
| `nv_gpu_utilization` | gauge | ratio | `gpu_uuid` | GPU utilization from 0.0 to 1.0. |
| `nv_gpu_memory_total_bytes` | gauge | bytes | `gpu_uuid` | Total GPU memory. |
| `nv_gpu_memory_used_bytes` | gauge | bytes | `gpu_uuid` | Used GPU memory. |
| `nv_cpu_utilization` | gauge | ratio | — | Total CPU utilization from 0.0 to 1.0. Linux only. |
| `nv_cpu_memory_total_bytes` | gauge | bytes | — | Total system memory. Linux only. |
| `nv_cpu_memory_used_bytes` | gauge | bytes | — | Used system memory. Linux only. |
| `nv_pinned_memory_pool_total_bytes` | gauge | bytes | — | Total pinned-memory pool capacity. |
| `nv_pinned_memory_pool_used_bytes` | gauge | bytes | — | Used pinned-memory pool. |

Response-cache metrics are emitted only when Triton's response cache is enabled.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `nv_cache_num_hits_per_model` | counter | requests | `model`, `version` | Response-cache hits per model. |
| `nv_cache_num_misses_per_model` | counter | requests | `model`, `version` | Response-cache misses per model. |
| `nv_cache_hit_duration_per_model` | counter | microseconds | `model`, `version` | Cumulative cache-hit lookup duration. |
| `nv_cache_miss_duration_per_model` | counter | microseconds | `model`, `version` | Cumulative cache-miss lookup/insert duration. |

#### TensorRT-LLM Triton Backend Custom Metrics

When TensorRT-LLM runs as a Triton backend, the backend can expose additional custom families using the `nv_trt_llm_*` and `nv_llm_*` prefixes.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `nv_trt_llm_request_metrics` | gauge | requests | `model`, `version`, `request_type` | TensorRT-LLM backend request counts by request type. |
| `nv_trt_llm_runtime_memory_metrics` | gauge | bytes | `model`, `version`, `memory_type` | Runtime memory usage by memory type. |
| `nv_trt_llm_kv_cache_block_metrics` | gauge | blocks | `model`, `version`, `kv_cache_block_type` | KV-cache block counts by block type. |
| `nv_trt_llm_disaggregated_serving_metrics` | gauge | — | `model`, `version`, `disaggregated_serving_type` | Disaggregated-serving state and transfer metrics. |
| `nv_trt_llm_v1_metrics` | gauge | — | `model`, `version`, metric-specific labels | TensorRT-LLM v1 backend metrics. |
| `nv_trt_llm_inflight_batcher_metrics` | gauge | — | `model`, `version`, metric-specific labels | TensorRT-LLM inflight-batcher backend metrics. |
| `nv_trt_llm_general_metrics` | gauge | — | `model`, `version`, metric-specific labels | General TensorRT-LLM backend metrics. |
| `nv_llm_output_token_len` | histogram | tokens | `model`, `version` | Output-token length distribution. |
| `nv_llm_input_token_len` | histogram | tokens | `model`, `version` | Input-token length distribution. |

---

### KVBM (KV Block Manager)

**Note:** These metrics are only available with Dynamo deployments using the KV Block Manager feature for advanced KV cache management.

#### Block Transfer Operations

All metrics are counters tracking cumulative block movement operations.

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `kvbm_matched_tokens` | counter | tokens | The number of matched tokens (prefix cache hits). |
| `kvbm_host_cache_hit_rate` | gauge | ratio | Host cache hit rate from the sliding window. |
| `kvbm_disk_cache_hit_rate` | gauge | ratio | Disk cache hit rate from the sliding window. |
| `kvbm_object_cache_hit_rate` | gauge | ratio | Object-storage cache hit rate from the sliding window. |
| `kvbm_offload_blocks_d2d` | counter | blocks | The number of offload blocks from device to disk (bypassing host memory). |
| `kvbm_offload_blocks_d2h` | counter | blocks | The number of offload blocks from device to host memory. |
| `kvbm_offload_blocks_h2d` | counter | blocks | The number of offload blocks from host memory to disk. |
| `kvbm_offload_blocks_d2o` | counter | blocks | The number of blocks offloaded from device to object storage. |
| `kvbm_onboard_blocks_d2d` | counter | blocks | The number of onboard blocks from disk to device (bypassing host memory). |
| `kvbm_onboard_blocks_h2d` | counter | blocks | The number of onboard blocks from host memory to device. |
| `kvbm_onboard_blocks_o2d` | counter | blocks | The number of blocks onboarded from object storage to device. |
| `kvbm_object_read_failures` | counter | blocks | Failed object-storage read operations. |
| `kvbm_object_write_failures` | counter | blocks | Failed object-storage write operations. |

**Block transfer patterns:**
- **d2d**: Device ↔ Disk (direct, fast path)
- **d2h**: Device → Host (offload to CPU memory)
- **h2d**: Host → Device (onboard from CPU memory) or Host → Disk for offload persistence
- **d2o**: Device → Object storage
- **o2d**: Object storage → Device

#### Logical Pool Metrics

Dynamo's logical KVBM pool collector also exports pool-scoped counters and gauges. These carry a `pool` label and may include external deployment labels such as `instance_id`.

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `kvbm_allocations_total` | counter | allocations | Blocks allocated from logical pools. |
| `kvbm_allocations_from_reset_total` | counter | allocations | Blocks allocated from the reset pool. |
| `kvbm_evictions_total` | counter | evictions | Blocks evicted from the inactive pool. |
| `kvbm_registrations_total` | counter | registrations | CompleteBlock to ImmutableBlock registrations. |
| `kvbm_duplicate_blocks_total` | counter | blocks | Duplicate blocks created by the allow-duplicates policy. |
| `kvbm_registration_dedup_total` | counter | registrations | Registrations deduplicated by the reject-duplicates policy. |
| `kvbm_stagings_total` | counter | stagings | MutableBlock to CompleteBlock transitions. |
| `kvbm_match_hashes_requested_total` | counter | hashes | Hashes requested in `match_blocks`. |
| `kvbm_match_blocks_returned_total` | counter | blocks | Blocks returned from `match_blocks`. |
| `kvbm_scan_hashes_requested_total` | counter | hashes | Hashes requested in `scan_matches`. |
| `kvbm_scan_blocks_returned_total` | counter | blocks | Blocks returned from `scan_matches`. |
| `kvbm_eager_primary_to_inactive_total` | counter | transitions | Lookup-driven Primary-to-Inactive race-window transitions. |
| `kvbm_allocate_atomic_rollback_total` | counter | rollbacks | Allocation rollbacks after inactive backend under-allocation. |
| `kvbm_release_primary_noop_total` | counter | releases | Primary drop no-ops after concurrent transition or resurrection. |
| `kvbm_release_duplicate_noop_total` | counter | releases | Duplicate drop no-ops due to slot identity mismatch. |
| `kvbm_inflight_mutable` | gauge | blocks | Mutable blocks currently held outside the pool. |
| `kvbm_inflight_immutable` | gauge | blocks | Immutable blocks currently held outside the pool. |
| `kvbm_reset_pool_size` | gauge | blocks | Current reset-pool size. |
| `kvbm_inactive_pool_size` | gauge | blocks | Current inactive-pool size. |

---

## Appendix

### Common Metric Labels

Labels that appear across multiple metrics:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `model` | Model identifier (Dynamo/Triton) | `qwen/qwen3-0.6b` |
| `model_namespace` | Triton model namespace | namespace configured in Triton |
| `_custom_tag` | Triton model tag label | tag labels are prefixed with `_` |
| `gpu_uuid` | Triton GPU UUID | GPU UUID string |
| `model_name` | Model identifier (backends) | `Qwen/Qwen3-0.6B` |
| `endpoint` | API endpoint | `chat_completions`, `completions` |
| `request_type` | Request type | `stream`, `unary` |
| `status` | Request outcome | `success`, `error` |
| `engine` | Engine identifier (vLLM) | `0`, `1`, ... |
| `engine_type` | Engine type | `pytorch`, `_autodeploy`, `unified`, `prefill`, `decode` |
| `tp_rank` | Tensor parallel rank | `0`, `1`, ... |
| `pp_rank` | Pipeline parallel rank | `0`, `1`, ... |
| `moe_ep_rank` | SGLang MoE expert-parallel rank | `0`, `1`, ... |
| `dp_rank` | Data-parallel rank | `0`, `1`, ... |
| `priority` | SGLang priority scheduling value | empty string, `0`, `1`, ... |
| `stage` | Processing stage (SGLang) | `prefill_forward`, `decode_transferred` |
| `finished_reason` | Completion reason | `stop`, `length`, `abort`, `error`, `repetition`, `timeout`, `cancelled` |
| `version` | Triton model version | `1`, ... |
| `reason` | vLLM waiting reason or Triton failure reason | `capacity`, `deferred`, `REJECTED`, `CANCELED`, `BACKEND`, `OTHER` |
| `source` | vLLM prompt-token source | `local_compute`, `local_cache_hit`, `external_kv_transfer` |
| `sleep_state` | vLLM engine sleep state | `awake`, `weights_offloaded`, `discard_all` |
| `position` | Speculative-decoding draft position | `0`, `1`, ... |
| `transfer_type` | KV offload transfer type | Backend-specific transfer type |
| `cache_source` | SGLang cache source | `device`, `host`, `storage_<backend>`, `total` |
| `forward_mode` | SGLang forward mode | Backend-specific forward mode |
| `layer` | SGLang model layer | `0`, `1`, ... |
| `dynamo_component` | Component identifier | Worker name/ID |
| `dynamo_endpoint` | Internal endpoint | Internal routing info |
| `dynamo_namespace` | Namespace | Deployment namespace |
| `worker_id` | Dynamo worker identifier | Worker ID |
| `worker_type` | Dynamo worker type | `prefill`, `decode` |
| `router_id` | Dynamo router identifier | Router ID |
| `operation` | Dynamo operation name | `tokenize`, `detokenize` |
| `migration_type` | Dynamo request migration type | `new_request`, `ongoing_request` |
| `event_type` | Dynamo KV publisher event type | Event kind |
| `worker` | Tokio worker index | `0`, `1`, ... |
| `pool` | Dynamo KVBM logical pool name | Pool identifier |
| `instance_id` | Dynamo KVBM external instance label | Deployment instance ID |
| `error_type` | Error classification | Error category |
| `service_name` | NATS service name | Service identifier |

### Notes on Metric Usage

1. **Dynamo vs backend metrics**: Dynamo metrics measure at the HTTP/routing layer (user-facing), while vLLM/SGLang/TensorRT-LLM metrics measure inside the inference engine. Triton metrics measure Triton core/backend scheduling plus system telemetry. Use Dynamo for user-facing SLAs, backend/Triton metrics for debugging performance.

2. **Counter vs Gauge interpretation**:
   - **Counters**: Use `stats.total` for total change during benchmark, `stats.rate` for rate of change (per second)
   - **Gauges**: Use `stats.avg` for typical value, `stats.max` for peak, `stats.p99` for tail behavior

3. **Histogram percentiles**: Histogram percentiles (`stats.p50_estimate`, `stats.p90_estimate`, `stats.p95_estimate`, `stats.p99_estimate`) are *estimated* from bucket boundaries. Exact values depend on bucket configuration.

4. **Multiple endpoints**: When scraping multiple instances, each series includes an `endpoint_url` label to identify the source.

5. **Backend-specific capabilities**:
   - **vLLM**: Most comprehensive metrics including full request phase breakdown, cache statistics, and batch efficiency
   - **SGLang**: RadixAttention cache metrics, disaggregated inference support, speculative decoding stats, per-stage latency breakdowns
   - **TensorRT-LLM**: Core latency, queue, token, KV-cache, memory, and speculative decoding metrics when Prometheus output is enabled
   - **Triton**: Triton core request counts, queue depth, cumulative latency counters, optional first-response histograms, GPU/CPU/pinned-memory telemetry, and response-cache metrics

---

*For detailed implementation and usage examples, see the [Server Metrics Tutorial](server-metrics.md). For aggregated statistics, see the [JSON Schema Reference](server-metrics-json-schema.md). For raw time-series analysis, see the [Parquet Schema Reference](server-metrics-parquet-schema.md).*
