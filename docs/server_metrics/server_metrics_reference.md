<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Server Metrics Reference

Comprehensive reference for server metrics collected during AIPerf benchmark runs from NVIDIA Dynamo, vLLM, SGLang, and TensorRT-LLM inference servers.

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
| `sglang:gen_throughput` | `stats.avg` | Real-time generation throughput (SGLang) |

### "What is my latency?"

| Metric | Field | Description |
|--------|-------|-------------|
| `dynamo_frontend_request_duration_seconds` | `stats.p99_estimate` | End-to-end p99 latency |
| `dynamo_frontend_request_duration_seconds` | `stats.avg` | Average request latency |
| `dynamo_frontend_time_to_first_token_seconds` | `stats.p99_estimate` | Time to first token (TTFT) p99 |
| `dynamo_frontend_inter_token_latency_seconds` | `stats.p99_estimate` | Inter-token latency (ITL) p99 |
| `vllm:time_to_first_token_seconds` | `stats.p99_estimate` | TTFT p99 (vLLM) |
| `sglang:queue_time_seconds` | `stats.p99_estimate` | Queue time p99 (SGLang) |
| `trtllm:time_to_first_token_seconds` | `stats.p99_estimate` | TTFT p99 (TensorRT-LLM) |

### "Am I hitting capacity limits?"

| Metric | Field | Threshold | Meaning |
|--------|-------|-----------|---------|
| `vllm:kv_cache_usage_perc` | `stats.max` | >0.9 | KV cache near full capacity |
| `vllm:num_preemptions` | `stats.total` | >0 | Memory pressure causing preemptions |
| `vllm:num_requests_waiting` | `stats.avg` | Growing | Queue building up |
| `dynamo_frontend_queued_requests` | `stats.avg` | High | Requests awaiting first token |
| `sglang:token_usage` | `stats.max` | >0.9 | High memory utilization (SGLang) |
| `sglang:num_queue_reqs` | `stats.avg` | Growing | Saturation (SGLang) |
| `trtllm:request_queue_time_seconds` | `stats.avg` | High | Saturation (TensorRT-LLM) |

### "What does my workload look like?"

| Metric | Field | Description |
|--------|-------|-------------|
| `dynamo_frontend_input_sequence_tokens` | `stats.avg` | Average prompt length |
| `dynamo_frontend_input_sequence_tokens` | `stats.p99_estimate` | Longest prompts (p99) |
| `dynamo_frontend_output_sequence_tokens` | `stats.avg` | Average response length |
| `dynamo_frontend_output_sequence_tokens` | `stats.p99_estimate` | Longest responses (p99) |

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
| `prefill_waiting` | Waiting before prefill |
| `prefill_bootstrap` | Prefill scheduling overhead |
| `prefill_prepare` | Preparing prefill batch |
| `prefill_forward` | Prefill forward pass execution |
| `prefill_transfer_kv_cache` | KV cache transfer (disaggregated mode) |
| `decode_waiting` | Waiting before decode |
| `decode_transferred` | Decode phase execution |

**TensorRT-LLM latency breakdown:**

| Phase | Metric | What it means |
|-------|--------|---------------|
| Queue | `trtllm:request_queue_time_seconds` | Waiting for GPU resources |
| TTFT | `trtllm:time_to_first_token_seconds` | Time to first output token |
| Total | `trtllm:e2e_request_latency_seconds` | Complete request duration |

---

## Backend Comparison Matrix

Key equivalent metrics across backends:

| Capability | Dynamo Frontend | vLLM | SGLang | TensorRT-LLM |
|------------|----------------|------|--------|--------------|
| **End-to-end latency** | `dynamo_frontend_request_duration_seconds` | `vllm:e2e_request_latency_seconds` | — | `trtllm:e2e_request_latency_seconds` |
| **TTFT** | `dynamo_frontend_time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` | — | `trtllm:time_to_first_token_seconds` |
| **ITL** | `dynamo_frontend_inter_token_latency_seconds` | `vllm:inter_token_latency_seconds` | — | `trtllm:time_per_output_token_seconds` |
| **Queue time** | — | `vllm:request_queue_time_seconds` | `sglang:queue_time_seconds` | `trtllm:request_queue_time_seconds` |
| **KV cache usage** | `dynamo_component_kvstats_gpu_cache_usage_percent` | `vllm:kv_cache_usage_perc` | `sglang:token_usage` | — |
| **Requests running** | `dynamo_frontend_inflight_requests` | `vllm:num_requests_running` | `sglang:num_running_reqs` | — |
| **Requests queued** | `dynamo_frontend_queued_requests` | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | — |
| **Successful requests** | `dynamo_frontend_requests` | `vllm:request_success` | — | `trtllm:request_success` |
| **Prompt tokens** | `dynamo_frontend_input_sequence_tokens` | `vllm:request_prompt_tokens` | — | — |
| **Generation tokens** | `dynamo_frontend_output_sequence_tokens` | `vllm:request_generation_tokens` | — | — |

**Key insight:** Dynamo metrics measure at the HTTP/routing layer (user-facing), while backend metrics measure inside the inference engine (debugging). Use both for complete visibility.

---

## Metric Interpretation Guide

### Metric Types

**Counter** (cumulative, monotonically increasing):
- `stats.total` = Total change during benchmark
- `stats.rate` = Rate of change (per second)
- Example: `vllm:prompt_tokens` with `stats.rate` = prefill throughput

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

## Dynamo Frontend

The Dynamo frontend is the HTTP entry point that receives client requests and routes them to backend workers. These metrics provide user-facing visibility into request processing.

### Request Flow

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_frontend_requests` | counter | requests | `endpoint`, `model`, `request_type`, `status` | Total LLM requests processed. Use `stats.total` for count during benchmark, `stats.rate` for throughput (req/s). |
| `dynamo_frontend_inflight_requests` | gauge | requests | `model` | Requests currently being processed. High values indicate saturation. |
| `dynamo_frontend_queued_requests` | gauge | requests | `model` | Requests that have not yet received the first token. |
| `dynamo_frontend_disconnected_clients` | gauge | clients | — | Client connections that disconnected (possibly due to timeouts). |

**Label values:**
- `endpoint`: `chat_completions`, `completions`
- `request_type`: `stream`, `unary`
- `status`: `success`, `error`

### Latency

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `dynamo_frontend_request_duration_seconds` | histogram | seconds | `model` | `0.0, 1.9, 3.4, 6.3, 12.0, 22.0, 40.0, 75.0, 140.0, 260.0, +Inf` | **End-to-end request latency** from HTTP receive to response complete. Key metric for SLA compliance. Use `stats.p99_estimate` for tail latency. |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | seconds | `model` | `0.0, 0.0022, 0.0047, 0.01, 0.022, 0.047, 0.1, 0.22, 0.47, 1.0, 2.2, 4.7, 10.0, 22.0, 48.0, 100.0, 220.0, 480.0, +Inf` | **Time to first token (TTFT)** - latency until first token is generated. Critical for perceived responsiveness. |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | seconds | `model` | `0.0, 0.0019, 0.0035, 0.0067, 0.013, 0.024, 0.045, 0.084, 0.16, 0.3, 0.56, 1.1, 2.0, +Inf` | **Inter-token latency (ITL)** - time between consecutive tokens. Lower is better for streaming UX. |

### Tokens

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `dynamo_frontend_output_tokens` | counter | tokens | `model` | — | Total output tokens generated. `stats.rate` = output token throughput (tokens/s). |
| `dynamo_frontend_input_sequence_tokens` | histogram | tokens | `model` | `0.0, 100.0, 210.0, 430.0, 870.0, 1800.0, 3600.0, 7400.0, 15000.0, 31000.0, 63000.0, 130000.0, +Inf` | **Input sequence length distribution**. `stats.avg` = mean prompt length, `stats.p99_estimate` = longest prompts. |
| `dynamo_frontend_output_sequence_tokens` | histogram | tokens | `model` | `0.0, 100.0, 210.0, 430.0, 880.0, 1800.0, 3700.0, 7600.0, 16000.0, 32000.0, +Inf` | **Output sequence length distribution**. `stats.avg` = mean response length. |

### Model Configuration (Static Gauges)

These are constant values that don't change during the benchmark. Only `stats.avg` is meaningful.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_frontend_model_context_length` | gauge | `model` | Maximum context window size in tokens (e.g., 40960). |
| `dynamo_frontend_model_kv_cache_block_size` | gauge | `model` | KV cache block size in tokens (e.g., 16). |
| `dynamo_frontend_model_max_num_batched_tokens` | gauge | `model` | Maximum tokens that can be batched together. |
| `dynamo_frontend_model_max_num_seqs` | gauge | `model` | Maximum concurrent sequences per worker. *(vLLM, TensorRT-LLM only)* |
| `dynamo_frontend_model_total_kv_blocks` | gauge | `model` | Total KV cache blocks available per worker. *(vLLM, SGLang only)* |
| `dynamo_frontend_model_migration_limit` | gauge | `model` | Maximum request migrations allowed (0 = disabled). |

---

## Dynamo Component

Dynamo components are backend workers that execute inference. These metrics come from the worker process level and provide visibility into backend-level request processing.

### Request Processing

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_requests` | counter | requests | `dynamo_component`, `dynamo_endpoint`, `dynamo_namespace`, `model` | Requests processed by this worker. Compare across workers to check load balancing. |
| `dynamo_component_inflight_requests` | gauge | requests | `dynamo_component`, `dynamo_endpoint`, `dynamo_namespace`, `model` | Requests currently executing on this worker. |
| `dynamo_component_errors` | counter | errors | `dynamo_component`, `dynamo_endpoint`, `dynamo_namespace`, `error_type`, `model` | Errors in work handler. Non-zero indicates problems. |

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `dynamo_component_request_duration_seconds` | histogram | seconds | `dynamo_component`, `dynamo_endpoint`, `dynamo_namespace`, `model` | `0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf` | Worker-level request processing time. Compare to frontend duration to measure routing overhead. |

### Data Transfer

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_request_bytes` | counter | bytes | `dynamo_component`, `dynamo_endpoint`, `dynamo_namespace`, `model` | Total bytes received in requests. `stats.rate` = inbound bandwidth. |
| `dynamo_component_response_bytes` | counter | bytes | `dynamo_component`, `dynamo_endpoint`, `dynamo_namespace`, `model` | Total bytes sent in responses. `stats.rate` = outbound bandwidth. |

### KV Cache Statistics

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_kvstats_active_blocks` | gauge | blocks | `dynamo_component`, `dynamo_namespace` | KV cache blocks currently in use. |
| `dynamo_component_kvstats_total_blocks` | gauge | blocks | `dynamo_component`, `dynamo_namespace` | Total KV cache blocks available. |
| `dynamo_component_kvstats_gpu_cache_usage_percent` | gauge | ratio | `dynamo_component`, `dynamo_namespace` | **GPU cache utilization** (0.0-1.0). High values (>0.9) may cause preemptions. |
| `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | gauge | ratio | `dynamo_component`, `dynamo_namespace` | Prefix cache hit rate (0.0-1.0). Higher = better reuse of cached prefixes. |

### NATS Messaging (Internal)

NATS metrics track the internal messaging system used for component communication within Dynamo.

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_component_nats_client_connection_state` | gauge | — | Connection state: 0=disconnected, 1=connected, 2=reconnecting. |
| `dynamo_component_nats_client_current_connections` | gauge | connections | Active NATS connections. |
| `dynamo_component_nats_client_in_messages` | gauge | messages | Messages received via NATS. |
| `dynamo_component_nats_client_out_messages` | gauge | messages | Messages sent via NATS. |
| `dynamo_component_nats_client_in_total_bytes` | gauge | bytes | Bytes received via NATS. |
| `dynamo_component_nats_client_out_overhead_bytes` | gauge | bytes | Bytes sent via NATS (including protocol overhead). |

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `dynamo_component_nats_service_active_services` | gauge | services | `dynamo_component`, `dynamo_namespace`, `service_name` | Active NATS services in component. |
| `dynamo_component_nats_service_active_endpoints` | gauge | endpoints | `dynamo_component`, `dynamo_namespace`, `service_name` | Active NATS endpoints. |
| `dynamo_component_nats_service_requests_total` | gauge | requests | `dynamo_component`, `dynamo_namespace`, `service_name` | Total NATS service requests. |
| `dynamo_component_nats_service_errors_total` | gauge | errors | `dynamo_component`, `dynamo_namespace`, `service_name` | NATS service errors. |
| `dynamo_component_nats_service_processing_ms_total` | gauge | milliseconds | `dynamo_component`, `dynamo_namespace`, `service_name` | Total NATS processing time. |
| `dynamo_component_nats_service_processing_ms_avg` | gauge | milliseconds | `dynamo_component`, `dynamo_namespace`, `service_name` | Average NATS processing time. |

---

## vLLM

vLLM is a high-performance inference engine. These metrics provide deep visibility into model execution, cache usage, and request processing phases.

### Cache & Memory

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:kv_cache_usage_perc` | gauge | ratio | `engine`, `model_name` | **KV cache utilization** (0.0-1.0). Key capacity indicator. Values near 1.0 cause performance degradation. Monitor `stats.max`. |
| `vllm:prefix_cache_hits` | counter | tokens | `engine`, `model_name` | Tokens served from prefix cache. Higher = better prompt reuse. |
| `vllm:prefix_cache_queries` | counter | tokens | `engine`, `model_name` | Tokens queried against prefix cache. `hits/queries` = hit rate. |
| `vllm:num_preemptions` | counter | preemptions | `engine`, `model_name` | Requests preempted due to memory pressure. Non-zero indicates capacity issues. |

### Queue State

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:num_requests_running` | gauge | requests | `engine`, `model_name` | Requests currently in model execution batch. Indicates batch size. |
| `vllm:num_requests_waiting` | gauge | requests | `engine`, `model_name` | Requests queued waiting for execution. High values indicate saturation. |

### Token Throughput

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `vllm:prompt_tokens` | counter | tokens | `engine`, `model_name` | Prefill tokens processed. `stats.rate` = prefill throughput. |
| `vllm:generation_tokens` | counter | tokens | `engine`, `model_name` | Generation tokens produced. `stats.rate` = decode throughput. |
| `vllm:request_success` | counter | requests | `engine`, `finished_reason`, `model_name` | Successfully completed requests. |

**Common `finished_reason` values:** `length`, `stop`, `error`

### Request-Level Latency Breakdown

These histograms show where time is spent for each request. Together they decompose the end-to-end latency.

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `vllm:e2e_request_latency_seconds` | histogram | seconds | `engine`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | **Total request latency** inside vLLM (queue + inference). |
| `vllm:request_queue_time_seconds` | histogram | seconds | `engine`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | Time spent in **WAITING** phase (queued before execution). |
| `vllm:request_prefill_time_seconds` | histogram | seconds | `engine`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | Time spent in **PREFILL** phase (processing input tokens). |
| `vllm:request_decode_time_seconds` | histogram | seconds | `engine`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | Time spent in **DECODE** phase (generating output tokens). |
| `vllm:request_inference_time_seconds` | histogram | seconds | `engine`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | Time spent in **RUNNING** phase (prefill + decode). |

### Token-Level Latency

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `vllm:time_to_first_token_seconds` | histogram | seconds | `engine`, `model_name` | `0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf` | **TTFT** - time from request start to first output token. |
| `vllm:inter_token_latency_seconds` | histogram | seconds | `engine`, `model_name` | `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf` | **ITL** - time between consecutive output tokens. |
| `vllm:request_time_per_output_token_seconds` | histogram | seconds | `engine`, `model_name` | `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf` | Average time per token for each request (total_time / num_tokens). |
| `vllm:time_per_output_token_seconds` | histogram | seconds | `engine`, `model_name` | `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf` | *(Deprecated)* Use `vllm:inter_token_latency_seconds` instead. |

### Request Parameters

These histograms show the distribution of request parameters processed by vLLM.

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `vllm:request_prompt_tokens` | histogram | tokens | `engine`, `model_name` | `1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, +Inf` | Input token count per request. Same as `dynamo_frontend_input_sequence_tokens`. |
| `vllm:request_generation_tokens` | histogram | tokens | `engine`, `model_name` | `1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, +Inf` | Output token count per request. Same as `dynamo_frontend_output_sequence_tokens`. |
| `vllm:request_max_num_generation_tokens` | histogram | tokens | `engine`, `model_name` | `1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, +Inf` | Maximum tokens requested per request (`max_tokens` parameter). |
| `vllm:request_params_max_tokens` | histogram | tokens | `engine`, `model_name` | `1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, +Inf` | Distribution of `max_tokens` API parameter. |
| `vllm:request_params_n` | histogram | — | `engine`, `model_name` | `1.0, 2.0, 5.0, 10.0, 20.0, +Inf` | Distribution of `n` parameter (number of completions per request). |
| `vllm:iteration_tokens_total` | histogram | tokens | `engine`, `model_name` | `1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, +Inf` | Tokens processed per engine step. Indicates batch efficiency. |

### Configuration Info

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:cache_config_info` | info | `engine`, `block_size`, `cache_dtype`, `enable_prefix_caching`, `gpu_memory_utilization`, `num_gpu_blocks`, etc. | Static cache configuration. Info is exposed as labels on a gauge metric with value 1.0. |

**Common cache config labels:**
- `block_size`: KV cache block size in tokens (e.g., `16`)
- `cache_dtype`: Cache data type (e.g., `auto`)
- `enable_prefix_caching`: Whether prefix caching is enabled (`True`/`False`)
- `gpu_memory_utilization`: GPU memory utilization target (e.g., `0.9`)
- `num_gpu_blocks`: Total GPU blocks allocated (e.g., `71671`)

---

## SGLang

SGLang is a fast inference engine with RadixAttention for efficient prefix caching. These metrics provide visibility into SGLang's scheduling, execution, and advanced features like disaggregated inference and speculative decoding.

### Throughput & Performance

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:gen_throughput` | gauge | tokens/s | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | **Generation throughput** in tokens per second. Real-time throughput indicator. |
| `sglang:cache_hit_rate` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Prefix cache hit rate (0.0-1.0). Higher = better prompt reuse via RadixAttention. |
| `sglang:token_usage` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Token usage ratio (0.0-1.0). Indicates memory utilization. |
| `sglang:utilization` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Overall utilization. -1.0 indicates idle, 0.0+ indicates active. |

### Queue State

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:num_running_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests currently executing in the batch. |
| `sglang:num_queue_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests in the waiting queue. High values indicate saturation. |
| `sglang:num_used_tokens` | gauge | tokens | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Total tokens currently in use across all requests. |
| `sglang:num_running_reqs_offline_batch` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Low-priority offline batch requests running. |
| `sglang:num_paused_reqs` | gauge | requests | `engine_type`, `model_name`, `pid`, `pp_rank`, `tp_rank` | Requests paused by async weight sync. |
| `sglang:num_retracted_reqs` | gauge | requests | `engine_type`, `model_name`, `pid`, `pp_rank`, `tp_rank` | Requests that were retracted/preempted. |
| `sglang:num_grammar_queue_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests waiting for grammar processing. |

### Disaggregated Inference Queues

For disaggregated prefill/decode deployments where prefill and decode run on separate instances.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:num_prefill_prealloc_queue_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests in prefill preallocation queue. |
| `sglang:num_prefill_inflight_queue_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests in prefill inflight queue. |
| `sglang:num_decode_prealloc_queue_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests in decode preallocation queue. |
| `sglang:num_decode_transfer_queue_reqs` | gauge | requests | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Requests in decode transfer queue. |

### Request Latency Breakdown

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `sglang:queue_time_seconds` | histogram | seconds | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | `0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2500.0, 3000.0, +Inf` | Time spent in **WAITING** queue before execution starts. |
| `sglang:per_stage_req_latency_seconds` | histogram | seconds | `engine_type`, `model_name`, `pp_rank`, `stage`, `tp_rank` | *(see below)* | Per-stage latency breakdown. `stage` label identifies the phase. |

**Histogram buckets for `sglang:per_stage_req_latency_seconds`:**
```
0.001, 0.0016, 0.0026, 0.0043, 0.0069, 0.0112, 0.0181, 0.0293, 0.0474, 0.0768, 0.1245, 0.2017, 0.3267, 0.5293, 0.8575, 1.3891, 2.2503, 3.6455, 5.9057, 9.5672, 15.4989, 25.1082, 40.6753, 65.8939, 106.7481, 172.9320, 280.1498, 453.8427, 735.2252, 1191.0649, +Inf
```

**Stage labels for `sglang:per_stage_req_latency_seconds`:**

| Stage | Description |
|-------|-------------|
| `prefill_waiting` | Time waiting before prefill begins |
| `prefill_bootstrap` | Time to bootstrap prefill (scheduling overhead) |
| `prefill_prepare` | Time preparing prefill batch |
| `prefill_forward` | Time executing prefill forward pass |
| `prefill_transfer_kv_cache` | Time transferring KV cache (disaggregated mode) |
| `decode_waiting` | Time waiting before decode begins |
| `decode_bootstrap` | Time to bootstrap decode |
| `decode_prepare` | Time preparing decode batch |
| `decode_transferred` | Total time in transferred/decode phase |

### KV Cache Transfer (Disaggregated)

For disaggregated prefill/decode deployments, these metrics track KV cache transfer between instances.

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:kv_transfer_latency_ms` | gauge | milliseconds | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | KV cache transfer latency. |
| `sglang:kv_transfer_speed_gb_s` | gauge | GB/s | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | KV cache transfer throughput. |
| `sglang:kv_transfer_alloc_ms` | gauge | milliseconds | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Time waiting for KV cache allocation. |
| `sglang:kv_transfer_bootstrap_ms` | gauge | milliseconds | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | KV transfer bootstrap time. |
| `sglang:pending_prealloc_token_usage` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Token usage for pending preallocated tokens (not preallocated yet). |

### Speculative Decoding

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:spec_accept_rate` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Speculative decoding acceptance rate (accepted tokens / total draft tokens in batch). Higher = better speculation. |
| `sglang:spec_accept_length` | gauge | tokens | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Average acceptance length of speculative decoding. |

### System Configuration

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `sglang:is_cuda_graph` | gauge | — | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Whether batch is using CUDA graph (1=yes, 0=no). |
| `sglang:engine_startup_time` | gauge | seconds | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Engine startup time. |
| `sglang:engine_load_weights_time` | gauge | seconds | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Time to load model weights. |
| `sglang:mamba_usage` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Token usage for Mamba layers (hybrid models). |
| `sglang:swa_token_usage` | gauge | ratio | `engine_type`, `model_name`, `pp_rank`, `tp_rank` | Token usage for sliding window attention layers. |

**Common label values:**
- `engine_type`: `unified`
- `model_name`: Model identifier (e.g., `Qwen/Qwen3-0.6B`)
- `tp_rank`: Tensor parallel rank (e.g., `0`, `1`, ...)
- `pp_rank`: Pipeline parallel rank (e.g., `0`, `1`, ...)
- `pid`: Process ID

---

## TensorRT-LLM

TensorRT-LLM (trtllm) is NVIDIA's high-performance inference engine optimized for NVIDIA GPUs. These metrics focus on request latency and completion tracking.

### Request Latency

| Metric | Type | Unit | Labels | Histogram Buckets | Description |
|--------|------|------|--------|-------------------|-------------|
| `trtllm:e2e_request_latency_seconds` | histogram | seconds | `engine_type`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | **End-to-end request latency** from submission to completion. Use `stats.p99_estimate` for tail latency. |
| `trtllm:request_queue_time_seconds` | histogram | seconds | `engine_type`, `model_name` | `0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf` | Time spent in **WAITING** phase (queued before execution). |
| `trtllm:time_to_first_token_seconds` | histogram | seconds | `engine_type`, `model_name` | `0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf` | **TTFT** - time from request start to first output token. |
| `trtllm:time_per_output_token_seconds` | histogram | seconds | `engine_type`, `model_name` | `0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf` | Time per output token (inter-token latency). |

### Request Completion

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `trtllm:request_success` | counter | requests | `engine_type`, `finished_reason`, `model_name` | Successfully completed requests. `finished_reason` label indicates completion reason. |

**Common label values:**
- `engine_type`: `trtllm`
- `model_name`: Model identifier (e.g., `Qwen/Qwen3-0.6B`)
- `finished_reason`: `length` (reached max_tokens), `stop` (stop sequence), `error` (error occurred)

---

## KVBM (KV Block Manager)

**Note:** These metrics are only available with Dynamo deployments using the KV Block Manager feature for advanced KV cache management.

### Block Transfer Operations

All metrics are counters tracking cumulative block movement operations.

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `kvbm_matched_tokens` | counter | tokens | The number of matched tokens (prefix cache hits). |
| `kvbm_offload_blocks_d2d` | counter | blocks | The number of offload blocks from device to disk (bypassing host memory). |
| `kvbm_offload_blocks_d2h` | counter | blocks | The number of offload blocks from device to host memory. |
| `kvbm_offload_blocks_h2d` | counter | blocks | The number of offload blocks from host memory to disk. |
| `kvbm_onboard_blocks_d2d` | counter | blocks | The number of onboard blocks from disk to device (bypassing host memory). |
| `kvbm_onboard_blocks_h2d` | counter | blocks | The number of onboard blocks from host memory to device. |

**Block transfer patterns:**
- **d2d**: Device ↔ Disk (direct, fast path)
- **d2h**: Device → Host (offload to CPU memory)
- **h2d**: Host → Device (onboard from CPU memory)
- **h2d** (disk): Host → Disk (persist to storage)

---

## Appendix

### Common Metric Labels

Labels that appear across multiple metrics:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `model` | Model identifier (Dynamo) | `qwen/qwen3-0.6b` |
| `model_name` | Model identifier (backends) | `Qwen/Qwen3-0.6B` |
| `endpoint` | API endpoint | `chat_completions`, `completions` |
| `request_type` | Request type | `stream`, `unary` |
| `status` | Request outcome | `success`, `error` |
| `engine` | Engine identifier (vLLM) | `0`, `1`, ... |
| `engine_type` | Engine type | `trtllm`, `unified` |
| `tp_rank` | Tensor parallel rank | `0`, `1`, ... |
| `pp_rank` | Pipeline parallel rank | `0`, `1`, ... |
| `stage` | Processing stage (SGLang) | `prefill_forward`, `decode_transferred` |
| `finished_reason` | Completion reason | `length`, `stop`, `error` |
| `dynamo_component` | Component identifier | Worker name/ID |
| `dynamo_endpoint` | Internal endpoint | Internal routing info |
| `dynamo_namespace` | Namespace | Deployment namespace |
| `error_type` | Error classification | Error category |
| `service_name` | NATS service name | Service identifier |

### Notes on Metric Usage

1. **Dynamo vs backend metrics**: Dynamo metrics measure at the HTTP/routing layer (user-facing), while vLLM/SGLang/TensorRT-LLM metrics measure inside the inference engine. Use Dynamo for user-facing SLAs, backend metrics for debugging performance.

2. **Counter vs Gauge interpretation**:
   - **Counters**: Use `stats.total` for total change during benchmark, `stats.rate` for rate of change (per second)
   - **Gauges**: Use `stats.avg` for typical value, `stats.max` for peak, `stats.p99` for tail behavior

3. **Histogram percentiles**: Histogram percentiles (`stats.p50_estimate`, `stats.p90_estimate`, `stats.p95_estimate`, `stats.p99_estimate`) are *estimated* from bucket boundaries. Exact values depend on bucket configuration.

4. **Multiple endpoints**: When scraping multiple instances, each series includes an `endpoint_url` label to identify the source.

5. **Backend-specific capabilities**:
   - **vLLM**: Most comprehensive metrics including full request phase breakdown, cache statistics, and batch efficiency
   - **SGLang**: RadixAttention cache metrics, disaggregated inference support, speculative decoding stats, per-stage latency breakdowns
   - **TensorRT-LLM**: Focused on core latency metrics (queue, TTFT, e2e) with minimal overhead

---

*For detailed implementation and usage examples, see the [Server Metrics Tutorial](server-metrics.md). For aggregated statistics, see the [JSON Schema Reference](server_metrics_json_schema.md). For raw time-series analysis, see the [Parquet Schema Reference](server_metrics_parquet_schema.md).*
