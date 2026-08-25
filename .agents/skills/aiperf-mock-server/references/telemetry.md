<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Telemetry & metrics endpoints

The mock exposes several Prometheus/OpenMetrics scrape endpoints so you can exercise
AIPerf's two side-channel scrapers without a real server. All routes are in
`rust/mock-server/src/app.rs`; the registries are in `rust/mock-server/src/prom.rs` and the
DCGM faker in `rust/mock-server/src/dcgm.rs`.

The runner scrapes these two ways (Config v2 / `aiperf profile` flags):
- **Server metrics** (`aiperf::server_metrics`): point `--server-metrics <url>` at a
  `/metrics` or backend-dialect endpoint below. Emits `server_metrics.{json,csv,parquet}`.
- **GPU telemetry** (`aiperf::gpu_telemetry`): point `--gpu-telemetry <url>` at a
  `/dcgm*/metrics` endpoint below. Feeds the `telemetry_data` summary.

## Endpoint catalog

| Route | Content |
|---|---|
| `GET /metrics` | The mock's own `aiperf_mock_*` counters/histograms (see below), plus the appended live `aiperf_mock_accuracy_*` tally when an accuracy dataset is loaded |
| `GET /vllm/metrics` | vLLM dialect (`vllm:*`) |
| `GET /sglang/metrics` | SGLang dialect (`sglang:*`) |
| `GET /trtllm/metrics` | TensorRT-LLM dialect (`trtllm:*`) |
| `GET /dynamo_frontend/metrics` | Dynamo frontend (`dynamo_frontend_*`) |
| `GET /dynamo_component/prefill/metrics` | Dynamo prefill worker (`dynamo_component_*`) |
| `GET /dynamo_component/decode/metrics` | Dynamo decode worker (`dynamo_component_*`) |
| `GET /dcgm1/metrics`, `GET /dcgm2/metrics` | Two independent synthetic DCGM exporters |

## Exposition format: classic Prometheus vs OpenMetrics (`--openmetrics`)

By default every `/metrics`-family endpoint serves **classic Prometheus text**
(`Content-Type: text/plain; version=0.0.4`) via the tikv `prometheus` crate's `TextEncoder`.
This matches the **Dynamo** frontend, which also uses tikv `prometheus` (v0.14) +
`TextEncoder.encode(registry.gather())`.

`--openmetrics` (env `MOCK_SERVER_OPENMETRICS`) switches all of them to **OpenMetrics text**,
matching the **vLLM Rust frontend** (`rust/src/metrics/` + `server/src/routes/metrics.rs`,
which uses the `prometheus-client` OpenMetrics encoder). The mock produces this by
post-processing the classic exposition rather than swapping libraries:

- `Content-Type: application/openmetrics-text; version=1.0.0; charset=utf-8`
- body terminated with a lone `# EOF` line
- the `_total` suffix stripped from counter **MetricFamily metadata** (`# TYPE`/`# HELP`) while
  retained on the sample line (`foo_total{...} N`) — the OpenMetrics counter-family rule

Why not switch to `prometheus-client`? A source-verified microbench showed it ~30% faster on
cached `observe`/`inc` and ~18% better under shared-histogram contention, but ~26% slower on
uncached labeled lookups — and the mock already caches child handles, so the net is a few % on
the metrics-on path only, not worth re-registering ~100 metric families. Dynamo (the closer
analog to what AIPerf drives) uses the same tikv crate we do.

**Disabling metrics entirely:** `--no-metrics` (env `MOCK_SERVER_NO_METRICS`) makes every
recording call a no-op — the endpoints still respond but report zeros. Under saturating load
the per-request histogram observes + shared-counter increments are a large fraction of server
CPU; this trades that telemetry for ~+15-30% throughput. Response content is unaffected.

## `/metrics` — the mock's own instrumentation (`aiperf_mock_*`)

Counters/gauges/histograms labeled by endpoint/model, e.g. `aiperf_mock_requests_total`,
`aiperf_mock_request_latency_seconds`, `aiperf_mock_prompt_tokens_total`,
`aiperf_mock_completion_tokens_total`, `aiperf_mock_time_to_first_token_seconds`,
`aiperf_mock_inter_token_latency_seconds`, `aiperf_mock_tokens_streamed_total`,
`aiperf_mock_errors_total`, `aiperf_mock_embeddings_generated_total`,
`aiperf_mock_rankings_generated_total`, `aiperf_mock_images_processed_total`,
`aiperf_mock_request_bytes_total` / `aiperf_mock_response_bytes_total`,
`aiperf_mock_uptime_seconds`. (Accuracy tally names live in `references/accuracy.md`.)

## vLLM dialect (`/vllm/metrics`)

Histograms: `vllm:e2e_request_latency_seconds`, `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds`, `vllm:request_queue_time_seconds`,
`vllm:iteration_tokens_total`. Counters/gauges: `vllm:prompt_tokens`,
`vllm:generation_tokens`, `vllm:request_success`, `vllm:num_requests_running`,
`vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:cpu_cache_usage_perc`,
`vllm:num_preemptions`, `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`,
`vllm:external_prefix_cache_hits`, `vllm:external_prefix_cache_queries`.
(The `cpu_cache_usage_perc`, `num_preemptions`, and `external_prefix_cache_*` families are
the newer fills for external-cache / CPU-offload / preemption metrics.)

## SGLang dialect (`/sglang/metrics`)

`sglang:gen_throughput`, `sglang:num_queue_reqs`, `sglang:num_running_reqs`,
`sglang:cache_hit_rate`, `sglang:num_used_tokens`, `sglang:token_usage`,
`sglang:spec_accept_rate`, `sglang:spec_accept_length`,
`sglang:cached_tokens`, `sglang:prompt_tokens`, `sglang:generation_tokens`,
`sglang:num_retracted_reqs`, `sglang:queue_time_seconds`, `sglang:e2e_request_latency_seconds`,
`sglang:time_to_first_token_seconds`. (`cached_tokens`, `num_retracted_reqs`, and the token
counters are the newer SGLang counters.) The speculative gauges expose one
deterministic leader fixture: `model_name="mock-model"`, `pp_rank="0"`,
`tp_rank="0"`, raw acceptance-rate `0.75`, and accepted-length `2.5`.

## TensorRT-LLM dialect (`/trtllm/metrics`)

`trtllm:e2e_request_latency_seconds`, `trtllm:time_to_first_token_seconds`,
`trtllm:time_per_output_token_seconds`, `trtllm:request_queue_time_seconds`,
`trtllm:request_success`.

## Dynamo dialects

Frontend (`dynamo_frontend_*`, labeled by `model`): `request_duration_seconds`,
`time_to_first_token_seconds`, `inter_token_latency_seconds`, `requests`,
`input_sequence_tokens`, `output_sequence_tokens`, `output_tokens`, `queued_requests`,
`inflight_requests`, `disconnected_clients`, `model_context_length`,
`model_kv_cache_block_size`, `model_total_kv_blocks`.

Component prefill/decode (`dynamo_component_*`, labeled by `dynamo_endpoint`, `model`):
`request_duration_seconds`, `requests`, `inflight_requests`, plus KV stats
`kvstats_active_blocks`, `kvstats_total_blocks`, `kvstats_gpu_cache_usage_percent`.

## DCGM GPU telemetry (`/dcgm1/metrics`, `/dcgm2/metrics`)

Synthetic per-GPU DCGM gauges labeled `gpu`, `UUID`, `pci_bus_id`, `device`, `modelName`,
`Hostname`. Emitted fields (18 families):

`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_POWER_MGMT_LIMIT`,
`DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_TOTAL`, `DCGM_FI_DEV_FB_FREE`,
`DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_MEMORY_TEMP`, `DCGM_FI_DEV_SM_CLOCK`,
`DCGM_FI_DEV_MEM_CLOCK`, `DCGM_FI_DEV_MEM_COPY_UTIL`, `DCGM_FI_DEV_ENC_UTIL`,
`DCGM_FI_DEV_DEC_UTIL`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`,
`DCGM_FI_DEV_XID_ERRORS`, `DCGM_FI_DEV_POWER_VIOLATION`, `DCGM_FI_DEV_THERMAL_VIOLATION`.

The newer fields consumed by the runner's GPU-telemetry decoder are the encoder/decoder
engine utilizations (`DCGM_FI_DEV_ENC_UTIL`, `DCGM_FI_DEV_DEC_UTIL`) and the SM-activity
profiling ratio (`DCGM_FI_PROF_SM_ACTIVE`, a `[0, 1]` ratio the runner scales ×100).

DCGM knobs (all `MOCK_SERVER_DCGM_*` env twins):

| Flag | Default | Effect |
|---|---|---|
| `--dcgm-gpu-name` | h200 | GPU model: `rtx6000`/`a100`/`h100`/`h100-sxm`/`h200`/`b200`/`gb200` (sets memory/power/clock/temp envelope) |
| `--dcgm-num-gpus` | 2 | Number of per-GPU series emitted |
| `--dcgm-min-throughput` | 100 | Minimum synthetic throughput floor |
| `--dcgm-window-sec` | 1.0 | Cadence window |
| `--dcgm-hostname` | localhost | `Hostname` label |
| `--dcgm-seed` | — | Deterministic DCGM values (UUIDs, noise) |
| `--dcgm-auto-load` | true | Auto-drive synthetic GPU load |

Output is deterministic under `--dcgm-seed`; load drives utilization/power/temp/clock/energy
higher, with occasional XID errors and power/thermal violation accumulation.
