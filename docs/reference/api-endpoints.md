---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: HTTP API Endpoints
---

# HTTP API Endpoints

When `aiperf profile` is launched with `--api-port <PORT>` (or `AIPERF_API_SERVER_PORT` is set), AIPerf starts an in-process FastAPI server alongside the benchmark and exposes HTTP and Prometheus endpoints for the lifetime of the run. The server is opt-in — if `--api-port` is not set, no listener is created.

`--api-host` (default `127.0.0.1`) controls the bind address; pass `--api-host 0.0.0.0` to expose externally (e.g. in Kubernetes).

The router classes that back these routes live under [`src/aiperf/api/routers/`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/api/routers/) and are loaded via the `api_router` plugin category in [`plugins.yaml`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/plugin/plugins.yaml).

## Endpoints

Paths written with `{filename:path}` / `{name:path}` / `{asset:path}` use FastAPI's `:path` converter, so the parameter matches nested relative paths containing `/` (e.g. `checkpoints/run_0001/profile_export.json`) rather than a single segment.

| Method | Path | Response shape | Description |
|---|---|---|---|
| `GET` | `/api/config` | `BenchmarkConfig` | Declarative benchmark configuration for the current run. `endpoint.api_key` is excluded. Run-identity fields (`benchmark_id`, `cli_command`, etc.) live on `/api/run`, not here. |
| `GET` | `/api/run` | `RunInfo` | Run-identity metadata: `benchmark_id`, `sweep_id`, `trial`, `random_seed`, `run_label`, `variation_*`, and the redacted `cli_command`. Same shape as the `run_info` block in `profile_export_aiperf.json`. |
| `GET` | `/api/metrics` | `MetricsResponse` | Live metric values (JSON). Updated as the benchmark progresses. |
| `GET` | `/metrics` | `text/plain` | Prometheus exposition format. Scrape target for Prometheus and compatible collectors. |
| `GET` | `/api/server-metrics` | JSON (`endpoint_summaries` map) | Real-time inference-server Prometheus metrics (queue depth, cache usage, latency, throughput) collected during the run. Returns `{"endpoint_summaries": {}, "message": ...}` until the first scrape lands. |
| `GET` | `/api/progress` | `ProgressResponse` | Per-phase credit/request progress. |
| `GET` | `/api/workers` | `WorkersResponse` | Per-worker stats (request counts, errors, lifecycle state). |
| `GET` | `/api/results` | `BenchmarkResultsResponse` | Final benchmark results once profiling has completed. |
| `GET` | `/api/results/list` | `ResultsListResponse` | List of result artifact files under the artifact directory. |
| `GET` | `/api/results/files/{filename:path}` | file stream | Download an individual artifact file by relative path. |
| `POST` | `/api/results/upload/{filename:path}` | `201` | Upload an artifact file into the artifact directory (used by the cluster results-sidecar / harvest path). |
| `GET` | `/api/dataset/data` | JSON (zstd-encodable) | The materialized dataset payload for the run. Honors `Accept-Encoding: zstd`. |
| `GET` | `/api/dataset/index` | JSON (zstd-encodable) | The dataset index. Honors `Accept-Encoding: zstd`. |
| `GET` | `/api/dataset/state` | `DatasetStateResponse` | Dataset preparation / loading state. |
| `GET` | `/api/tokenizer/{name:path}/bundle` | stream (bundle bytes) | Download a materialized tokenizer bundle by name (used by group-managed worker warmup). Not in the OpenAPI schema. |
| `GET` | `/api/debug/pod-states` | JSON | Debug snapshot of per-pod states. |
| `GET` | `/api/debug/worker-startup-states` | JSON | Debug snapshot of per-worker startup states. |
| `POST` | `/api/shutdown` | JSON `{status}` | Request a graceful shutdown of the run. |
| `WEBSOCKET` | `/ws` | JSON messages | Real-time message streaming — subscribe by message type; supports `ping`/`pong`. Connection-capped. |
| `GET` | `/`, `/dashboard`, `/dashboard-v2`, `/dashboard-v2/{asset:path}` | HTML / static assets | Bundled web dashboards. `/` is the index page, `/dashboard` the legacy single-file dashboard, and `/dashboard-v2` the v2 SPA (redirects to `/dashboard-v2/`; assets served under that prefix). Not in the OpenAPI schema. |
| `GET` | `/healthz` | `text/plain` (`ok` / `unhealthy`) | Kubernetes liveness probe. 503 when the service is in `FAILED` state. |
| `GET` | `/readyz` | `text/plain` (`ok` / `not ready`) | Kubernetes readiness probe. 200 only when the service is `RUNNING`. |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | (FastAPI defaults) | Auto-generated OpenAPI documentation. |

## Secret redaction

`cli_command` is captured from `sys.argv` at `BenchmarkRun` construction and redacted by [`build_cli_command()`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/common/redact.py) before it is stored — so the string returned from `/api/run` and written to `profile_export_aiperf.json` is the same redacted form. `--api-key`-shaped flags, credential-bearing `--header` values (e.g. `Authorization: Bearer …`), and userinfo embedded in URL-typed flags (`--url`, `-u`, `--otel-url`, `--mlflow-tracking-uri`) are all scrubbed to `<redacted>`.

`/api/config` independently excludes `endpoint.api_key` from its response.

## Example

```bash
aiperf profile --model my-model --url http://localhost:8000 \
    --api-key sk-secret-12345 --api-port 9097 --request-count 1000 &

# Inspect the redacted launch command
curl -s http://127.0.0.1:9097/api/run | jq .cli_command
# "aiperf profile --model 'my-model' --url 'http://localhost:8000' \
#  --api-key '<redacted>' --api-port 9097 --request-count 1000"

# Liveness probe
curl -s http://127.0.0.1:9097/healthz
# ok
```
