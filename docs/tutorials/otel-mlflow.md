---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: OTel and MLflow Telemetry
---

# OTel and MLflow Telemetry

## What You Will Learn

This tutorial walks through AIPerf's telemetry integrations:

- **Live OTel streaming** — push GenAI-spec metrics to an OpenTelemetry Collector in real time during a benchmark run.
- **Live MLflow logging** — record per-request scalars to an MLflow tracking server as the run executes.
- **Post-run artifact upload** — automatically upload the JSON/CSV exports, metadata, and plots to the same MLflow run after profiling completes.

By the end you will have a single `aiperf profile` command that streams metrics to both sinks and a follow-up `aiperf plot` command that attaches visualizations to the MLflow run.

## Prerequisites

Install AIPerf with the optional telemetry extras:

```bash
pip install "aiperf[otel,mlflow]"
```

You also need:

| Component | Purpose | Quick start |
|-----------|---------|-------------|
| OTel Collector | Receives OTLP/HTTP metrics | `docker run -p 4318:4318 otel/opentelemetry-collector-contrib` |
| MLflow Tracking Server | Stores runs, metrics, artifacts | `mlflow ui` (uses `file:./mlruns` by default) |

Verify both are reachable before continuing:

```bash
# OTel Collector health (returns 200 on the base path when running)
curl -sf http://localhost:4318/ && echo "OTel Collector reachable"

# MLflow tracking server health
curl -sf http://localhost:5000/health && echo "MLflow reachable"
```

## Run a Profile with Telemetry Enabled

```bash
aiperf profile \
    --url http://localhost:8000 \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --synthetic-input-tokens-mean 128 \
    --output-tokens-mean 128 \
    --concurrency 4 \
    --request-count 64 \
    --otel-url http://localhost:4318 \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-experiment my-experiment \
    --stream default
```

### Flag breakdown

| Flag | Effect |
|------|--------|
| `--otel-url http://localhost:4318` | Enables OTLP/HTTP export to the collector. Accepts `host`, `host:port`, or a full URL. |
| `--mlflow-tracking-uri http://localhost:5000` | Enables MLflow integration and points to your tracking server. |
| `--mlflow-experiment my-experiment` | Creates or reuses the named experiment. |
| `--stream default` | Activates the default streaming strategy (metrics + timing). Use `--stream metrics` for metrics only. |

## Inspect Live OTel Data

While the benchmark runs, metrics flow to your OTel Collector and from there to any configured backend (Prometheus, Grafana, Jaeger, etc.).

AIPerf emits metrics using [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| OTel metric name | Description | Unit |
|-----------------|-------------|------|
| `gen_ai.client.operation.duration` | End-to-end request latency | s |
| `gen_ai.client.token.usage` | Token counts (input/output) | tokens |
| `gen_ai.client.operation.time_per_output_chunk` | Inter-token latency | s |
| `gen_ai.client.operation.time_to_first_chunk` | Time to first token | s |

Metrics carry standard GenAI attributes (`gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`) plus AIPerf-specific dimensions prefixed with `aiperf.*`.

### Example Grafana query

```promql
histogram_quantile(0.99,
  sum(rate(gen_ai_client_operation_duration_bucket[1m])) by (le, gen_ai_request_model)
)
```

## Inspect Live MLflow Data

Open the MLflow UI at `http://localhost:5000`. Navigate to the experiment you specified and select the active run.

During profiling you will see live scalars logged under the `live.*` namespace:

- `live.gen_ai.client.operation.duration`
- `live.gen_ai.client.token.usage`
- `live.gen_ai.client.operation.time_per_output_chunk`
- `live.gen_ai.client.operation.time_to_first_chunk`

These update in near real time as each request completes. Refresh the MLflow metrics tab to see the curves build up.

## Post-Run Artifact Upload

When the benchmark finishes, AIPerf performs a deferred export:

1. Local exporters write JSON and CSV files to the output directory.
2. The MLflow data exporter detects the live run via `mlflow_export.json` (written during the run).
3. All artifacts (JSON export, CSV export, GPU telemetry, metadata) are uploaded to the same MLflow run.

The `mlflow_export.json` file records the mapping between the local run and the MLflow run:

```json
{
  "tracking_uri": "http://localhost:5000",
  "experiment_name": "my-experiment",
  "run_id": "a1b2c3d4e5f6...",
  "reused_live_run": true
}
```

### Grouping benchmarks under a parent run

Use `--mlflow-parent-run-id` to organize multiple benchmarks as child runs under a single parent. This is useful for parameter sweeps or A/B comparisons.

```bash
# Create a parent run
mlflow runs create --experiment-name my-sweep
# Note the run_id from the output, e.g. "abc123def456"

# Run benchmarks as children
aiperf profile \
    --url http://localhost:8000 \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --synthetic-input-tokens-mean 128 \
    --output-tokens-mean 128 \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-parent-run-id abc123def456 \
    --concurrency 4 \
    --request-count 64

aiperf profile \
    --url http://localhost:8000 \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --synthetic-input-tokens-mean 128 \
    --output-tokens-mean 128 \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-parent-run-id abc123def456 \
    --concurrency 8 \
    --request-count 64
```

In the MLflow UI the parent run shows both child runs nested beneath it, making it straightforward to compare concurrency=4 vs concurrency=8 side by side.

## Attach Plots

After profiling, generate and upload plots to the same MLflow run:

```bash
aiperf plot --paths ./artifacts/my-model-chat-concurrency4 --mlflow-upload
```

The `--mlflow-upload` flag reads `mlflow_export.json` from the input directory and uploads the generated PNG files as artifacts on the existing run. The plots appear under the run's artifact tab in MLflow.

## Customising provider.name

AIPerf infers the `gen_ai.provider.name` attribute (provider name) from the endpoint URL hostname. For example, requests to `api.openai.com` resolve to `openai`.

When auto-inference doesn't match your setup (e.g. you're running vLLM on `localhost`), override it explicitly:

```bash
aiperf profile \
    --url http://localhost:8000 \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --otel-url http://localhost:4318 \
    --stream default \
    --gen-ai-provider vllm \
    --concurrency 4 \
    --request-count 64
```

The value you pass appears as `gen_ai.provider.name` on every emitted metric and as the `gen_ai.provider.name` tag in MLflow.

## Troubleshooting

### Metric name migration

If you previously relied on `aiperf.*` metric names (`aiperf.request_latency_ns`, etc.), AIPerf now emits OTel GenAI spec names (`gen_ai.client.operation.duration`, etc.) in seconds. Dashboards querying the old names must be updated; see the mapping table in `docs/metrics-reference.md`.

### Collector unreachable

```
WARNING  OTel collector at http://localhost:4318 is not reachable. Metrics will not be exported.
```

AIPerf logs a warning and continues the benchmark without streaming. The run itself is not affected. Check that your collector is running and the port is correct. If using Docker, ensure the container port is mapped to the host.

### MLflow timeout

```
WARNING  MLflow tracking server at http://localhost:5000 did not respond within 5s.
```

Verify the tracking server is running. For remote servers, check network connectivity and firewall rules. You can also set `MLFLOW_HTTP_REQUEST_TIMEOUT` to increase the timeout.

### Missing optional dependencies

```
ImportError: MLflow integration requires the 'mlflow' package. Install with: pip install "aiperf[mlflow]"
```

Install the required extras:

```bash
# OTel only
pip install "aiperf[otel]"

# MLflow only
pip install "aiperf[mlflow]"

# Both
pip install "aiperf[otel,mlflow]"
```
