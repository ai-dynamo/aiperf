---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Environment Variables
---

# Environment Variables

AIPerf can be configured using environment variables with the `AIPERF_` prefix.
All settings are organized into logical subsystems for better discoverability.

**Pattern:** `AIPERF_{SUBSYSTEM}_{SETTING_NAME}`

**Examples:**
```bash
export AIPERF_HTTP_CONNECTION_LIMIT=5000
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8
export AIPERF_ZMQ_RCVTIMEO=600000
```

> [!WARNING]
> Environment variable names, default values, and definitions are subject to change.
> These settings may be modified, renamed, or removed in future releases.

## CLI RUNNER

CLI runner post-run callback behavior. Controls whether OnComplete callback exceptions abort the run after all callbacks attempt or are isolated and logged. Default is isolated so that a single misbehaving callback (e.g. auto-plot in strict mode, third-party hook) cannot bypass the deliberate ``os._exit`` hang-protection that guards against multiprocessing/ZMQ teardown hangs in the parent process.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RAISE_ON_CALLBACK_ERROR` | `False` | — | When true, re-raise the first OnComplete callback exception after running all remaining callbacks but before os._exit. Provides a strict-mode contract where a callback raise propagates out of the runner. When false (default) the exception is logged with full traceback, the exit code is forced non-zero, and the process still terminates via os._exit so leftover ZMQ/multiprocessing state cannot hang the interpreter. |

## OPERATOR ENVIRONMENT

Root operator environment configuration. Loads from environment variables. Nested settings use their own prefixes.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DEFAULT_IMAGE` | `'nvcr.io/nvidia/aiperf:latest'` | — | Default container image for benchmark jobs |
| `AIPERF_JOB_TIMEOUT_SECONDS` | `0` | ≥ 0 | Job timeout in seconds (0 = no timeout) |
| `AIPERF_POD_RESTART_THRESHOLD` | `3` | ≥ 0, ≤ 100 | Pod restart count before emitting a warning event |
| `AIPERF_METRICS_PORT` | `9090` | ≥ 0, ≤ 65535 | Port for the Prometheus /metrics endpoint exposed by the kopf operator process. Set to 0 to disable. Scraped by ServiceMonitor. |
| `AIPERF_ENDPOINT_CHECK_TIMEOUT` | `10.0` | > 0, ≤ 300 | Seconds to wait for endpoint health check |
| `AIPERF_PREFLIGHT_TIMEOUT` | `30.0` | > 0, ≤ 120 | Seconds to wait for all pre-flight checks to complete |
| `AIPERF_CONFIGMAP_PROPAGATION_DELAY_SECONDS` | `10.0` | ≥ 0, ≤ 60 | Seconds to wait after creating the benchmark ConfigMap before creating the JobSet. Allows kubelet caches on worker nodes to sync the ConfigMap before pods start mounting it, preventing FailedMount races on first deployment with a freshly pulled image. |
| `AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED` | `False` | — | Enable results-server HTTP routes that mutate Kubernetes state. Defaults false so read-only results APIs remain exposed while POST routes fail closed unless an operator explicitly opts in. |
| `AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN` | `''` | — | Bearer token required by enabled results-server mutating routes. Leave empty to fail closed even when MUTATING_ROUTES_ENABLED is true. |
| `AIPERF_OPERATOR_CLUSTER_NAME` | `''` | — | Optional human-readable cluster name surfaced in the UI top banner (e.g. 'dgx-prod', 'kind-aiperf'). When unset the banner falls back to the Kubernetes server version. Set via AIPERF_OPERATOR_CLUSTER_NAME on the operator deployment. |
| `AIPERF_MONITOR` | — | — | Monitor timer settings |
| `AIPERF_RESULTS` | — | — | Results fetching and storage settings |
| `AIPERF_PROGRESS` | — | — | Progress-client retry settings (controller HTTP polling). |
| `AIPERF_SWEEP_CONTROLLER` | — | — | Sweep-controller pod settings (child-create collision handling). |
| `AIPERF_SERVICE` | — | — | Operator-service network identity (base URL stamped onto CR status). |
| `AIPERF_DASHBOARD` | — | — | Plotly Dashboard sidecar wiring. |

## ACCURACY

Accuracy benchmark settings. Tunables for the accuracy benchmark loaders. Currently only pins the LiveCodeBench dataset release so accuracy numbers are reproducible across runs without requiring source edits.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_ACCURACY_LCB_RELEASE_TAG` | `'v4_v5'` | — | LiveCodeBench dataset subset (HF config name) passed as the positional ``name`` arg to ``load_dataset("livecodebench/code_generation_lite", name, split="test", trust_remote_code=True)``. Pins which monthly snapshot LCB serves so accuracy numbers are reproducible across runs and branches. Default ``v4_v5`` matches lighteval's base subset; bump (e.g. to ``v6``) when the team rebaselines against a newer snapshot. The loader always passes ``trust_remote_code=True`` so LCB's dataset-loading script can execute on ``datasets`` v4+ (mirrors lighteval's reference opt-in). Consumed by ``aiperf.accuracy.benchmarks.lcb_codegeneration``. |

## APISERVER

API server settings. Controls the host and port of the API server.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_API_SERVER_HOST` | `'127.0.0.1'` | — | Host to bind the API server to |
| `AIPERF_API_SERVER_PORT` | `None` | ≥ 1, ≤ 65535 | Port to bind the API server to |
| `AIPERF_API_SERVER_CORS_ORIGINS` | `[]` | — | List of CORS origins to allow (empty = no CORS, ['*'] = all origins) |
| `AIPERF_API_SERVER_SHUTDOWN_TIMEOUT` | `5.0` | ≥ 1.0, ≤ 300.0 | Timeout in seconds for graceful API server shutdown before force-cancelling |
| `AIPERF_API_SERVER_GET_POD_STATES_TIMEOUT` | `2.0` | ≥ 0.1, ≤ 60.0 | Timeout in seconds for the GET_POD_STATES RPC issued by the /api/progress and /api/debug/* routers. Kept short so a slow / unresponsive SystemController falls back to the bus-fed cache rather than hanging the endpoint. |
| `AIPERF_API_SERVER_POST_COMPLETE_GRACE` | `5.0` | ≥ 0.0, ≤ 300.0 | Seconds the API listener stays open after a benchmark terminates so polling clients can observe the final status before the server shuts down. Set to 0 to skip the grace window and shut down immediately. |

## COMPRESSION

Compression settings for streaming file transfers. Controls chunk size and compression levels for zstd and gzip encodings used in dataset and results file transfers.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_COMPRESSION_CHUNK_SIZE` | `65536` | ≥ 1024, ≤ 1048576 | Chunk size in bytes for streaming compressed data (default: 64KB) |
| `AIPERF_COMPRESSION_ZSTD_LEVEL` | `3` | ≥ 1, ≤ 22 | Zstandard compression level (1=fastest, 22=best compression, default: 3) |
| `AIPERF_COMPRESSION_GZIP_LEVEL` | `6` | ≥ 1, ≤ 9 | Gzip compression level (1=fastest, 9=best compression, default: 6) |

## DAG

Settings for DAG benchmark mode (`dag_jsonl` input type).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DAG_FAIL_FAST` | `False` | — | When True, abort the whole run on the first DAG child error (cancel pending siblings, raise to PhaseRunner, terminate phase). Default False - the orchestrator counts the error in BranchStats.children_errored, releases the join slot, drains pending siblings, and continues the run. Set via AIPERF_DAG_FAIL_FAST=1 for strict CI assertions. |

## DASHBOARD

Plotly Dashboard sidecar wiring (operator + results-server). The dashboard is an opt-in third container in the operator Pod; these settings let other containers locate it.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DASHBOARD_PORT` | `0` | ≥ 0, ≤ 65535 | Pod-local HTTP port the dashboard sidecar listens on. 0 means the sidecar is disabled / absent. results-server uses this to reverse-proxy /dashboard/*; the operator uses it to fire fire-and-forget refresh POSTs after a benchmark completion claim. |
| `AIPERF_DASHBOARD_PROXY_ENABLED` | `False` | — | When true, results-server forwards /dashboard/* to the sidecar at localhost:PORT and the SPA shows the 'Plots ↗' top-nav entry. When false, /dashboard/* returns 503 and the link is hidden. Set independently from PORT so a misconfigured chart fails closed. |

## DATASET

Dataset loading and configuration. Controls timeouts and behavior for dataset loading operations, as well as memory-mapped dataset storage settings.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DATASET_CONFIGURATION_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for dataset configuration operations |
| `AIPERF_DATASET_MMAP_BASE_PATH` | `None` | — | Base path for memory-mapped dataset files. If None, uses system temp directory. Set to a shared filesystem path for Kubernetes mounted volumes. Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/ |
| `AIPERF_DATASET_DOWNLOAD_MAX_RETRIES` | `3` | ≥ 0, ≤ 20 | Maximum number of retries for dataset download in Kubernetes worker pods |
| `AIPERF_DATASET_DOWNLOAD_RETRY_DELAY` | `2.0` | ≥ 0.1, ≤ 60.0 | Initial delay in seconds between dataset download retries (doubles each retry) |
| `AIPERF_DATASET_PUBLIC_DATASET_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for public dataset loading operations |
| `AIPERF_DATASET_MEDIA_DOWNLOAD_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds per media URL download when inline encoding is required |
| `AIPERF_DATASET_MEDIA_DOWNLOAD_MAX_CONCURRENCY` | `10` | ≥ 1, ≤ 100 | Maximum number of concurrent media URL downloads |
| `AIPERF_DATASET_INLINE_RECORDS_WARN_THRESHOLD` | `500` | ≥ 1 | Soft warning threshold for the number of inline `records:` entries on a `FileDataset`. When total inline records exceed this value, the config loader logs a warning suggesting the user move the dataset to a JSONL file. No hard cap. |

## GPU

GPU telemetry collection configuration. Controls GPU metrics collection frequency, endpoint detection, and shutdown behavior. Metrics are collected from DCGM endpoints at the specified interval.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_GPU_COLLECTION_INTERVAL` | `0.333` | ≥ 0.01, ≤ 300.0 | GPU telemetry metrics collection interval in seconds (default: 333ms, ~3Hz) |
| `AIPERF_GPU_DEFAULT_DCGM_ENDPOINTS` | `['http://localhost:9400/metrics', 'http://localhost:9401/metrics']` | — | Default DCGM endpoint URLs to check for GPU telemetry (comma-separated string or JSON array) |
| `AIPERF_GPU_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for telemetry record export results processor |
| `AIPERF_GPU_FINAL_SCRAPE_GRACE_NS` | `666000000` | ≥ 0, ≤ 60000000000 | Grace window in nanoseconds appended to phase end_ns when computing the GPU energy-counter delta. Energy is scraped on a cadence (see COLLECTION_INTERVAL), so the trailing scrape often lands after the phase ends; this grace lets it be included while bounding the window so cooldown/idle samples and subsequent-phase samples don't leak into the delta. Default 666_000_000 ns ~= 2x the default 333 ms COLLECTION_INTERVAL; raise this if you also raise COLLECTION_INTERVAL. |
| `AIPERF_GPU_REACHABILITY_TIMEOUT` | `10` | ≥ 1, ≤ 300 | Timeout in seconds for checking GPU telemetry endpoint reachability during init |
| `AIPERF_GPU_SHUTDOWN_DELAY` | `5.0` | ≥ 1.0, ≤ 300.0 | Delay in seconds before shutting down GPU telemetry service to allow command response transmission |
| `AIPERF_GPU_THREAD_JOIN_TIMEOUT` | `5.0` | ≥ 1.0, ≤ 300.0 | Timeout in seconds for joining GPU telemetry collection threads during shutdown |

## HTTP

HTTP client socket and connection configuration. Controls low-level socket options, keepalive settings, DNS caching, and connection pooling for HTTP clients. These settings optimize performance for high-throughput streaming workloads. Video Generation Polling: For async video generation APIs that use job polling (e.g., SGLang /v1/videos), the poll interval is controlled by AIPERF_HTTP_VIDEO_POLL_INTERVAL. The max poll time uses the --request-timeout-seconds CLI argument.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_HTTP_CONNECTION_LIMIT` | `2500` | ≥ 1, ≤ 65000 | Maximum number of concurrent HTTP connections |
| `AIPERF_HTTP_KEEPALIVE_TIMEOUT` | `300` | ≥ 0, ≤ 10000 | HTTP connection keepalive timeout in seconds for connection pooling |
| `AIPERF_HTTP_SO_RCVBUF` | `10485760` | ≥ 1024 | Socket receive buffer size in bytes (default: 10MB for high-throughput streaming) |
| `AIPERF_HTTP_SO_RCVTIMEO` | `30` | ≥ 1, ≤ 100000 | Socket receive timeout in seconds |
| `AIPERF_HTTP_SO_SNDBUF` | `10485760` | ≥ 1024 | Socket send buffer size in bytes (default: 10MB for high-throughput streaming) |
| `AIPERF_HTTP_SO_SNDTIMEO` | `30` | ≥ 1, ≤ 100000 | Socket send timeout in seconds |
| `AIPERF_HTTP_TCP_KEEPCNT` | `1` | ≥ 1, ≤ 100 | Maximum number of keepalive probes to send before considering the connection dead |
| `AIPERF_HTTP_TCP_KEEPIDLE` | `60` | ≥ 1, ≤ 100000 | Time in seconds before starting TCP keepalive probes on idle connections |
| `AIPERF_HTTP_TCP_KEEPINTVL` | `30` | ≥ 1, ≤ 100000 | Interval in seconds between TCP keepalive probes |
| `AIPERF_HTTP_TCP_USER_TIMEOUT` | `30000` | ≥ 1, ≤ 1000000 | TCP user timeout in milliseconds (Linux-specific, detects dead connections) |
| `AIPERF_HTTP_TTL_DNS_CACHE` | `300` | ≥ 0, ≤ 1000000 | DNS cache TTL in seconds for aiohttp client sessions |
| `AIPERF_HTTP_FORCE_CLOSE` | `False` | — | Force close connections after each request |
| `AIPERF_HTTP_ENABLE_CLEANUP_CLOSED` | `False` | — | Enable cleanup of closed ssl connections |
| `AIPERF_HTTP_USE_DNS_CACHE` | `True` | — | Enable DNS cache |
| `AIPERF_HTTP_SSL_VERIFY` | `True` | — | Enable SSL certificate verification. Set to False to disable verification. WARNING: Disabling this is insecure and should only be used for testing in a trusted environment. |
| `AIPERF_HTTP_REQUEST_CANCELLATION_SEND_TIMEOUT` | `300.0` | ≥ 10.0, ≤ 3600.0 | Safety net timeout in seconds for waiting for HTTP request to be fully sent when request cancellation is enabled. Used as fallback when no explicit timeout is configured to prevent hanging indefinitely while waiting for the request to be written to the socket. |
| `AIPERF_HTTP_IP_VERSION` | `'4'` | — | IP version for HTTP socket connections. Options: '4' (AF_INET, default), '6' (AF_INET6), or 'auto' (AF_UNSPEC, system chooses). |
| `AIPERF_HTTP_TRUST_ENV` | `False` | — | Trust environment variables for HTTP client configuration. When enabled, aiohttp will read proxy settings from HTTP_PROXY, HTTPS_PROXY, and NO_PROXY environment variables. |
| `AIPERF_HTTP_VIDEO_POLL_INTERVAL` | `0.1` | ≥ 0.001, ≤ 10.0 | Interval in seconds between status polls for async video generation jobs. Lower values provide faster completion detection but increase server load. Applies to the aiohttp transport. |

## LOGGING

Logging system configuration. Controls multiprocessing log queue size and other logging behavior.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_LOGGING_QUEUE_MAXSIZE` | `1000` | ≥ 1, ≤ 1000000 | Maximum size of the multiprocessing logging queue |

## METRICS

Metrics collection and storage configuration. Controls metrics storage allocation and collection behavior.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_METRICS_ARRAY_INITIAL_CAPACITY` | `10000` | ≥ 100, ≤ 1000000 | Initial array capacity for metric storage dictionaries to minimize reallocation |
| `AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD` | `10.0` | ≥ 0.0, ≤ 100.0 | Percentage difference threshold for flagging discrepancies between API usage and client token counts (default: 10%) |
| `AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD` | `5.0` | ≥ 0.0, ≤ 100.0 | Percentage difference threshold for flagging discrepancies between requested and actual output sequence length (default: 5%) |
| `AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD` | `50` | ≥ 1 | Maximum absolute token threshold for OSL mismatch. The effective threshold is min(requested_osl * pct_threshold, this value). Makes threshold tighter for large OSL values (default: 50 tokens) |
| `AIPERF_METRICS_TDIGEST_COMPRESSION` | `500` | ≥ 20, ≤ 10000 | t-digest sketch compression for list-valued record metric aggregation. Higher = more centroids, tighter percentile accuracy, larger sketch. Default 500 measured to keep worst-case relative percentile error under 0.05% on 50M-sample workloads (40x under the 0.5% claimed accuracy band) at ~4 KB sketch size. |
| `AIPERF_METRICS_LIST_BACKEND` | `'ragged'` | — | Storage backend for list-valued RECORD metrics (today: only inter_chunk_latency). 'ragged' (default) keeps every value, enabling exact percentiles and ICL-aware throughput / tokens-in-flight sweep curves. 'tdigest' uses a bounded-memory crick.TDigest sketch (~4 KB regardless of sample count) — percentiles are approximate (<=0.05% relative error at default compression), and ICL-aware sweep curves silently fall back to their non-ICL equivalents that use only request-level (start_ns, generation_start_ns, end_ns) timing. Choose tdigest when records-manager pod memory at 1M+ request scale is the binding constraint. |

## MLFLOW

MLflow export configuration. Controls timeout behavior for post-run MLflow artifact uploads.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_MLFLOW_EXPORT_TIMEOUT_SECONDS` | `30.0` | ≥ 1.0, ≤ 600.0 | Timeout in seconds for the post-run MLflow export operation. If the MLflow tracking server is unreachable, the export will be abandoned after this duration rather than blocking indefinitely. |

## OPERATOR

Operator-service network identity. The operator Pod has three containers but only ONE FastAPI app: the ``results-server`` sidecar on ``resultsServer.port`` (8081 in the chart) hosts every ``/api/v1/*`` router (jobs, sweeps, results, config, admin, analytics, dashboard_proxy). The ``operator`` container on port 8080 runs kopf only — its sole HTTP surface is ``/healthz`` plus Prometheus ``/metrics``. So there is no separate "sweeps API URL" and "results API URL" — there is one base URL for everything, pointing at the results-server. Used when the operator stamps absolute URLs onto CR status (e.g. ``AIPerfSweep.status.apiUrl``, ``AIPerfSweep.status.runsTruncated.fetchURL``) that external clients dereference to fetch results, and when in-pod consumers (e.g. the sweep-controller's empty-summary fallback) need the operator's API endpoint.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_OPERATOR_BASE_URL` | `'http://aiperf-operator.aiperf-system:8081'` | — | Base URL (no trailing slash) for the operator's HTTP API. All ``/api/v1/*`` routers — jobs, sweeps, results, config, admin, analytics, dashboard_proxy — are served by the ``results-server`` container on this port; the operator container exposes only ``/healthz`` + ``/metrics`` on port 8080. Stamped onto ``AIPerfSweep.status.apiUrl`` and ``AIPerfSweep.status.runsTruncated.fetchURL`` so external clients can fetch per-sweep summaries; also consumed by the sweep-controller's per-child summary fallback. Override via ``AIPERF_OPERATOR_BASE_URL`` when the operator's Service+Namespace differ from the Helm chart defaults (e.g. a non-default ``Release.Name`` or an alternate namespace). |

## OPERATORMONITOR

Timer settings for the kopf monitor handler.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_OPERATOR_MONITOR_INTERVAL` | `10.0` | > 0, ≤ 3600 | Seconds between progress checks |
| `AIPERF_OPERATOR_MONITOR_INITIAL_DELAY` | `5.0` | ≥ 0, ≤ 300 | Seconds before first progress check after job creation |

## OPERATORPROGRESS

Operator progress-client retry settings. Used by ``aiperf.operator.progress_client.ProgressClient`` when polling the controller pod's HTTP progress API. Retries apply to transient failures (connection errors, retryable HTTP statuses); other errors propagate.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_OPERATOR_PROGRESS_MAX_RETRIES` | `3` | ≥ 0, ≤ 20 | Max retry attempts on transient progress-API failures. |
| `AIPERF_OPERATOR_PROGRESS_INITIAL_BACKOFF_SEC` | `0.5` | > 0, ≤ 60 | Initial backoff (seconds) between progress-API retries. |
| `AIPERF_OPERATOR_PROGRESS_BACKOFF_MULTIPLIER` | `2.0` | ≥ 1.0, ≤ 10.0 | Multiplicative backoff factor between progress-API retries. |

## OTEL

OpenTelemetry metrics streaming configuration. Controls buffering and flush behavior for OTLP metric streaming.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_OTEL_FLUSH_INTERVAL_SECONDS` | `2.0` | ≥ 0.1, ≤ 60.0 | Interval in seconds between periodic OTel metrics flushes |
| `AIPERF_OTEL_MAX_BATCH_RECORDS` | `500` | ≥ 1, ≤ 1000000 | Maximum number of metric records to include in a single OTel flush |
| `AIPERF_OTEL_MAX_BUFFERED_RECORDS` | `10000` | ≥ 1, ≤ 10000000 | Maximum number of buffered metric records before oldest records are dropped |
| `AIPERF_OTEL_REQUEST_TIMEOUT_SECONDS` | `10.0` | ≥ 0.1, ≤ 300.0 | Timeout in seconds for OTel collector HTTP requests |

## RECORD

Record processing and export configuration. Controls batch sizes, processor scaling, and progress reporting for record processing.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RECORD_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for record export results processor |
| `AIPERF_RECORD_EXPORT_FLUSH_INTERVAL` | `2.0` | ≥ 0.1, ≤ 300.0 | Maximum seconds record JSONL data may remain buffered before being flushed to disk |
| `AIPERF_RECORD_RAW_EXPORT_BATCH_SIZE` | `10` | ≥ 1, ≤ 1000000 | Batch size for raw record writer processor |
| `AIPERF_RECORD_INGEST_BATCH_SIZE` | `64` | ≥ 1, ≤ 1000000 | Batch size for record-processor to records-manager ingestion |
| `AIPERF_RECORD_INGEST_BATCH_FLUSH_INTERVAL` | `0.01` | ≥ 0.001, ≤ 300.0 | Maximum seconds metric records may remain buffered before ingestion flush |
| `AIPERF_RECORD_PROCESSOR_SCALE_FACTOR` | `4` | ≥ 1, ≤ 100 | Scale factor for number of record processors to spawn based on worker count. Formula: 1 record processor for every X workers |
| `AIPERF_RECORD_PROGRESS_REPORT_INTERVAL` | `2.0` | ≥ 0.1, ≤ 600.0 | Interval in seconds between records progress report messages |
| `AIPERF_RECORD_PROCESS_RECORDS_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for processing record results |
| `AIPERF_RECORD_CHECKPOINT_INTERVAL` | `30.0` | ≥ 1.0, ≤ 3600.0 | Interval in seconds between controller-side partial checkpoint writes |

## RESULTS

Results fetching and storage settings.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RESULTS_DIR` | `Path('/data')` | — | Base directory for storing benchmark results (mounted PVC) |
| `AIPERF_RESULTS_MAX_RETRIES` | `5` | ≥ 0, ≤ 50 | Max retries when fetching results from controller |
| `AIPERF_RESULTS_RETRY_DELAY` | `2.0` | ≥ 0, ≤ 60 | Seconds between result fetch retries |
| `AIPERF_RESULTS_TTL_DAYS` | `30` | ≥ 0, ≤ 3650 | Days to keep results before cleanup (0 = never clean) |
| `AIPERF_RESULTS_COMPRESS_ON_DISK` | `True` | — | Store downloaded result files as zstd-compressed (.zst) on disk |
| `AIPERF_RESULTS_RETAIN_RUNS` | `10` | ≥ 1, ≤ 10000 | Max per-run result dirs to keep under <namespace>/<name>/ before retention trimming. Applied after every successful completion; the just-written epoch is always protected from deletion. |
| `AIPERF_RESULTS_RETAIN_DAYS` | `0` | ≥ 0, ≤ 36500 | Age-based retention cap in days. 0 disables age policy. A run is deleted only when BOTH this age cap AND RETAIN_RUNS agree the run is outside the keep window; protect_epoch still wins. |
| `AIPERF_RESULTS_TRANSIENT_FETCH_RETRY_BUDGET_SEC` | `60.0` | ≥ 0.0, ≤ 600.0 | Wall-clock budget (seconds, measured from the completion-claim annotation timestamp) within which a transient HTTP fetch failure is converted to a kopf.TemporaryError so the next monitor tick retries via the orphan-claim recovery path. Past this budget the operator gives up and marks the AIPerfJob Failed with the ResultsFetchFailed condition. WHY: sub-second benchmarks can race the controller's post-export shutdown — the marker has been written and key files exist on the controller PVC, but the operator's HTTP fetch hits a connection-refused or empty list as the controller container terminates. Set 0 to disable retries. |
| `AIPERF_RESULTS_TRANSIENT_FETCH_RETRY_DELAY_SEC` | `5.0` | ≥ 0.5, ≤ 60.0 | Delay (seconds) passed to ``kopf.TemporaryError`` when retrying a transient results-fetch failure. Each retry runs through the orphan-claim recovery path on the next monitor tick. |

## SEARCHPLANNER

Adaptive-search planner tunables. Controls precision targets, warmup-phase injection, and request-count presets for the smooth-isotonic and monotonic SLA-saturation search planners. All values are read at planner-construction or iteration-mutate time, so changes take effect on the next search run.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SEARCH_PLANNER_SLA_PRECISION_DEFAULT` | `0.05` | > 0.0, &lt; 1.0 | Default SLA boundary search precision target. The bisection / smooth-isotonic bracket halts when (infeasible_min - feasible_max) / infeasible_min < this value, and the cliff detector requires bracket_gap > this * x_hi to report a cliff. 5% mirrors perf_analyzer's --binary-search default. |
| `AIPERF_SEARCH_PLANNER_DEFAULT_WARMUP_SECONDS` | `30.0` | > 0.0, ≤ 100000.0 | Smooth-isotonic SLA planner: default warmup phase duration in seconds injected into each iteration's cfg when ``cfg.sla_warmup_seconds`` is unset. Spec calls for max(30s, 3*inter-batch-time) but inter-batch-time is unknown at planner-time, so 30s is the safe floor. Must be strictly positive: zero defeats the cold-KV-cache rationale that motivates the floor. |
| `AIPERF_SEARCH_PLANNER_FIRST_PROBE_WARMUP_FLOOR` | `60.0` | > 0.0, ≤ 100000.0 | Smooth-isotonic SLA planner: minimum warmup duration in seconds for the first probe at each swept-dim value. Cold KV-cache and CUDA-graph compilation cost is largest the first time we hit a given concurrency, so floor that probe at 60s. Must be strictly positive: zero defeats the cold-KV-cache rationale. |
| `AIPERF_SEARCH_PLANNER_REPLICATE_WARMUP_FLOOR` | `15.0` | > 0.0, ≤ 100000.0 | Smooth-isotonic SLA planner: minimum warmup duration in seconds for replicate probes at an already-probed swept-dim value. Replicates reuse the warm KV-cache / CUDA-graph state, so a shorter warmup suffices. Must be strictly positive: zero defeats the floor. |
| `AIPERF_SEARCH_PLANNER_SLA_PRECISION_REQUESTS` | `{'tight': 10000, 'normal': 1000, 'coarse': 300}` | — | Mapping from ``cfg.sla_precision`` preset name to the ``phases.profiling.requests`` value injected when the user did not set ``requests`` explicitly on the profiling phase. Drives p99 CI width. Each value must be strictly positive — zero/negative request counts surface as iteration-time failures otherwise. Override via JSON, e.g. ``AIPERF_SEARCH_PLANNER_SLA_PRECISION_REQUESTS='{"tight": 20000}'``. |

## SERVERMETRICS

Server metrics collection configuration. Controls server metrics collection frequency, endpoint detection, and shutdown behavior. Metrics are collected from Prometheus-compatible endpoints at the specified interval. Use `--no-server-metrics` CLI flag to disable collection.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | `2.0` | ≥ 0.0, ≤ 30.0 | Time in seconds to continue collecting metrics after profiling completes, allowing server-side metrics to flush/finalize before shutting down (default: 2.0s) |
| `AIPERF_SERVER_METRICS_PROFILE_COMPLETE_RELAY_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 600.0 | Seconds the records manager waits for the system controller to relay PROFILE_COMPLETE to GPU telemetry, server metrics, and worker group manager. The relay blocks on the slowest scrape (Prometheus query + Parquet write), which can exceed 30s on contended clusters. A timeout here is non-fatal: the records manager logs a warning and continues to _process_results so the operator's results-fetch loop still sees a complete export. |
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | `0.333` | ≥ 0.001, ≤ 300.0 | Server metrics collection interval in seconds (default: 333ms, ~3Hz) |
| `AIPERF_SERVER_METRICS_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for server metrics jsonl writer export results processor |
| `AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT` | `10` | ≥ 1, ≤ 300 | Timeout in seconds for checking server metrics endpoint reachability during init |
| `AIPERF_SERVER_METRICS_SHUTDOWN_DELAY` | `5.0` | ≥ 1.0, ≤ 300.0 | Delay in seconds before shutting down server metrics service to allow command response transmission |

## SERVICE

Service lifecycle and inter-service communication configuration. Controls timeouts for service registration, startup, shutdown, command handling, connection probing, heartbeats, and profile operations.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SERVICE_COMMAND_RESPONSE_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 1000.0 | Timeout in seconds for command responses |
| `AIPERF_SERVICE_COMMS_REQUEST_TIMEOUT` | `90.0` | ≥ 1.0, ≤ 1000.0 | Timeout in seconds for requests from req_clients to rep_clients |
| `AIPERF_SERVICE_CONNECTION_PROBE_INTERVAL` | `0.1` | ≥ 0.1, ≤ 600.0 | Interval in seconds for connection probes while waiting for initial connection to the zmq message bus |
| `AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT` | `90.0` | ≥ 1.0, ≤ 100000.0 | Maximum time in seconds to wait for connection probe response while waiting for initial connection to the zmq message bus |
| `AIPERF_SERVICE_CREDIT_PROGRESS_REPORT_INTERVAL` | `2.0` | ≥ 1, ≤ 100000.0 | Interval in seconds between credit progress report messages |
| `AIPERF_SERVICE_DISABLE_UVLOOP` | `False` | — | Disable uvloop and use default asyncio event loop instead |
| `AIPERF_SERVICE_HEARTBEAT_INTERVAL` | `5.0` | ≥ 1.0, ≤ 100000.0 | Interval in seconds between heartbeat messages for component services |
| `AIPERF_SERVICE_HEARTBEAT_MISSED_THRESHOLD` | `3` | ≥ 1, ≤ 100 | Number of missed heartbeat intervals before a service is considered stale |
| `AIPERF_SERVICE_POD_FAILURE_ABORT_THRESHOLD_PERCENT` | `100` | ≥ 0, ≤ 100 | Percentage of worker pods that must fail before aborting the benchmark. For example, 50 means abort when 50%+ of worker pods have failed. Set to 100 to abort only when all workers are gone. Set to 0 to disable pod failure abort. |
| `AIPERF_SERVICE_PROCESS_MONITOR_INTERVAL` | `0.5` | ≥ 0.1, ≤ 30.0 | Interval in seconds between process liveness checks in MultiProcessServiceManager |
| `AIPERF_SERVICE_SHUTDOWN_PROPAGATION_DELAY` | `0.5` | ≥ 0.0, ≤ 10.0 | Delay in seconds after broadcasting shutdown command to allow message propagation before stopping services |
| `AIPERF_SERVICE_FAILURE_SHUTDOWN_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 300.0 | Wall-clock cap on the shutdown path inside AIPerfLifecycleMixin._fail. If cleanup (on_stop hooks, task cancellation) does not complete within this window after a failed on_init/on_start transition, the process hard-exits via os._exit(1). Prevents silent zombie containers when cleanup blocks on a cancelled C-ext call. |
| `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` | `600.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile configure command |
| `AIPERF_SERVICE_PROFILE_START_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile start command |
| `AIPERF_SERVICE_PROFILE_CANCEL_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile cancel command |
| `AIPERF_SERVICE_RAW_RECORD_UPLOAD_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 600.0 | Timeout in seconds to wait for worker pods to upload raw record files to the controller API after benchmark completion. |
| `AIPERF_SERVICE_REGISTRATION_INTERVAL` | `1.0` | ≥ 1.0, ≤ 100000.0 | Interval in seconds between registration attempts for component services |
| `AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS` | `10` | ≥ 1, ≤ 100000 | Maximum number of registration attempts before giving up |
| `AIPERF_SERVICE_REGISTRATION_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for service registration |
| `AIPERF_SERVICE_START_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for service start operations. Also bounds the per-phase wait for the first worker to register with the credit router before credit issuance begins; exceeding it fails the phase. |
| `AIPERF_SERVICE_TASK_CANCEL_TIMEOUT_SHORT` | `2.0` | ≥ 1.0, ≤ 100000.0 | Maximum time in seconds to wait for simple tasks to complete when cancelling |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED` | `True` | — | Enable event loop health monitoring to detect blocked event loops. When enabled, TimingManager and Worker services periodically check if the event loop is responsive and log warnings when latency exceeds the threshold. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_INTERVAL` | `0.25` | ≥ 0.05, ≤ 10.0 | Interval in seconds between event loop health checks (default: 250ms). The monitor sleeps for this duration and measures actual elapsed time to detect blocking. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS` | `25.0` | > 1.0, ≤ 10000.0 | Warning threshold in milliseconds for event loop latency (default: 25ms). If the actual sleep duration exceeds the expected duration by this amount, a warning is logged. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_STACKTRACE` | `False` | — | Enable watchdog thread that captures event loop thread stack traces when blocked. A daemon thread pings the event loop and captures sys._current_frames() when it fails to respond within the warning threshold. Adds minimal overhead (one thread per monitored service). |
| `AIPERF_SERVICE_HEALTH_ENABLED` | `False` | — | Enable the lightweight health server for Kubernetes liveness/readiness probes. When enabled, non-API services will start an HTTP server serving /healthz and /readyz endpoints. |
| `AIPERF_SERVICE_HEALTH_HOST` | `'127.0.0.1'` | — | Host to bind the health server to. Use '0.0.0.0' for Kubernetes deployments. |
| `AIPERF_SERVICE_HEALTH_PORT` | `8080` | ≥ 1, ≤ 65535 | Port for the health server HTTP endpoints (/healthz, /readyz). |
| `AIPERF_SERVICE_HEALTH_REQUEST_TIMEOUT` | `5.0` | ≥ 0.1, ≤ 60.0 | Timeout in seconds for reading health check HTTP requests. |
| `AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT` | `28000` | ≥ 1024, ≤ 65535 | Windows-only: starting port for the ZMQ IPC TCP-loopback fallback range. Per-endpoint ports are derived as ``base + (sha256_hash mod range)``. No-op on POSIX where ipc:// is used directly. |
| `AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE` | `20000` | ≥ 64, ≤ 60000 | Windows-only: size of the TCP-loopback port window for the ZMQ IPC fallback. Birthday-paradox collision probability for n sockets is ``1 - exp(-n*n/(2*range))``. Widen if AIPerf grows to many more sockets per run, or relocate via ``AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT`` if 28000-48000 conflicts. |

## STEADYSTATE

Steady-state detection configuration. Controls the automatic ramp detection algorithm parameters. CLI flags (``--steady-state``, ``--steady-state-start-pct``, etc.) take precedence over these environment variables.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_STEADY_STATE_BOOTSTRAP_ITERATIONS` | `None` | > 0 | Number of bootstrap iterations for boundary confidence intervals. None disables bootstrap. |
| `AIPERF_STEADY_STATE_MIN_WINDOW_PCT` | `10.0` | > 0.0, ≤ 100.0 | Minimum steady-state window size as percentage of total duration; below this, falls back to full range |

## SWEEPCONTROLLER

Sweep-controller pod settings. Used by the sweep-controller pod (`aiperf.sweep_controller.k8s_executor`) when creating child AIPerfJob CRs.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SWEEP_CONTROLLER_STALE_CHILD_DELETION_TIMEOUT_SECONDS` | `60.0` | > 0, ≤ 600 | Max seconds the sweep-controller will wait for a same-named AIPerfJob from a prior sweep run to finish cascade-deletion before raising ChildNameConflictError. Hit when a user deletes and recreates a sweep with the same name while old children are still terminating. |
| `AIPERF_SWEEP_CONTROLLER_STALE_CHILD_POLL_INTERVAL_SECONDS` | `2.0` | > 0, ≤ 30 | Poll interval (seconds) while waiting for a deleting same-named AIPerfJob to disappear. See STALE_CHILD_DELETION_TIMEOUT_SECONDS. |

## TIMING

Timing manager configuration. Controls timing-related settings for credit phase execution and scheduling.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 300.0 | Timeout in seconds for waiting for cancelled credits to drain after phase timeout |
| `AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL` | `0.1` | ≥ 0.01, ≤ 10.0 | Update interval in seconds for continuous rate ramping (default 0.1s = 100ms) |
| `AIPERF_TIMING_RECONCILIATION_INTERVAL` | `1.0` | ≥ 0.1, ≤ 60.0 | Interval in seconds between reconciliation ticks in the sticky credit router |

## TOKENIZER

Tokenizer pre-warm and loading configuration. Controls how the CLI parent pre-warms tokenizer caches before spawning AIPerf services. Pre-warming runs in subprocesses so the parent never imports the heavy native libraries (``transformers``, Rust-backed ``tokenizers``, ``tiktoken``).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_TOKENIZER_PRELOAD_TIMEOUT` | `120.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for the parent's tokenizer pre-warm phase. Bounds the total wall-clock time for all parallel subprocess pre-warms. On timeout, subprocesses are killed and AIPerf continues; child services may then download tokenizers themselves on first use. |
| `AIPERF_TOKENIZER_SKIP_PRELOAD` | `False` | — | Skip parent-process tokenizer cache pre-warming. Intended for test harnesses that replace tokenizer loading and must avoid forked prefetch subprocesses. Production defaults to preloading. |

## UI

User interface and dashboard configuration. Controls refresh rates, update thresholds, and notification behavior for the various UI modes (dashboard, tqdm, etc.).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_UI_LOG_REFRESH_INTERVAL` | `0.1` | ≥ 0.01, ≤ 100000.0 | Log viewer refresh interval in seconds (default: 10 FPS) |
| `AIPERF_UI_MIN_UPDATE_PERCENT` | `1.0` | ≥ 0.01, ≤ 100.0 | Minimum percentage difference from last update to trigger a UI update (for non-dashboard UIs) |
| `AIPERF_UI_NOTIFICATION_TIMEOUT` | `3` | ≥ 1, ≤ 100000 | Duration in seconds to display UI notifications before auto-dismissing |
| `AIPERF_UI_REALTIME_METRICS_INTERVAL` | `5.0` | ≥ 1.0, ≤ 1000.0 | Interval in seconds between real-time metrics messages |
| `AIPERF_UI_REALTIME_METRICS_ENABLED` | `False` | — | Enable real-time metrics collection and reporting despite UI type |
| `AIPERF_UI_SPINNER_REFRESH_RATE` | `0.1` | ≥ 0.1, ≤ 100.0 | Progress spinner refresh rate in seconds (default: 10 FPS) |
| `AIPERF_UI_STATUS_LOG_INTERVAL` | `30.0` | ≥ 1.0, ≤ 3600.0 | Interval in seconds between periodic status log messages when using --ui none |

## WANDB

Weights & Biases export configuration. Controls timeout behavior for the post-run W&B upload.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_WANDB_EXPORT_TIMEOUT_SECONDS` | `30.0` | ≥ 1.0, ≤ 600.0 | Timeout in seconds for the post-run Weights & Biases export operation. If the W&B backend is unreachable, the export will be abandoned after this duration rather than blocking indefinitely. |

## WORKER

Worker management and auto-scaling configuration. Controls worker pool sizing, health monitoring, load detection, and recovery behavior. The CPU_UTILIZATION_FACTOR is used in the auto-scaling formula: max_workers = max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP))

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_WORKER_CHECK_INTERVAL` | `1.0` | ≥ 0.1, ≤ 100000.0 | Interval in seconds between worker status checks by WorkerManager |
| `AIPERF_WORKER_CPU_UTILIZATION_FACTOR` | `0.75` | ≥ 0.1, ≤ 1.0 | Factor multiplied by CPU count to determine default max workers (0.0-1.0). Formula: max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP)) |
| `AIPERF_WORKER_ERROR_RECOVERY_TIME` | `3.0` | ≥ 0.1, ≤ 1000.0 | Time in seconds from last error before worker is considered healthy again |
| `AIPERF_WORKER_HEALTH_CHECK_INTERVAL` | `2.0` | ≥ 0.1, ≤ 1000.0 | Interval in seconds between worker health check messages |
| `AIPERF_WORKER_HIGH_LOAD_CPU_USAGE` | `85.0` | ≥ 50.0, ≤ 100.0 | CPU usage percentage threshold for considering a worker under high load |
| `AIPERF_WORKER_HIGH_LOAD_RECOVERY_TIME` | `5.0` | ≥ 0.1, ≤ 1000.0 | Time in seconds from last high load before worker is considered recovered |
| `AIPERF_WORKER_MAX_WORKERS_CAP` | `32` | ≥ 1, ≤ 10000 | Absolute maximum number of workers to spawn, regardless of CPU count |
| `AIPERF_WORKER_STALE_TIME` | `10.0` | ≥ 0.1, ≤ 1000.0 | Time in seconds from last status report before worker is considered stale |
| `AIPERF_WORKER_STATUS_SUMMARY_INTERVAL` | `0.5` | ≥ 0.1, ≤ 1000.0 | Interval in seconds between worker status summary messages |
| `AIPERF_WORKER_DEFAULT_WORKERS_PER_POD` | `10` | ≥ 1, ≤ 100 | Default number of worker subprocesses per Kubernetes worker pod. Each pod downloads the dataset once and shares it across workers via mmap. |

## ZMQ

ZMQ socket and communication configuration. Controls ZMQ socket timeouts, keepalive settings, retry behavior, and concurrency limits. These settings affect reliability and performance of the internal message bus.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_ZMQ_CONTEXT_TERM_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for terminating the ZMQ context during shutdown |
| `AIPERF_ZMQ_PULL_YIELD_INTERVAL` | `10` | ≥ 0, ≤ 1000000 | Yield to the event loop after every N received messages from ZMQ PULL clients. Prevents event loop starvation during message bursts. 0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc. |
| `AIPERF_ZMQ_REPLY_YIELD_INTERVAL` | `10` | ≥ 0, ≤ 1000000 | Yield to the event loop after every N received requests from ZMQ ROUTER reply clients. Prevents event loop starvation during request bursts. 0 disables yielding, 1 yields after every request, 10 yields every 10 requests, etc. |
| `AIPERF_ZMQ_REQUEST_YIELD_INTERVAL` | `10` | ≥ 0, ≤ 1000000 | Yield to the event loop after every N received responses from ZMQ DEALER request clients. Prevents event loop starvation during response bursts. 0 disables yielding, 1 yields after every response, 10 yields every 10 responses, etc. |
| `AIPERF_ZMQ_STREAMING_DEALER_YIELD_INTERVAL` | `10` | ≥ 0, ≤ 1000000 | Yield to the event loop after every N received messages from ZMQ streaming DEALER clients. Prevents event loop starvation during message bursts. 0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc. |
| `AIPERF_ZMQ_STREAMING_ROUTER_YIELD_INTERVAL` | `10` | ≥ 0, ≤ 1000000 | Yield to the event loop after every N received messages from ZMQ streaming ROUTER clients. Prevents event loop starvation during message bursts. 0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc. |
| `AIPERF_ZMQ_SUB_YIELD_INTERVAL` | `10` | ≥ 0, ≤ 1000000 | Yield to the event loop after every N received messages from ZMQ SUB clients. Prevents event loop starvation during message bursts. 0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc. |
| `AIPERF_ZMQ_PULL_MAX_CONCURRENCY` | `100000` | ≥ 1, ≤ 10000000 | Maximum concurrency for ZMQ PULL clients |
| `AIPERF_ZMQ_PUSH_MAX_RETRIES` | `2` | ≥ 1, ≤ 100 | Maximum number of retry attempts when pushing messages to ZMQ PUSH socket |
| `AIPERF_ZMQ_PUSH_RETRY_DELAY` | `0.1` | ≥ 0.1, ≤ 1000.0 | Delay in seconds between retry attempts for ZMQ PUSH operations |
| `AIPERF_ZMQ_RCVTIMEO` | `300000` | ≥ 1, ≤ 10000000 | Socket receive timeout in milliseconds (default: 5 minutes) |
| `AIPERF_ZMQ_SNDTIMEO` | `300000` | ≥ 1, ≤ 10000000 | Socket send timeout in milliseconds (default: 5 minutes) |
| `AIPERF_ZMQ_TCP_KEEPALIVE_IDLE` | `60` | ≥ 1, ≤ 100000 | Time in seconds before starting TCP keepalive probes on idle ZMQ connections |
| `AIPERF_ZMQ_TCP_KEEPALIVE_INTVL` | `10` | ≥ 1, ≤ 100000 | Interval in seconds between TCP keepalive probes for ZMQ connections |
| `AIPERF_ZMQ_EVENT_BUS_PROXY_FRONTEND_PORT` | `5663` | ≥ 1, ≤ 65535 | Default TCP port for the event-bus XPUB/XSUB proxy frontend (producers connect here). Single source of truth for the non-k8s comm configs (TCP, dual-bind); k8s pod manifests pull the same value via ``K8sEnvironment.PORTS.EVENT_BUS_PROXY_PUB_FRONTEND`` (defaults match). |
| `AIPERF_ZMQ_EVENT_BUS_PROXY_BACKEND_PORT` | `5664` | ≥ 1, ≤ 65535 | Default TCP port for the event-bus XPUB/XSUB proxy backend (subscribers connect here). See ``EVENT_BUS_PROXY_FRONTEND_PORT``. |

## DEV

Development and debugging configuration. Controls developer-focused features like debug logging, profiling, and internal metrics. These settings are typically disabled in production environments.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DEV_DEBUG_SERVICES` | `None` | — | List of services to enable DEBUG logging for (comma-separated or multiple flags) |
| `AIPERF_DEV_ENABLE_YAPPI` | `False` | — | Enable yappi profiling (Yet Another Python Profiler) for performance analysis. Requires 'pip install yappi snakeviz' |
| `AIPERF_DEV_MEMORY_PROFILE_ENABLED` | `False` | — | Enable memory profiling using tracemalloc. Logs memory usage and top allocators periodically. |
| `AIPERF_DEV_MEMORY_PROFILE_INTERVAL` | `10.0` | ≥ 1.0, ≤ 3600.0 | Interval in seconds between memory profile snapshots when profiling is enabled. |
| `AIPERF_DEV_MEMORY_PROFILE_TOP_N` | `10` | ≥ 1, ≤ 100 | Number of top memory allocators to log in each snapshot. |
| `AIPERF_DEV_MODE` | `False` | — | Enable AIPerf Developer mode for internal metrics and debugging |
| `AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS` | `False` | — | [Developer use only] Show experimental metrics in output (requires DEV_MODE) |
| `AIPERF_DEV_SHOW_INTERNAL_METRICS` | `False` | — | [Developer use only] Show internal and hidden metrics in output (requires DEV_MODE) |
| `AIPERF_DEV_TRACE_SERVICES` | `None` | — | List of services to enable TRACE logging for (comma-separated or multiple flags) |
