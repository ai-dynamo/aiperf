<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

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

## CONFIG

Configuration file paths for distributed deployments. Controls paths to configuration files loaded by services running in containers. These are primarily used by `aiperf service` when running in Kubernetes.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_CONFIG_SERVICE_FILE` | `None` | — | Path to service configuration JSON/YAML file. Default: /etc/aiperf/service_config.json in Kubernetes deployments. |
| `AIPERF_CONFIG_USER_FILE` | `None` | — | Path to user configuration JSON/YAML file. Default: /etc/aiperf/user_config.json in Kubernetes deployments. |

## DATASET

Dataset loading and configuration. Controls timeouts and behavior for dataset loading operations, as well as memory-mapped dataset storage settings.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DATASET_CONFIGURATION_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for dataset configuration operations |
| `AIPERF_DATASET_MMAP_BASE_PATH` | `None` | — | Base path for memory-mapped dataset files. If None, uses system temp directory. Set to a shared filesystem path for Kubernetes mounted volumes. Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/ |
| `AIPERF_DATASET_PUBLIC_DATASET_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for public dataset loading operations |

## GPU

GPU telemetry collection configuration. Controls GPU metrics collection frequency, endpoint detection, and shutdown behavior. Metrics are collected from DCGM endpoints at the specified interval.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_GPU_COLLECTION_INTERVAL` | `0.333` | ≥ 0.01, ≤ 300.0 | GPU telemetry metrics collection interval in seconds (default: 333ms, ~3Hz) |
| `AIPERF_GPU_DEFAULT_DCGM_ENDPOINTS` | `['http://localhost:9400/metrics', 'http://localhost:9401/metrics']` | — | Default DCGM endpoint URLs to check for GPU telemetry (comma-separated string or JSON array) |
| `AIPERF_GPU_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for telemetry record export results processor |
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

## RECORD

Record processing and export configuration. Controls batch sizes, processor scaling, and progress reporting for record processing.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RECORD_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for record export results processor |
| `AIPERF_RECORD_RAW_EXPORT_BATCH_SIZE` | `10` | ≥ 1, ≤ 1000000 | Batch size for raw record writer processor |
| `AIPERF_RECORD_PROCESSOR_SCALE_FACTOR` | `4` | ≥ 1, ≤ 100 | Scale factor for number of record processors to spawn based on worker count. Formula: 1 record processor for every X workers |
| `AIPERF_RECORD_PROGRESS_REPORT_INTERVAL` | `2.0` | ≥ 0.1, ≤ 600.0 | Interval in seconds between records progress report messages |
| `AIPERF_RECORD_PROCESS_RECORDS_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for processing record results |

## SERVERMETRICS

Server metrics collection configuration. Controls server metrics collection frequency, endpoint detection, and shutdown behavior. Metrics are collected from Prometheus-compatible endpoints at the specified interval. Use `--no-server-metrics` CLI flag to disable collection.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | `2.0` | ≥ 0.0, ≤ 30.0 | Time in seconds to continue collecting metrics after profiling completes, allowing server-side metrics to flush/finalize before shutting down (default: 2.0s) |
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
| `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile configure command |
| `AIPERF_SERVICE_PROFILE_START_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile start command |
| `AIPERF_SERVICE_PROFILE_CANCEL_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile cancel command |
| `AIPERF_SERVICE_REGISTRATION_INTERVAL` | `1.0` | ≥ 1.0, ≤ 100000.0 | Interval in seconds between registration attempts for component services |
| `AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS` | `10` | ≥ 1, ≤ 100000 | Maximum number of registration attempts before giving up |
| `AIPERF_SERVICE_REGISTRATION_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for service registration |
| `AIPERF_SERVICE_START_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for service start operations |
| `AIPERF_SERVICE_TASK_CANCEL_TIMEOUT_SHORT` | `2.0` | ≥ 1.0, ≤ 100000.0 | Maximum time in seconds to wait for simple tasks to complete when cancelling |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED` | `True` | — | Enable event loop health monitoring to detect blocked event loops. When enabled, TimingManager and Worker services periodically check if the event loop is responsive and log warnings when latency exceeds the threshold. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_INTERVAL` | `0.25` | ≥ 0.05, ≤ 10.0 | Interval in seconds between event loop health checks (default: 250ms). The monitor sleeps for this duration and measures actual elapsed time to detect blocking. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS` | `10.0` | > 1.0, ≤ 10000.0 | Warning threshold in milliseconds for event loop latency (default: 10ms). If the actual sleep duration exceeds the expected duration by this amount, a warning is logged. |
| `AIPERF_SERVICE_HEALTH_ENABLED` | `False` | — | Enable the lightweight health server for Kubernetes liveness/readiness probes. When enabled, non-API services will start an HTTP server serving /healthz and /readyz endpoints. |
| `AIPERF_SERVICE_HEALTH_HOST` | `'127.0.0.1'` | — | Host to bind the health server to. Use '0.0.0.0' for Kubernetes deployments. |
| `AIPERF_SERVICE_HEALTH_PORT` | `8080` | ≥ 1, ≤ 65535 | Port for the health server HTTP endpoints (/healthz, /readyz). |
| `AIPERF_SERVICE_HEALTH_REQUEST_TIMEOUT` | `5.0` | ≥ 0.1, ≤ 60.0 | Timeout in seconds for reading health check HTTP requests. |

## TIMING

Timing manager configuration. Controls timing-related settings for credit phase execution and scheduling.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 300.0 | Timeout in seconds for waiting for cancelled credits to drain after phase timeout |
| `AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL` | `0.1` | ≥ 0.01, ≤ 10.0 | Update interval in seconds for continuous rate ramping (default 0.1s = 100ms) |

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

## DEV

Development and debugging configuration. Controls developer-focused features like debug logging, profiling, and internal metrics. These settings are typically disabled in production environments.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DEV_DEBUG_SERVICES` | `None` | — | List of services to enable DEBUG logging for (comma-separated or multiple flags) |
| `AIPERF_DEV_ENABLE_YAPPI` | `False` | — | Enable yappi profiling (Yet Another Python Profiler) for performance analysis. Requires 'pip install yappi snakeviz' |
| `AIPERF_DEV_MODE` | `False` | — | Enable AIPerf Developer mode for internal metrics and debugging |
| `AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS` | `False` | — | [Developer use only] Show experimental metrics in output (requires DEV_MODE) |
| `AIPERF_DEV_SHOW_INTERNAL_METRICS` | `False` | — | [Developer use only] Show internal and hidden metrics in output (requires DEV_MODE) |
| `AIPERF_DEV_TRACE_SERVICES` | `None` | — | List of services to enable TRACE logging for (comma-separated or multiple flags) |
