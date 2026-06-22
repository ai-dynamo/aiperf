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

## AGENTX

Settings for the InferenceX AgentX scenario family. Controls runtime detection knobs for the agentx scenario, currently the substring allowlist used to classify a server response as a context-overflow error (RFC 2026-04-26 §7).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS` | `['context length', 'maximum context', 'context_length_exceeded', 'prompt is too long']` | — | Case-insensitive substring allowlist used to classify a server error response as a context-overflow event. Matched against the raw response body and the OpenAI-style nested 'error.message' field. Extend via AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS to support additional inference-server vocabularies (vLLM, TGI, TensorRT-LLM, ...). Empty list disables runtime detection. |
| `AIPERF_AGENTX_CONTEXT_OVERFLOW_RATE_LIMIT` | `0.01` | ≥ 0.0, ≤ 1.0 | Strict upper bound on the per-run context-overflow rate (context_overflow_count / total_responses) before a scenario submission is flipped to submission_valid=false with reason 'context_overflow_rate_exceeded'. Default 0.01 (1%) matches the scenario spec RFC 2026-04-26 §7. Comparison is strictly greater-than: rate exactly equal to the limit is accepted. Has no effect on non-scenario runs (no --scenario flag) or runs with zero responses. |
| `AIPERF_AGENTX_RECYCLE_GUARD_MAX_WINDOW` | `1000000` | ≥ 1 | Maximum number of recently-recycled root correlation_ids retained by AgenticReplayStrategy's double-recycle guard (which raises if a final-turn credit return is delivered twice and would re-spawn a session). Without a bound the guard retains one entry per recycled session for the entire PROFILING phase -- hundreds of MB of unreclaimable memory on long, high-throughput durability ramps. Oldest entries are evicted FIFO once the window is full; a duplicate delivered after this many intervening recycles is no longer caught. Duplicate deliveries are near-immediate in practice, so the default window is far larger than any real gap; raise it for very high concurrency. |

## APISERVER

API server settings. Controls the host and port of the API server.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_API_SERVER_HOST` | `'127.0.0.1'` | — | Host to bind the API server to |
| `AIPERF_API_SERVER_PORT` | `None` | ≥ 1, ≤ 65535 | Port to bind the API server to |
| `AIPERF_API_SERVER_CORS_ORIGINS` | `[]` | — | List of CORS origins to allow (empty = no CORS, ['*'] = all origins) |
| `AIPERF_API_SERVER_SHUTDOWN_TIMEOUT` | `5.0` | ≥ 1.0, ≤ 300.0 | Timeout in seconds for graceful API server shutdown before force-cancelling |

## COMPRESSION

Compression settings for streaming file transfers. Controls chunk size and compression levels for zstd and gzip encodings used in dataset and results file transfers.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_COMPRESSION_CHUNK_SIZE` | `65536` | ≥ 1024, ≤ 1048576 | Chunk size in bytes for streaming compressed data (default: 64KB) |
| `AIPERF_COMPRESSION_ZSTD_LEVEL` | `3` | ≥ 1, ≤ 22 | Zstandard compression level (1=fastest, 22=best compression, default: 3) |
| `AIPERF_COMPRESSION_GZIP_LEVEL` | `6` | ≥ 1, ≤ 9 | Gzip compression level (1=fastest, 9=best compression, default: 6) |

## CONFIG

Configuration file paths for distributed deployments. Controls paths to configuration files loaded by services running in containers. These are primarily used by `aiperf service` when running in Kubernetes.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_CONFIG_SERVICE_FILE` | `None` | — | Path to service configuration JSON/YAML file. Default: /etc/aiperf/service_config.json in Kubernetes deployments. |
| `AIPERF_CONFIG_USER_FILE` | `None` | — | Path to user configuration JSON/YAML file. Default: /etc/aiperf/user_config.json in Kubernetes deployments. |

## DAG

DAG branch orchestration configuration. Controls runtime behaviour of ``BranchOrchestrator`` for FORK-mode DAG benchmarks (``dag_jsonl`` input type).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DAG_FAIL_FAST` | `False` | — | When True, a single child error aborts the parent and every orphan sibling under the same DAG branch (releases sticky refcounts and calls issuer.abort_session). When False (default), a child error is treated as leaf-reached for join counting and the parent's join still fires. Inspected once at BranchOrchestrator construction. |

## DATASET

Dataset loading and configuration. Controls timeouts and behavior for dataset loading operations, as well as memory-mapped dataset storage settings.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_DATASET_CONFIGURATION_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for dataset configuration operations |
| `AIPERF_DATASET_MMAP_BASE_PATH` | `None` | — | Base path for memory-mapped dataset files. If None, uses system temp directory. Set to a shared filesystem path for Kubernetes mounted volumes. Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/ |
| `AIPERF_DATASET_MMAP_CACHE_ENABLED` | `True` | — | If True, AIPerf reuses memory-mapped dataset files across runs whose input bytes, tokenizer identity, and prompt/input settings are byte-identical. Set to False to force every run to re-tokenize and re-write its mmap files. Cache misses still produce byte-identical mmap files to a non-cached run. |
| `AIPERF_DATASET_MMAP_CACHE_DIR` | `None` | — | Directory holding the content-addressed mmap cache. If None, defaults to ~/.cache/aiperf/dataset_mmap. Each cache entry lives under <dir>/<key>/ and contains dataset.dat, index.dat, manifest.json, and (when produced) inputs.json. No automatic eviction is implemented yet -- delete the directory to reclaim disk. |
| `AIPERF_DATASET_PUBLIC_DATASET_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for public dataset loading operations |
| `AIPERF_DATASET_MEDIA_DOWNLOAD_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds per media URL download when inline encoding is required |
| `AIPERF_DATASET_MEDIA_DOWNLOAD_MAX_CONCURRENCY` | `10` | ≥ 1, ≤ 100 | Maximum number of concurrent media URL downloads |
| `AIPERF_DATASET_WEKA_PARALLEL_WORKERS` | `0` | ≥ 0, ≤ 256 | Number of worker processes for WekaTraceLoader parallel reconstruction. 0 = auto (min(cpu_count - 1, 16, num_traces)). Set to 1 to force serial reconstruction. |
| `AIPERF_DATASET_WEKA_PARALLEL_THRESHOLD` | `8` | ≥ 1, ≤ 100000 | Minimum number of parent traces required before WekaTraceLoader switches to the multi-process parallel reconstruction path. Below this, the in-process serial path is used (Pool startup overhead exceeds the speedup for tiny corpora). |
| `AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES` | `False` | — | When True, WekaTraceLoader emits user-only deltas and selects ConversationContextMode.DELTAS_WITHOUT_RESPONSES so the worker threads the server's live assistant response back into the session's turn_list between turns. Preserves the server's just-generated KV blocks across turn boundaries (real cache-hit rate) at the cost of hash-id fidelity past turn 0 (server-generated assistant length will not exactly match the trace's recorded output_length, so subsequent user-turn block alignment drifts from the trace's hash_ids). Default False preserves the pre-canned-assistant behavior that matches recorded hash_ids byte-for-byte. |
| `AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS` | `True` | — | When True (default), WekaTraceLoader runs hash_id LCP chain detection at both layers: untagged agent fan-outs recorded as flat top-level requests split into per-agent child conversations (::fa:NNN), and each subagent entry's inner requests split into per-context-chain children (::sa:<agent_id> plus :fa:NNN siblings), all with SPAWN/SPAWN_JOIN linkage so replay reproduces the recorded concurrency. Set to False to disable detection at both layers: all top-level requests serialize into one root conversation and each subagent emits exactly one child with its inner requests in time order. Detected chains at both layers are further split into genuine agents and auxiliary one-shot sidecars (top-level ::fa: vs ::aux:; subagent overflow :fa: vs :aux:) per WEKA_AUX_MAX_REQUESTS / WEKA_AUX_ISL_RATIO / WEKA_AUX_ISL_FLOOR. |
| `AIPERF_DATASET_WEKA_TOOL_SHAPED_MESSAGES` | `False` | — | When True, WekaTraceLoader emits the OpenAI tool-call wire shape for turns classified as tool-result continuations: the same-delta assistant message gains a synthetic tool_calls entry and the turn's new input is sent as a role='tool' message instead of plain user text (content unchanged). Exercises the server's tool-message chat-template path at the cost of exact ISL fidelity (tool messages tokenize differently than plain user text). Only turns with a recorded tool signal (input_types / prior stop) shape; legacy traces are unaffected. Default False keeps the byte-exact plain-user replay shape. |
| `AIPERF_DATASET_WEKA_SEAM_MAX_GAP_SECONDS` | `3600.0` | ≥ 0.0 | LCP chain-detection seam guard: the maximum wall-clock gap (seconds) between a chain's last request and a candidate continuation before that continuation is only accepted when it also keeps enough of the prior context (see WEKA_SEAM_MIN_OVERLAP_RATIO). A genuine context compaction continues promptly (seconds to minutes), so a low-overlap join hours later is treated as a distinct session that merely shares a base prefix and is spawned as its own conversation instead of being stitched onto the chain (which would fabricate a multi-hour intra-conversation idle gap). The guard fires only when BOTH this gap is exceeded AND overlap is below the ratio, so prompt compactions at any overlap and verbatim long-gap resumes at high overlap are preserved. Raise toward infinity to disable the temporal half of the guard. |
| `AIPERF_DATASET_WEKA_SEAM_MIN_OVERLAP_RATIO` | `0.5` | ≥ 0.0, ≤ 1.0 | LCP chain-detection seam guard: the minimum shared-prefix ratio (continuation's fork depth / the chain tail's block count) for a far-future continuation to still be accepted as the same agent. Below this, a continuation past WEKA_SEAM_MAX_GAP_SECONDS is spawned as a new conversation rather than spliced on. Corpus data is bimodal -- real compactions and verbatim resumes keep >=94% of the prefix, while coincidental base-prefix mis-merges keep <50% -- so 0.5 sits in a wide safe valley. Set to 0.0 to disable the overlap half of the guard. |
| `AIPERF_DATASET_WEKA_AUX_MAX_REQUESTS` | `1` | ≥ 0 | Auxiliary (sidecar) classification: a detected worker chain with at most this many requests is eligible to be reclassified as an auxiliary one-shot call -- a tool-issued sidecar (web fetch/search summary, title generation, a classifier) rather than a sustained agent -- when it also passes the WEKA_AUX_ISL_* size test. Applies to both top-level flat chains (::fa: -> ::aux:) and a subagent's nested-LCP overflow (:fa: -> :aux:). Corpus sidecars are overwhelmingly single-request, so the default is 1. Set to 0 to disable aux classification (every worker chain keeps its agent tag). Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True. |
| `AIPERF_DATASET_WEKA_AUX_ISL_RATIO` | `0.1` | ≥ 0.0 | Auxiliary (sidecar) classification: an aux-eligible chain (see WEKA_AUX_MAX_REQUESTS) is reclassified to a sidecar only when its first request's input length is below max(WEKA_AUX_ISL_FLOOR, this ratio * the enclosing main chain's peak input length -- the trace's for flat chains, the subagent's for overflow). The ratio catches calls small relative to a large conversation's accumulated context; the floor catches them in absolute terms. Sidecars start from a fresh few-thousand-token context vs the agent's tens-to-hundreds of thousands. |
| `AIPERF_DATASET_WEKA_AUX_ISL_FLOOR` | `16384` | ≥ 0 | Auxiliary (sidecar) classification: absolute input-length floor (tokens) for the aux size test (see WEKA_AUX_ISL_RATIO). A chain whose first-request input length is below max(this, ratio * main peak ISL) is treated as an auxiliary one-shot sidecar. Keeps small fresh-context calls classified as sidecars even when the enclosing conversation is itself small. |
| `AIPERF_DATASET_WEKA_AUX_CROSS_MODEL` | `True` | — | Auxiliary (sidecar) classification: when True (default), an aux-eligible chain (<= WEKA_AUX_MAX_REQUESTS requests) whose first request runs on a different model than its enclosing main chain is treated as a sidecar regardless of input length. An agent does not switch models for its own reasoning, so a one-shot on a different model is a tool-internal call -- e.g. a Haiku WebFetch summary fired by an Opus agent, which can carry a large fetched-page payload and so escape the WEKA_AUX_ISL_* size test. Set to False to classify purely by size. |
| `AIPERF_DATASET_WEKA_AUX_REDUCTION_OSL_MAX` | `4000` | ≥ 0 | Auxiliary (sidecar) classification, reduction arm: a single-request worker chain on the SAME model as its enclosing main chain is reclassified to an auxiliary one-shot when its output length is in (0, this) tokens AND its input length is at least WEKA_AUX_ISL_FLOOR AND its input/output ratio exceeds WEKA_AUX_REDUCTION_RATIO. This catches large-input/short-output reductions (context compaction, subagent-result summaries, tool-output digests) that the size and cross-model arms miss because they are same-model and large. The bound separates a bounded summary from generative agent output (a real agent emits long completions); corpus reductions cap well below 4k output across every capture. Reductions are emitted as ::aux:red: (still aux, distinguishable from fetch/size sidecars). Set to 0 to disable the reduction arm. Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True. |
| `AIPERF_DATASET_WEKA_AUX_REDUCTION_RATIO` | `20.0` | ≥ 0.0 | Auxiliary (sidecar) classification, reduction arm: the minimum input-to-output token ratio for a same-model single-request large-input chain to be treated as a reduction sidecar (see WEKA_AUX_REDUCTION_OSL_MAX). A reduction consumes a large body and emits a short summary, so input/output is high (corpus median ~120); 20 is a conservative floor that still excludes balanced request/response calls. Only applies when WEKA_AUX_REDUCTION_OSL_MAX > 0. |
| `AIPERF_DATASET_WEKA_WORKER_GROUP_MIN` | `3` | ≥ 0 | Parallel worker-group tagging: a coordinated parallel fan-out must BOTH share a deep spawned context AND run concurrently. Workers that forked from shared context (fork depth > 0) are first scoped by their fork point (the parent request they branched off), then within each scope split into connected components of overlapping active [t0, t1) intervals; a component with at least this many members is emitted as ::wg:{group}_{member} (group = the concurrent fan-out, member = index by start time) instead of the generic ::fa: agent marker. The fork-point scope keeps unrelated fan-outs apart (pure interval overlap bridges a busy trace into one blob); the overlap split drops members that share the fork point but never run concurrently. This isolates genuine parallel sub-agent fan-out (the dominant agent population) from solo agents, unlike keying on the first context block (shared by ~every worker all session). Auxiliary chains are classified first, so a one-shot sidecar never becomes a worker-group member. Set to 0 to disable worker-group tagging (parallel workers keep the generic ::fa: tag). Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True. |

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
| `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` | `False` | — | Also send X-Session-ID with the stable X-Correlation-ID value. Use this when an external router requires a session-affinity header. |
| `AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID` | `False` | — | Also send X-SMG-Routing-Key with the stable X-Correlation-ID value. Use this with the SGLang Model Gateway manual routing policy. |
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
| `AIPERF_METRICS_LIST_BACKEND` | `'ragged'` | — | Storage backend for list-valued RECORD metrics (today: only inter_chunk_latency). 'ragged' (default) keeps every value, enabling exact percentiles and ICL-aware throughput / tokens-in-flight sweep curves. 'tdigest' uses a bounded-memory crick.TDigest sketch (~4 KB regardless of sample count) — percentiles are approximate (≤0.05% relative error at default compression), and ICL-aware sweep curves silently fall back to their non-ICL equivalents that use only request-level (start_ns, generation_start_ns, end_ns) timing. Choose tdigest when records-manager pod memory at 1M+ request scale is the binding constraint. |
| `AIPERF_METRICS_EXPORT_FLUSH_INTERVAL` | `1.0` | ≥ 0.05, ≤ 60.0 | Periodic flush interval (seconds) for buffered JSONL stream exporters (raw record writer, record export, gpu/server-metrics JSONL writers). Bounds the worst-case freshness of low-throughput export files when the in-memory batch never reaches batch_size. |

## RECORD

Record processing and export configuration. Controls batch sizes, processor scaling, and progress reporting for record processing.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RECORD_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for record export results processor |
| `AIPERF_RECORD_RAW_EXPORT_BATCH_SIZE` | `10` | ≥ 1, ≤ 1000000 | Batch size for raw record writer processor |
| `AIPERF_RECORD_PROCESSOR_SCALE_FACTOR` | `4` | ≥ 1, ≤ 100 | Scale factor for number of record processors to spawn based on worker count. Formula: 1 record processor for every X workers |
| `AIPERF_RECORD_PROGRESS_REPORT_INTERVAL` | `2.0` | ≥ 0.1, ≤ 600.0 | Interval in seconds between records progress report messages |
| `AIPERF_RECORD_PROCESS_RECORDS_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for processing record results |
| `AIPERF_RECORD_STRIP_PAYLOAD_BYTES` | `None` | — | Tri-state control for omitting canonical request payload bytes from RecordContext after a request is sent, which substantially reduces record-pipeline memory for very large prompts. None (default) auto-detects: bytes are stripped only when no downstream record consumer needs them (client-side input tokenization disabled, no synthetic image/audio/video inputs, and raw payload export off). True forces stripping even when a consumer wants the bytes, disabling client-side input tokenization, media counting from request bodies, and raw request payload export. False always retains them. Auto-detection does not see media embedded in custom dataset payloads under server-token-count mode; set False explicitly for that case. |

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
| `AIPERF_SERVICE_WARMUP_PROGRESS_LOG_INTERVAL` | `30.0` | ≥ 0.0, ≤ 100000.0 | Interval in seconds between warmup progress heartbeat log messages. Set to 0 to disable. |
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
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS` | `25.0` | > 1.0, ≤ 10000.0 | Warning threshold in milliseconds for event loop latency (default: 25ms). If the actual sleep duration exceeds the expected duration by this amount, a warning is logged. |
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
| `AIPERF_UI_REALTIME_METRICS_INTERVAL` | `None` | ≥ 0.0, ≤ 1000.0 | Interval in seconds between real-time metrics publishes (and the per-tick stats log block). 0 disables the log block; dashboards still poll. When unset, defaults to 5.0 under --ui dashboard, 30.0 otherwise. |
| `AIPERF_UI_SPINNER_REFRESH_RATE` | `0.1` | ≥ 0.1, ≤ 100.0 | Progress spinner refresh rate in seconds (default: 10 FPS) |
| `AIPERF_UI_CONSOLE_EXPORT_WIDTH` | `140` | ≥ 40, ≤ 10000 | Fixed column width used to render the post-run console exporter tables. Applied both to the recording console that produces profile_export_console.txt and to the live console when stdout is not a tty (so non-tty CI logs match the saved artifact). |

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
