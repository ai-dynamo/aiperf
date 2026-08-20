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
| `AIPERF_JOB_TIMEOUT_SECONDS` | `0` | ≥ 0 | Job timeout in seconds (0 = no timeout) |
| `AIPERF_POD_RESTART_THRESHOLD` | `3` | ≥ 0, ≤ 100 | Pod restart count before emitting a warning event |
| `AIPERF_METRICS_PORT` | `9090` | ≥ 0, ≤ 65535 | Port for the Prometheus /metrics endpoint exposed by the kopf operator process. Set to 0 to disable. Scraped by ServiceMonitor. |
| `AIPERF_ENDPOINT_CHECK_TIMEOUT` | `10.0` | > 0, ≤ 300 | Seconds to wait for endpoint health check |
| `AIPERF_PREFLIGHT_TIMEOUT` | `30.0` | > 0, ≤ 120 | Seconds to wait for all pre-flight checks to complete |
| `AIPERF_CONFIGMAP_PROPAGATION_DELAY_SECONDS` | `10.0` | ≥ 0, ≤ 60 | Seconds to wait after creating the benchmark ConfigMap before creating the JobSet. Allows kubelet caches on worker nodes to sync the ConfigMap before pods start mounting it, preventing FailedMount races on first deployment with a freshly pulled image. |
| `AIPERF_MUTATING_ROUTES_ENABLED` | `False` | — | Enable results-server HTTP routes that mutate Kubernetes state. Defaults false so read-only results APIs remain exposed while POST routes fail closed unless an operator explicitly opts in. |
| `AIPERF_MUTATING_ROUTES_TOKEN` | `''` | — | Bearer token required by enabled results-server mutating routes. Leave empty to fail closed even when MUTATING_ROUTES_ENABLED is true. |
| `AIPERF_CLUSTER_NAME` | `''` | — | Optional human-readable cluster name surfaced in the UI top banner (e.g. 'dgx-prod', 'kind-aiperf'). When unset the banner falls back to the Kubernetes server version. Set via AIPERF_CLUSTER_NAME on the operator deployment. |

## ACCURACY

Accuracy benchmark settings. Tunables for accuracy benchmarking: the cancel-path result-wait timeout and the LiveCodeBench dataset release pin, so accuracy behavior and numbers are reproducible across runs without requiring source edits.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_ACCURACY_CANCEL_RESULT_WAIT_SEC` | `5.0` | ≥ 0.0 | Bounded time (seconds) the SystemController waits on the cancel (Ctrl+C) path for the RecordsManager's ProcessAccuracyResultMessage before stopping. The normal completion path blocks on the accuracy shutdown gate indefinitely, but the cancel path must not hang forever, so it waits at most this long for the graded accuracy summary to arrive over pub/sub before proceeding to export. Set to 0 to skip the wait entirely. |
| `AIPERF_ACCURACY_LCB_GRADE_TIMEOUT_MAX_S` | `300.0` | > 0.0 | Hard ceiling (seconds) on the client-side wall-clock timeout for a single LiveCodeBench code-execution grade. The per-grade timeout scales with the problem's test-case count (lighteval's internal budget plus a margin) but is capped here so one wedged grading worker cannot stall the whole run. Raise it if legitimately slow large problems are being prematurely failed; lower it to fail wedged workers faster. Consumed by ``aiperf.accuracy.graders.code_execution._derive_grade_timeout``. |
| `AIPERF_ACCURACY_LCB_RELEASE_TAG` | `'v4_v5'` | — | LiveCodeBench dataset subset (HF config name) passed as the positional ``name`` arg to ``load_dataset("livecodebench/code_generation_lite", name, split="test", trust_remote_code=True)``. Pins which monthly snapshot LCB serves so accuracy numbers are reproducible across runs and branches. Default ``v4_v5`` matches lighteval's base subset; bump (e.g. to ``v6``) when the team rebaselines against a newer snapshot. ``trust_remote_code=True`` is required because LCB ships a repository loading script; this is only compatible with ``datasets<4`` (``datasets>=4`` dropped loading-script support entirely — the loader surfaces a clear error with a ``datasets<4`` pin). Consumed by ``aiperf.accuracy.benchmarks.lcb_codegeneration``. |

## AGENTX

Settings for the InferenceX AgentX scenario family. Controls runtime knobs for the agentx scenario: the substring allowlist and rate limit used to classify and gate context-overflow errors (RFC 2026-04-26 §7), and the AgenticReplayStrategy double-recycle guard window (RECYCLE_GUARD_MAX_WINDOW).

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
| `AIPERF_API_SERVER_GET_POD_STATES_TIMEOUT` | `2.0` | ≥ 0.1, ≤ 60.0 | Timeout in seconds for API worker-state queries to the SystemController. A short timeout lets progress and debug endpoints fall back to their bus-fed cache while the controller is unavailable. |
| `AIPERF_API_SERVER_POST_COMPLETE_GRACE` | `5.0` | ≥ 0.0, ≤ 300.0 | Seconds the API listener stays open after a benchmark terminates so polling clients can observe the final status before the server shuts down. Set to 0 to skip the grace window and shut down immediately. Not used under ServiceRunType.KUBERNETES, where the controller pod outlives its benchmark and is retired explicitly via POST /api/shutdown instead -- see FastAPIService._on_shutdown_command. A fixed window there is shorter than the operator's monitor interval and strands the AIPerfJob in a pre-terminal phase. |

## CHAT

Settings for the interactive ``aiperf chat`` command.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_CHAT_CONNECT_TIMEOUT` | `10.0` | > 0.0 | Seconds to wait to establish a connection to the endpoint before a turn fails. Kept short so an unreachable URL fails fast. |
| `AIPERF_CHAT_READ_TIMEOUT` | `300.0` | > 0.0 | Seconds to wait for the next streamed chunk before a turn fails. No overall (total) timeout is applied, so long generations are never truncated mid-reply; this only fires if the server stalls. |

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
| `AIPERF_DAG_FAIL_FAST` | `False` | — | When True, a single DAG child error aborts the parent and every orphan sibling under the same branch (releases sticky refcounts and calls issuer.abort_session); unrelated root sessions continue. Default False - the orchestrator counts the error in BranchStats.children_errored, releases the join slot, drains pending siblings, and continues. Set via AIPERF_DAG_FAIL_FAST=1 for strict CI assertions. |

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
| `AIPERF_DATASET_REBROADCAST_INTERVAL` | `2.0` | > 0.0, ≤ 60.0 | Seconds between re-announcements of the dataset-configured notification in Kubernetes. The notification is a one-shot broadcast and sibling worker pods start seconds apart, so a pod that subscribes late would otherwise never learn a dataset exists. |
| `AIPERF_DATASET_REBROADCAST_WINDOW` | `120.0` | ≥ 0.0, ≤ 3600.0 | Total seconds to keep re-announcing the dataset-configured notification for late-joining worker pods. Set 0 to disable. |
| `AIPERF_DATASET_BASETEN_SESSION_COLUMN` | `'provided_session_id'` | one of: 'provided_session_id' / 'poor_man_session_id' | Session column used by the Baseten trace loader when both supported columns exist. Set to poor_man_session_id for legacy traces. If the selected column is absent, the loader uses the available column. |
| `AIPERF_DATASET_STATE_POLL_INTERVAL` | `1.0` | > 0.0, ≤ 60.0 | Seconds between polls of pod-local dataset state while a worker waits to become dispatchable. The dataset arrives via one-shot broadcasts; a worker container that subscribes after they fire recovers by polling its WorkerGroupManager at this interval instead of waiting forever. |
| `AIPERF_DATASET_MMAP_BASE_PATH` | `None` | — | Base path for memory-mapped dataset files. If None, uses system temp directory. Set to a shared filesystem path for Kubernetes mounted volumes. Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/ |
| `AIPERF_DATASET_MMAP_CACHE_ENABLED` | `True` | — | If True, AIPerf reuses memory-mapped dataset files across runs whose input bytes, tokenizer identity, and prompt/input settings are byte-identical. Set to False to force every run to re-tokenize and re-write its mmap files. Cache misses still produce byte-identical mmap files to a non-cached run. |
| `AIPERF_DATASET_MMAP_CACHE_DIR` | `None` | — | Directory holding the content-addressed mmap cache. If None, defaults to ~/.cache/aiperf/dataset_mmap. Each cache entry lives under a `dir/key` subpath and contains dataset.dat, index.dat, manifest.json, and (when produced) inputs.json. No automatic eviction is implemented yet -- delete the directory to reclaim disk. |
| `AIPERF_DATASET_PREFORMAT_PAYLOADS` | `False` | — | If True, pre-encode single-turn / self-contained synthetic conversations to the PAYLOAD_BYTES mmap fast path at dataset-build time so workers stream the bytes verbatim and skip per-request encoding. This is a throughput optimization that DROPS input-tokenization metrics (input_sequence_length, image counts) because the structured prompt is discarded. Default False keeps the structured-turns (CONVERSATION) path so those metrics are computed. Datasets that natively ship raw payloads (raw_payload / inputs_json / mooncake-with-payload) always use PAYLOAD_BYTES regardless of this flag; cache-bust runs always use CONVERSATION regardless. |
| `AIPERF_DATASET_DOWNLOAD_MAX_RETRIES` | `3` | ≥ 0, ≤ 20 | Maximum number of retries for dataset download in Kubernetes worker pods |
| `AIPERF_DATASET_DOWNLOAD_RETRY_DELAY` | `2.0` | ≥ 0.1, ≤ 60.0 | Initial delay in seconds between dataset download retries (doubles each retry) |
| `AIPERF_DATASET_PUBLIC_DATASET_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for public dataset loading operations |
| `AIPERF_DATASET_MEDIA_DOWNLOAD_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds per media URL download when inline encoding is required |
| `AIPERF_DATASET_MEDIA_DOWNLOAD_MAX_CONCURRENCY` | `10` | ≥ 1, ≤ 100 | Maximum number of concurrent media URL downloads |
| `AIPERF_DATASET_INLINE_RECORDS_WARN_THRESHOLD` | `500` | ≥ 1 | Soft warning threshold for the number of inline `records:` entries on a `FileDataset`. When total inline records exceed this value, the config loader logs a warning suggesting the user move the dataset to a JSONL file. No hard cap. |
| `AIPERF_DATASET_TRACELAB_SUBAGENT_JOIN` | `True` | — | When True (default), TraceLabTraceDatasetLoader recovers subagent parent/child links by timing containment and nests each recovered child as a subagent entry inside its parent trace. Set to False to emit every recorded session as an independent flat trace, which is the shape the corpus literally records. |
| `AIPERF_DATASET_TRACELAB_CODEX_SUBAGENT_JOIN` | `True` | — | When True (default), the TraceLab subagent join also runs over codex sessions. Codex uses an async spawn/wait/close agent lifecycle whose handles are stripped from the released corpus, so only a coarse session-level window is available there and a session fanning out several agents collapses them into one window. Set to False to keep only the precise blocking-tool-call join. |
| `AIPERF_DATASET_TRACELAB_MIN_SPAWN_MS` | `10000` | ≥ 0 | Minimum wall latency, in milliseconds, for a spawning tool call to be treated as a subagent round-trip by the TraceLab join. Short calls are overwhelmingly no-op or error returns, and admitting them widens the containment window enough to start capturing unrelated concurrent sessions. |
| `AIPERF_DATASET_WEKA_PARALLEL_WORKERS` | `0` | ≥ 0, ≤ 256 | Number of worker processes for WekaTraceLoader parallel reconstruction. 0 = auto (min(cpu_count - 1, 16, num_traces)). Set to 1 to force serial reconstruction. |
| `AIPERF_DATASET_WEKA_PARALLEL_THRESHOLD` | `8` | ≥ 1, ≤ 100000 | Minimum number of parent traces required before WekaTraceLoader switches to the multi-process parallel reconstruction path. Below this, the in-process serial path is used (Pool startup overhead exceeds the speedup for tiny corpora). |
| `AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS` | `True` | — | When True (default), WekaTraceLoader runs hash_id LCP chain detection at both layers: untagged agent fan-outs recorded as flat top-level requests split into per-agent child conversations (::fa:NNN), and each subagent entry's inner requests split into per-context-chain children (`::sa:agent_id` plus :fa:NNN siblings), all with SPAWN/SPAWN_JOIN linkage so replay reproduces the recorded concurrency. Set to False to disable detection at both layers: all top-level requests serialize into one root conversation and each subagent emits exactly one child with its inner requests in time order. Detected chains at both layers are further split into genuine agents and auxiliary one-shot sidecars (top-level ::fa: vs ::aux:; subagent overflow :fa: vs :aux:) per WEKA_AUX_MAX_REQUESTS / WEKA_AUX_ISL_RATIO / WEKA_AUX_ISL_FLOOR. |
| `AIPERF_DATASET_WEKA_TOOL_SHAPED_MESSAGES` | `False` | — | When True, WekaTraceLoader emits the OpenAI tool-call wire shape for turns classified as tool-result continuations: the same-delta assistant message gains a synthetic tool_calls entry and the turn's new input is sent as a role='tool' message instead of plain user text (content unchanged). Exercises the server's tool-message chat-template path at the cost of exact ISL fidelity (tool messages tokenize differently than plain user text). Only turns with a recorded tool signal (input_types / prior stop) shape; legacy traces are unaffected. Default False keeps the byte-exact plain-user replay shape. |
| `AIPERF_DATASET_WEKA_SEAM_MAX_GAP_SECONDS` | `3600.0` | ≥ 0.0 | LCP chain-detection seam guard: the maximum wall-clock gap (seconds) between a chain's last request and a candidate continuation before that continuation is only accepted when it also keeps enough of the prior context (see WEKA_SEAM_MIN_OVERLAP_RATIO). A genuine context compaction continues promptly (seconds to minutes), so a low-overlap join hours later is treated as a distinct session that merely shares a base prefix and is spawned as its own conversation instead of being stitched onto the chain (which would fabricate a multi-hour intra-conversation idle gap). The guard fires only when BOTH this gap is exceeded AND overlap is below the ratio, so prompt compactions at any overlap and verbatim long-gap resumes at high overlap are preserved. Raise toward infinity to disable the temporal half of the guard. |
| `AIPERF_DATASET_WEKA_SEAM_MIN_OVERLAP_RATIO` | `0.5` | ≥ 0.0, ≤ 1.0 | LCP chain-detection seam guard: the minimum shared-prefix ratio (continuation's fork depth / the chain tail's block count) for a far-future continuation to still be accepted as the same agent. Below this, a continuation past WEKA_SEAM_MAX_GAP_SECONDS is spawned as a new conversation rather than spliced on. Corpus data is bimodal -- real compactions and verbatim resumes keep at least 94% of the prefix, while coincidental base-prefix mis-merges keep under 50% -- so 0.5 sits in a wide safe valley. Set to 0.0 to disable the overlap half of the guard. |
| `AIPERF_DATASET_WEKA_AUX_MAX_REQUESTS` | `1` | ≥ 0 | Auxiliary (sidecar) classification: a detected worker chain with at most this many requests is eligible to be reclassified as an auxiliary one-shot call -- a tool-issued sidecar (web fetch/search summary, title generation, a classifier) rather than a sustained agent -- when it also passes the WEKA_AUX_ISL_* size test. Applies to both top-level flat chains (::fa: -> ::aux:) and a subagent's nested-LCP overflow (:fa: -> :aux:). Corpus sidecars are overwhelmingly single-request, so the default is 1. Set to 0 to disable aux classification (every worker chain keeps its agent tag). Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True. |
| `AIPERF_DATASET_WEKA_AUX_ISL_RATIO` | `0.1` | ≥ 0.0 | Auxiliary (sidecar) classification: an aux-eligible chain (see WEKA_AUX_MAX_REQUESTS) is reclassified to a sidecar only when its first request's input length is below max(WEKA_AUX_ISL_FLOOR, this ratio * the enclosing main chain's peak input length -- the trace's for flat chains, the subagent's for overflow). The ratio catches calls small relative to a large conversation's accumulated context; the floor catches them in absolute terms. Sidecars start from a fresh few-thousand-token context vs the agent's tens-to-hundreds of thousands. |
| `AIPERF_DATASET_WEKA_AUX_ISL_FLOOR` | `16384` | ≥ 0 | Auxiliary (sidecar) classification: absolute input-length floor (tokens) for the aux size test (see WEKA_AUX_ISL_RATIO). A chain whose first-request input length is below max(this, ratio * main peak ISL) is treated as an auxiliary one-shot sidecar. Keeps small fresh-context calls classified as sidecars even when the enclosing conversation is itself small. |
| `AIPERF_DATASET_WEKA_AUX_CROSS_MODEL` | `True` | — | Auxiliary (sidecar) classification: when True (default), an aux-eligible chain (at most WEKA_AUX_MAX_REQUESTS requests) whose first request runs on a different model than its enclosing main chain is treated as a sidecar regardless of input length. An agent does not switch models for its own reasoning, so a one-shot on a different model is a tool-internal call -- e.g. a Haiku WebFetch summary fired by an Opus agent, which can carry a large fetched-page payload and so escape the WEKA_AUX_ISL_* size test. Set to False to classify purely by size. |
| `AIPERF_DATASET_WEKA_AUX_REDUCTION_OSL_MAX` | `4000` | ≥ 0 | Auxiliary (sidecar) classification, reduction arm: a single-request worker chain on the SAME model as its enclosing main chain is reclassified to an auxiliary one-shot when its output length is in (0, this) tokens AND its input length is at least WEKA_AUX_ISL_FLOOR AND its input/output ratio exceeds WEKA_AUX_REDUCTION_RATIO. This catches large-input/short-output reductions (context compaction, subagent-result summaries, tool-output digests) that the size and cross-model arms miss because they are same-model and large. The bound separates a bounded summary from generative agent output (a real agent emits long completions); corpus reductions cap well below 4k output across every capture. Reductions are emitted as ordinary ::aux: sidecars. Set to 0 to disable the reduction arm. Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True. |
| `AIPERF_DATASET_WEKA_AUX_REDUCTION_RATIO` | `20.0` | ≥ 0.0 | Auxiliary (sidecar) classification, reduction arm: the minimum input-to-output token ratio for a same-model single-request large-input chain to be treated as a reduction sidecar (see WEKA_AUX_REDUCTION_OSL_MAX). A reduction consumes a large body and emits a short summary, so input/output is high (corpus median ~120); 20 is a conservative floor that still excludes balanced request/response calls. Only applies when WEKA_AUX_REDUCTION_OSL_MAX > 0. |
| `AIPERF_DATASET_WEKA_WORKER_GROUP_MIN` | `3` | ≥ 0 | Parallel worker-group tagging: a coordinated parallel fan-out must BOTH share a deep spawned context AND run concurrently. Workers that forked from shared context (fork depth > 0) are first scoped by their fork point (the parent request they branched off), then within each scope split into connected components of overlapping active [t0, t1) intervals; a component with at least this many members is emitted as ::wg:{group}_{member} (group = the concurrent fan-out, member = index by start time) instead of the generic ::fa: agent marker. The fork-point scope keeps unrelated fan-outs apart (pure interval overlap bridges a busy trace into one blob); the overlap split drops members that share the fork point but never run concurrently. This isolates genuine parallel sub-agent fan-out (the dominant agent population) from solo agents, unlike keying on the first context block (shared by ~every worker all session). Auxiliary chains are classified first, so a one-shot sidecar never becomes a worker-group member. Set to 0 to disable worker-group tagging (parallel workers keep the generic ::fa: tag). Only applies when WEKA_SPLIT_FLATTENED_AGENTS is True. |

## ENDPOINT

Endpoint wire-format configuration. Controls how AIPerf serializes message content when building request payloads. The main knob is FORCE_CONTENT_PARTS, which overrides the single-text fast path that emits a plain string for simple turns.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_ENDPOINT_FORCE_CONTENT_PARTS` | `False` | — | When True, always emit the multi-part content array (e.g. [{"type": "text", "text": "..."}]) for synthetic turns, even when there is only a single text with no media. By default (False) single-text turns emit a plain string to stay compatible with servers that reject list-of-parts content for non-multimodal inputs (e.g. OpenAI Dynamo). Enable when the target server requires the structured content-parts shape unconditionally. |

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

## HTTP

HTTP client socket and connection configuration. Controls low-level socket options, keepalive settings, DNS caching, and connection pooling for HTTP clients. These settings optimize performance for high-throughput streaming workloads. Video Generation Polling: For async video generation APIs that use job polling (e.g., SGLang /v1/videos), the poll interval is controlled by AIPERF_HTTP_VIDEO_POLL_INTERVAL. The max poll time uses the --request-timeout-seconds CLI argument.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_HTTP_CONNECTION_LIMIT` | `2500` | ≥ 1, ≤ 65000 | Maximum number of concurrent HTTP connections |
| `AIPERF_HTTP_KEEPALIVE_TIMEOUT` | `300` | ≥ 0, ≤ 10000 | HTTP connection keepalive timeout in seconds for connection pooling |
| `AIPERF_HTTP_SO_RCVBUF` | `10485760` | ≥ 1024 | Socket receive buffer size in bytes (default: 10MB for high-throughput streaming) |
| `AIPERF_HTTP_SO_SNDBUF` | `10485760` | ≥ 1024 | Socket send buffer size in bytes (default: 10MB for high-throughput streaming) |
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
| `AIPERF_HTTP_IP_VERSION` | `'4'` | one of: '4' / '6' / 'auto' | IP version for HTTP socket connections. Options: '4' (AF_INET, default), '6' (AF_INET6), or 'auto' (AF_UNSPEC, system chooses). |
| `AIPERF_HTTP_TRUST_ENV` | `False` | — | Trust environment variables for HTTP client configuration. When enabled, aiohttp will read proxy settings from HTTP_PROXY, HTTPS_PROXY, and NO_PROXY environment variables. |
| `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` | `False` | — | Also send X-Session-ID with the stable X-Correlation-ID value. This transport setting is the supported way to enable generic HTTP session affinity. It is ADDITIVE (both headers are sent); --session-header only RENAMES the single correlation header. |
| `AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID` | `False` | — | Also send X-SMG-Routing-Key with the stable X-Correlation-ID value. This transport setting is the supported affinity path for the SGLang Model Gateway manual routing policy. |
| `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID` | `False` | — | Also send X-Dynamo-Session-ID with the stable X-Correlation-ID value, plus X-Dynamo-Parent-Session-ID on subagent children. This transport setting is the supported affinity path for a Dynamo frontend running --router-session-affinity-ttl-secs, pinning every turn of a session to the replica holding its KV prefix. |
| `AIPERF_HTTP_VIDEO_POLL_INTERVAL` | `0.1` | ≥ 0.001, ≤ 10.0 | Interval in seconds between status polls for async video generation jobs. Lower values provide faster completion detection but increase server load. Applies to the aiohttp transport. |

## K8S

Root Kubernetes environment configuration. Loads configuration from environment variables with the AIPERF_K8S_ prefix. Resource settings per container type are created via _resource_settings() with service-specific env prefixes and defaults.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST` | `None` | — | Optional per-record-processor CPU request override inside worker pods |
| `AIPERF_K8S_RECORD_PROCESSOR_SCALE_FACTOR` | `1` | ≥ 1, ≤ 100 | Kubernetes-only default scale factor for record processors per worker pod. Formula: 1 record processor for every X workers. Default: 1 record processor per worker. |
| `AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED` | `True` | — | Run the XPUB/XSUB event-bus proxy as a dedicated sidecar container in the controller pod rather than inside the control-plane (SystemController) container. Isolates pub/sub socket accept/forward from the control plane's event loop so large fan-ins (hundreds of simultaneous RP/worker connections) at startup don't starve the SystemController. Set to false to revert to the pre-sidecar behavior where SystemController owns the event-bus proxy. |
| `AIPERF_K8S_SHARE_PROCESS_NAMESPACE` | `False` | — | When true, JobSet pods spawned by the operator set podSpec.shareProcessNamespace=true so all containers share a PID namespace. Enables cross-container `kubectl exec kill -9 <pid>` for chaos-testing workflows. Keep false in production; chaos fixtures flip it on via AIPERF_K8S_SHARE_PROCESS_NAMESPACE=true. |
| `AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE` | `None` | — | Chaos-test hook: when set, the operator's progress-client uses this base URL (scheme+host+port, e.g. http://toxiproxy.aiperf-chaos-toxiproxy.svc:20002) instead of the per-CR JobSet pod DNS + API_SERVICE port for controller HTTP calls. Production MUST leave unset — it collapses multi-job isolation because every CR funnels through the same URL. Chaos fixtures set it via AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE to steer traffic through toxiproxy for latency/blackhole injection. |
| `AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE` | `None` | — | Chaos-test hook: when KUBERNETES_SERVICE_HOST points at an L4 proxy rather than kubernetes.default.svc, verify the apiserver certificate against this hostname while still dialing the proxy. Production MUST leave unset; C15 sets it to kubernetes.default.svc. |

## K8SAPI

Api container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_API_CPU` | `75m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_API_MEMORY` | `256Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SDATASETMANAGER

Dataset Manager container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_DATASET_MANAGER_CPU` | `50m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_DATASET_MANAGER_MEMORY` | `256Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SDIAGNOSIS

Thresholds for ``aiperf.kubernetes.benchmark_diagnosis`` heuristics.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_DIAGNOSIS_STALLED_PENDING_THRESHOLD_SECONDS` | `60.0` | ≥ 1.0, ≤ 3600.0 | Pending job is flagged as stalled after this many seconds. |
| `AIPERF_K8S_DIAGNOSIS_STALLED_RUNNING_THRESHOLD_SECONDS` | `30.0` | ≥ 1.0, ≤ 3600.0 | Running job with no throughput and no completed requests is flagged as stalled after this many seconds. |
| `AIPERF_K8S_DIAGNOSIS_HIGH_ERROR_RATE_THRESHOLD` | `0.05` | ≥ 0.0, ≤ 1.0 | Error rate (0.0-1.0) above which a high-error-rate finding is reported. |
| `AIPERF_K8S_DIAGNOSIS_FAIL_ABOVE_ERROR_RATE` | `1.0` | > 0.0, ≤ 1.0 | Error rate (0.0-1.0) at or above which a finished benchmark is reported as Failed instead of Completed. Defaults to 1.0, so only a run in which every single request errored is failed outright; a run that merely errored heavily still completes and is flagged by the high-error-rate diagnosis. Lower it to enforce a stricter success bar. |
| `AIPERF_K8S_DIAGNOSIS_HIGH_LATENCY_P99_MULTIPLIER` | `10.0` | ≥ 1.0, ≤ 1000.0 | Multiplier on average latency above which p99 is flagged as a tail-latency outlier. |

## K8SEVENTBUSPROXY

Event Bus Proxy container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_EVENT_BUS_PROXY_CPU` | `50m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_EVENT_BUS_PROXY_MEMORY` | `64Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SGPUTELEMETRYMANAGER

Gpu Telemetry Manager container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_GPU_TELEMETRY_MANAGER_CPU` | `25m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_GPU_TELEMETRY_MANAGER_MEMORY` | `192Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SHEALTH

Health probe configuration for all containers.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_HEALTH_INITIAL_DELAY_SECONDS` | `5` | ≥ 0, ≤ 300 | Seconds before starting probes after container starts |
| `AIPERF_K8S_HEALTH_PERIOD_SECONDS` | `10` | ≥ 1, ≤ 300 | Interval in seconds between probe checks |
| `AIPERF_K8S_HEALTH_TIMEOUT_SECONDS` | `5` | ≥ 1, ≤ 60 | Seconds before probe times out |
| `AIPERF_K8S_HEALTH_FAILURE_THRESHOLD` | `10` | ≥ 1, ≤ 20 | Consecutive failures before container is restarted/marked unready |
| `AIPERF_K8S_HEALTH_SUCCESS_THRESHOLD` | `1` | ≥ 1, ≤ 10 | Consecutive successes before container is marked healthy |
| `AIPERF_K8S_HEALTH_STARTUP_PERIOD_SECONDS` | `5` | ≥ 1, ≤ 30 | Interval between startup probe checks |
| `AIPERF_K8S_HEALTH_STARTUP_FAILURE_THRESHOLD` | `30` | ≥ 1, ≤ 120 | Consecutive startup probe failures before pod is killed. Total startup time = STARTUP_PERIOD_SECONDS * STARTUP_FAILURE_THRESHOLD |

## K8SJOBSET

JobSet-level configuration.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` | `300` | ≥ 0 | Seconds to keep JobSet after completion (None to disable) |
| `AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS` | `28800` | ≥ 0 | TTL for operator-less (direct) deployments. Pods stay alive for manual results retrieval. Default 8 hours (28800s). |
| `AIPERF_K8S_JOBSET_CONTROLLER_BACKOFF_LIMIT` | `0` | ≥ 0, ≤ 10 | Job backoff limit for controller (0 = no retries) |
| `AIPERF_K8S_JOBSET_WORKER_BACKOFF_LIMIT` | `20` | ≥ 0, ≤ 20 | Job backoff limit for workers (allows retries for transient failures) |
| `AIPERF_K8S_JOBSET_WORKER_CONNECTION_PROBE_TIMEOUT` | `60.0` | ≥ 30.0, ≤ 600.0 | Seconds worker pods wait for the PUB/SUB connection probe to succeed. Overrides AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT for k8s worker containers only. Pods that cannot connect exit cleanly so Kubernetes restarts them with a fresh ZMQ context; WORKER_BACKOFF_LIMIT absorbs transient first-deploy flakes. |
| `AIPERF_K8S_JOBSET_CONFIG_MOUNT_PATH` | `'/etc/aiperf'` | — | Path to mount ConfigMap with configs |
| `AIPERF_K8S_JOBSET_DATASETS_PATH` | `'/aiperf/datasets'` | — | Shared path for dataset files (dataset-manager writes, API serves) |
| `AIPERF_K8S_JOBSET_SWEEP_AGGREGATE_INLINE_MAX_BYTES` | `600000` | ≥ 10000, ≤ 900000 | Max encoded size of the AIPerfSweep aggregate bundle inlined into status.aggregate. K8s rejects CR patches over ~1 MiB with HTTP 413; if the bundle exceeds this cap, the sweep-controller drops `confidence` (the largest contributor on big sweeps) and relies on the disk-backed results sidecar to serve the full document. Default 600 KB leaves headroom for status fields and apiserver framing under the 1 MiB ceiling. |
| `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME` | `''` | — | Operator-side default for Kueue gang-scheduling. When the AIPerfJob CR's spec.scheduling.queue_name is unset, the JobSet manifest falls back to this value. When non-empty, the JobSet gets the kueue.x-k8s.io/queue-name label, which Kueue's JobSet integration uses to admit the workload as a unit (gang-scheduling: controller + all worker pods admitted atomically, or none). Safe to leave unset on clusters without Kueue — the label is then never added. Set to e.g. 'aiperf-lq' on clusters where Kueue is installed and a LocalQueue of that name exists in the benchmark namespace. |
| `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_PRIORITY_CLASS` | `''` | — | Operator-side default for Kueue WorkloadPriorityClass. Companion to KUEUE_DEFAULT_QUEUE_NAME. When unset, the JobSet gets no kueue.x-k8s.io/priority-class label and Kueue's default fairness applies. |

## K8SPORT

Container port assignments.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_PORT_SYSTEM_CONTROLLER_HEALTH` | `8080` | ≥ 1, ≤ 65535 | System controller health port |
| `AIPERF_K8S_PORT_WORKER_MANAGER_HEALTH` | `8081` | ≥ 1, ≤ 65535 | Worker manager health port |
| `AIPERF_K8S_PORT_TIMING_MANAGER_HEALTH` | `8082` | ≥ 1, ≤ 65535 | Timing manager health port |
| `AIPERF_K8S_PORT_DATASET_MANAGER_HEALTH` | `8083` | ≥ 1, ≤ 65535 | Dataset manager health port |
| `AIPERF_K8S_PORT_RECORDS_MANAGER_HEALTH` | `8084` | ≥ 1, ≤ 65535 | Records manager health port |
| `AIPERF_K8S_PORT_API_SERVICE` | `9090` | ≥ 1, ≤ 65535 | API service port |
| `AIPERF_K8S_PORT_RESULTS_SIDECAR` | `9091` | ≥ 1, ≤ 65535 | Results sidecar port for serving exported files after controller failure |
| `AIPERF_K8S_PORT_API_SERVICE_HEALTH` | `8085` | ≥ 1, ≤ 65535 | API service health port |
| `AIPERF_K8S_PORT_GPU_TELEMETRY_MANAGER_HEALTH` | `8086` | ≥ 1, ≤ 65535 | GPU telemetry manager health port |
| `AIPERF_K8S_PORT_SERVER_METRICS_MANAGER_HEALTH` | `8087` | ≥ 1, ≤ 65535 | Server metrics manager health port |
| `AIPERF_K8S_PORT_EVENT_BUS_PROXY_HEALTH` | `8088` | ≥ 1, ≤ 65535 | Event-bus proxy sidecar health port |
| `AIPERF_K8S_PORT_EVENT_BUS_PROXY_PUB_FRONTEND` | `5663` | ≥ 1, ≤ 65535 | Event-bus XPUB/XSUB proxy publisher-frontend bind port (producers connect to this). |
| `AIPERF_K8S_PORT_EVENT_BUS_PROXY_SUB_BACKEND` | `5664` | ≥ 1, ≤ 65535 | Event-bus XPUB/XSUB proxy subscriber-backend bind port (subscribers connect to this). |
| `AIPERF_K8S_PORT_WORKER_HEALTH` | `8080` | ≥ 1, ≤ 65535 | Worker health port |
| `AIPERF_K8S_PORT_RECORD_PROCESSOR_HEALTH` | `8081` | ≥ 1, ≤ 65535 | Record processor health port |

## K8SPORTFORWARD

Tunables for ``aiperf.kubernetes.port_forward`` kubectl-based forwards.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_PORT_FORWARD_TIMEOUT_SECONDS` | `60.0` | ≥ 1.0, ≤ 600.0 | Total seconds to wait for kubectl port-forward to start and (optionally) for the API to respond. |
| `AIPERF_K8S_PORT_FORWARD_API_INITIAL_DELAY_SECONDS` | `0.5` | ≥ 0.0, ≤ 10.0 | Seconds to wait after the tunnel comes up before the first API health check. |
| `AIPERF_K8S_PORT_FORWARD_API_RETRY_DELAY_SECONDS` | `2.0` | ≥ 0.1, ≤ 30.0 | Seconds to back off between port-forward restart attempts while the API isn't ready. |
| `AIPERF_K8S_PORT_FORWARD_API_MAX_RETRIES` | `10` | ≥ 0, ≤ 50 | Maximum number of port-forward restarts before giving up on the API readiness probe. |
| `AIPERF_K8S_PORT_FORWARD_PROCESS_CLEANUP_TIMEOUT_SECONDS` | `5.0` | ≥ 0.1, ≤ 60.0 | Seconds to wait for graceful kubectl termination before escalating to SIGKILL. |

## K8SPROGRESSSTREAM

Tunables for ``aiperf.kubernetes.progress_stream`` WebSocket reconnects.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_PROGRESS_STREAM_WS_INITIAL_BACKOFF_SECONDS` | `1.0` | ≥ 0.1, ≤ 60.0 | Initial reconnect backoff after a WebSocket transport error. |
| `AIPERF_K8S_PROGRESS_STREAM_WS_MAX_BACKOFF_SECONDS` | `30.0` | ≥ 1.0, ≤ 300.0 | Cap on the exponential reconnect backoff. |
| `AIPERF_K8S_PROGRESS_STREAM_WS_HEARTBEAT_SECONDS` | `30` | ≥ 1, ≤ 300 | Seconds between aiohttp WebSocket heartbeats. |

## K8SRECORDSMANAGER

Records Manager container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_RECORDS_MANAGER_CPU` | `75m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_RECORDS_MANAGER_MEMORY` | `256Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SRESULTSSIDECAR

Results Sidecar container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_RESULTS_SIDECAR_CPU` | `25m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_RESULTS_SIDECAR_MEMORY` | `192Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SSERVERMETRICSMANAGER

Server Metrics Manager container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_SERVER_METRICS_MANAGER_CPU` | `25m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_SERVER_METRICS_MANAGER_MEMORY` | `192Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SSWEEPCONTROLLER

Sweep Controller container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_SWEEP_CONTROLLER_CPU` | `75m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_SWEEP_CONTROLLER_MEMORY` | `512Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SSYSTEMCONTROLLER

System Controller container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_SYSTEM_CONTROLLER_CPU` | `75m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_SYSTEM_CONTROLLER_MEMORY` | `192Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8STIMINGMANAGER

Timing Manager container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_TIMING_MANAGER_CPU` | `50m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_TIMING_MANAGER_MEMORY` | `192Mi` | — | Memory request and limit (Guaranteed QoS) |

## K8SWATCHDOG

Thresholds for ``aiperf.kubernetes.watchdog`` pod-health heuristics. These were plain keyword defaults on ``BenchmarkWatchdog.__init__`` with no environment binding, so a cluster with slow image pulls or an intentionally restart-tolerant workload had no way to raise them.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_WATCHDOG_POLL_INTERVAL_SECONDS` | `5.0` | ≥ 0.0, ≤ 300.0 | Seconds between watchdog pod-state polls. |
| `AIPERF_K8S_WATCHDOG_STATUS_INTERVAL_SECONDS` | `10.0` | ≥ 0.0, ≤ 3600.0 | Seconds between watchdog status log lines. |
| `AIPERF_K8S_WATCHDOG_PENDING_THRESHOLD_SECONDS` | `30.0` | ≥ 1.0, ≤ 3600.0 | Seconds a pod startup blocker may remain stable before the CLI watchdog or operator raises a warning. |
| `AIPERF_K8S_WATCHDOG_PENDING_CRITICAL_THRESHOLD_SECONDS` | `90.0` | ≥ 1.0, ≤ 3600.0 | Seconds a pod startup blocker may remain stable before escalation to critical. The operator fails only known non-recoverable image, configuration, crash-loop, or structural scheduling blockers; capacity-related scheduling remains retryable. |
| `AIPERF_K8S_WATCHDOG_CRASHLOOP_RESTART_THRESHOLD` | `2` | ≥ 1, ≤ 100 | Container restart count at which a crash-loop warning is raised and the operator may treat a stable CrashLoopBackOff as terminal. |

## K8SWORKERPOD

Worker Pod container CPU and memory (Guaranteed QoS).

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_WORKER_POD_CPU` | `150m` | — | CPU request and limit (Guaranteed QoS) |
| `AIPERF_K8S_WORKER_POD_MEMORY` | `4Gi` | — | Memory request and limit (Guaranteed QoS) |

## K8SZMQ

ZMQ communication settings for Kubernetes deployments.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_K8S_ZMQ_CONTROLLER_HOST` | `None` | — | Controller hostname for ZMQ dual-bind mode. Set on worker pods to connect via TCP to controller. When None, services use IPC (controller mode). |
| `AIPERF_K8S_ZMQ_IPC_PATH` | `'/aiperf/ipc'` | — | Path for IPC socket files in pods |

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
| `AIPERF_METRICS_EXPORT_FLUSH_INTERVAL` | `1.0` | ≥ 0.05, ≤ 60.0 | Periodic flush interval (seconds) for buffered JSONL stream exporters (raw record writer, record export, gpu/server-metrics JSONL writers). Bounds the worst-case freshness of low-throughput export files when the in-memory batch never reaches batch_size. |
| `AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD` | `10.0` | ≥ 0.0, ≤ 100.0 | Percentage difference threshold for flagging discrepancies between API usage and client token counts (default: 10%) |
| `AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD` | `5.0` | ≥ 0.0, ≤ 100.0 | Percentage difference threshold for flagging discrepancies between requested and actual output sequence length (default: 5%) |
| `AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD` | `50` | ≥ 1 | Maximum absolute token threshold for OSL mismatch. The effective threshold is min(requested_osl * pct_threshold, this value). Makes threshold tighter for large OSL values (default: 50 tokens) |
| `AIPERF_METRICS_TDIGEST_COMPRESSION` | `500` | ≥ 20, ≤ 10000 | t-digest sketch compression for list-valued record metric aggregation. Higher = more centroids, tighter percentile accuracy, larger sketch. Default 500 measured to keep worst-case relative percentile error under 0.05% on 50M-sample workloads (40x under the 0.5% claimed accuracy band) at ~4 KB sketch size. |
| `AIPERF_METRICS_LIST_BACKEND` | `'ragged'` | one of: 'ragged' / 'tdigest' | Storage backend for list-valued RECORD metrics (today: only inter_chunk_latency). 'ragged' (default) keeps every value, enabling exact percentiles and ICL-aware throughput / tokens-in-flight sweep curves. 'tdigest' uses a bounded-memory crick.TDigest sketch (~4 KB regardless of sample count) — percentiles are approximate (at most 0.05% relative error at default compression), and ICL-aware sweep curves silently fall back to their non-ICL equivalents that use only request-level (start_ns, generation_start_ns, end_ns) timing. Choose tdigest when records-manager pod memory at 1M+ request scale is the binding constraint. |

## MLFLOW

MLflow export configuration. Controls timeout behavior for post-run MLflow artifact uploads.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_MLFLOW_EXPORT_TIMEOUT_SECONDS` | `30.0` | ≥ 1.0, ≤ 600.0 | Timeout in seconds for the post-run MLflow export operation. If the MLflow tracking server is unreachable, the export will be abandoned after this duration rather than blocking indefinitely. |

## NETWORKLATENCY

Network latency calibration configuration. Controls the TCP-handshake RTT probes used to estimate the client-to-endpoint network round-trip time so it can be subtracted from latency metrics. Probes run throughout the profiling phase. Enable with `--network-latency-automatic`.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_NETWORK_LATENCY_DEFAULT_PROBE_INTERVAL` | `1.0` | ≥ 0.001, ≤ 300.0 | Default seconds between RTT probes when --network-latency-ping-interval is unset (default: 1.0s, ~1Hz) |
| `AIPERF_NETWORK_LATENCY_MIN_SAMPLES` | `5` | ≥ 1, ≤ 100000 | Minimum number of successful RTT samples to collect; extra probes are issued at profile completion if a short run did not reach this floor |
| `AIPERF_NETWORK_LATENCY_CONNECT_TIMEOUT` | `5.0` | ≥ 0.001, ≤ 300.0 | Timeout in seconds for a single TCP-handshake RTT probe |
| `AIPERF_NETWORK_LATENCY_COMPLETE_TOPUP_TIMEOUT` | `3.0` | ≥ 0.0, ≤ 30.0 | Wall-clock budget in seconds for the final MIN_SAMPLES top-up probes at PROFILE_COMPLETE, kept well under the command-response budget so a slow endpoint cannot stall completion |
| `AIPERF_NETWORK_LATENCY_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for the network latency jsonl writer export results processor |

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

## POD

Kubernetes worker-pod monitoring configuration. Consumed by the Kubernetes service manager, which polls the Kubernetes API for worker-pod phases and container-level failures.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_POD_MONITOR_INTERVAL` | `5.0` | ≥ 0.1, ≤ 100000.0 | Interval in seconds between worker-pod monitoring sweeps. Bounds how quickly a Failed/Unknown worker pod is detected via the Kubernetes API |
| `AIPERF_POD_FAILURE_ABORT_THRESHOLD_PERCENT` | `50.0` | ≥ 0.0, ≤ 100.0 | Percentage of failed worker pods at which the Kubernetes service manager signals the controller to abort the benchmark. Set to 0 to never abort on pod failures |

## RECORD

Record processing and export configuration. Controls batch sizes, processor scaling, and progress reporting for record processing.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RECORD_CHECKPOINT_INTERVAL` | `30.0` | ≥ 0.0, ≤ 3600.0 | Seconds between partial-checkpoint writes during a Kubernetes run. The results sidecar serves these before the results-ready marker exists, and the operator treats a growing checkpoint as evidence the controller is alive. Set to 0 to disable. |
| `AIPERF_RECORD_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for record export results processor |
| `AIPERF_RECORD_COMPLETION_STALL_TIMEOUT` | `300.0` | ≥ 0.0, ≤ 86400.0 | Seconds of ZERO record progress, after all credits are complete, before the RecordsManager stops waiting and finalizes the run as degraded. The completion barrier is event-driven: it needs one record per completed request, so a request that completes without ever emitting a record leaves the barrier permanently short and nothing re-triggers it. This bounds that into a loud failure instead of an unbounded hang. The timer measures time since the last record arrived, not total elapsed, so legitimately slow aggregation is never cut short. Set 0 to disable. |
| `AIPERF_RECORD_COMPLETION_STALL_CHECK_INTERVAL` | `10.0` | > 0.0, ≤ 3600.0 | Seconds between record-progress stall checks after credits complete. |
| `AIPERF_RECORD_RAW_EXPORT_BATCH_SIZE` | `10` | ≥ 1, ≤ 1000000 | Batch size for raw record writer processor |
| `AIPERF_RECORD_PROCESSOR_SCALE_FACTOR` | `4` | ≥ 1, ≤ 100 | Scale factor for number of record processors to spawn based on worker count. Formula: 1 record processor for every X workers. The default of 4 is the ratio the Kubernetes pod-sizing design was built around, alongside ~500 concurrent connections per worker; see RuntimeConfig.record_processors_per_pod |
| `AIPERF_RECORD_PROGRESS_REPORT_INTERVAL` | `2.0` | ≥ 0.1, ≤ 600.0 | Interval in seconds between records progress report messages |
| `AIPERF_RECORD_PROCESS_RECORDS_TIMEOUT` | `300.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for processing record results |
| `AIPERF_RECORD_STRIP_PAYLOAD_BYTES` | `None` | — | Tri-state control for omitting canonical request payload bytes from RecordContext after a request is sent, which substantially reduces record-pipeline memory for very large prompts. None (default) auto-detects: bytes are stripped only when no downstream record consumer needs them (client-side input tokenization disabled, no synthetic image/audio/video inputs, and raw payload export off). True forces stripping even when a consumer wants the bytes, disabling client-side input tokenization, media counting from request bodies, and raw request payload export. False always retains them. Auto-detection does not see media embedded in custom dataset payloads under server-token-count mode; set False explicitly for that case. |

## RESULTS

Results fetching and storage settings.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_RESULTS_DIR` | `Path('/data')` | — | Base directory for storing benchmark results (mounted PVC) |
| `AIPERF_RESULTS_K8S_INIT_TIMEOUT_SEC` | `10.0` | > 0, ≤ 120 | Seconds the results-server waits for its Kubernetes client to initialize at startup before giving up and serving PVC-only. The live-job endpoints need a cluster, but every results, sweeps, and artifact route reads the disk, so an unreachable apiserver must degrade the server rather than prevent it from starting. |
| `AIPERF_RESULTS_MAX_RETRIES` | `5` | ≥ 0, ≤ 50 | Max retries when fetching results from controller |
| `AIPERF_RESULTS_RETRY_DELAY` | `2.0` | ≥ 0, ≤ 60 | Seconds between result fetch retries |
| `AIPERF_RESULTS_TTL_DAYS` | `30` | ≥ 0, ≤ 3650 | Days to keep results before cleanup (0 = never clean) |
| `AIPERF_RESULTS_COMPRESS_ON_DISK` | `True` | — | Store downloaded result files as zstd-compressed (.zst) on disk |
| `AIPERF_RESULTS_RETAIN_RUNS` | `10` | ≥ 1, ≤ 10000 | Max per-run result dirs to keep under <namespace>/<name>/ before retention trimming. Applied after every successful completion; the just-written epoch is always protected from deletion. |
| `AIPERF_RESULTS_RETAIN_DAYS` | `0` | ≥ 0, ≤ 36500 | Age-based retention cap in days. 0 disables age policy. A run is deleted only when BOTH this age cap AND RETAIN_RUNS agree the run is outside the keep window; protect_epoch still wins. |
| `AIPERF_RESULTS_TRANSIENT_FETCH_RETRY_BUDGET_SEC` | `60.0` | ≥ 0.0, ≤ 600.0 | Wall-clock budget (seconds, measured from the completion-claim annotation timestamp) within which a transient HTTP fetch failure is converted to a kopf.TemporaryError so the next monitor tick retries via the orphan-claim recovery path. Past this budget the operator gives up and marks the AIPerfJob Failed with the ResultsFetchFailed condition. WHY: sub-second benchmarks can race the controller's post-export shutdown — the marker has been written and key files exist on the controller PVC, but the operator's HTTP fetch hits a connection-refused or empty list as the controller container terminates. Set 0 to disable retries. |
| `AIPERF_RESULTS_TRANSIENT_FETCH_RETRY_DELAY_SEC` | `5.0` | ≥ 0.5, ≤ 60.0 | Delay (seconds) passed to ``kopf.TemporaryError`` when retrying a transient results-fetch failure. Each retry runs through the orphan-claim recovery path on the next monitor tick. |
| `AIPERF_RESULTS_PHASE_SETTLE_ATTEMPTS` | `3` | ≥ 0, ≤ 20 | How many times the completion handler re-samples controller progress while a phase reports its requests finished but its records still aggregating. Record aggregation trails the last request by a beat, so a single sample can leave status.phases showing isRecordsComplete=false on a run whose exports are complete. Set 0 to take exactly one sample. |
| `AIPERF_RESULTS_PHASE_SETTLE_DELAY_SEC` | `2.0` | ≥ 0.1, ≤ 30.0 | Delay between the re-samples controlled by ``PHASE_SETTLE_ATTEMPTS``. The total wait is a hard ceiling on how long completion is delayed for a cosmetic status mirror. |

## SEARCHPLANNER

Adaptive-search planner tunables. Controls precision targets, warmup-phase injection, and request-count presets for the smooth-isotonic and monotonic SLA-saturation search planners. All values are read at planner-construction or iteration-mutate time, so changes take effect on the next search run.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SEARCH_PLANNER_SLA_PRECISION_DEFAULT` | `0.05` | > 0.0, &lt; 1.0 | Default SLA boundary search precision target. The bisection / smooth-isotonic bracket halts when (infeasible_min - feasible_max) / infeasible_min stays below this value, and the cliff detector requires bracket_gap > this * x_hi to report a cliff. 5% mirrors perf_analyzer's --binary-search default. |
| `AIPERF_SEARCH_PLANNER_DEFAULT_WARMUP_SECONDS` | `30.0` | > 0.0, ≤ 100000.0 | Smooth-isotonic SLA planner: default warmup phase duration in seconds injected into each iteration's cfg when ``cfg.sla_warmup_seconds`` is unset. Spec calls for max(30s, 3*inter-batch-time) but inter-batch-time is unknown at planner-time, so 30s is the safe floor. Must be strictly positive: zero defeats the cold-KV-cache rationale that motivates the floor. |
| `AIPERF_SEARCH_PLANNER_FIRST_PROBE_WARMUP_FLOOR` | `60.0` | > 0.0, ≤ 100000.0 | Smooth-isotonic SLA planner: minimum warmup duration in seconds for the first probe at each swept-dim value. Cold KV-cache and CUDA-graph compilation cost is largest the first time we hit a given concurrency, so floor that probe at 60s. Must be strictly positive: zero defeats the cold-KV-cache rationale. |
| `AIPERF_SEARCH_PLANNER_REPLICATE_WARMUP_FLOOR` | `15.0` | > 0.0, ≤ 100000.0 | Smooth-isotonic SLA planner: minimum warmup duration in seconds for replicate probes at an already-probed swept-dim value. Replicates reuse the warm KV-cache / CUDA-graph state, so a shorter warmup suffices. Must be strictly positive: zero defeats the floor. |
| `AIPERF_SEARCH_PLANNER_SLA_PRECISION_REQUESTS` | `{'tight': 10000, 'normal': 1000, 'coarse': 300}` | — | Mapping from ``cfg.sla_precision`` preset name to the ``phases.profiling.requests`` value injected when the user did not set ``requests`` explicitly on the profiling phase. Drives p99 CI width. Each value must be strictly positive — zero/negative request counts surface as iteration-time failures otherwise. Override via JSON, e.g. ``AIPERF_SEARCH_PLANNER_SLA_PRECISION_REQUESTS='{"tight": 20000}'``. |

## SERVERMETRICS

Server metrics collection configuration. Controls server metrics collection frequency, endpoint detection, and shutdown behavior. Metrics are collected from Prometheus-compatible endpoints at the specified interval. Use `--no-server-metrics` CLI flag to disable collection.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | `2.0` | ≥ 0.0, ≤ 30.0 | Time in seconds to continue collecting metrics after profiling completes, allowing server-side metrics to flush/finalize before shutting down (default: 2.0s) |
| `AIPERF_SERVER_METRICS_PROFILE_COMPLETE_RELAY_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 600.0 | Seconds RecordsManager waits for the final server-metrics scrape command response. A timeout is non-fatal because the controller's result join remains the authoritative completion barrier. |
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | `0.333` | ≥ 0.001, ≤ 300.0 | Server metrics collection interval in seconds (default: 333ms, ~3Hz) |
| `AIPERF_SERVER_METRICS_EXPORT_BATCH_SIZE` | `100` | ≥ 1, ≤ 1000000 | Batch size for server metrics jsonl writer export results processor |
| `AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT` | `10` | ≥ 1, ≤ 300 | Timeout in seconds for checking server metrics endpoint reachability during init |
| `AIPERF_SERVER_METRICS_SHUTDOWN_DELAY` | `5.0` | ≥ 1.0, ≤ 300.0 | Delay in seconds before shutting down server metrics service to allow command response transmission |
| `AIPERF_SERVER_METRICS_CR_PROJECTION_MAX_SERIES` | `32` | ≥ 1, ≤ 10000 | Maximum series a single metric may carry into the Kubernetes AIPerfJob status.serverMetrics projection. A metric with more series than this is dropped whole rather than truncated, because a partial series list would decode as a valid-but-wrong aggregate. Guards the 1.5 MB apiserver object ceiling; the WebSocket feed and server_metrics_export.json are unaffected. |
| `AIPERF_SERVER_METRICS_CR_PROJECTION_MAX_LABELS` | `16` | ≥ 1, ≤ 1000 | Maximum Prometheus labels a single series may carry into the Kubernetes AIPerfJob status.serverMetrics projection. Labels form the series identity in the dashboard, so a metric with an over-labeled series is dropped whole rather than having its labels trimmed. |

## SERVICE

Service lifecycle and inter-service communication configuration. Controls timeouts for service registration, startup, shutdown, command handling, connection probing, heartbeats, and profile operations.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SERVICE_COMMAND_RESPONSE_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 1000.0 | Timeout in seconds for command responses |
| `AIPERF_SERVICE_COMMS_REQUEST_TIMEOUT` | `90.0` | ≥ 1.0, ≤ 1000.0 | Timeout in seconds for requests from req_clients to rep_clients |
| `AIPERF_SERVICE_PARENT_LIVENESS_POLL_INTERVAL` | `2.0` | ≥ 0.1, ≤ 60.0 | Interval in seconds at which a child service polls whether the SystemController is still alive. Only used when the direct parent is not the controller (forkserver/spawn start methods), where the kernel's PR_SET_PDEATHSIG cannot cover controller death. |
| `AIPERF_SERVICE_CONNECTION_PROBE_INTERVAL` | `0.1` | ≥ 0.1, ≤ 600.0 | Interval in seconds for connection probes while waiting for initial connection to the zmq message bus |
| `AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT` | `90.0` | ≥ 1.0, ≤ 100000.0 | Maximum time in seconds to wait for connection probe response while waiting for initial connection to the zmq message bus |
| `AIPERF_SERVICE_CREDIT_PROGRESS_REPORT_INTERVAL` | `2.0` | ≥ 1, ≤ 100000.0 | Interval in seconds between credit progress report messages |
| `AIPERF_SERVICE_WARMUP_PROGRESS_LOG_INTERVAL` | `30.0` | ≥ 0.0, ≤ 100000.0 | Interval in seconds between warmup progress heartbeat log messages. Set to 0 to disable. |
| `AIPERF_SERVICE_DISABLE_UVLOOP` | `False` | — | Disable uvloop and use default asyncio event loop instead |
| `AIPERF_SERVICE_HEARTBEAT_INTERVAL` | `5.0` | ≥ 1.0, ≤ 100000.0 | Interval in seconds between heartbeat messages for component services |
| `AIPERF_SERVICE_HEARTBEAT_MISSED_THRESHOLD` | `3` | ≥ 1, ≤ 100 | Consecutive heartbeat intervals a registered service may miss before the watchdog suspects it. A service is only failed after appearing stale on two consecutive watchdog ticks, so worst-case detection is HEARTBEAT_INTERVAL * (threshold + 1) seconds. |
| `AIPERF_SERVICE_FAILURE_SHUTDOWN_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 300.0 | Wall-clock cap on the shutdown path inside AIPerfLifecycleMixin._fail. If cleanup (on_stop hooks, task cancellation) does not complete within this window after a failed on_init/on_start transition, a containerized (operator-managed) service hard-exits via os._exit(1), preventing silent zombie containers when cleanup blocks on a cancelled C-extension call. A local run logs the wedged shutdown and reports the failure normally instead, so the traceback and artifact export are not discarded. |
| `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` | `600.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile configure command |
| `AIPERF_SERVICE_PROFILE_START_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile start command |
| `AIPERF_SERVICE_PROFILE_CANCEL_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for profile cancel command |
| `AIPERF_SERVICE_REGISTRATION_INTERVAL` | `1.0` | ≥ 1.0, ≤ 100000.0 | Interval in seconds between registration attempts for component services |
| `AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS` | `10` | ≥ 1, ≤ 100000 | Maximum number of registration attempts before giving up |
| `AIPERF_SERVICE_REGISTRATION_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for service registration |
| `AIPERF_SERVICE_REGISTRATION_PROGRESS_LOG_INTERVAL` | `5.0` | ≥ 0.1, ≤ 100000.0 | Interval in seconds between 'still waiting for services to register' progress logs emitted by the service registry while blocked |
| `AIPERF_SERVICE_GROUP_HELLO_ATTEMPT_TIMEOUT` | `2.0` | ≥ 0.1, ≤ 1000.0 | Timeout in seconds for a single group-local GroupPeerHello attempt before it is retried against the worker group manager |
| `AIPERF_SERVICE_GROUP_HELLO_TOTAL_TIMEOUT` | `120.0` | ≥ 1.0, ≤ 100000.0 | Total deadline in seconds for a group-local peer to get its GroupPeerHello acknowledged before startup is failed |
| `AIPERF_SERVICE_START_TIMEOUT` | `30.0` | ≥ 1.0, ≤ 100000.0 | Timeout in seconds for service start operations. Also bounds the per-phase wait for the first worker to register with the credit router before credit issuance begins; exceeding it fails the phase. |
| `AIPERF_SERVICE_TASK_CANCEL_TIMEOUT_SHORT` | `2.0` | ≥ 1.0, ≤ 100000.0 | Maximum time in seconds to wait for simple tasks to complete when cancelling |
| `AIPERF_SERVICE_SPAWN_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 100000.0 | Safety-net timeout in seconds for multiprocessing Process.start(). Normal spawns complete in milliseconds; this guards against extreme system conditions (memory pressure, exhausted forkserver) blocking the event loop indefinitely. |
| `AIPERF_SERVICE_ERROR_QUEUE_MAXSIZE` | `256` | ≥ 1, ≤ 65536 | Maximum number of pending subprocess error reports held in the multiprocessing error queue before further reports are dropped. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED` | `True` | — | Enable event loop health monitoring to detect blocked event loops. When enabled, TimingManager and Worker services periodically check if the event loop is responsive and log warnings when latency exceeds the threshold. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_INTERVAL` | `0.25` | ≥ 0.05, ≤ 10.0 | Interval in seconds between event loop health checks (default: 250ms). The monitor sleeps for this duration and measures actual elapsed time to detect blocking. |
| `AIPERF_SERVICE_EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS` | `25.0` | > 1.0, ≤ 10000.0 | Warning threshold in milliseconds for event loop latency (default: 25ms). If the actual sleep duration exceeds the expected duration by this amount, a warning is logged. |
| `AIPERF_SERVICE_HEALTH_ENABLED` | `False` | — | Enable the lightweight health server for Kubernetes liveness/readiness probes. When enabled, non-API services will start an HTTP server serving /healthz and /readyz endpoints. |
| `AIPERF_SERVICE_HEALTH_HOST` | `'127.0.0.1'` | — | Host to bind the health server to. Use '0.0.0.0' for Kubernetes deployments. |
| `AIPERF_SERVICE_HEALTH_PORT` | `8080` | ≥ 1, ≤ 65535 | Port for the health server HTTP endpoints (/healthz, /readyz). |
| `AIPERF_SERVICE_HEALTH_REQUEST_TIMEOUT` | `5.0` | ≥ 0.1, ≤ 60.0 | Timeout in seconds for reading health check HTTP requests. |
| `AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT` | `28000` | ≥ 1024, ≤ 65535 | Windows-only: starting port for the ZMQ IPC TCP-loopback fallback range. Per-endpoint ports are derived as ``base + (sha256_hash mod range)``. No-op on POSIX where ipc:// is used directly. |
| `AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE` | `20000` | ≥ 64, ≤ 60000 | Windows-only: size of the TCP-loopback port window for the ZMQ IPC fallback. Birthday-paradox collision probability for n sockets is ``1 - exp(-n*n/(2*range))``. Widen if AIPerf grows to many more sockets per run, or relocate via ``AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT`` if 28000-48000 conflicts. |

## SWEEPCONTROLLER

Sweep-controller pod settings. Used by the sweep-controller pod (`aiperf.sweep_controller.k8s_executor`) when creating child AIPerfJob CRs.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_SWEEP_CONTROLLER_STALE_CHILD_DELETION_TIMEOUT_SECONDS` | `60.0` | > 0, ≤ 600 | Max seconds the sweep-controller will wait for a same-named AIPerfJob from a prior sweep run to finish cascade-deletion before raising ChildNameConflictError. Hit when a user deletes and recreates a sweep with the same name while old children are still terminating. |
| `AIPERF_SWEEP_CONTROLLER_STALE_CHILD_POLL_INTERVAL_SECONDS` | `2.0` | > 0, ≤ 30 | Poll interval (seconds) while waiting for a deleting same-named AIPerfJob to disappear. See STALE_CHILD_DELETION_TIMEOUT_SECONDS. |
| `AIPERF_SWEEP_CONTROLLER_CANCEL_GRACE_SECONDS` | `120.0` | > 0, ≤ 3600 | Max seconds the sweep-controller will keep polling a child AIPerfJob for a terminal phase after requesting cancel before giving up and advancing the sweep. Bounds the post-cancel wait so a stuck child (stalled operator cancel path, wedged pod, repeatedly-failing JobSet delete) cannot wedge the whole sweep indefinitely. |
| `AIPERF_SWEEP_CONTROLLER_CHILD_MISSING_TIMEOUT_SECONDS` | `300.0` | > 0, ≤ 3600 | Max seconds the sweep-controller will keep polling for a child AIPerfJob that has gone missing (404) before its terminal phase, with no cancel requested, before giving up and advancing the sweep. Hit when a user (or the kube garbage collector) deletes a child AIPerfJob out-of-band mid-run; without this bound the sequential sweep wedges forever on the deleted variation. |

## TIMING

Timing manager configuration. Controls timing-related settings for credit phase execution and scheduling.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
| `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT` | `10.0` | ≥ 1.0, ≤ 300.0 | Timeout in seconds for waiting for cancelled credits to drain after phase timeout |
| `AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL` | `0.1` | ≥ 0.01, ≤ 10.0 | Update interval in seconds for continuous rate ramping (default 0.1s = 100ms) |
| `AIPERF_TIMING_HIGH_RES_TIMER` | `True` | — | Use high-resolution rate-loop pacing instead of event-loop timers, which quantize sub-millisecond sleeps to ~1ms granularity. Restores exact rate delivery and arrival-distribution fidelity at high request rates. Uses a Linux timerfd (kernel hrtimer, ~50us wakeup precision) when available, and a dedicated sleep thread on other platforms (~100us POSIX, ~0.5ms Windows). Set to false to force event-loop timer pacing. |
| `AIPERF_TIMING_MAX_CATCHUP_SECONDS` | `0.01` | ≥ 0.0, ≤ 10.0 | Maximum schedule backlog in seconds the rate loop is allowed to catch up on before re-anchoring to the current time. Event-loop timers oversleep sub-millisecond waits (~1ms granularity under uvloop/libuv); without a catch-up window every oversleep permanently forfeits schedule and high request rates silently under-deliver. Bounded so a genuine multi-second stall still re-anchors instead of firing a burst storm. |

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
| `AIPERF_UI_CONSOLE_EXPORT_WIDTH` | `140` | ≥ 40, ≤ 10000 | Fixed column width used to render the post-run console exporter tables. Applied both to the recording console that produces profile_export_console.txt and to the live console when stdout is not a tty (so non-tty CI logs match the saved artifact). |
| `AIPERF_UI_LOG_REFRESH_INTERVAL` | `0.1` | ≥ 0.01, ≤ 100000.0 | Log viewer refresh interval in seconds (default: 10 FPS) |
| `AIPERF_UI_MIN_UPDATE_PERCENT` | `1.0` | ≥ 0.01, ≤ 100.0 | Minimum percentage difference from last update to trigger a UI update (for non-dashboard UIs) |
| `AIPERF_UI_REALTIME_METRICS_INTERVAL` | `None` | ≥ 0.0, ≤ 1000.0 | Interval in seconds between real-time metrics messages (and the per-tick stats log block). 0 disables the log block; dashboards still poll. When None, `realtime_metrics_interval(ui_type)` auto-defaults to 5.0 under --ui dashboard, 30.0 otherwise. |
| `AIPERF_UI_REALTIME_METRICS_ENABLED` | `False` | — | Enable real-time metrics collection and reporting despite UI type |
| `AIPERF_UI_SPINNER_REFRESH_RATE` | `0.1` | ≥ 0.1, ≤ 100.0 | Progress spinner refresh rate in seconds (default: 10 FPS) |

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
| `AIPERF_WORKER_CLOCK_PROBE_TIMEOUT` | `1.0` | ≥ 0.1, ≤ 100000.0 | Per-probe timeout in seconds for a single startup TimePing/TimePong round trip. Kept short so an unreachable credit ROUTER costs a fast retry instead of consuming AIPERF_WORKER_CLOCK_PROBE_BUDGET on one probe. Kubernetes mode only. |
| `AIPERF_WORKER_CLOCK_PROBE_BUDGET` | `30.0` | ≥ 0.1, ≤ 100000.0 | Total seconds a worker may spend on startup TimePing/TimePong clock-offset RTT probes before giving up and announcing readiness uncalibrated. Hard ceiling on the whole probe sequence, not per probe, so a router that never echoes cannot stall worker registration past AIPERF_SERVICE_REGISTRATION_TIMEOUT. Sized for real cluster startup, where the credit ROUTER is commonly not echoing for the first several seconds after a worker container starts. Kubernetes mode only. |
| `AIPERF_WORKER_CPU_UTILIZATION_FACTOR` | `0.75` | ≥ 0.1, ≤ 1.0 | Factor multiplied by CPU count to determine default max workers (0.0-1.0). Formula: max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP)) |
| `AIPERF_WORKER_ERROR_RECOVERY_TIME` | `3.0` | ≥ 0.1, ≤ 1000.0 | Time in seconds from last error before worker is considered healthy again |
| `AIPERF_WORKER_HEALTH_CHECK_INTERVAL` | `2.0` | ≥ 0.1, ≤ 1000.0 | Interval in seconds between worker health check messages |
| `AIPERF_WORKER_HIGH_LOAD_CPU_USAGE` | `85.0` | ≥ 50.0, ≤ 100.0 | CPU usage percentage threshold for considering a worker under high load |
| `AIPERF_WORKER_HIGH_LOAD_RECOVERY_TIME` | `5.0` | ≥ 0.1, ≤ 1000.0 | Time in seconds from last high load before worker is considered recovered |
| `AIPERF_WORKER_MAX_WORKERS_CAP` | `32` | ≥ 1, ≤ 10000 | Absolute maximum number of workers to spawn, regardless of CPU count |
| `AIPERF_WORKER_MIN_ALIVE_FRACTION` | `0.0` | ≥ 0.0, ≤ 1.0 | Fail the benchmark when the number of dispatchable workers remains below this fraction of the peak ever registered for one worker staleness interval. TimingManager evaluates the count after worker deregistration, so Kubernetes replacement pods can become dispatchable before a transient loss is fatal. 0 disables the check. |
| `AIPERF_WORKER_STALE_TIME` | `10.0` | ≥ 0.1, ≤ 1000.0 | Time in seconds from last status report before worker is considered stale |
| `AIPERF_WORKER_STATUS_SUMMARY_INTERVAL` | `0.5` | ≥ 0.1, ≤ 1000.0 | Interval in seconds between worker status summary messages |
| `AIPERF_WORKER_RAW_RECORD_UPLOAD_TIMEOUT` | `60.0` | ≥ 1.0, ≤ 600.0 | Timeout in seconds to wait for worker pods to upload raw record files to the controller API after benchmark completion. |
| `AIPERF_WORKER_DEFAULT_WORKERS_PER_POD` | `10` | ≥ 1, ≤ 100 | Default number of worker subprocesses per Kubernetes worker pod. Each pod downloads the dataset once and shares it across workers via mmap. Packing exists because a node holds only ~65k ephemeral ports, which caps concurrent connections per node; see RuntimeConfig.workers_per_pod |

## ZMQ

ZMQ socket and communication configuration. Controls ZMQ socket timeouts, keepalive settings, retry behavior, and concurrency limits. These settings affect reliability and performance of the internal message bus.

| Environment Variable | Default | Constraints | Description |
|----------------------|---------|-------------|-------------|
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
| `AIPERF_DEV_MODE` | `False` | — | Enable AIPerf Developer mode for internal metrics and debugging |
| `AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS` | `False` | — | [Developer use only] Show experimental metrics in output (requires DEV_MODE) |
| `AIPERF_DEV_SHOW_INTERNAL_METRICS` | `False` | — | [Developer use only] Show internal and hidden metrics in output (requires DEV_MODE) |
| `AIPERF_DEV_TRACE_SERVICES` | `None` | — | List of services to enable TRACE logging for (comma-separated or multiple flags) |
