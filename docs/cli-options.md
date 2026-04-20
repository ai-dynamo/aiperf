---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Command Line Options
---

# Command Line Options

## `aiperf` Commands

### [`--install-completion`](#aiperf---install-completion)

Install shell completion for this application.

### [`analyze-trace`](#aiperf-analyze-trace)

Analyze a mooncake trace file for ISL/OSL distributions

### [`profile`](#aiperf-profile)

Benchmark AI models and measure performance metrics

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Accuracy](#accuracy) • [Multi-Run](#multi-run)

### [`plot`](#aiperf-plot)

Generate visualizations from profiling data

### [`plugins`](#aiperf-plugins)

Explore and validate AIPerf plugins

### [`service`](#aiperf-service)

Run an individual AIPerf service in a single process

### [`config init`](#aiperf-config-init)

Generate a starter configuration from bundled templates.

### [`config validate`](#aiperf-config-validate)

Validate a YAML configuration file.

### [`config show`](#aiperf-config-show)

Display a configuration file with all defaults expanded.

### [`config schema`](#aiperf-config-schema)

Output the JSON schema for AIPerfConfig.

### [`config diff`](#aiperf-config-diff)

Compare two configuration files and show differences.

### [`config generate`](#aiperf-config-generate)

Generate YAML configuration from CLI options.

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Accuracy](#accuracy) • [Multi-Run](#multi-run) • [Parameters](#parameters)

### [`kube init`](#aiperf-kube-init)

Generate a starter configuration template

### [`kube validate`](#aiperf-kube-validate)

Validate AIPerfJob YAML files against the CRD schema

### [`kube profile`](#aiperf-kube-profile)

Run a benchmark in Kubernetes

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Accuracy](#accuracy) • [Multi-Run](#multi-run) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets) • [Parameters](#parameters)

### [`kube generate`](#aiperf-kube-generate)

Generate Kubernetes YAML manifests

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Accuracy](#accuracy) • [Multi-Run](#multi-run) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets) • [Parameters](#parameters)

### [`kube attach`](#aiperf-kube-attach)

Attach to a running benchmark and stream progress

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube list`](#aiperf-kube-list)

List benchmark jobs and their status

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube logs`](#aiperf-kube-logs)

Retrieve logs from benchmark pods

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube results`](#aiperf-kube-results)

Retrieve benchmark results

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube debug`](#aiperf-kube-debug)

Run diagnostic analysis on a deployment

### [`kube watch`](#aiperf-kube-watch)

Watch a running benchmark with live status and diagnostics

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube preflight`](#aiperf-kube-preflight)

Run pre-flight checks against the target Kubernetes cluster

[Kubernetes](#kubernetes) • [Parameters](#parameters)

### [`kube dashboard`](#aiperf-kube-dashboard)

Open the operator results server UI in your browser

[Kubernetes](#kubernetes) • [Parameters](#parameters)

<hr/>

## `aiperf --install-completion`

Install shell completion for this application.

This command generates and installs the completion script to the appropriate location for your shell. After installation, you may need to restart your shell or source your shell configuration file.

#### `--shell` `<str>`

Shell type for completion. If not specified, attempts to auto-detect current shell.

#### `-o`, `--output` `<str>`

Output path for the completion script. If not specified, uses shell-specific default.

<hr/>

## `aiperf analyze-trace`

Analyze a mooncake trace file for ISL/OSL distributions

#### `--input-file` `<str>` _(Required)_

Path to input mooncake trace JSONL file.

#### `--block-size` `<int>`

KV cache block size for analysis.
<br/>_Default: `512`_

#### `--output-file` `<str>`

Output path for analysis report (JSON).

<hr/>

## `aiperf profile`

Benchmark AI models and measure performance metrics

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds (0 = no timeout). Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--extra-inputs` `<list>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<list>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. 'sharegpt4o'). Only applies when --public-dataset uses a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0).

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0).

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

#### `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases.
<br/>_Default: `1`_

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0).

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases. If not set, uses the --concurrency value.
<br/>_Default: `1`_

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Concurrency-controlled immediate dispatch. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean value.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean value.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

Mean value.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

Mean value.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Default: `44100`_

#### `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Default: `0`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Auto-set random seed if not specified for workload consistency.
<br/>_Default: `True`_

<hr/>

## `aiperf plot`

Generate visualizations from profiling data

#### `--paths`, `--empty-paths` `<list>`

Paths to profiling run directories. Defaults to ./artifacts if not specified.

#### `--output` `<str>`

Directory to save generated plots. Defaults to &lt;first_path>/plots if not specified.

#### `--theme` `<str>`

Plot theme: 'light' (white background) or 'dark' (dark background).
<br/>_Default: `light`_

#### `--config` `<str>`

Path to custom plot configuration YAML file. If not specified, auto-creates and uses ~/.aiperf/plot_config.yaml.

#### `--verbose`, `--no-verbose`

Show detailed error tracebacks in console (errors are always logged to ~/.aiperf/plot.log).

#### `--dashboard`, `--no-dashboard`

Launch interactive dashboard server instead of generating static PNGs.

#### `--host` `<str>`

Host for dashboard server (only used with --dashboard).
<br/>_Default: `127.0.0.1`_

#### `--port` `<int>`

Port for dashboard server (only used with --dashboard).
<br/>_Default: `8050`_

<hr/>

## `aiperf plugins`

Explore and validate AIPerf plugins

#### `--category` `<str>`

Category to explore.
<br/>_Choices: [`accuracy_benchmark`, `accuracy_grader`, `api_router`, `arrival_pattern`, `communication`, `communication_client`, `console_exporter`, `custom_dataset_loader`, `data_exporter`, `dataset_backing_store`, `dataset_client_store`, `dataset_composer`, `dataset_sampler`, `endpoint`, `gpu_telemetry_collector`, `plot`, `public_dataset_loader`, `ramp`, `record_processor`, `results_processor`, `service`, `service_manager`, `timing_strategy`, `transport`, `ui`, `url_selection_strategy`, `zmq_proxy`]_

#### `--name` `<str>`

Type name for details.

#### `-a`, `--all`, `--no-all`

Show all categories and plugins.

#### `-v`, `--validate`, `--no-validate`

Validate plugins.yaml.

<hr/>

## `aiperf service`

Run an individual AIPerf service in a single process

#### `--type` `<str>` _(Required)_

Service type to run.
<br/>_Choices: [`api`, `dataset_manager`, `gpu_telemetry_manager`, `record_processor`, `records_manager`, `server_metrics_manager`, `system_controller`, `timing_manager`, `worker`, `worker_manager`, `worker_group_manager`]_

#### `--benchmark-run` `<str>`

Path to a BenchmarkRun JSON file. The service bootstraps with a fully-built BenchmarkRun including metadata, variation, and trial info.

#### `--service-id` `<str>`

Unique identifier for the service instance. Useful when running multiple instances of the same service type.

#### `--health-host` `<str>`

Host to bind the health server to. Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable.

#### `--health-port` `<int>`

HTTP port for health endpoints (/healthz, /readyz). Required for Kubernetes liveness and readiness probes. Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable.

#### `--api-port` `<int>`

HTTP port for API endpoints (e.g., /api/dataset, /api/progress). Only used by services that expose HTTP APIs.

<hr/>

## `aiperf config init`

Generate a starter configuration from bundled templates.

Without arguments, generates the 'minimal' template. Use --list to browse all 19 bundled templates organized by category, or --search to find templates by keyword.

Use --model and --url to pre-fill the two fields every config needs, so the generated file is ready to run without editing.

**Examples:**

```bash
# List all available templates
aiperf config init --list

# List with tags, features, and difficulty
aiperf config init --list --verbose

# Search for sweep-related templates
aiperf config init --search sweep

# Filter by category
aiperf config init --list --category "Load Testing"

# Generate the minimal starter config
aiperf config init

# Generate a specific template
aiperf config init --template goodput_slo

# Generate with your model and endpoint pre-filled
aiperf config init --template latency_test \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --url http://my-server:8000/v1/chat/completions

# Save to a file
aiperf config init --template sweep_distributions --output benchmark.yaml

# Pipe to a file
aiperf config init --template latency_test > my_benchmark.yaml
```

#### `-t`, `--template` `<str>`

Template name to use (e.g. 'minimal', 'goodput_slo'). Run with --list to see all available templates.

#### `-l`, `--list`, `--no-list`

List all available templates grouped by category.

#### `-s`, `--search` `<str>`

Search templates by keyword (matches name, description, tags, features).

#### `-c`, `--category` `<str>`

Filter templates by category (substring match).

#### `-v`, `--verbose`, `--no-verbose`

Show tags, features, and difficulty in template listings.

#### `--model` `<str>`

Override model name in the generated config.

#### `--url` `<str>`

Override endpoint URL in the generated config.

#### `-o`, `--output` `<str>`

Output file path. If not specified, prints to stdout.

<hr/>

## `aiperf config validate`

Validate a YAML configuration file.

Checks that the configuration is valid YAML, conforms to the AIPerfConfig schema, and has no missing required fields.

**Examples:**

```bash
# Validate a config file
aiperf config validate benchmark.yaml

# Validate without environment variable interpolation
aiperf config validate benchmark.yaml --no-interpolate

# Strict validation (fail on warnings)
aiperf config validate benchmark.yaml --strict
```

#### `--path` `<str>` _(Required)_

Path to the YAML configuration file.

#### `--interpolate`, `--no-interpolate`

Whether to interpolate environment variables.
<br/>_Default: `True`_

#### `--strict`, `--no-strict`

Fail on warnings (e.g., unused fields).

<hr/>

## `aiperf config show`

Display a configuration file with all defaults expanded.

Loads the configuration, applies all defaults, and outputs the complete configuration in the specified format.

**Examples:**

```bash
# Show config with defaults as YAML
aiperf config show benchmark.yaml

# Show config as JSON
aiperf config show benchmark.yaml --format json

# Show config without environment interpolation
aiperf config show benchmark.yaml --no-interpolate
```

#### `--path` `<str>` _(Required)_

Path to the YAML configuration file.

#### `--format` `<str>`

Output format: 'yaml' or 'json'.
<br/>_Default: `yaml`_

#### `--interpolate`, `--no-interpolate`

Whether to interpolate environment variables.
<br/>_Default: `True`_

<hr/>

## `aiperf config schema`

Output the JSON schema for AIPerfConfig.

Generates a JSON Schema document that describes the complete AIPerfConfig structure. Useful for IDE integration and validation tooling.

**Examples:**

```bash
# Print schema to stdout
aiperf config schema

# Save schema to file
aiperf config schema --output aiperf-schema.json
```

#### `--output` `<str>`

Path to write the schema file. If not provided, prints to stdout.

<hr/>

## `aiperf config diff`

Compare two configuration files and show differences.

Loads both configurations, normalizes them with defaults, and shows the differences between them. Useful for understanding how configs differ or verifying changes.

**Examples:**

```bash
# Compare two configs (text output)
aiperf config diff baseline.yaml experiment.yaml

# Compare with JSON output
aiperf config diff baseline.yaml experiment.yaml --format json
```

#### `--config1` `<str>` _(Required)_

Path to first YAML configuration file.

#### `--config2` `<str>` _(Required)_

Path to second YAML configuration file.

#### `--format` `<str>`

Output format: 'text' or 'json'.
<br/>_Default: `text`_

<hr/>

## `aiperf config generate`

Generate YAML configuration from CLI options.

Takes the same CLI flags as 'aiperf profile' and outputs the equivalent YAML configuration. Useful for migrating from CLI-based to YAML-based configuration.

**Examples:**

```bash
# Generate YAML config from CLI options (prints to stdout)
aiperf config generate --model llama-3.1-8B --url localhost:8000 \
    --request-rate 10 --request-count 1000

# Save to file
aiperf config generate --model llama-3.1-8B --url localhost:8000 \
    --concurrency 32 --request-count 1000 --output benchmark.yaml

# Generate as JSON
aiperf config generate --model llama-3.1-8B --url localhost:8000 \
    --format json
```

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds (0 = no timeout). Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--extra-inputs` `<list>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<list>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. 'sharegpt4o'). Only applies when --public-dataset uses a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0).

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0).

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

#### `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases.
<br/>_Default: `1`_

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0).

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases. If not set, uses the --concurrency value.
<br/>_Default: `1`_

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Concurrency-controlled immediate dispatch. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean value.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean value.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

Mean value.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

Mean value.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Default: `44100`_

#### `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Default: `0`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Auto-set random seed if not specified for workload consistency.
<br/>_Default: `True`_

### Parameters

#### `--output` `<str>`

Path to write the config. If not provided, prints to stdout.

#### `--format` `<str>`

Output format: 'yaml' or 'json'.
<br/>_Default: `yaml`_

<hr/>

## `aiperf kube init`

Generate a starter configuration template

#### `-o`, `--output` `<str>`

Output file path. If not specified, prints to stdout.

<hr/>

## `aiperf kube validate`

Validate AIPerfJob YAML files against the CRD schema

#### `--files` `<list>` _(Required)_

One or more AIPerfJob YAML file paths to validate.

#### `-s`, `--strict`, `--no-strict`

Fail on warnings such as unknown spec fields.

#### `-o`, `--output` `<str>`

Output format: "text" for human-readable, "json" for machine-parseable.
<br/>_Default: `text`_

<hr/>

## `aiperf kube profile`

Run a benchmark in Kubernetes

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds (0 = no timeout). Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--extra-inputs` `<list>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<list>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. 'sharegpt4o'). Only applies when --public-dataset uses a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0).

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0).

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

#### `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases.
<br/>_Default: `1`_

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0).

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases. If not set, uses the --concurrency value.
<br/>_Default: `1`_

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Concurrency-controlled immediate dispatch. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean value.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean value.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

Mean value.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

Mean value.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Default: `44100`_

#### `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Default: `0`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Auto-set random seed if not specified for workload consistency.
<br/>_Default: `True`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

#### `--name` `<str>`

Human-readable name for the benchmark job (DNS label, max 40 chars).

#### `--image` `<str>` _(Required)_

AIPerf container image to use for Kubernetes deployment.
<br/>_Constraints: min: 1_

#### `--image-pull-policy` `<str>`

Image pull policy (Always, IfNotPresent, Never). Use 'Never' for minikube (or local clusters) with locally loaded images.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `Always` |  | Every time the kubelet launches a container, it queries the registry to resolve the name to a digest. Uses cached image if digest matches, otherwise pulls the image. |
| `Never` |  | The kubelet does not try fetching the image. Startup fails if the image is not already present locally. |
| `IfNotPresent` |  | The image is pulled only if it is not already present locally. |

#### `--workers-max` `<int>`

Total number of workers. Automatically distributed across pods based on --workers-per-pod (default 10). E.g., --workers-max 50 = 5 pods × 10 workers.
<br/>_Constraints: > 0_
<br/>_Default: `10`_

#### `--ttl-seconds` `<int>`

Seconds to keep pods after completion (None to disable TTL).
<br/>_Default: `300`_

### Kubernetes Node Placement

#### `--node-selector` `<str>`

Node selector labels (e.g., {'gpu': 'true'}).
<br/>_Default: `{}`_

#### `--tolerations` `<list>`

Pod tolerations for scheduling on tainted nodes.
<br/>_Default: `[]`_

### Kubernetes Scheduling

#### `--queue-name` `<str>`

Kueue LocalQueue name for gang-scheduling. When set, the JobSet is submitted to Kueue for quota-managed admission.

#### `--priority-class` `<str>`

Kueue WorkloadPriorityClass name for scheduling priority.

### Kubernetes Metadata

#### `--annotations` `<str>`

Additional pod annotations.
<br/>_Default: `{}`_

#### `--labels` `<str>`

Additional pod labels.
<br/>_Default: `{}`_

### Kubernetes Secrets

#### `--image-pull-secrets` `<list>`

Image pull secret names.
<br/>_Default: `[]`_

#### `--env-vars` `<str>`

Extra environment variables (key: value).
<br/>_Default: `{}`_

#### `--env-from-secrets` `<str>`

Environment variables from secrets (ENV_NAME: secret_name/key).
<br/>_Default: `{}`_

#### `--secret-mounts` `<list>`

Secret volume mounts.
<br/>_Default: `[]`_

#### `--service-account` `<str>`

Service account name for pods.

### Parameters

#### `-d`, `--detach`, `--no-detach`

Exit immediately after deploying (don't wait for completion). Automatically enabled in non-interactive environments (pipes, CI/CD).

#### `--no-wait`

Don't wait for pods to be ready before attaching (advanced).

#### `--attach-port` `<int>`

Local port for API port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

#### `--skip-endpoint-check`

Skip endpoint health validation before deploying.
<br/>_Flag (no value required)_

#### `--dry-run`

Print the AIPerfJob CR as JSON without submitting it.
<br/>_Flag (no value required)_

#### `--no-operator`

Force direct deployment without the operator. Automatically enabled if the AIPerfJob CRD is not installed on the cluster.

<hr/>

## `aiperf kube generate`

Generate Kubernetes YAML manifests

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds (0 = no timeout). Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--extra-inputs` `<list>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<list>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. 'sharegpt4o'). Only applies when --public-dataset uses a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0).

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0).

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

#### `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases.
<br/>_Default: `1`_

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0).

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). Primary control for concurrency phases. If not set, uses the --concurrency value.
<br/>_Default: `1`_

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Concurrency-controlled immediate dispatch. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests after duration expires (must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean value.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean value.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

Mean value.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

Mean value.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

Mean value.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

Mean value.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

Standard deviation. 0 = deterministic.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Default: `44100`_

#### `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Default: `0`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Auto-set random seed if not specified for workload consistency.
<br/>_Default: `True`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

#### `--name` `<str>`

Human-readable name for the benchmark job (DNS label, max 40 chars).

#### `--image` `<str>` _(Required)_

AIPerf container image to use for Kubernetes deployment.
<br/>_Constraints: min: 1_

#### `--image-pull-policy` `<str>`

Image pull policy (Always, IfNotPresent, Never). Use 'Never' for minikube (or local clusters) with locally loaded images.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `Always` |  | Every time the kubelet launches a container, it queries the registry to resolve the name to a digest. Uses cached image if digest matches, otherwise pulls the image. |
| `Never` |  | The kubelet does not try fetching the image. Startup fails if the image is not already present locally. |
| `IfNotPresent` |  | The image is pulled only if it is not already present locally. |

#### `--workers-max` `<int>`

Total number of workers. Automatically distributed across pods based on --workers-per-pod (default 10). E.g., --workers-max 50 = 5 pods × 10 workers.
<br/>_Constraints: > 0_
<br/>_Default: `10`_

#### `--ttl-seconds` `<int>`

Seconds to keep pods after completion (None to disable TTL).
<br/>_Default: `300`_

### Kubernetes Node Placement

#### `--node-selector` `<str>`

Node selector labels (e.g., {'gpu': 'true'}).
<br/>_Default: `{}`_

#### `--tolerations` `<list>`

Pod tolerations for scheduling on tainted nodes.
<br/>_Default: `[]`_

### Kubernetes Scheduling

#### `--queue-name` `<str>`

Kueue LocalQueue name for gang-scheduling. When set, the JobSet is submitted to Kueue for quota-managed admission.

#### `--priority-class` `<str>`

Kueue WorkloadPriorityClass name for scheduling priority.

### Kubernetes Metadata

#### `--annotations` `<str>`

Additional pod annotations.
<br/>_Default: `{}`_

#### `--labels` `<str>`

Additional pod labels.
<br/>_Default: `{}`_

### Kubernetes Secrets

#### `--image-pull-secrets` `<list>`

Image pull secret names.
<br/>_Default: `[]`_

#### `--env-vars` `<str>`

Extra environment variables (key: value).
<br/>_Default: `{}`_

#### `--env-from-secrets` `<str>`

Environment variables from secrets (ENV_NAME: secret_name/key).
<br/>_Default: `{}`_

#### `--secret-mounts` `<list>`

Secret volume mounts.
<br/>_Default: `[]`_

#### `--service-account` `<str>`

Service account name for pods.

### Parameters

#### `--operator`

Output an AIPerfJob CR (requires operator on target cluster).
<br/>_Flag (no value required)_

#### `--no-operator`

Output raw K8s manifests (Namespace, RBAC, ConfigMap, JobSet).

<hr/>

## `aiperf kube attach`

Attach to a running benchmark and stream progress

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to attach to (default: last deployed job).

#### `-p`, `--port` `<int>`

Local port for port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube list`

List benchmark jobs and their status

### Parameters

#### `--job-id` `<str>`

Specific job ID to check.

#### `-A`, `--all-namespaces`, `--no-all-namespaces`

Search in all namespaces.
<br/>_Default: `True`_

#### `--running`, `--no-running`

Show only running jobs.

#### `--completed`, `--no-completed`

Show only completed jobs.

#### `--failed`, `--no-failed`

Show only failed jobs.

#### `-w`, `--wide`, `--no-wide`

Show additional columns (model, endpoint, error).

#### `--watch`, `--no-watch`

Refresh the list every few seconds until interrupted.

#### `--interval` `<int>`

Refresh interval in seconds (used with --watch).
<br/>_Default: `5`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube logs`

Retrieve logs from benchmark pods

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to get logs from (default: last deployed job).

#### `--container` `<str>`

Specific container name to get logs from.

#### `-f`, `--follow`, `--no-follow`

Follow log output in real-time.

#### `--tail` `<int>`

Number of lines from the end to show.

#### `-o`, `--output` `<str>`

Directory to save log files (one per pod). Prints to stdout if not set.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube results`

Retrieve benchmark results

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to get results from (default: last deployed job).

#### `--output` `<str>`

Output directory for results (default: ./artifacts/{name}).

#### `--from-pods`, `--no-from-pods`

Retrieve results from benchmark pods instead of the operator. Tries the controller API first, falls back to kubectl cp.

#### `-a`, `--all`, `--summary-only`

Download all artifacts. Use --summary-only to download only summary results.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--shutdown`, `--no-shutdown`

Shut down the API service after downloading results. Only takes effect with --from-pods.

#### `--port` `<int>`

Local port for API port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

#### `--operator-namespace` `<str>`

Namespace where the operator is deployed.
<br/>_Default: `aiperf-system`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube debug`

Run diagnostic analysis on a deployment

#### `-n`, `--namespace` `<str>`

Kubernetes namespace to inspect.

#### `-j`, `--job-id` `<str>`

Specific AIPerf job ID to diagnose.

#### `--kubeconfig` `<str>`

Path to kubeconfig file.

#### `--kube-context` `<str>`

Kubernetes context to use.

#### `-v`, `--verbose`, `--no-verbose`

Show detailed output including pod logs.

#### `-A`, `--all-namespaces`, `--no-all-namespaces`

Inspect all namespaces with AIPerf deployments.

<hr/>

## `aiperf kube watch`

Watch a running benchmark with live status and diagnostics

### Parameters

#### `--job-id` `<str>`

Job to watch (default: last deployed / auto-detect).

#### `-A`, `--all`, `--no-all`

Watch all running jobs.

#### `-o`, `--output` `<str>`

Output format: rich (TUI), text (plain log lines), or json (NDJSON).
<br/>_Default: `rich`_

#### `-i`, `--interval` `<float>`

Refresh interval in seconds.
<br/>_Default: `2.0`_

#### `-f`, `--follow-logs`, `--no-follow-logs`

Include live log tail in output.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube preflight`

Run pre-flight checks against the target Kubernetes cluster

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

### Parameters

#### `-i`, `--image` `<str>`

Container image to verify accessibility.

#### `-e`, `--endpoint-url` `<str>`

LLM endpoint URL to test connectivity.

#### `-w`, `--workers` `<int>`

Planned number of worker pods (for resource projection).
<br/>_Default: `1`_

#### `-o`, `--output` `<str>`

Output format: "text" for human-readable, "json" for machine-parseable.
<br/>_Default: `text`_

<hr/>

## `aiperf kube dashboard`

Open the operator results server UI in your browser

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

### Parameters

#### `--port` `<int>`

Local port to bind (default: 0 = ephemeral).
<br/>_Default: `0`_

#### `--operator-namespace` `<str>`

Namespace where the operator is deployed.
<br/>_Default: `aiperf-system`_

#### `--no-browser`, `--no-no-browser`

Print the URL instead of opening a browser.
