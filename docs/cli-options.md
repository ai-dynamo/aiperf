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

### [`speed-bench-report`](#aiperf-speed-bench-report)

Assemble per-category SPEED-Bench aiperf results into a matrix report

### [`profile`](#aiperf-profile)

Benchmark AI models and measure performance metrics

[Endpoint](#endpoint) • [Input](#input) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Prompt](#prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Conversation Input](#conversation-input) • [Output](#output) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [GPU Telemetry](#gpu-telemetry) • [Server Metrics](#server-metrics) • [ZMQ Communication](#zmq-communication) • [Parameters](#parameters) • [Workers](#workers) • [Service](#service)

### [`plot`](#aiperf-plot)

Generate visualizations from profiling data

### [`plugins`](#aiperf-plugins)

Explore and validate AIPerf plugins

### [`service`](#aiperf-service)

Run an individual AIPerf service in a single process

### [`proxy`](#aiperf-proxy)

Run a standalone ZMQ proxy (for Kubernetes sidecar deployments)

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

[Endpoint](#endpoint) • [Input](#input) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Prompt](#prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Conversation Input](#conversation-input) • [Output](#output) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [GPU Telemetry](#gpu-telemetry) • [Server Metrics](#server-metrics) • [ZMQ Communication](#zmq-communication) • [Parameters](#parameters) • [Workers](#workers) • [Service](#service)

### [`config init`](#aiperf-config-init)

Generate a starter configuration from bundled templates.

### [`kube init`](#aiperf-kube-init)

Generate a starter configuration template

### [`kube validate`](#aiperf-kube-validate)

Validate AIPerfJob YAML files against the CRD schema

### [`kube profile`](#aiperf-kube-profile)

Run a benchmark in Kubernetes

[Endpoint](#endpoint) • [Input](#input) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Prompt](#prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Conversation Input](#conversation-input) • [Output](#output) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [GPU Telemetry](#gpu-telemetry) • [Server Metrics](#server-metrics) • [ZMQ Communication](#zmq-communication) • [Parameters](#parameters) • [Workers](#workers) • [Service](#service) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets)

### [`kube sweep`](#aiperf-kube-sweep)

Run a parameter sweep or multi-run benchmark in Kubernetes

[Endpoint](#endpoint) • [Input](#input) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Prompt](#prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Conversation Input](#conversation-input) • [Output](#output) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [GPU Telemetry](#gpu-telemetry) • [Server Metrics](#server-metrics) • [ZMQ Communication](#zmq-communication) • [Parameters](#parameters) • [Workers](#workers) • [Service](#service) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets)

### [`kube generate`](#aiperf-kube-generate)

Generate Kubernetes YAML manifests

[Endpoint](#endpoint) • [Input](#input) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Prompt](#prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Conversation Input](#conversation-input) • [Output](#output) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [GPU Telemetry](#gpu-telemetry) • [Server Metrics](#server-metrics) • [ZMQ Communication](#zmq-communication) • [Parameters](#parameters) • [Workers](#workers) • [Service](#service) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets)

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

### [`kube results list-runs`](#aiperf-kube-results-list-runs)

List all historical runs of a benchmark job.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube show`](#aiperf-kube-show)

Render an AIPerfJob CR with Jinja2/env-vars resolved

### [`kube debug`](#aiperf-kube-debug)

Run diagnostic analysis on a deployment

[Kubernetes](#kubernetes) • [Parameters](#parameters)

### [`kube watch`](#aiperf-kube-watch)

Watch a running benchmark with live status and diagnostics

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube preflight`](#aiperf-kube-preflight)

Run pre-flight checks against the target Kubernetes cluster

[Kubernetes](#kubernetes) • [Parameters](#parameters)

### [`kube dashboard`](#aiperf-kube-dashboard)

Open the operator results server UI in your browser

[Kubernetes](#kubernetes) • [Parameters](#parameters)

### [`synthesize`](#aiperf-synthesize)

Synthesize a dataset workload (e.g. agentic-code)

### [`validate`](#aiperf-validate)

Validate a benchmark artifact (e.g. mooncake-trace)

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

## `aiperf speed-bench-report`

Assemble per-category SPEED-Bench aiperf results into a matrix report

#### `--paths`, `--empty-paths` `<list>` _(Required)_

Run directories or parent directories containing run subdirectories.

#### `--output` `<str>`

Output CSV file path. Defaults to ./speed_bench_report.csv.
<br/>_Default: `speed_bench_report.csv`_

#### `--format` `<str>`

Output format - 'csv', 'table', or 'both'. Defaults to 'both'.
<br/>_Default: `both`_

#### `--metric` `<str>`

Which metric to report - 'accept_length', 'accept_rate', or 'throughput'. Defaults to 'accept_length'.
<br/>_Default: `accept_length`_

<hr/>

## `aiperf profile`

Benchmark AI models and measure performance metrics

### Endpoint

#### `-m -m`, `--model-names`, `--model`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy`, `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `--custom-endpoint`, `--endpoint`, `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type`, `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`, `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u -u`, `--url`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`).
<br/>_Constraints: min: 1_
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy`, `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds`, `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Default: `21600`_

#### `--ready-check-timeout`, `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--api-key`, `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type`, `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`, `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`, `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy`, `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`, `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type`, `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

### Input

#### `--extra-inputs`, `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H -H`, `--header`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file`, `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--fixed-schedule`, `--fixed-schedule`, `--no-fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.

#### `--fixed-schedule-auto-offset`, `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset`, `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset`, `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--public-dataset`, `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset`, `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type`, `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy`, `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed`, `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.

#### `--goodput`, `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format`, `--audio-format`, `--audio-format`, `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths`, `--audio-depths`, `--audio-depths`, `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format`, `--image-format`, `--image-format`, `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration`, `--video-duration`, `--video-duration`, `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps`, `--video-fps`, `--video-fps`, `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width`, `--video-width`, `--video-width`, `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height`, `--video-height`, `--video-height`, `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type`, `--video-synth-type`, `--video-synth-type`, `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format`, `--video-format`, `--video-format`, `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec`, `--video-codec`, `--video-codec`, `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8000, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Prompt

#### `-b -b -b -b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets.
<br/>_Constraints: ≥ 0_
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).

#### `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length.
<br/>_Constraints: ≥ 0_

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Rankings

#### `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete.
<br/>_Constraints: ≥ 1_

#### `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir`, `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file`, `--profile-export-prefix`, `--profile-export-file` `<str>`

Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: `.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.

#### `--export-level`, `--profile-export-level`, `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration`, `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).

#### `--export-http-trace`, `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--export-per-chunk-data`, `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

#### `--show-trace-timing`, `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer`, `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision`, `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`, `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration`, `--benchmark-duration` `<float>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics.
<br/>_Constraints: > 0_

#### `--benchmark-grace-period`, `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency`, `--concurrency` `<int>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate.
<br/>_Constraints: ≥ 1_

#### `--prefill-concurrency`, `--prefill-concurrency` `<int>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously.
<br/>_Constraints: ≥ 1_

#### `--request-rate`, `--request-rate` `<float>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds).
<br/>_Constraints: > 0_

#### `--arrival-pattern`, `--request-rate-mode`, `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness`, `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests`, `--request-count`, `--num-requests` `<int>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-count`, `--num-warmup-requests`, `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration`, `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions`, `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency`, `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency`, `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate`, `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern`, `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period`, `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--request-cancellation-rate`, `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay`, `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

#### `--user-centric-rate`, `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users`, `--num-users` `<int>`

The number of initial users to use for --user-centric-rate mode.
<br/>_Constraints: ≥ 1_

#### `--concurrency-ramp-duration`, `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration`, `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--warmup-concurrency-ramp-duration`, `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration`, `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration`, `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration`, `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### Multi-Run

#### `--num-profile-runs`, `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds`, `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level`, `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--convergence-metric`, `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat`, `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold`, `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean (default 0.10 = 10%). For cv mode: maximum coefficient of variation (default 0.10 = 10%). For distribution mode: KS test p-value threshold (default 0.10). Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.1`_

#### `--convergence-mode`, `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `ci_width` | _default_ | Stop when Student's t confidence interval width relative to mean is below threshold. |
| `cv` |  | Stop when coefficient of variation (std/mean) is below threshold. |
| `distribution` |  | Stop when KS test p-value indicates latest run matches prior runs. |

### Accuracy

#### `--accuracy-benchmark`, `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks`, `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots`, `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Constraints: ≥ 0, ≤ 8_
<br/>_Default: `0`_

#### `--accuracy-enable-cot`, `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader`, `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt`, `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`, `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Automatically collects from inference endpoint base_url + `/metrics`. Optionally specify additional custom Prometheus-compatible endpoint URLs (e.g., http://node1:8081/metrics, http://node2:9090/metrics). Use `--no-server-metrics` to disable collection. Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Specify which output formats to generate for server metrics. Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` | _default_ | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` | _default_ | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` | _default_ | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### ZMQ Communication

#### `--zmq-host`, `--zmq-host`, `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path`, `--zmq-ipc-path`, `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

### Parameters

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmqdualbind.event-bus-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmqdualbind.event-bus-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmqdualbind.dataset-manager-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmqdualbind.dataset-manager-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmqdualbind.raw-inference-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmqdualbind.raw-inference-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmqdualbind.ipc-path` `<str>`

Directory path for IPC socket files.

#### `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmqdualbind.tcp-host` `<str>`

TCP bind host for proxies (Defaults to 127.0.0.1 for localhost, use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmqdualbind.controller-host` `<str>`

Controller host for remote TCP connections. When set, services connect via TCP to this host instead of IPC. Set via JobSet DNS in Kubernetes.

#### `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmqdualbind.records-push-pull-tcp-port` `<int>`

TCP port for records push/pull communication with remote workers.
<br/>_Default: `5557`_

#### `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmqdualbind.credit-router-tcp-port` `<int>`

TCP port for credit router communication with remote workers.
<br/>_Default: `5564`_

### Workers

#### `--workers-max`, `--max-workers`, `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service

#### `--log-level` `<str>`

Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.

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

Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of `RecordProcessor` services to spawn for parallel metric computation. Higher request rates require more processors to keep up with incoming records. If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).
<br/>_Constraints: ≥ 1_

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

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
<br/>_Choices: [`accuracy_benchmark`, `accuracy_grader`, `api_router`, `arrival_pattern`, `communication`, `communication_client`, `console_exporter`, `custom_dataset_loader`, `data_exporter`, `dataset_backing_store`, `dataset_client_store`, `dataset_composer`, `dataset_sampler`, `endpoint`, `gpu_telemetry_collector`, `gpu_telemetry_processor`, `plot`, `public_dataset_loader`, `ramp`, `record_processor`, `results_processor`, `server_metrics_processor`, `service`, `service_manager`, `timing_strategy`, `transport`, `ui`, `url_selection_strategy`, `zmq_proxy`]_

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

## `aiperf proxy`

Run a standalone ZMQ proxy (for Kubernetes sidecar deployments)

#### `--benchmark-run` `<str>` _(Required)_

Path to the BenchmarkRun JSON file. Used to resolve the proxy's bind addresses from the resolved communication config.

#### `--kind` `<str>`

Which proxy to run. Currently only 'event_bus' is supported.
<br/>_Default: `event_bus`_

#### `--health-port` `<int>`

HTTP port for /healthz and /readyz. Falls back to AIPERF_SERVICE_HEALTH_PORT.

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

#### `-m -m`, `--model-names`, `--model`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy`, `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `--custom-endpoint`, `--endpoint`, `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type`, `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`, `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u -u`, `--url`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`).
<br/>_Constraints: min: 1_
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy`, `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds`, `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Default: `21600`_

#### `--ready-check-timeout`, `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--api-key`, `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type`, `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`, `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`, `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy`, `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`, `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type`, `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

### Input

#### `--extra-inputs`, `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H -H`, `--header`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file`, `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--fixed-schedule`, `--fixed-schedule`, `--no-fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.

#### `--fixed-schedule-auto-offset`, `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset`, `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset`, `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--public-dataset`, `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset`, `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type`, `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy`, `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed`, `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.

#### `--goodput`, `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format`, `--audio-format`, `--audio-format`, `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths`, `--audio-depths`, `--audio-depths`, `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format`, `--image-format`, `--image-format`, `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration`, `--video-duration`, `--video-duration`, `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps`, `--video-fps`, `--video-fps`, `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width`, `--video-width`, `--video-width`, `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height`, `--video-height`, `--video-height`, `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type`, `--video-synth-type`, `--video-synth-type`, `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format`, `--video-format`, `--video-format`, `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec`, `--video-codec`, `--video-codec`, `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8000, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Prompt

#### `-b -b -b -b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets.
<br/>_Constraints: ≥ 0_
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).

#### `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length.
<br/>_Constraints: ≥ 0_

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Rankings

#### `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete.
<br/>_Constraints: ≥ 1_

#### `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir`, `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file`, `--profile-export-prefix`, `--profile-export-file` `<str>`

Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: `.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.

#### `--export-level`, `--profile-export-level`, `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration`, `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).

#### `--export-http-trace`, `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--export-per-chunk-data`, `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

#### `--show-trace-timing`, `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer`, `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision`, `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`, `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration`, `--benchmark-duration` `<float>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics.
<br/>_Constraints: > 0_

#### `--benchmark-grace-period`, `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency`, `--concurrency` `<int>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate.
<br/>_Constraints: ≥ 1_

#### `--prefill-concurrency`, `--prefill-concurrency` `<int>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously.
<br/>_Constraints: ≥ 1_

#### `--request-rate`, `--request-rate` `<float>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds).
<br/>_Constraints: > 0_

#### `--arrival-pattern`, `--request-rate-mode`, `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness`, `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests`, `--request-count`, `--num-requests` `<int>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-count`, `--num-warmup-requests`, `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration`, `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions`, `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency`, `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency`, `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate`, `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern`, `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period`, `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--request-cancellation-rate`, `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay`, `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

#### `--user-centric-rate`, `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users`, `--num-users` `<int>`

The number of initial users to use for --user-centric-rate mode.
<br/>_Constraints: ≥ 1_

#### `--concurrency-ramp-duration`, `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration`, `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--warmup-concurrency-ramp-duration`, `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration`, `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration`, `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration`, `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### Multi-Run

#### `--num-profile-runs`, `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds`, `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level`, `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--convergence-metric`, `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat`, `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold`, `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean (default 0.10 = 10%). For cv mode: maximum coefficient of variation (default 0.10 = 10%). For distribution mode: KS test p-value threshold (default 0.10). Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.1`_

#### `--convergence-mode`, `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `ci_width` | _default_ | Stop when Student's t confidence interval width relative to mean is below threshold. |
| `cv` |  | Stop when coefficient of variation (std/mean) is below threshold. |
| `distribution` |  | Stop when KS test p-value indicates latest run matches prior runs. |

### Accuracy

#### `--accuracy-benchmark`, `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks`, `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots`, `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Constraints: ≥ 0, ≤ 8_
<br/>_Default: `0`_

#### `--accuracy-enable-cot`, `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader`, `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt`, `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`, `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Automatically collects from inference endpoint base_url + `/metrics`. Optionally specify additional custom Prometheus-compatible endpoint URLs (e.g., http://node1:8081/metrics, http://node2:9090/metrics). Use `--no-server-metrics` to disable collection. Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Specify which output formats to generate for server metrics. Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` | _default_ | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` | _default_ | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` | _default_ | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### ZMQ Communication

#### `--zmq-host`, `--zmq-host`, `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path`, `--zmq-ipc-path`, `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

### Parameters

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmqdualbind.event-bus-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmqdualbind.event-bus-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmqdualbind.dataset-manager-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmqdualbind.dataset-manager-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmqdualbind.raw-inference-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmqdualbind.raw-inference-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmqdualbind.ipc-path` `<str>`

Directory path for IPC socket files.

#### `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmqdualbind.tcp-host` `<str>`

TCP bind host for proxies (Defaults to 127.0.0.1 for localhost, use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmqdualbind.controller-host` `<str>`

Controller host for remote TCP connections. When set, services connect via TCP to this host instead of IPC. Set via JobSet DNS in Kubernetes.

#### `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmqdualbind.records-push-pull-tcp-port` `<int>`

TCP port for records push/pull communication with remote workers.
<br/>_Default: `5557`_

#### `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmqdualbind.credit-router-tcp-port` `<int>`

TCP port for credit router communication with remote workers.
<br/>_Default: `5564`_

#### `--output` `<str>`

Path to write the config. If not provided, prints to stdout.

#### `--format` `<str>`

Output format: 'yaml' or 'json'.
<br/>_Default: `yaml`_

### Workers

#### `--workers-max`, `--max-workers`, `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service

#### `--log-level` `<str>`

Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.

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

Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of `RecordProcessor` services to spawn for parallel metric computation. Higher request rates require more processors to keep up with incoming records. If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).
<br/>_Constraints: ≥ 1_

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

<hr/>

## `aiperf config init`

Generate a starter configuration from bundled templates.

Without arguments, generates the 'minimal' template. Use --list to browse available templates, --search to find by keyword, --model/--url to pre-fill the two fields every config needs.

**Examples:**

```bash
aiperf config init --list
aiperf config init --template goodput_slo --model my-model --url http://host/v1
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

## `aiperf kube init`

Generate a starter configuration template

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

#### `--job-name` `<str>`

Value for metadata.name on the generated AIPerfJob.
<br/>_Default: `my-benchmark`_

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

#### `-m -m`, `--model-names`, `--model`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy`, `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `--custom-endpoint`, `--endpoint`, `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type`, `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`, `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u -u`, `--url`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`).
<br/>_Constraints: min: 1_
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy`, `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds`, `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Default: `21600`_

#### `--ready-check-timeout`, `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--api-key`, `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type`, `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`, `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`, `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy`, `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`, `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type`, `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

### Input

#### `--extra-inputs`, `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H -H`, `--header`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file`, `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--fixed-schedule`, `--fixed-schedule`, `--no-fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.

#### `--fixed-schedule-auto-offset`, `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset`, `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset`, `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--public-dataset`, `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset`, `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type`, `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy`, `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed`, `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.

#### `--goodput`, `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format`, `--audio-format`, `--audio-format`, `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths`, `--audio-depths`, `--audio-depths`, `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format`, `--image-format`, `--image-format`, `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration`, `--video-duration`, `--video-duration`, `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps`, `--video-fps`, `--video-fps`, `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width`, `--video-width`, `--video-width`, `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height`, `--video-height`, `--video-height`, `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type`, `--video-synth-type`, `--video-synth-type`, `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format`, `--video-format`, `--video-format`, `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec`, `--video-codec`, `--video-codec`, `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8000, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Prompt

#### `-b -b -b -b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets.
<br/>_Constraints: ≥ 0_
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).

#### `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length.
<br/>_Constraints: ≥ 0_

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Rankings

#### `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete.
<br/>_Constraints: ≥ 1_

#### `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir`, `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file`, `--profile-export-prefix`, `--profile-export-file` `<str>`

Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: `.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.

#### `--export-level`, `--profile-export-level`, `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration`, `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).

#### `--export-http-trace`, `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--export-per-chunk-data`, `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

#### `--show-trace-timing`, `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer`, `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision`, `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`, `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration`, `--benchmark-duration` `<float>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics.
<br/>_Constraints: > 0_

#### `--benchmark-grace-period`, `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency`, `--concurrency` `<int>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate.
<br/>_Constraints: ≥ 1_

#### `--prefill-concurrency`, `--prefill-concurrency` `<int>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously.
<br/>_Constraints: ≥ 1_

#### `--request-rate`, `--request-rate` `<float>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds).
<br/>_Constraints: > 0_

#### `--arrival-pattern`, `--request-rate-mode`, `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness`, `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests`, `--request-count`, `--num-requests` `<int>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-count`, `--num-warmup-requests`, `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration`, `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions`, `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency`, `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency`, `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate`, `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern`, `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period`, `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--request-cancellation-rate`, `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay`, `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

#### `--user-centric-rate`, `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users`, `--num-users` `<int>`

The number of initial users to use for --user-centric-rate mode.
<br/>_Constraints: ≥ 1_

#### `--concurrency-ramp-duration`, `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration`, `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--warmup-concurrency-ramp-duration`, `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration`, `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration`, `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration`, `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### Multi-Run

#### `--num-profile-runs`, `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds`, `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level`, `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--convergence-metric`, `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat`, `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold`, `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean (default 0.10 = 10%). For cv mode: maximum coefficient of variation (default 0.10 = 10%). For distribution mode: KS test p-value threshold (default 0.10). Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.1`_

#### `--convergence-mode`, `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `ci_width` | _default_ | Stop when Student's t confidence interval width relative to mean is below threshold. |
| `cv` |  | Stop when coefficient of variation (std/mean) is below threshold. |
| `distribution` |  | Stop when KS test p-value indicates latest run matches prior runs. |

### Accuracy

#### `--accuracy-benchmark`, `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks`, `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots`, `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Constraints: ≥ 0, ≤ 8_
<br/>_Default: `0`_

#### `--accuracy-enable-cot`, `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader`, `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt`, `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`, `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Automatically collects from inference endpoint base_url + `/metrics`. Optionally specify additional custom Prometheus-compatible endpoint URLs (e.g., http://node1:8081/metrics, http://node2:9090/metrics). Use `--no-server-metrics` to disable collection. Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Specify which output formats to generate for server metrics. Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` | _default_ | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` | _default_ | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` | _default_ | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### ZMQ Communication

#### `--zmq-host`, `--zmq-host`, `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path`, `--zmq-ipc-path`, `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

### Parameters

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmqdualbind.event-bus-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmqdualbind.event-bus-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmqdualbind.dataset-manager-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmqdualbind.dataset-manager-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmqdualbind.raw-inference-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmqdualbind.raw-inference-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmqdualbind.ipc-path` `<str>`

Directory path for IPC socket files.

#### `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmqdualbind.tcp-host` `<str>`

TCP bind host for proxies (Defaults to 127.0.0.1 for localhost, use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmqdualbind.controller-host` `<str>`

Controller host for remote TCP connections. When set, services connect via TCP to this host instead of IPC. Set via JobSet DNS in Kubernetes.

#### `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmqdualbind.records-push-pull-tcp-port` `<int>`

TCP port for records push/pull communication with remote workers.
<br/>_Default: `5557`_

#### `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmqdualbind.credit-router-tcp-port` `<int>`

TCP port for credit router communication with remote workers.
<br/>_Default: `5564`_

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

### Workers

#### `--workers-max`, `--max-workers`, `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service

#### `--log-level` `<str>`

Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.

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

Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of `RecordProcessor` services to spawn for parallel metric computation. Higher request rates require more processors to keep up with incoming records. If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).
<br/>_Constraints: ≥ 1_

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

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

<hr/>

## `aiperf kube sweep`

Run a parameter sweep or multi-run benchmark in Kubernetes

### Endpoint

#### `-m -m`, `--model-names`, `--model`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy`, `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `--custom-endpoint`, `--endpoint`, `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type`, `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`, `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u -u`, `--url`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`).
<br/>_Constraints: min: 1_
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy`, `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds`, `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Default: `21600`_

#### `--ready-check-timeout`, `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--api-key`, `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type`, `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`, `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`, `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy`, `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`, `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type`, `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

### Input

#### `--extra-inputs`, `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H -H`, `--header`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file`, `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--fixed-schedule`, `--fixed-schedule`, `--no-fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.

#### `--fixed-schedule-auto-offset`, `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset`, `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset`, `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--public-dataset`, `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset`, `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type`, `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy`, `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed`, `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.

#### `--goodput`, `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format`, `--audio-format`, `--audio-format`, `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths`, `--audio-depths`, `--audio-depths`, `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format`, `--image-format`, `--image-format`, `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration`, `--video-duration`, `--video-duration`, `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps`, `--video-fps`, `--video-fps`, `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width`, `--video-width`, `--video-width`, `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height`, `--video-height`, `--video-height`, `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type`, `--video-synth-type`, `--video-synth-type`, `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format`, `--video-format`, `--video-format`, `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec`, `--video-codec`, `--video-codec`, `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8000, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Prompt

#### `-b -b -b -b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets.
<br/>_Constraints: ≥ 0_
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).

#### `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length.
<br/>_Constraints: ≥ 0_

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Rankings

#### `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete.
<br/>_Constraints: ≥ 1_

#### `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir`, `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file`, `--profile-export-prefix`, `--profile-export-file` `<str>`

Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: `.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.

#### `--export-level`, `--profile-export-level`, `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration`, `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).

#### `--export-http-trace`, `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--export-per-chunk-data`, `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

#### `--show-trace-timing`, `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer`, `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision`, `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`, `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration`, `--benchmark-duration` `<float>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics.
<br/>_Constraints: > 0_

#### `--benchmark-grace-period`, `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency`, `--concurrency` `<int>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate.
<br/>_Constraints: ≥ 1_

#### `--prefill-concurrency`, `--prefill-concurrency` `<int>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously.
<br/>_Constraints: ≥ 1_

#### `--request-rate`, `--request-rate` `<float>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds).
<br/>_Constraints: > 0_

#### `--arrival-pattern`, `--request-rate-mode`, `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness`, `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests`, `--request-count`, `--num-requests` `<int>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-count`, `--num-warmup-requests`, `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration`, `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions`, `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency`, `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency`, `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate`, `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern`, `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period`, `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--request-cancellation-rate`, `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay`, `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

#### `--user-centric-rate`, `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users`, `--num-users` `<int>`

The number of initial users to use for --user-centric-rate mode.
<br/>_Constraints: ≥ 1_

#### `--concurrency-ramp-duration`, `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration`, `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--warmup-concurrency-ramp-duration`, `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration`, `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration`, `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration`, `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### Multi-Run

#### `--num-profile-runs`, `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds`, `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level`, `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--convergence-metric`, `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat`, `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold`, `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean (default 0.10 = 10%). For cv mode: maximum coefficient of variation (default 0.10 = 10%). For distribution mode: KS test p-value threshold (default 0.10). Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.1`_

#### `--convergence-mode`, `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `ci_width` | _default_ | Stop when Student's t confidence interval width relative to mean is below threshold. |
| `cv` |  | Stop when coefficient of variation (std/mean) is below threshold. |
| `distribution` |  | Stop when KS test p-value indicates latest run matches prior runs. |

### Accuracy

#### `--accuracy-benchmark`, `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks`, `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots`, `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Constraints: ≥ 0, ≤ 8_
<br/>_Default: `0`_

#### `--accuracy-enable-cot`, `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader`, `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt`, `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`, `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Automatically collects from inference endpoint base_url + `/metrics`. Optionally specify additional custom Prometheus-compatible endpoint URLs (e.g., http://node1:8081/metrics, http://node2:9090/metrics). Use `--no-server-metrics` to disable collection. Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Specify which output formats to generate for server metrics. Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` | _default_ | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` | _default_ | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` | _default_ | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### ZMQ Communication

#### `--zmq-host`, `--zmq-host`, `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path`, `--zmq-ipc-path`, `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

### Parameters

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmqdualbind.event-bus-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmqdualbind.event-bus-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmqdualbind.dataset-manager-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmqdualbind.dataset-manager-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmqdualbind.raw-inference-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmqdualbind.raw-inference-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmqdualbind.ipc-path` `<str>`

Directory path for IPC socket files.

#### `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmqdualbind.tcp-host` `<str>`

TCP bind host for proxies (Defaults to 127.0.0.1 for localhost, use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmqdualbind.controller-host` `<str>`

Controller host for remote TCP connections. When set, services connect via TCP to this host instead of IPC. Set via JobSet DNS in Kubernetes.

#### `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmqdualbind.records-push-pull-tcp-port` `<int>`

TCP port for records push/pull communication with remote workers.
<br/>_Default: `5557`_

#### `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmqdualbind.credit-router-tcp-port` `<int>`

TCP port for credit router communication with remote workers.
<br/>_Default: `5564`_

#### `--trials` `<int>`

Multi-run trials per sweep cell; overrides multi_run.trials in the YAML.

#### `--cooldown` `<float>`

Cooldown seconds between multi-run trials (overrides YAML).
<br/>_Default: `0.0`_

#### `--convergence-metric` `<str>`

Stop multi-run early when this metric converges (e.g. ttft_p99).

#### `--min-runs` `<int>`

Minimum runs before convergence is checked (default 3).
<br/>_Default: `3`_

#### `--max-runs` `<int>`

Hard cap on runs even if not converged (default 10).
<br/>_Default: `10`_

#### `--convergence-threshold` `<float>`

Relative convergence threshold (default 0.05 = 5%).
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.05`_

#### `-d`, `--detach`, `--no-detach`

Exit after submitting (don't tail). v1 always behaves as detach=True.

#### `--dry-run`

Print the AIPerfSweep CR as JSON without submitting it.
<br/>_Flag (no value required)_

### Workers

#### `--workers-max`, `--max-workers`, `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service

#### `--log-level` `<str>`

Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.

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

Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of `RecordProcessor` services to spawn for parallel metric computation. Higher request rates require more processors to keep up with incoming records. If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).
<br/>_Constraints: ≥ 1_

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

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

<hr/>

## `aiperf kube generate`

Generate Kubernetes YAML manifests

### Endpoint

#### `-m -m`, `--model-names`, `--model`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy`, `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `--custom-endpoint`, `--endpoint`, `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type`, `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`, `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u -u`, `--url`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`).
<br/>_Constraints: min: 1_
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy`, `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds`, `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Default: `21600`_

#### `--ready-check-timeout`, `--ready-check-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Default: `0.0`_

#### `--api-key`, `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type`, `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`, `http2`]_

#### `--use-legacy-max-tokens`, `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`, `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy`, `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`, `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type`, `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

### Input

#### `--extra-inputs`, `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H -H`, `--header`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file`, `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--fixed-schedule`, `--fixed-schedule`, `--no-fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.

#### `--fixed-schedule-auto-offset`, `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset`, `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset`, `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--public-dataset`, `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`]_

#### `--hf-subset`, `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type`, `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy`, `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed`, `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.

#### `--goodput`, `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio`, `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean`, `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev`, `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format`, `--audio-format`, `--audio-format`, `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths`, `--audio-depths`, `--audio-depths`, `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates`, `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels`, `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean`, `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev`, `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean`, `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev`, `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image`, `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format`, `--image-format`, `--image-format`, `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video`, `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration`, `--video-duration`, `--video-duration`, `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps`, `--video-fps`, `--video-fps`, `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width`, `--video-width`, `--video-width`, `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height`, `--video-height`, `--video-height`, `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type`, `--video-synth-type`, `--video-synth-type`, `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format`, `--video-format`, `--video-format`, `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec`, `--video-codec`, `--video-codec`, `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate`, `--video-audio-sample-rate` `<int>`

Audio sample rate in Hz for the embedded audio track. Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), 48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8000, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels`, `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec`, `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth`, `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Prompt

#### `-b -b -b -b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl`, `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets.
<br/>_Constraints: ≥ 0_
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev`, `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size`, `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).

#### `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution`, `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl`, `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length.
<br/>_Constraints: ≥ 0_

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev`, `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts`, `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length`, `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length`, `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length`, `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Rankings

#### `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean`, `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev`, `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev`, `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev`, `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio`, `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier`, `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier`, `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier`, `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl`, `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl`, `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions`, `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete.
<br/>_Constraints: ≥ 1_

#### `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts`, `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean`, `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev`, `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean`, `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev`, `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio`, `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir`, `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file`, `--profile-export-prefix`, `--profile-export-file` `<str>`

Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: `.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.

#### `--export-level`, `--profile-export-level`, `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration`, `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).

#### `--export-http-trace`, `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--export-per-chunk-data`, `--export-per-chunk-data`

Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports. These arrays contain one timing value per SSE chunk and can be very large for long responses.
<br/>_Flag (no value required)_

#### `--show-trace-timing`, `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer`, `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision`, `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`, `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration`, `--benchmark-duration` `<float>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics.
<br/>_Constraints: > 0_

#### `--benchmark-grace-period`, `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency`, `--concurrency` `<int>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate.
<br/>_Constraints: ≥ 1_

#### `--prefill-concurrency`, `--prefill-concurrency` `<int>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously.
<br/>_Constraints: ≥ 1_

#### `--request-rate`, `--request-rate` `<float>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds).
<br/>_Constraints: > 0_

#### `--arrival-pattern`, `--request-rate-mode`, `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness`, `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests`, `--request-count`, `--num-requests` `<int>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-count`, `--num-warmup-requests`, `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration`, `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions`, `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency`, `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency`, `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate`, `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern`, `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period`, `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--request-cancellation-rate`, `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay`, `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

#### `--user-centric-rate`, `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users`, `--num-users` `<int>`

The number of initial users to use for --user-centric-rate mode.
<br/>_Constraints: ≥ 1_

#### `--concurrency-ramp-duration`, `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration`, `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--warmup-concurrency-ramp-duration`, `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration`, `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration`, `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration`, `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### Multi-Run

#### `--num-profile-runs`, `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds`, `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level`, `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--convergence-metric`, `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat`, `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold`, `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean (default 0.10 = 10%). For cv mode: maximum coefficient of variation (default 0.10 = 10%). For distribution mode: KS test p-value threshold (default 0.10). Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.1`_

#### `--convergence-mode`, `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `ci_width` | _default_ | Stop when Student's t confidence interval width relative to mean is below threshold. |
| `cv` |  | Stop when coefficient of variation (std/mean) is below threshold. |
| `distribution` |  | Stop when KS test p-value indicates latest run matches prior runs. |

### Accuracy

#### `--accuracy-benchmark`, `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks`, `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots`, `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Constraints: ≥ 0, ≤ 8_
<br/>_Default: `0`_

#### `--accuracy-enable-cot`, `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt.
<br/>_Flag (no value required)_

#### `--accuracy-grader`, `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`]_

#### `--accuracy-system-prompt`, `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`, `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Automatically collects from inference endpoint base_url + `/metrics`. Optionally specify additional custom Prometheus-compatible endpoint URLs (e.g., http://node1:8081/metrics, http://node2:9090/metrics). Use `--no-server-metrics` to disable collection. Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Specify which output formats to generate for server metrics. Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` | _default_ | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` | _default_ | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` | _default_ | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### ZMQ Communication

#### `--zmq-host`, `--zmq-host`, `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path`, `--zmq-ipc-path`, `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

### Parameters

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.event-bus-proxy-config.ipc-path`, `--service-config.zmqdualbind.event-bus-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmq-dual-bind.event-bus-proxy-config.name`, `--service-config.zmqdualbind.event-bus-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-host`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.event-bus-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.event-bus-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-control`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.event-bus-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.event-bus-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.ipc-path`, `--service-config.zmqdualbind.dataset-manager-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.name`, `--service-config.zmqdualbind.dataset-manager-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-host`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.dataset-manager-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-control`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.dataset-manager-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.dataset-manager-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.ipc-path`, `--service-config.zmqdualbind.raw-inference-proxy-config.ipc-path` `<str>`

Path for IPC sockets.

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.name`, `--service-config.zmqdualbind.raw-inference-proxy-config.name` `<str>`

Name for IPC sockets.
<br/>_Default: `proxy`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-host`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-host` `<str>`

TCP bind host (use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-frontend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-frontend-port` `<int>`

TCP port for frontend.
<br/>_Default: `15555`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.tcp-backend-port`, `--service-config.zmqdualbind.raw-inference-proxy-config.tcp-backend-port` `<int>`

TCP port for backend.
<br/>_Default: `15556`_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-control`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-control`

Enable control socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmq-dual-bind.raw-inference-proxy-config.no-enable-capture`, `--service-config.zmqdualbind.raw-inference-proxy-config.no-enable-capture`

Enable capture socket.
<br/>_Flag (no value required)_

#### `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmq-dual-bind.ipc-path`, `--service-config.zmqdualbind.ipc-path` `<str>`

Directory path for IPC socket files.

#### `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmq-dual-bind.tcp-host`, `--service-config.zmqdualbind.tcp-host` `<str>`

TCP bind host for proxies (Defaults to 127.0.0.1 for localhost, use 0.0.0.0 for all interfaces).
<br/>_Default: `127.0.0.1`_

#### `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmq-dual-bind.controller-host`, `--service-config.zmqdualbind.controller-host` `<str>`

Controller host for remote TCP connections. When set, services connect via TCP to this host instead of IPC. Set via JobSet DNS in Kubernetes.

#### `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmq-dual-bind.records-push-pull-tcp-port`, `--service-config.zmqdualbind.records-push-pull-tcp-port` `<int>`

TCP port for records push/pull communication with remote workers.
<br/>_Default: `5557`_

#### `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmq-dual-bind.credit-router-tcp-port`, `--service-config.zmqdualbind.credit-router-tcp-port` `<int>`

TCP port for credit router communication with remote workers.
<br/>_Default: `5564`_

#### `--operator`

Output an AIPerfJob CR (requires operator on target cluster).
<br/>_Flag (no value required)_

#### `--no-operator`

Output raw K8s manifests (Namespace, RBAC, ConfigMap, JobSet).

### Workers

#### `--workers-max`, `--max-workers`, `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service

#### `--log-level` `<str>`

Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.

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

Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of `RecordProcessor` services to spawn for parallel metric computation. Higher request rates require more processors to keep up with incoming records. If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).
<br/>_Constraints: ≥ 1_

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

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

#### `-n`, `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube list`

List benchmark jobs and their status

### Parameters

#### `--job-id` `<str>`

Specific job ID to check.

#### `-A`, `--all-namespaces`, `--no-all-namespaces`

Search in all namespaces (default). Ignored when --namespace is set.
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

#### `-n`, `--namespace` `<str>`

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

#### `-n`, `--namespace` `<str>`

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

Namespace where the operator is deployed. Auto-detected (cluster-wide pod search) when omitted.

#### `--run` `<str>`

Pin to a specific historical run (epoch from `aiperf kube results list-runs`). Default: latest.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube results list-runs`

List all historical runs of a benchmark job.

Queries the operator's ``/api/v1/results/<ns>/<job_id>/runs`` endpoint and prints either a table (default) or the raw JSON payload. With ``--preview``, also fetches ``/api/v1/config/retention`` and annotates each row with whether the current policy would reap it (latest is always protected).

**Examples:**

```bash
aiperf kube results list-runs                 # last deployed job
aiperf kube results list-runs foo             # specific job
aiperf kube results list-runs foo --output json
aiperf kube results list-runs foo --preview   # mark reap candidates
```

### Parameters

#### `--job-id` `<str>`

AIPerf job ID to list runs for (default: last deployed job).

#### `-o`, `--output` `<str>`

Output format: 'text' for table, 'json' for machine-parseable.
<br/>_Default: `text`_

#### `--preview`, `--no-preview`

Show which runs would be reaped under current retention settings (read-only; no deletion).

#### `--operator-namespace` `<str>`

Namespace where the operator is deployed. Auto-detected (cluster-wide pod search) when omitted.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube show`

Render an AIPerfJob CR with Jinja2/env-vars resolved

#### `-f`, `--path` `<str>` _(Required)_

Path to an AIPerfJob YAML file.

<hr/>

## `aiperf kube debug`

Run diagnostic analysis on a deployment

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

### Parameters

#### `-j`, `--job-id` `<str>`

Specific AIPerf job ID to diagnose.

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

#### `-n`, `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

<hr/>

## `aiperf kube preflight`

Run pre-flight checks against the target Kubernetes cluster

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `-n`, `--namespace` `<str>`

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

#### `-n`, `--namespace` `<str>`

Kubernetes namespace (default: aiperf-benchmarks).

### Parameters

#### `--port` `<int>`

Local port to bind (default: 0 = ephemeral).
<br/>_Default: `0`_

#### `--operator-namespace` `<str>`

Namespace where the operator is deployed. Auto-detected (cluster-wide pod search) when omitted.

#### `--no-browser`, `--no-no-browser`

Print the URL instead of opening a browser.

<hr/>

## `aiperf synthesize`

Synthesize a dataset workload (e.g. agentic-code)

#### `--target` `<str>` _(Required)_

Dataset workload to synthesize.

#### `--num-sessions` `<int>`

Number of sessions to generate.
<br/>_Default: `1000`_

#### `--output` `<str>`

Parent directory for the run directory.
<br/>_Default: `.`_

#### `--config` `<str>`

Path to config/manifest JSON.

#### `--seed` `<int>`

Random seed for reproducibility.
<br/>_Default: `42`_

#### `--max-isl` `<int>`

Maximum input sequence length.

#### `--max-osl` `<int>`

Maximum output sequence length.

<hr/>

## `aiperf validate`

Validate a benchmark artifact (e.g. mooncake-trace)

#### `--target` `<str>` _(Required)_

Artifact format to validate.

#### `--input` `<str>` _(Required)_

Path to the artifact file.
