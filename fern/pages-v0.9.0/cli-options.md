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

Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

### [`config init`](#aiperf-config-init)

Generate, list, or search bundled AIPerf config templates.

### [`config expand`](#aiperf-config-expand)

Expand a sweep config and print the resulting variations.

### [`config validate`](#aiperf-config-validate)

Validate an AIPerf config file.

### [`profile`](#aiperf-profile)

Run the Profile subcommand.

[Endpoint](#endpoint) • [Tokenizer](#tokenizer) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Conversation Input](#conversation-input) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Output](#output) • [HTTP Trace](#http-trace) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [Service](#service) • [Workers](#workers) • [ZMQ Communication](#zmq-communication)

### [`plot`](#aiperf-plot)

Generate visualizations from AIPerf profiling data.

### [`plugins`](#aiperf-plugins)

Explore AIPerf plugins: aiperf plugins [category] [type]

### [`service`](#aiperf-service)

Run an AIPerf service in a single process.

[Parameters](#parameters) • [Endpoint](#endpoint) • [Tokenizer](#tokenizer) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Conversation Input](#conversation-input) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Output](#output) • [HTTP Trace](#http-trace) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [Service](#service) • [Workers](#workers) • [ZMQ Communication](#zmq-communication)

### [`speed-bench-report`](#aiperf-speed-bench-report)

Assemble per-category SPEED-Bench aiperf results into a matrix report.

### [`synthesize`](#aiperf-synthesize)

Synthesize a dataset workload.

### [`validate`](#aiperf-validate)

Validate a benchmark artifact.

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

Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

#### `--input-file` `<str>` _(Required)_

Path to input mooncake trace JSONL file.

#### `--block-size` `<int>`

KV cache block size for analysis (default: 512).
<br/>_Default: `512`_

#### `--output-file` `<str>`

Optional output path for analysis report (JSON).

<hr/>

## `aiperf config init`

Generate, list, or search bundled AIPerf config templates.

Without ``--output``, selected template YAML is printed to stdout. With ``--output``, the customized template is written to that path after applying ``--model`` and ``--url`` overrides.

#### `-t`, `--template` `<str>`

Template name to generate (e.g. 'minimal', 'goodput_slo').

#### `-l`, `--list`, `--no-list`

List bundled templates and exit.

#### `-s`, `--search` `<str>`

Filter templates by keyword.

#### `-c`, `--category` `<str>`

Filter --list by category.

#### `-v`, `--verbose`, `--no-verbose`

Show tags and difficulty columns.

#### `-m`, `--model` `<str>`

Override the template's model name.

#### `-u`, `--url` `<str>`

Override the template's endpoint URL.

#### `-o`, `--output` `<str>`

Write to file instead of stdout.

<hr/>

## `aiperf config expand`

Expand a sweep config and print the resulting variations.

Drives the same `load_config` -> `build_benchmark_plan` pipeline that `aiperf profile` uses, then prints what the orchestrator would have iterated over - without launching any benchmarks. Useful for verifying sweep paths, dir_name conventions, and per-variation merges before spending compute.

#### `--config-file` `<str>` _(Required)_

Path to an AIPerf YAML config containing a `sweep:` block.

#### `-F`, `--full`, `--no-full`

Also emit each variation's fully-merged BenchmarkConfig body.

#### `-i`, `--index` `<int>`

Show only the variation at this zero-based index (implies --full).

#### `-f`, `--format` `<str>`

Output format: text (default human-readable), yaml, or json.
<br/>_Default: `text`_

<hr/>

## `aiperf config validate`

Validate an AIPerf config file.

Loads the config through the same pipeline as `aiperf profile`, surfacing fatal errors (exit 1) and non-fatal warnings (printed to stderr; exit 0). Useful as a pre-flight check or in CI before kicking off a benchmark.

#### `--config-file` `<str>` _(Required)_

Path to an AIPerf YAML config to validate.

<hr/>

## `aiperf profile`

Run the Profile subcommand.

Benchmark generative AI models and measure performance metrics including throughput, latency, token statistics, and resource utilization.

**Examples:**

```bash
# Basic profiling with streaming
aiperf profile --model Qwen/Qwen3-0.6B --url localhost:8000 --endpoint-type chat --streaming

# Concurrency-based benchmarking
aiperf profile --model your_model --url localhost:8000 --concurrency 10 --request-count 100

# Request rate benchmarking (Poisson distribution)
aiperf profile --model your_model --url localhost:8000 --request-rate 5.0 --benchmark-duration 60

# Time-based benchmarking with grace period
aiperf profile --model your_model --url localhost:8000 --benchmark-duration 300 --benchmark-grace-period 30

# Custom dataset with fixed schedule replay
aiperf profile --model your_model --url localhost:8000 --input-file trace.jsonl --fixed-schedule

# Multi-turn conversations with ShareGPT dataset
aiperf profile --model your_model --url localhost:8000 --public-dataset sharegpt --num-sessions 50

# Goodput measurement with SLOs
aiperf profile --model your_model --url localhost:8000 --goodput "request_latency:250 inter_token_latency:10"
```

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select a model with probability proportional to a per-model weight. |

#### `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `image_edit`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `raw`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`). URLs that do not include a scheme (no `://`) have `http://` prepended automatically.
<br/>_Constraints: min: 1_
<br/>_Default: `['http://localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Constraints: > 0_
<br/>_Default: `21600`_

#### `--wait-for-model-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--wait-for-model-mode` `<str>`

How readiness probes the endpoint: 'models' checks /v1/models, 'inference' sends a canned one-token inference request, and 'both' runs the models check before inference.
<br/>_Default: `inference`_

#### `--wait-for-model-interval` `<float>`

Seconds between endpoint readiness probe attempts.
<br/>_Constraints: > 0.0_
<br/>_Default: `5.0`_

#### `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`]_

#### `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

#### `--session-header` `<str>`

HTTP header name used to carry the per-session affinity identifier. When set, replaces the default `X-Correlation-ID` header with the provided name (e.g., `--session-header X-Session-ID`).

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. If `--tokenizer` is not set and the model name looks like an obvious placeholder (e.g. `mock-model`, `test-model`, `fake-model`), AIPerf substitutes `builtin` automatically and emits a warning. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Input

#### `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`, `librispeech`, `voxpopuli`, `gigaspeech`, `ami`, `spgispeech`]_

#### `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `raw_payload`, `inputs_json`, `dag_jsonl`, `sagemaker_data_capture`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.
<br/>_Constraints: ≥ 0_

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Fixed Schedule

#### `--fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.
<br/>_Flag (no value required)_

#### `--no-fixed-schedule`

Suppress the automatic switch to fixed-schedule mode for trace datasets that carry per-record timestamps. By default a trace input (e.g. mooncake_trace) with timestamps in the first record auto-promotes the profiling phase to fixed_schedule. Pass --no-fixed-schedule to keep the user-selected timing mode (e.g. concurrency, request_rate) and ignore the trace timestamps.

#### `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

### Goodput

#### `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<str>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete. Pass a comma-separated list (e.g. `--num-conversations 50,100,200`) to sweep over session-bounded run lengths; the converter promotes the list to a sweep on phases.profiling.sessions before AIPerfConfig validation. The synthetic dataset pool is sized to max(list) so every variation has its full unique-session set.

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling. Pass a comma-separated list (e.g. `--conversation-turn-mean 1,3,8`) to sweep over multiple turn-mean values; the converter promotes the list to a sweep on datasets.main.turns.mean before AIPerfConfig validation.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

#### `--inter-turn-delay-cap-seconds` `<float>`

Clamp per-turn replay delays (read from JSONL trace files) to at most this many seconds. ``None`` disables the cap. Used by the DAG JSONL loader to keep long pre-recorded waits from stalling the benchmark; the loader reports the clamp count at end of load. Routes onto the active FileDataset's ``inter_turn_delay_cap_seconds`` field at config-resolution time.
<br/>_Constraints: ≥ 0.0_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets. Pass a comma-separated list (e.g. `--isl 128,512,2048`) to sweep over multiple input lengths; the converter promotes the list to a sweep on datasets.main.prompts.isl.mean before AIPerfConfig validation.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation. Pass a comma-separated list (e.g. `--isl-stddev 10,50,200`) to sweep over multiple stddev values; the converter promotes the list to a sweep on datasets.main.prompts.isl.stddev. Pair with a zip-mode `--isl` sweep to model realistic small/medium/large traffic shapes.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).
<br/>_Constraints: ≥ 1_

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<str>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length. Pass a comma-separated list (e.g. `--osl 128,256,512`) to sweep over multiple output lengths; the converter promotes the list to a sweep on datasets.main.prompts.osl.mean before AIPerfConfig validation.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<int>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set. Pass a comma-separated list (e.g. `--osl-stddev 5,25,100`) to sweep over multiple stddev values; the converter promotes the list to a sweep on datasets.main.prompts.osl.stddev. Pair with a zip-mode `--osl` sweep to model realistic output-length variance across traffic tiers.
<br/>_Default: `0`_

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

#### `--image-source` `<str>`

Source image generation mode (default `noise`). `noise` generates random noise images on the fly at the requested dimensions — no files on disk required, and the pool is effectively unbounded so servers cannot dedupe on identical inputs. `assets` loads images from the built-in `assets/source_images` directory (ships with a small set of 4 images) and resizes them to the requested dimensions. A path to a directory loads images from the given directory (e.g. `--image-source ./source_images`). Note: random-noise images are roughly incompressible, so payload bytes are larger than equivalent natural images.
<br/>_Default: `noise`_

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate` `<float>`

Audio sample rate in Hz or kHz for the embedded audio track. Common values: 8/8000 (telephony), 16/16000 (speech), 44.1/44100 (CD quality), 48/48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Rankings

#### `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Load Generator

#### `--benchmark-duration` `<str>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics. Pass a comma-separated list (e.g. `--benchmark-duration 30,60,120`) to sweep over multiple durations; the converter promotes the list to a sweep on phases.profiling.duration before AIPerfConfig validation.

#### `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency` `<str>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate. Pass a comma-separated list (e.g. `--concurrency 10,20,30`) to sweep over multiple concurrencies; the converter promotes the list to a sweep before AIPerfConfig validation.

#### `--prefill-concurrency` `<str>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously. Pass a comma-separated list (e.g. `--prefill-concurrency 1,2,4`) to sweep over multiple values; the converter promotes the list to a sweep before AIPerfConfig validation.

#### `--request-rate` `<str>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds). Pass a comma-separated list (e.g. `--request-rate 10,20,50`) to sweep over multiple rates; the converter promotes the list to a sweep before AIPerfConfig validation.

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests` `<str>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`. Pass a comma-separated list (e.g. `--request-count 100,500,1000`) to sweep over multiple request counts; the converter promotes the list to a sweep on phases.profiling.requests before AIPerfConfig validation.

#### `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

### Warmup

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users` `<str>`

The number of initial users to use for --user-centric-rate mode. Pass a comma-separated list (e.g. `--num-users 4,8,16`) to sweep over user counts; the converter promotes the list to a sweep on phases.profiling.users before AIPerfConfig validation.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Base filename for ALL exported files. With prefix='foo' every output becomes `foo.csv`, `foo.json`, `foo_timeslices.{csv,json}`, `foo.jsonl`, `foo_raw.jsonl`, `foo_gpu_telemetry.jsonl`, and `foo_server_metrics.{jsonl,json,csv,parquet}`. When unset (the default), historical per-file names are used: `profile_export_aiperf.{csv,json}` for the summary, `profile_export.jsonl` and `profile_export_raw.jsonl` for records, `gpu_telemetry_export.jsonl`, and `server_metrics_export.*`. Known suffixes (e.g. `_raw.jsonl`, `_timeslices.csv`, `_server_metrics.parquet`) are stripped from the supplied value.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).
<br/>_Constraints: > 0_

#### `--auto-plot`, `--no-auto-plot`

Auto-invoke `aiperf plot` against the artifact directory after the benchmark completes. None = defer to recipe default (False if no recipe). True/False = explicit override. Failures are logged but do not fail the command unless --plot-required is set.

#### `--plot-required`

Treat auto-plot failures as fatal: re-raise so `aiperf profile` exits non-zero. Only meaningful when auto-plot is on. Default False = warn and continue.
<br/>_Flag (no value required)_

#### `--otel-url` `<str>`

OTLP/HTTP metrics endpoint URL.

#### `--stream` `<list>`

Select which AIPerf telemetry domains to stream over OTel. Valid values: 'metrics', 'timing', or 'default'. 'default' streams both metrics and timing. Examples: --stream metrics | --stream timing | --stream metrics timing.

#### `--otel-resource-attributes` `<list>`

Custom OTel resource attributes as key=value pairs. Merged into the default resource attributes on every exported metric.

#### `--gen-ai-provider` `<str>`

GenAI semantic convention provider override.

#### `--mlflow-tracking-uri` `<str>`

MLflow tracking URI.

#### `--mlflow-experiment` `<str>`

MLflow experiment name.

#### `--mlflow-run-name` `<str>`

MLflow run name.

#### `--mlflow-tag` `<list>`

Additional MLflow run tags to attach on upload. Specify as key:value pairs (e.g., --mlflow-tag team:perf) or as JSON string.

#### `--mlflow-parent-run-id` `<str>`

Optional MLflow parent run ID.

#### `--mlflow-artifact-glob` `<list>`

Artifact glob overrides for MLflow upload. Can be specified multiple times or as a comma-separated list.

### HTTP Trace

#### `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

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
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' or 'amdsmi' to use a local GPU library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry amdsmi | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--vary-seed-per-trial`, `--no-vary-seed-per-trial`

When True, derive a distinct seed for each trial of a variation via SHA-256 over (envelope_seed, variation.label, trial). When False (default), all trials of a variation share the same seed, giving pure-runtime variance for confidence intervals. Enable when you want trials to also sample different inputs (captures end-to-end variance at the cost of conflating input noise with runtime noise in the resulting confidence statistics).

#### `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean. For cv mode: maximum coefficient of variation. For distribution mode: KS test p-value threshold. When unset, each mode uses its own algorithm-specific default. Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_

#### `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.
<br/>_Choices: [`ci_width`, `cv`, `distribution`]_
<br/>_Default: `ci_width`_

#### `--parameter-sweep-cooldown-seconds` `<float>`

Cooldown seconds between sweep variations (e.g. between --concurrency 10 and --concurrency 20). Honored by MultiRunOrchestrator when iterating plan.configs. Default 0.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--parameter-sweep-same-seed`, `--no-parameter-sweep-same-seed`

If true, every sweep variation reuses the same random seed (correlated comparisons). If false (default), each variation derives a unique seed `base_seed + variation.index` so independent draws exercise different inputs. Requires --random-seed when true.

#### `--parameter-sweep-mode` `<str>`

Execution order for sweep + multi-trial composition. 'repeated' (default) iterates trials as the outer loop and variations as the inner loop, so all variations run within trial 1, then within trial 2, etc. 'independent' inverts the loops: all trials at one variation complete before the next variation starts. Both modes produce the same total runs, only the artifact-path layout and submit order differ.
<br/>_Choices: [`independent`, `repeated`]_
<br/>_Default: `repeated`_

#### `--sweep-type` `<str>`

Topology used when multiple CLI magic-list flags (--concurrency, --request-rate, --isl, --osl, ...) are passed together. 'grid' (default) takes the Cartesian product of all lists; 'zip' pairs them element-wise (all lists must have equal length, like the YAML `sweep: {type: zip}` block). Ignored when only one magic-list flag is set or when the sweep is declared in YAML.
<br/>_Default: `grid`_

#### `--no-sweep-table`

Suppress the per-cell streaming sweep table during multi-variation sweeps. Auto-suppressed when stdout is non-interactive, when the dashboard UI is active, or for single-cell sweeps.

#### `--search-space` `<list>`

Adaptive-search space dimensions. Repeatable. Each value is 'path:lo,hi[:kind]', e.g. 'phases.profiling.concurrency:1,1000:int'. Mutually exclusive with magic-list flags (--concurrency 10,20,30) and with explicit sweep blocks. See docs/sweeping/bayesian-optimization.md.

#### `--search-metric` `<str>`

Metric tag to optimize, e.g. 'output_token_throughput'. Required when --search-space is set. Must match a key in RunResult.summary_metrics produced by the run (NOT the flattened '_avg' / '_p99' aggregator-suffixed key).

#### `--search-stat` `<str>`

Statistic on the metric: avg / p50 / p90 / p95 / p99. Defaults to 'avg' when omitted (set by the CLIConfig -> AIPerfConfig converter).

#### `--search-direction` `<str>`

Optimization direction. Required when --search-space is set.

#### `--search-max-iterations` `<int>`

Maximum number of search iterations. Each iteration runs --num-profile-runs benchmarks. Required when --search-space is set.
<br/>_Constraints: ≥ 2, ≤ 200_

#### `--search-initial-points` `<int>`

Random Sobol points before fitting the GP. Defaults to 5 when omitted. Must be &lt; --search-max-iterations.
<br/>_Constraints: ≥ 1_

#### `--search-random-seed` `<int>`

Random seed for reproducible search trajectories. When unset, the planner uses non-deterministic randomness.
<br/>_Constraints: ≥ 0_

#### `--search-planner` `<str>`

Outer-loop search planner plugin. Default `bayesian` is a curated Optuna preset that uses BoTorch qLogNEI/qLogNEHVI when the optional `botorch` extra is installed and otherwise falls back to Optuna TPE with a warning. `optuna` is the expert-mode alternative exposing `--optuna-sampler` (tpe / gp / botorch) and `--optuna-acquisition`. Explicit unavailable optional samplers raise. Third-party planners registered under the `search_planner` plugin category are accepted here. Only applies when --search-space is set.
<br/>_Choices: [`bayesian`, `monotonic_sla`, `smooth_isotonic`, `optuna`]_

#### `--optuna-sampler` `<str>`

Optuna sampler selection. Only consulted when --search-planner=optuna. ``botorch`` is the preferred implicit default and requires the optional ``botorch`` extra; when the implicit default is unavailable, the planner warns and falls back to ``tpe``. Explicit ``botorch`` requests raise if the optional stack is unavailable. ``tpe`` is dep-light and ships with Optuna core. ``gp`` is Optuna's native GP-EI with inequality constraints (Optuna 4.2+) but requires ``torch``.

#### `--optuna-acquisition` `<str>`

Acquisition function override for the Optuna BoTorch sampler. Only consulted when --search-planner=optuna AND --optuna-sampler=botorch; rejected otherwise. ``None`` (default) lets Optuna pick (single-objective unconstrained -> LogEI per Optuna v4.x). ``logei``/``qlogei`` make that explicit. ``qnei`` selects plain noisy EI (Letham 2017). ``qlognei`` selects qLogNoisyExpectedImprovement (Ament 2023, https://arxiv.org/abs/2310.20708) -- BoTorch's strongly recommended modern noisy-EI default; requires ``botorch>=0.10``. Multi-objective variants (``qehvi``/``qnehvi``/``qlognehvi``) are accepted when ``objectives`` has length > 1; the planner rejects them on single-objective configs.

#### `--optuna-terminator` `<str>`

Optional posterior-regret stopping rule layered on top of the three-signal convergence check. Only consulted when --search-planner=optuna. ``regret`` selects Optuna's ``RegretBoundEvaluator`` (Makarova et al. 2022, https://proceedings.mlr.press/v188/makarova22a.html). ``emmr`` selects ``EMMREvaluator`` (Ishibashi et al. 2023, https://proceedings.mlr.press/v206/ishibashi23a.html). Both are in the same family as Wilson 2024's PRB stopping rule and ship in Optuna core (no extra dep). ``none`` (default) disables; convergence is then driven by --search-max-iterations / --improvement-patience / --plateau-cv only.

#### `--search-percentile-pooling` `<str>`

Percentile aggregation strategy when --search-stat is a percentile (p50/p90/p95/p99). ``mean`` (default) computes the BO objective as the arithmetic mean of per-trial percentiles across --num-profile-runs trials. ``pooled`` walks each trial's per-request profile_export.jsonl, accumulates raw samples, and computes ``np.percentile`` over the pooled bag -- exposing more tail mass than mean-of-percentiles (correct for SLO claims; same argmax for ranking on monotone problems). ``pooled`` requires --export-level records; if the JSONL is missing the planner falls back to mean with a one-time warning. Rejected when --search-stat is ``avg``.

#### `--bo-constraint-mode` `<str>`

Deprecated and ignored. The bayesian preset and the optuna expert mode both use Optuna's native ``constraints_func`` (Letham et al. 2019, arXiv:1706.07094), which subsumes both the soft-penalty and EIC formulations. Accepted for backwards compatibility but has no effect: the value flows through ``_converter_optionals._SWEEP_OPTIONAL_FIELDS`` to ``AdaptiveSearchSweep.constraint_mode`` (see ``aiperf.config.sweep.config``), and that field is not read by any planner.

#### `--variant`, `--sweep-variant` `<list>`

Repeatable: each occurrence describes one sweep variation. Format: '[name:] key=value, key=value, ...'. Keys are CLI flag names with the leading '--' stripped, in either kebab-case or snake_case (isl, osl, concurrency, request-rate / request_rate, request-count / request_count, benchmark-duration / benchmark_duration, ...). Multi-occurrence emits a ScenarioSweep. Mutually exclusive with magic-list flags, --search-recipe, and YAML-declared sweeps. Single-occurrence is rejected -- use the standalone --isl / --osl / --concurrency flags for a one-off.

#### `--search-sla` `<list>`

SLA filter to attach to the adaptive-search or grid path. Format: 'metric_tag:stat:op:threshold'. Stat in {avg, p50, p90, p95, p99}; op in {lt, le, gt, ge}; threshold is a float. Repeatable. Example: --search-sla 'time_to_first_token:p95:lt:200' --search-sla 'request_error_rate:p99:lt:0.05'. Composes with recipe-named SLA flags (--ttft-sla-ms etc.); the final filter list is recipe filters first, then --search-sla filters in CLI order.

#### `--search-recipe` `<str>`

Named search-recipe preset that expands to an adaptive-search or sweep block. Mutually exclusive with explicit --search-* flags. Recipes are registered under the search_recipe plugin category. Example: --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200.

#### `--ttft-sla-ms` `<float>`

Time-to-first-token SLA threshold in milliseconds. Required by TTFT-SLA recipes (e.g. max-throughput-ttft-sla); ignored otherwise. Must be > 0 — a 0 or negative threshold yields an unsatisfiable filter.
<br/>_Constraints: > 0_

#### `--isl-osl-pairs` `<str>`

Paired ISL/OSL workload shapes for the pareto-sweep recipe, e.g. '128/128,512/256,2048/512'. Each pair is '&lt;isl>/&lt;osl>' with positive ints; pairs are comma-separated and whitespace-tolerant. Recipe-only flag; ignored unless --search-recipe pareto-sweep is set.

#### `--itl-sla-ms` `<float>`

Inter-token-latency SLA threshold in milliseconds. Required by ITL-SLA recipes (e.g. max-throughput-itl-sla); ignored otherwise. Must be > 0 — a 0 or negative threshold yields an unsatisfiable filter.
<br/>_Constraints: > 0_

#### `--tpot-sla-ms` `<float>`

Time-per-output-token SLA threshold in milliseconds. Maps to the `inter_token_latency` metric tag (TPOT and ITL are equivalent in this codebase). Consumed by the max-concurrency-under-sla and max-goodput-under-slo recipes; ignored otherwise. Streaming required.
<br/>_Constraints: > 0_

#### `--e2e-sla-ms` `<float>`

End-to-end request-latency SLA threshold in milliseconds (p99). Maps to the `request_latency` metric tag. Consumed by the max-concurrency-under-sla and max-goodput-under-slo recipes; ignored otherwise. Available without streaming.
<br/>_Constraints: > 0_

#### `--error-rate-sla` `<float>`

Maximum acceptable request error rate as a fraction in (0, 1) (e.g. 0.05 = 5%). Maps to the `request_error_rate` metric tag (p99). Consumed by the max-concurrency-under-sla recipe; ignored otherwise. Available without streaming.
<br/>_Constraints: > 0, &lt; 1_

#### `--slo-attainment-fraction` `<float>`

Minimum fraction of requests that must satisfy ALL configured per-request SLOs (TTFT/TPOT/E2E) for a configuration to be considered feasible by the goodput recipe. Bounded in (0, 1]. Default 0.95 matches DistServe's canonical attainment-fraction convention (https://arxiv.org/pdf/2401.09670). Consumed by the max-goodput-under-slo recipe; ignored otherwise.
<br/>_Constraints: > 0, ≤ 1_

#### `--search-style` `<str>`

Search strategy for the max-concurrency-under-sla recipe. 'smooth_isotonic' (default) runs PAVA + PCHIP smooth-isotonic regression-based 1D SLA-saturation search. 'monotonic' runs a 1D binary-search via the MonotonicSLASearchPlanner (~10-20 iterations). 'bo' runs penalty Bayesian Optimization (~30 iterations). 'optuna' runs the same penalty-BO formulation via the OptunaSearchPlanner (TPE/GP/BoTorch samplers; BoTorch requires the optional botorch extra). 'grid' runs a log-spaced 8-step sweep + sla_breach_knee post-process. Recipe-only flag; ignored unless --search-recipe max-concurrency-under-sla is set.

#### `--degradation-threshold` `<float>`

Relative latency degradation threshold for the concurrency-ramp recipe (e.g. 0.20 = 20%). The recipe's post-process handler reports the first concurrency where p99 latency exceeds baseline * (1 + threshold). Recipe-only flag; ignored unless --search-recipe concurrency-ramp is set.
<br/>_Constraints: > 0, &lt; 1_

#### `--degradation-metric-tag` `<str>`

ConcurrencyRamp post-process: metric tag for knee detection (default: request_latency). Use, e.g., 'time_to_first_token' to detect the knee on TTFT instead of end-to-end request latency. Recipe-only flag; ignored unless --search-recipe concurrency-ramp is set.

#### `--degradation-stat` `<str>`

ConcurrencyRamp post-process: statistic for knee detection (default: p99). Recipe-only flag; ignored unless --search-recipe concurrency-ramp is set.

#### `--isl-min` `<int>`

Minimum input-sequence-length for the prefill-ttft-curve recipe (default 256 when omitted). The recipe sweeps ISL on a log scale from --isl-min to --isl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--isl-max` `<int>`

Maximum input-sequence-length for the prefill-ttft-curve recipe (default 32768 when omitted). The recipe sweeps ISL on a log scale from --isl-min to --isl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--isl-steps` `<int>`

Number of log-spaced steps for the prefill-ttft-curve recipe's ISL grid (default 8 when omitted). Must be >= 2 — a single-point ramp degenerates and post-process can't compute a baseline. Recipe-only flag.
<br/>_Constraints: ≥ 2_

#### `--concurrency-min` `<int>`

Lower bound for the concurrency sweep axis used by concurrency-ramp and decode-itl-curve recipes (defaults: 1 for concurrency-ramp, 1 for decode-itl-curve). Must be &lt; --concurrency-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--concurrency-max` `<int>`

Upper bound for the concurrency sweep axis used by concurrency-ramp and decode-itl-curve recipes (defaults: 1000 for concurrency-ramp, 200 for decode-itl-curve). Must be > --concurrency-min. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--concurrency-steps` `<int>`

Number of log-spaced steps for the concurrency sweep axis used by concurrency-ramp (default 8) and decode-itl-curve (default 6). Must be >= 2. Recipe-only flag.
<br/>_Constraints: ≥ 2_

#### `--osl-min` `<int>`

Minimum output-sequence-length for the decode-itl-curve recipe's OSL grid (default 64 when omitted). The recipe sweeps OSL on a log scale from --osl-min to --osl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--osl-max` `<int>`

Maximum output-sequence-length for the decode-itl-curve recipe's OSL grid (default 1024 when omitted). The recipe sweeps OSL on a log scale from --osl-min to --osl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--osl-steps` `<int>`

Number of log-spaced steps for the decode-itl-curve recipe's OSL grid (default 4 when omitted). Must be >= 2. Recipe-only flag.
<br/>_Constraints: ≥ 2_

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). Accepts comma-separated values (e.g. abstract_algebra,anatomy) or repeated flags. If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation, None uses the benchmark default (e.g. MMLU=5). Maximum 32.
<br/>_Constraints: ≥ 0, ≤ 32_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt. Defaults to the benchmark's ``default_enable_cot`` metadata when unset (e.g. AIME defaults to True).
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`, `lighteval_expr`, `lighteval_latex`, `lighteval_gpqa`]_

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

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

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.
<br/>_Constraints: ≥ 1_

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

#### `--zmq-dual-bind`

Select the ZMQ dual-bind communication backend (IPC + TCP). All dual-bind knobs are cluster-managed; this flag only selects the discriminator and the converter routes downstream to the default.
<br/>_Flag (no value required)_

<hr/>

## `aiperf plot`

Generate visualizations from AIPerf profiling data.

On first run, automatically creates ~/.aiperf/plot_config.yaml which you can edit to customize plots, including experiment classification (baseline vs treatment runs). Use --config to specify a different config file.

_**Note:** PNG export requires Chrome or Chromium to be installed on your system, as it is used by kaleido to render Plotly figures to static images._

_**Note:** The plot command expects default export filenames (e.g., `profile_export.jsonl`). Runs created with `--profile-export-file` or custom `--profile-export-prefix` use different filenames and will not be detected by the plot command._

**Examples:**

```bash
# Generate plots (auto-creates ~/.aiperf/plot_config.yaml on first run)
aiperf plot

# Use custom config
aiperf plot --config my_plots.yaml

# Show detailed error tracebacks
aiperf plot --verbose

# Generate plots and upload them to the MLflow run from mlflow_export.json
aiperf plot --paths artifacts/my-run --mlflow-upload

# Generate plots and upload to an explicit MLflow run
aiperf plot --paths artifacts/my-run --mlflow-upload --mlflow-tracking-uri http://127.0.0.1:5000 --mlflow-run-id <run_id>
```

#### `--paths`, `--empty-paths` `<list>`

Paths to profiling run directories. Defaults to ./artifacts if not specified.

#### `--output` `<str>`

Directory to save generated plots. Defaults to &lt;first_path>/plots if not specified.

#### `--theme` `<str>`

Plot theme to use: 'light' (white background) or 'dark' (dark background). Defaults to 'light'.
<br/>_Default: `light`_

#### `--config` `<str>`

Path to custom plot configuration YAML file. If not specified, auto-creates and uses ~/.aiperf/plot_config.yaml.

#### `--verbose`, `--no-verbose`

Show detailed error tracebacks in console (errors are always logged to ~/.aiperf/plot.log).

#### `--dashboard`, `--no-dashboard`

Launch interactive dashboard server instead of generating static PNGs.

#### `--host` `<str>`

Host for dashboard server (only used with --dashboard). Defaults to 127.0.0.1.
<br/>_Default: `127.0.0.1`_

#### `--port` `<int>`

Port for dashboard server (only used with --dashboard). Defaults to 8050.
<br/>_Default: `8050`_

#### `--mlflow-upload`, `--no-mlflow-upload`

Upload generated PNG plot artifacts to an existing MLflow run. Mutually exclusive with --dashboard.

#### `--mlflow-tracking-uri` `<str>`

Optional MLflow tracking URI override for plot upload.

#### `--mlflow-run-id` `<str>`

Optional MLflow run id override for plot upload.

<hr/>

## `aiperf plugins`

Explore AIPerf plugins: aiperf plugins [category] [type]

#### `--category` `<str>`

Category to explore.
<br/>_Choices: [`accuracy_benchmark`, `accuracy_grader`, `api_router`, `arrival_pattern`, `communication`, `communication_client`, `console_exporter`, `convergence_criterion`, `custom_dataset_loader`, `data_exporter`, `dataset_backing_store`, `dataset_client_store`, `dataset_composer`, `dataset_sampler`, `endpoint`, `gpu_telemetry_collector`, `gpu_telemetry_processor`, `plot`, `public_dataset_loader`, `ramp`, `record_processor`, `results_processor`, `search_planner`, `search_recipe`, `search_recipe_post_process`, `server_metrics_processor`, `service`, `service_manager`, `timing_strategy`, `transport`, `ui`, `url_selection_strategy`, `zmq_proxy`]_

#### `--name` `<str>`

Type name for details.

#### `-a`, `--all`, `--no-all`

Show all categories and plugins.

#### `-v`, `--validate`, `--no-validate`

Validate plugins.yaml.

<hr/>

## `aiperf service`

Run an AIPerf service in a single process.

_Advanced use only — intended for developers and Kubernetes/distributed deployments where services run in separate containers or nodes._

For standard single-node benchmarking, use the `aiperf profile` command instead.

### Parameters

#### `--type` `<str>` _(Required)_

Service type to run.
<br/>_Choices: [`api`, `dataset_manager`, `gpu_telemetry_manager`, `record_processor`, `records_manager`, `server_metrics_manager`, `system_controller`, `timing_manager`, `worker`, `worker_manager`]_

#### `--service-id` `<str>`

Unique identifier for the service instance. Useful when running multiple instances of the same service type.

#### `--health-host` `<str>`

Host to bind the health server to. Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable.

#### `--health-port` `<int>`

HTTP port for health endpoints (/healthz, /readyz). Required for Kubernetes liveness and readiness probes. Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable.

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select a model with probability proportional to a per-model weight. |

#### `--custom-endpoint`, `--endpoint` `<str>`

Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. Use this option to override the default path for non-standard API implementations.

#### `--endpoint-type` `<str>`

The API endpoint type to benchmark. Determines request/response format and supported features. Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). See enum documentation for all supported endpoint types.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `responses`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `image_edit`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `raw`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming responses. When enabled, the server streams tokens incrementally as they are generated. Automatically disabled if the selected endpoint type does not support streaming. Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.
<br/>_Flag (no value required)_

#### `-u`, `--url` `<list>`

Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`). URLs that do not include a scheme (no `://`) have `http://` prepended automatically.
<br/>_Constraints: min: 1_
<br/>_Default: `['http://localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for selecting URLs when multiple `--url` values are provided. 'round_robin' (default): distribute requests evenly across URLs in sequential order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--request-timeout-seconds` `<float>`

Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, request transmission, and response receipt. Applies to both streaming and non-streaming requests. Requests exceeding this timeout are cancelled and recorded as failures.
<br/>_Constraints: > 0_
<br/>_Default: `21600`_

#### `--wait-for-model-timeout` `<float>`

Seconds to wait for endpoint readiness before benchmarking (0 = skip). Sends a real inference request to verify the model is loaded and can generate output.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--wait-for-model-mode` `<str>`

How readiness probes the endpoint: 'models' checks /v1/models, 'inference' sends a canned one-token inference request, and 'both' runs the models check before inference.
<br/>_Default: `inference`_

#### `--wait-for-model-interval` `<float>`

Seconds between endpoint readiness probe attempts.
<br/>_Constraints: > 0.0_
<br/>_Default: `5.0`_

#### `--api-key` `<str>`

API authentication key for the endpoint. When provided, automatically included in request headers as `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type` `<str>`

Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme (`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.
<br/>_Choices: [`http`]_

#### `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from API usage fields instead of client-side tokenization. When enabled, tokenizers are still loaded (needed for dataset generation) but tokenizer.encode() is not called for computing metrics. Token count fields will be None if the server does not provide usage information. For OpenAI-compatible streaming endpoints (chat/completions), stream_options.include_usage is automatically configured when this flag is enabled.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

Transport connection reuse strategy. 'pooled' (default): connections are pooled and reused across all requests. 'never': new connection for each request, closed after response. 'sticky-user-sessions': connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled (default), only generation time is measured.
<br/>_Flag (no value required)_

#### `--request-content-type` `<str>`

Content type for request body serialization. By default, requests are sent as 'application/json'. Set to 'multipart/form-data' for servers that require form-encoded requests (e.g., vLLM video generation endpoints).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `application/json` |  | Standard JSON encoding. Default for all endpoints. |
| `multipart/form-data` |  | Multipart form encoding. Required by some video generation servers (e.g., vLLM). |

#### `--session-header` `<str>`

HTTP header name used to carry the per-session affinity identifier. When set, replaces the default `X-Correlation-ID` header with the provided name (e.g., `--session-header X-Session-ID`).

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). If not specified, defaults to the value of `--model-names`. If `--tokenizer` is not set and the model name looks like an obvious placeholder (e.g. `mock-model`, `test-model`, `fake-model`), AIPerf substitutes `builtin` automatically and emits a warning. Essential for accurate token-based metrics (input/output token counts, token throughput).

#### `--tokenizer-revision` `<str>`

Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning to a specific version. Defaults to `main` branch if not specified.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.
<br/>_Flag (no value required)_

### Input

#### `--extra-inputs` `<list>`

Additional input parameters to include in every API request payload. Specify as `key:value` pairs (e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{"temperature": 0.7}'`). These parameters are merged with request-specific inputs and sent directly to the endpoint API.
<br/>_Default: `[]`_

#### `-H`, `--header` `<list>`

Custom HTTP headers to include with every request. Specify as `Header:Value` pairs (e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers (e.g., `Authorization` from `--api-key`).
<br/>_Default: `[]`_

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. Run `aiperf plugins public_dataset_loader` to list available datasets. Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.
<br/>_Choices: [`sharegpt`, `aimo`, `mmstar`, `mmvu`, `vision_arena`, `llava_onevision`, `speed_bench_qualitative`, `speed_bench_coding`, `speed_bench_humanities`, `speed_bench_math`, `speed_bench_multilingual`, `speed_bench_qa`, `speed_bench_rag`, `speed_bench_reasoning`, `speed_bench_roleplay`, `speed_bench_stem`, `speed_bench_summarization`, `speed_bench_writing`, `speed_bench_throughput_1k`, `speed_bench_throughput_2k`, `speed_bench_throughput_8k`, `speed_bench_throughput_16k`, `speed_bench_throughput_32k`, `speed_bench_throughput_1k_low_entropy`, `speed_bench_throughput_1k_mixed`, `speed_bench_throughput_1k_high_entropy`, `speed_bench_throughput_2k_low_entropy`, `speed_bench_throughput_2k_mixed`, `speed_bench_throughput_2k_high_entropy`, `speed_bench_throughput_8k_low_entropy`, `speed_bench_throughput_8k_mixed`, `speed_bench_throughput_8k_high_entropy`, `speed_bench_throughput_16k_low_entropy`, `speed_bench_throughput_16k_mixed`, `speed_bench_throughput_16k_high_entropy`, `speed_bench_throughput_32k_low_entropy`, `speed_bench_throughput_32k_mixed`, `speed_bench_throughput_32k_high_entropy`, `aimo_aime`, `aimo_numina_cot`, `aimo_numina_1_5`, `spec_bench`, `instruct_coder`, `blazedit_5k`, `blazedit_10k`, `librispeech`, `voxpopuli`, `gigaspeech`, `ami`, `spgispeech`]_

#### `--hf-subset` `<str>`

HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). Only applies when using `--public-dataset` with a HuggingFace-backed loader. Takes priority over the subset defined in the plugin registry.

#### `--custom-dataset-type` `<str>`

Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), `mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; when using `random_pool`, `--conversation-num` defaults to 100 if not specified; batch sizes > 1 sample each modality independently from a flat pool and do not preserve per-entry associations - use `single_turn` if paired modalities must stay together). Requires `--input-file`. Mutually exclusive with `--public-dataset`.
<br/>_Choices: [`burst_gpt_trace`, `bailian_trace`, `mooncake_trace`, `raw_payload`, `inputs_json`, `dag_jsonl`, `sagemaker_data_capture`, `multi_turn`, `random_pool`, `single_turn`]_

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. `sequential`: Iterate through dataset in order, wrapping to start after end. `random`: Randomly sample with replacement (entries may repeat before all are used). `shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).
<br/>_Choices: [`random`, `sequential`, `shuffle`]_

#### `--random-seed` `<int>`

Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. Initialized globally at config creation.
<br/>_Constraints: ≥ 0_

#### `-f`, `--config` `<str>`

Path to a YAML configuration file. CLI flags override values from the config file.

### Fixed Schedule

#### `--fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets.
<br/>_Flag (no value required)_

#### `--no-fixed-schedule`

Suppress the automatic switch to fixed-schedule mode for trace datasets that carry per-record timestamps. By default a trace input (e.g. mooncake_trace) with timestamps in the first record auto-promotes the profiling phase to fixed_schedule. Pass --no-fixed-schedule to keep the user-selected timing mode (e.g. concurrency, request_rate) and ignore the trace timestamps.

#### `--fixed-schedule-auto-offset`

Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.
<br/>_Flag (no value required)_

#### `--fixed-schedule-start-offset` `<int>`

Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. Must be ≤ `--fixed-schedule-end-offset` if both specified.
<br/>_Constraints: ≥ 0_

#### `--fixed-schedule-end-offset` `<int>`

End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. Must be ≥ `--fixed-schedule-start-offset` if both specified.
<br/>_Constraints: ≥ 0_

### Goodput

#### `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric's display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

### Conversation Input

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<str>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete. Pass a comma-separated list (e.g. `--num-conversations 50,100,200`) to sweep over session-bounded run lengths; the converter promotes the list to a sweep on phases.profiling.sessions before AIPerfConfig validation. The synthetic dataset pool is sized to max(list) so every variation has its full unique-session set.

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents one user message that can be used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. Higher values provide more diversity.
<br/>_Constraints: ≥ 1_
<br/>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Each turn consists of a user message and model response. Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. Multi-turn conversations enable testing of context retention and conversation history handling. Pass a comma-separated list (e.g. `--conversation-turn-mean 1,3,8`) to sweep over multiple turn-mean values; the converter promotes the list to a sweep on datasets.main.turns.mean before AIPerfConfig validation.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. Set to 0 for uniform conversation lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. Delays follow normal distribution. Set to 0 for deterministic delays. Models realistic human interaction patterns with variable response times.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: `actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. Values &lt; 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1.0`_

#### `--inter-turn-delay-cap-seconds` `<float>`

Clamp per-turn replay delays (read from JSONL trace files) to at most this many seconds. ``None`` disables the cap. Used by the DAG JSONL loader to keep long pre-recorded waits from stalling the benchmark; the loader reports the clamp count at end of load. Routes onto the active FileDataset's ``inter_turn_delay_cap_seconds`` field at config-resolution time.
<br/>_Constraints: ≥ 0.0_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if `--num-prefix-prompts` is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`.
<br/>_Constraints: ≥ 1_

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Requires --num-dataset-entries to be specified. Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size.
<br/>_Constraints: ≥ 1_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, not custom or public datasets. Pass a comma-separated list (e.g. `--isl 128,512,2048`) to sweep over multiple input lengths; the converter promotes the list to a sweep on datasets.main.prompts.isl.mean before AIPerfConfig validation.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. Set to 0 for uniform prompt lengths. Applies only to synthetic data generation. Pass a comma-separated list (e.g. `--isl-stddev 10,50,200`) to sweep over multiple stddev values; the converter promotes the list to a sweep on datasets.main.prompts.isl.stddev. Pair with a zip-mode `--isl` sweep to model realistic small/medium/large traffic shapes.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).
<br/>_Constraints: ≥ 1_

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<str>`

Mean number of tokens to request in model outputs via `max_completion_tokens` field. Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit generation length. When not set, model determines output length. Pass a comma-separated list (e.g. `--osl 128,256,512`) to sweep over multiple output lengths; the converter promotes the list to a sweep on datasets.main.prompts.osl.mean before AIPerfConfig validation.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<int>`

Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, simulating mixed response length requirements. Lengths follow normal distribution. Only applies when `--prompt-output-tokens-mean` is set. Pass a comma-separated list (e.g. `--osl-stddev 5,25,100`) to sweep over multiple stddev values; the converter promotes the list to a sweep on datasets.main.prompts.osl.stddev. Pair with a zip-mode `--osl` sweep to model realistic output-length variance across traffic tiers.
<br/>_Default: `0`_

### Audio Input

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--audio-length-mean` `<float>`

Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. Generated audio is random noise with specified sample rate, bit depth, and format.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-length-stddev` `<float>`

Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, simulating mixed-duration audio inputs. Durations follow normal distribution. Set to 0 for uniform audio lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--audio-format` `<str>`

File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.
<br/>_Constraints: min: 1_
<br/>_Default: `[16]`_

#### `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br/>_Constraints: min: 1_
<br/>_Default: `[16.0]`_

#### `--audio-num-channels` `<int>`

Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. Most speech models use mono.
<br/>_Constraints: ≥ 1, ≤ 2_
<br/>_Default: `1`_

### Image Input

#### `--image-width-mean` `<float>`

Mean width in pixels for synthetically generated images. Image widths follow a normal distribution around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions and file sizes for multimodal benchmarking.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-width-stddev` `<float>`

Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, simulating mixed-resolution image inputs. Widths follow normal distribution. Set to 0 for uniform image widths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-mean` `<float>`

Mean height in pixels for synthetically generated images. Image heights follow a normal distribution around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. Generated images are resized from source images in `assets/source_images` directory.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-height-stddev` `<float>`

Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, simulating mixed-resolution image inputs. Heights follow normal distribution. Set to 0 for uniform image heights.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--image-format` `<str>`

Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), `jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` | _default_ | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` |  | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

#### `--image-source` `<str>`

Source image generation mode (default `noise`). `noise` generates random noise images on the fly at the requested dimensions — no files on disk required, and the pool is effectively unbounded so servers cannot dedupe on identical inputs. `assets` loads images from the built-in `assets/source_images` directory (ships with a small set of 4 images) and resizes them to the requested dimensions. A path to a directory loads images from the given directory (e.g. `--image-source ./source_images`). Note: random-noise images are roughly incompressible, so payload bytes are larger than equivalent natural images.
<br/>_Default: `noise`_

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. Higher batch sizes test multi-video understanding and significantly increase request payload size.
<br/>_Constraints: ≥ 0_
<br/>_Default: `1`_

#### `--video-duration` `<float>`

Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count (frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. Requires FFmpeg for video generation.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `5.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). Total frames = `--video-duration` × FPS.
<br/>_Constraints: ≥ 1_
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.
<br/>_Constraints: ≥ 1_

#### `--video-height` `<int>`

Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.
<br/>_Constraints: ≥ 1_

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). Format choice affects compatibility, file size, and encoding options. Use `webm` for open-source workflows, `mp4` for maximum compatibility.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

#### `--video-audio-sample-rate` `<float>`

Audio sample rate in Hz or kHz for the embedded audio track. Common values: 8/8000 (telephony), 16/16000 (speech), 44.1/44100 (CD quality), 48/48000 (professional). Higher sample rates increase audio fidelity and file size.
<br/>_Constraints: ≥ 8, ≤ 96000_
<br/>_Default: `44100`_

#### `--video-audio-num-channels` `<int>`

Number of audio channels to embed in generated video files. 0 = disabled (no audio track, default), 1 = mono, 2 = stereo. When set to 1 or 2, a Gaussian noise audio track matching the video duration is muxed into each video via FFmpeg.
<br/>_Constraints: ≥ 0, ≤ 2_
<br/>_Default: `0`_

#### `--video-audio-codec` `<str>`

Audio codec for the embedded audio track. If not specified, auto-selects based on video format: aac for MP4, libvorbis for WebM. Options: aac, libvorbis, libopus.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `aac` |  | AAC codec. Default for MP4 containers. |
| `libvorbis` |  | Vorbis codec. Default for WebM containers. |
| `libopus` |  | Opus codec. Alternative for WebM containers. |

#### `--video-audio-depth` `<str>`

Audio bit depth for the embedded audio track. Supported values: 8, 16, 24, or 32 bits. Higher bit depths provide greater dynamic range but increase file size.
<br/>_Default: `16`_

### Rankings

#### `--rankings-passages-mean` `<int>`

Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). Higher values test ranking at scale but increase request payload size and processing time.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--rankings-passages-stddev` `<int>`

Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). Longer passages increase input processing demands and request size.
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating realistic heterogeneous document collections. Token lengths follow normal distribution. Set to 0 for uniform passage lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`).
<br/>_Constraints: ≥ 1_
<br/>_Default: `550`_

#### `--rankings-query-prompt-token-stddev` `<int>`

Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating realistic user search patterns. Token lengths follow normal distribution. Set to 0 for uniform query lengths.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in radix tree.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across.
<br/>_Constraints: ≥ 1_
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.
<br/>_Constraints: ≥ 1_

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.
<br/>_Constraints: ≥ 1_

### Load Generator

#### `--benchmark-duration` `<str>`

Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, Responses received within `--benchmark-grace-period` after duration ends are included in metrics. Pass a comma-separated list (e.g. `--benchmark-duration 30,60,120`) to sweep over multiple durations; the converter promotes the list to a sweep on phases.profiling.duration before AIPerfConfig validation.

#### `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. Use 'inf' to wait indefinitely for all responses.
<br/>_Constraints: ≥ 0_
<br/>_Default: `30.0`_

#### `--concurrency` `<str>`

Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate. Pass a comma-separated list (e.g. `--concurrency 10,20,30`) to sweep over multiple concurrencies; the converter promotes the list to a sweep before AIPerfConfig validation.

#### `--prefill-concurrency` `<str>`

Max concurrent requests waiting for first token (prefill phase). Limits how many requests can be in the prefill/prompt-processing stage simultaneously. Pass a comma-separated list (e.g. `--prefill-concurrency 1,2,4`) to sweep over multiple values; the converter promotes the list to a sweep before AIPerfConfig validation.

#### `--request-rate` `<str>`

Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds). Pass a comma-separated list (e.g. `--request-rate 10,20,50`) to sweep over multiple rates; the converter promotes the list to a sweep before AIPerfConfig validation.

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. `constant`: Generate requests at a fixed rate. `poisson`: Generate requests using a poisson distribution. `gamma`: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). Controls the shape of the arrival pattern: - 1.0: Poisson-like (exponential inter-arrivals, default) - &lt;1.0: Bursty/clustered arrivals (higher variance) - >1.0: Smooth/regular arrivals (lower variance) Compatible with vLLM's --burstiness parameter (same value = same distribution).
<br/>_Constraints: > 0_

#### `--request-count`, `--num-requests` `<str>`

The maximum number of requests to send. If not set, will be automatically determined based on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`. Pass a comma-separated list (e.g. `--request-count 100,500,1000`) to sweep over multiple request counts; the converter promotes the list to a sweep on phases.profiling.requests before AIPerfConfig validation.

#### `--concurrency-ramp-duration` `<float>`

Duration in seconds to ramp session concurrency from 1 to target. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

#### `--prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp prefill concurrency from 1 to target.
<br/>_Constraints: > 0_

#### `--request-rate-ramp-duration` `<float>`

Duration in seconds to ramp request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration), ensuring correct behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.
<br/>_Constraints: > 0_

### Warmup

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

The maximum number of warmup requests to send before benchmarking. If not set and no --warmup-duration is set, then no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--warmup-duration` `<float>`

The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. If neither are set, no warmup phase will be used.
<br/>_Constraints: > 0_

#### `--num-warmup-sessions` `<int>`

The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-concurrency` `<int>`

The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-prefill-concurrency` `<int>`

The prefill concurrency value to use for the warmup phase. If not set, it will use the `--prefill-concurrency` value.
<br/>_Constraints: ≥ 1_

#### `--warmup-request-rate` `<float>`

The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.
<br/>_Constraints: > 0_

#### `--warmup-arrival-pattern` `<str>`

The arrival pattern to use for the warmup phase. If not set, it will use the `--arrival-pattern` value. Valid values: constant, poisson, gamma.

#### `--warmup-grace-period` `<float>`

The grace period in seconds to wait for responses after warmup phase ends. Only applies when warmup is enabled. Responses received within this period are included in warmup completion. If not set, waits indefinitely for all warmup responses.
<br/>_Constraints: ≥ 0_

#### `--warmup-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup session concurrency from 1 to target. If not set, uses `--concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Duration in seconds to ramp warmup prefill concurrency from 1 to target. If not set, uses `--prefill-concurrency-ramp-duration` value.
<br/>_Constraints: > 0_

#### `--warmup-request-rate-ramp-duration` `<float>`

Duration in seconds to ramp warmup request rate from a proportional minimum to target. Start rate is calculated as target * (update_interval / duration). If not set, uses `--request-rate-ramp-duration` value.
<br/>_Constraints: > 0_

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user). New users are spawned on a fixed schedule to maintain steady-state throughput. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.
<br/>_Constraints: > 0_

#### `--num-users` `<str>`

The number of initial users to use for --user-centric-rate mode. Pass a comma-separated list (e.g. `--num-users 4,8,16`) to sweep over user counts; the converter promotes the list to a sweep on phases.profiling.users before AIPerfConfig validation.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.
<br/>_Constraints: > 0.0, ≤ 100.0_

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means 'send the full request, then immediately disconnect'. Requires --request-cancellation-rate to be set.
<br/>_Constraints: ≥ 0.0_
<br/>_Default: `0.0`_

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. All output file paths are constructed relative to this directory.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Base filename for ALL exported files. With prefix='foo' every output becomes `foo.csv`, `foo.json`, `foo_timeslices.{csv,json}`, `foo.jsonl`, `foo_raw.jsonl`, `foo_gpu_telemetry.jsonl`, and `foo_server_metrics.{jsonl,json,csv,parquet}`. When unset (the default), historical per-file names are used: `profile_export_aiperf.{csv,json}` for the summary, `profile_export.jsonl` and `profile_export_raw.jsonl` for records, `gpu_telemetry_export.jsonl`, and `server_metrics_export.*`. Known suffixes (e.g. `_raw.jsonl`, `_timeslices.csv`, `_server_metrics.parquet`) are stripped from the supplied value.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. `summary`: Only aggregate metrics files (`.csv`, `.json`). `records`: Includes per-request metrics (`.jsonl`). `raw`: Includes raw request/response data (`_raw.jsonl`).

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` |  | Export only aggregated/summarized metrics (default, most compact) |
| `records` | _default_ | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time (e.g., warmup effects, degradation under sustained load).
<br/>_Constraints: > 0_

#### `--auto-plot`, `--no-auto-plot`

Auto-invoke `aiperf plot` against the artifact directory after the benchmark completes. None = defer to recipe default (False if no recipe). True/False = explicit override. Failures are logged but do not fail the command unless --plot-required is set.

#### `--plot-required`

Treat auto-plot failures as fatal: re-raise so `aiperf profile` exits non-zero. Only meaningful when auto-plot is on. Default False = warn and continue.
<br/>_Flag (no value required)_

#### `--otel-url` `<str>`

OTLP/HTTP metrics endpoint URL.

#### `--stream` `<list>`

Select which AIPerf telemetry domains to stream over OTel. Valid values: 'metrics', 'timing', or 'default'. 'default' streams both metrics and timing. Examples: --stream metrics | --stream timing | --stream metrics timing.

#### `--otel-resource-attributes` `<list>`

Custom OTel resource attributes as key=value pairs. Merged into the default resource attributes on every exported metric.

#### `--gen-ai-provider` `<str>`

GenAI semantic convention provider override.

#### `--mlflow-tracking-uri` `<str>`

MLflow tracking URI.

#### `--mlflow-experiment` `<str>`

MLflow experiment name.

#### `--mlflow-run-name` `<str>`

MLflow run name.

#### `--mlflow-tag` `<list>`

Additional MLflow run tags to attach on upload. Specify as key:value pairs (e.g., --mlflow-tag team:perf) or as JSON string.

#### `--mlflow-parent-run-id` `<str>`

Optional MLflow parent run ID.

#### `--mlflow-artifact-glob` `<list>`

Artifact glob overrides for MLflow upload. Can be specified multiple times or as a comma-separated list.

### HTTP Trace

#### `--export-http-trace`

Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. See the HTTP Trace Metrics guide for details on trace data fields.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in the console at the end of the benchmark. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration following k6 naming conventions.
<br/>_Flag (no value required)_

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
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'pynvml' or 'amdsmi' to use a local GPU library instead of DCGM HTTP endpoints, (2) 'dashboard' for realtime dashboard mode, (3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. Examples: --gpu-telemetry pynvml | --gpu-telemetry amdsmi | --gpu-telemetry dashboard node1:9400.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

Select the user interface type for displaying benchmark progress. `dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, `none` disables UI completely. Defaults to `dashboard` in interactive terminals, `none` when not a TTY (e.g., piped or redirected output). Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. Must be between 1 and 10. When set to 1 (default), runs a single benchmark. When set to >1, runs multiple benchmarks and computes aggregate statistics (mean, std, confidence intervals, coefficient of variation) across runs. Useful for quantifying variance and establishing confidence in results.
<br/>_Constraints: ≥ 1, ≤ 10_
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Only applies when --num-profile-runs > 1. Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). Default is 0 (no cooldown).
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Only applies when --num-profile-runs > 1. Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). Higher values produce wider confidence intervals.
<br/>_Constraints: > 0, &lt; 1_
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for profile runs after the first. Only applies when --num-profile-runs > 1. When True (default), only the first run includes warmup, subsequent runs measure steady-state performance for more accurate aggregate statistics. When False, all runs include warmup (useful for long cooldown periods or when testing cold-start performance).
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. Only applies when --num-profile-runs > 1. When True (default), automatically sets --random-seed=42 if not specified, ensuring identical workloads across all runs for valid statistical comparison. When False, preserves None seed, resulting in different workloads per run (not recommended for confidence reporting as it produces invalid statistics). If --random-seed is explicitly set, that value is always used regardless of this setting.
<br/>_Default: `True`_

#### `--vary-seed-per-trial`, `--no-vary-seed-per-trial`

When True, derive a distinct seed for each trial of a variation via SHA-256 over (envelope_seed, variation.label, trial). When False (default), all trials of a variation share the same seed, giving pure-runtime variance for confidence intervals. Enable when you want trials to also sample different inputs (captures end-to-end variance at the cost of conflating input noise with runtime noise in the resulting confidence statistics).

#### `--convergence-metric` `<str>`

Target metric name for adaptive convergence stopping. When set with --num-profile-runs > 1, enables adaptive mode that stops early once the metric stabilizes according to --convergence-mode. Uses --num-profile-runs as the maximum run cap. Example metrics: time_to_first_token, request_latency, inter_token_latency.

#### `--convergence-stat` `<str>`

Statistic to evaluate for convergence when using ci_width or cv mode. Common values: avg, p50, p90, p95, p99. Only applies when --convergence-metric is set.
<br/>_Choices: [`avg`, `p50`, `p90`, `p95`, `p99`, `min`, `max`]_
<br/>_Default: `avg`_

#### `--convergence-threshold` `<float>`

Threshold for convergence detection. For ci_width mode: maximum CI width as a fraction of the mean. For cv mode: maximum coefficient of variation. For distribution mode: KS test p-value threshold. When unset, each mode uses its own algorithm-specific default. Only applies when --convergence-metric is set.
<br/>_Constraints: > 0, &lt; 1_

#### `--convergence-mode` `<str>`

Statistical method for convergence detection. ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. cv: Stop when coefficient of variation (std/mean) is below threshold. distribution: Stop when KS test p-value indicates latest run matches prior runs (requires --export-level records or --export-level raw; rejected with --export-level summary). Only applies when --convergence-metric is set.
<br/>_Choices: [`ci_width`, `cv`, `distribution`]_
<br/>_Default: `ci_width`_

#### `--parameter-sweep-cooldown-seconds` `<float>`

Cooldown seconds between sweep variations (e.g. between --concurrency 10 and --concurrency 20). Honored by MultiRunOrchestrator when iterating plan.configs. Default 0.
<br/>_Constraints: ≥ 0_
<br/>_Default: `0.0`_

#### `--parameter-sweep-same-seed`, `--no-parameter-sweep-same-seed`

If true, every sweep variation reuses the same random seed (correlated comparisons). If false (default), each variation derives a unique seed `base_seed + variation.index` so independent draws exercise different inputs. Requires --random-seed when true.

#### `--parameter-sweep-mode` `<str>`

Execution order for sweep + multi-trial composition. 'repeated' (default) iterates trials as the outer loop and variations as the inner loop, so all variations run within trial 1, then within trial 2, etc. 'independent' inverts the loops: all trials at one variation complete before the next variation starts. Both modes produce the same total runs, only the artifact-path layout and submit order differ.
<br/>_Choices: [`independent`, `repeated`]_
<br/>_Default: `repeated`_

#### `--sweep-type` `<str>`

Topology used when multiple CLI magic-list flags (--concurrency, --request-rate, --isl, --osl, ...) are passed together. 'grid' (default) takes the Cartesian product of all lists; 'zip' pairs them element-wise (all lists must have equal length, like the YAML `sweep: {type: zip}` block). Ignored when only one magic-list flag is set or when the sweep is declared in YAML.
<br/>_Default: `grid`_

#### `--no-sweep-table`

Suppress the per-cell streaming sweep table during multi-variation sweeps. Auto-suppressed when stdout is non-interactive, when the dashboard UI is active, or for single-cell sweeps.

#### `--search-space` `<list>`

Adaptive-search space dimensions. Repeatable. Each value is 'path:lo,hi[:kind]', e.g. 'phases.profiling.concurrency:1,1000:int'. Mutually exclusive with magic-list flags (--concurrency 10,20,30) and with explicit sweep blocks. See docs/sweeping/bayesian-optimization.md.

#### `--search-metric` `<str>`

Metric tag to optimize, e.g. 'output_token_throughput'. Required when --search-space is set. Must match a key in RunResult.summary_metrics produced by the run (NOT the flattened '_avg' / '_p99' aggregator-suffixed key).

#### `--search-stat` `<str>`

Statistic on the metric: avg / p50 / p90 / p95 / p99. Defaults to 'avg' when omitted (set by the CLIConfig -> AIPerfConfig converter).

#### `--search-direction` `<str>`

Optimization direction. Required when --search-space is set.

#### `--search-max-iterations` `<int>`

Maximum number of search iterations. Each iteration runs --num-profile-runs benchmarks. Required when --search-space is set.
<br/>_Constraints: ≥ 2, ≤ 200_

#### `--search-initial-points` `<int>`

Random Sobol points before fitting the GP. Defaults to 5 when omitted. Must be &lt; --search-max-iterations.
<br/>_Constraints: ≥ 1_

#### `--search-random-seed` `<int>`

Random seed for reproducible search trajectories. When unset, the planner uses non-deterministic randomness.
<br/>_Constraints: ≥ 0_

#### `--search-planner` `<str>`

Outer-loop search planner plugin. Default `bayesian` is a curated Optuna preset that uses BoTorch qLogNEI/qLogNEHVI when the optional `botorch` extra is installed and otherwise falls back to Optuna TPE with a warning. `optuna` is the expert-mode alternative exposing `--optuna-sampler` (tpe / gp / botorch) and `--optuna-acquisition`. Explicit unavailable optional samplers raise. Third-party planners registered under the `search_planner` plugin category are accepted here. Only applies when --search-space is set.
<br/>_Choices: [`bayesian`, `monotonic_sla`, `smooth_isotonic`, `optuna`]_

#### `--optuna-sampler` `<str>`

Optuna sampler selection. Only consulted when --search-planner=optuna. ``botorch`` is the preferred implicit default and requires the optional ``botorch`` extra; when the implicit default is unavailable, the planner warns and falls back to ``tpe``. Explicit ``botorch`` requests raise if the optional stack is unavailable. ``tpe`` is dep-light and ships with Optuna core. ``gp`` is Optuna's native GP-EI with inequality constraints (Optuna 4.2+) but requires ``torch``.

#### `--optuna-acquisition` `<str>`

Acquisition function override for the Optuna BoTorch sampler. Only consulted when --search-planner=optuna AND --optuna-sampler=botorch; rejected otherwise. ``None`` (default) lets Optuna pick (single-objective unconstrained -> LogEI per Optuna v4.x). ``logei``/``qlogei`` make that explicit. ``qnei`` selects plain noisy EI (Letham 2017). ``qlognei`` selects qLogNoisyExpectedImprovement (Ament 2023, https://arxiv.org/abs/2310.20708) -- BoTorch's strongly recommended modern noisy-EI default; requires ``botorch>=0.10``. Multi-objective variants (``qehvi``/``qnehvi``/``qlognehvi``) are accepted when ``objectives`` has length > 1; the planner rejects them on single-objective configs.

#### `--optuna-terminator` `<str>`

Optional posterior-regret stopping rule layered on top of the three-signal convergence check. Only consulted when --search-planner=optuna. ``regret`` selects Optuna's ``RegretBoundEvaluator`` (Makarova et al. 2022, https://proceedings.mlr.press/v188/makarova22a.html). ``emmr`` selects ``EMMREvaluator`` (Ishibashi et al. 2023, https://proceedings.mlr.press/v206/ishibashi23a.html). Both are in the same family as Wilson 2024's PRB stopping rule and ship in Optuna core (no extra dep). ``none`` (default) disables; convergence is then driven by --search-max-iterations / --improvement-patience / --plateau-cv only.

#### `--search-percentile-pooling` `<str>`

Percentile aggregation strategy when --search-stat is a percentile (p50/p90/p95/p99). ``mean`` (default) computes the BO objective as the arithmetic mean of per-trial percentiles across --num-profile-runs trials. ``pooled`` walks each trial's per-request profile_export.jsonl, accumulates raw samples, and computes ``np.percentile`` over the pooled bag -- exposing more tail mass than mean-of-percentiles (correct for SLO claims; same argmax for ranking on monotone problems). ``pooled`` requires --export-level records; if the JSONL is missing the planner falls back to mean with a one-time warning. Rejected when --search-stat is ``avg``.

#### `--bo-constraint-mode` `<str>`

Deprecated and ignored. The bayesian preset and the optuna expert mode both use Optuna's native ``constraints_func`` (Letham et al. 2019, arXiv:1706.07094), which subsumes both the soft-penalty and EIC formulations. Accepted for backwards compatibility but has no effect: the value flows through ``_converter_optionals._SWEEP_OPTIONAL_FIELDS`` to ``AdaptiveSearchSweep.constraint_mode`` (see ``aiperf.config.sweep.config``), and that field is not read by any planner.

#### `--variant`, `--sweep-variant` `<list>`

Repeatable: each occurrence describes one sweep variation. Format: '[name:] key=value, key=value, ...'. Keys are CLI flag names with the leading '--' stripped, in either kebab-case or snake_case (isl, osl, concurrency, request-rate / request_rate, request-count / request_count, benchmark-duration / benchmark_duration, ...). Multi-occurrence emits a ScenarioSweep. Mutually exclusive with magic-list flags, --search-recipe, and YAML-declared sweeps. Single-occurrence is rejected -- use the standalone --isl / --osl / --concurrency flags for a one-off.

#### `--search-sla` `<list>`

SLA filter to attach to the adaptive-search or grid path. Format: 'metric_tag:stat:op:threshold'. Stat in {avg, p50, p90, p95, p99}; op in {lt, le, gt, ge}; threshold is a float. Repeatable. Example: --search-sla 'time_to_first_token:p95:lt:200' --search-sla 'request_error_rate:p99:lt:0.05'. Composes with recipe-named SLA flags (--ttft-sla-ms etc.); the final filter list is recipe filters first, then --search-sla filters in CLI order.

#### `--search-recipe` `<str>`

Named search-recipe preset that expands to an adaptive-search or sweep block. Mutually exclusive with explicit --search-* flags. Recipes are registered under the search_recipe plugin category. Example: --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200.

#### `--ttft-sla-ms` `<float>`

Time-to-first-token SLA threshold in milliseconds. Required by TTFT-SLA recipes (e.g. max-throughput-ttft-sla); ignored otherwise. Must be > 0 — a 0 or negative threshold yields an unsatisfiable filter.
<br/>_Constraints: > 0_

#### `--isl-osl-pairs` `<str>`

Paired ISL/OSL workload shapes for the pareto-sweep recipe, e.g. '128/128,512/256,2048/512'. Each pair is '&lt;isl>/&lt;osl>' with positive ints; pairs are comma-separated and whitespace-tolerant. Recipe-only flag; ignored unless --search-recipe pareto-sweep is set.

#### `--itl-sla-ms` `<float>`

Inter-token-latency SLA threshold in milliseconds. Required by ITL-SLA recipes (e.g. max-throughput-itl-sla); ignored otherwise. Must be > 0 — a 0 or negative threshold yields an unsatisfiable filter.
<br/>_Constraints: > 0_

#### `--tpot-sla-ms` `<float>`

Time-per-output-token SLA threshold in milliseconds. Maps to the `inter_token_latency` metric tag (TPOT and ITL are equivalent in this codebase). Consumed by the max-concurrency-under-sla and max-goodput-under-slo recipes; ignored otherwise. Streaming required.
<br/>_Constraints: > 0_

#### `--e2e-sla-ms` `<float>`

End-to-end request-latency SLA threshold in milliseconds (p99). Maps to the `request_latency` metric tag. Consumed by the max-concurrency-under-sla and max-goodput-under-slo recipes; ignored otherwise. Available without streaming.
<br/>_Constraints: > 0_

#### `--error-rate-sla` `<float>`

Maximum acceptable request error rate as a fraction in (0, 1) (e.g. 0.05 = 5%). Maps to the `request_error_rate` metric tag (p99). Consumed by the max-concurrency-under-sla recipe; ignored otherwise. Available without streaming.
<br/>_Constraints: > 0, &lt; 1_

#### `--slo-attainment-fraction` `<float>`

Minimum fraction of requests that must satisfy ALL configured per-request SLOs (TTFT/TPOT/E2E) for a configuration to be considered feasible by the goodput recipe. Bounded in (0, 1]. Default 0.95 matches DistServe's canonical attainment-fraction convention (https://arxiv.org/pdf/2401.09670). Consumed by the max-goodput-under-slo recipe; ignored otherwise.
<br/>_Constraints: > 0, ≤ 1_

#### `--search-style` `<str>`

Search strategy for the max-concurrency-under-sla recipe. 'smooth_isotonic' (default) runs PAVA + PCHIP smooth-isotonic regression-based 1D SLA-saturation search. 'monotonic' runs a 1D binary-search via the MonotonicSLASearchPlanner (~10-20 iterations). 'bo' runs penalty Bayesian Optimization (~30 iterations). 'optuna' runs the same penalty-BO formulation via the OptunaSearchPlanner (TPE/GP/BoTorch samplers; BoTorch requires the optional botorch extra). 'grid' runs a log-spaced 8-step sweep + sla_breach_knee post-process. Recipe-only flag; ignored unless --search-recipe max-concurrency-under-sla is set.

#### `--degradation-threshold` `<float>`

Relative latency degradation threshold for the concurrency-ramp recipe (e.g. 0.20 = 20%). The recipe's post-process handler reports the first concurrency where p99 latency exceeds baseline * (1 + threshold). Recipe-only flag; ignored unless --search-recipe concurrency-ramp is set.
<br/>_Constraints: > 0, &lt; 1_

#### `--degradation-metric-tag` `<str>`

ConcurrencyRamp post-process: metric tag for knee detection (default: request_latency). Use, e.g., 'time_to_first_token' to detect the knee on TTFT instead of end-to-end request latency. Recipe-only flag; ignored unless --search-recipe concurrency-ramp is set.

#### `--degradation-stat` `<str>`

ConcurrencyRamp post-process: statistic for knee detection (default: p99). Recipe-only flag; ignored unless --search-recipe concurrency-ramp is set.

#### `--isl-min` `<int>`

Minimum input-sequence-length for the prefill-ttft-curve recipe (default 256 when omitted). The recipe sweeps ISL on a log scale from --isl-min to --isl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--isl-max` `<int>`

Maximum input-sequence-length for the prefill-ttft-curve recipe (default 32768 when omitted). The recipe sweeps ISL on a log scale from --isl-min to --isl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--isl-steps` `<int>`

Number of log-spaced steps for the prefill-ttft-curve recipe's ISL grid (default 8 when omitted). Must be >= 2 — a single-point ramp degenerates and post-process can't compute a baseline. Recipe-only flag.
<br/>_Constraints: ≥ 2_

#### `--concurrency-min` `<int>`

Lower bound for the concurrency sweep axis used by concurrency-ramp and decode-itl-curve recipes (defaults: 1 for concurrency-ramp, 1 for decode-itl-curve). Must be &lt; --concurrency-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--concurrency-max` `<int>`

Upper bound for the concurrency sweep axis used by concurrency-ramp and decode-itl-curve recipes (defaults: 1000 for concurrency-ramp, 200 for decode-itl-curve). Must be > --concurrency-min. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--concurrency-steps` `<int>`

Number of log-spaced steps for the concurrency sweep axis used by concurrency-ramp (default 8) and decode-itl-curve (default 6). Must be >= 2. Recipe-only flag.
<br/>_Constraints: ≥ 2_

#### `--osl-min` `<int>`

Minimum output-sequence-length for the decode-itl-curve recipe's OSL grid (default 64 when omitted). The recipe sweeps OSL on a log scale from --osl-min to --osl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--osl-max` `<int>`

Maximum output-sequence-length for the decode-itl-curve recipe's OSL grid (default 1024 when omitted). The recipe sweeps OSL on a log scale from --osl-min to --osl-max. Recipe-only flag.
<br/>_Constraints: ≥ 1_

#### `--osl-steps` `<int>`

Number of log-spaced steps for the decode-itl-curve recipe's OSL grid (default 4 when omitted). Must be >= 2. Recipe-only flag.
<br/>_Constraints: ≥ 2_

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode alongside performance profiling.
<br/>_Choices: [`mmlu`, `aime`, `hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`]_

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). Accepts comma-separated values (e.g. abstract_algebra,anatomy) or repeated flags. If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation, None uses the benchmark default (e.g. MMLU=5). Maximum 32.
<br/>_Constraints: ≥ 0, ≤ 32_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation. Adds reasoning instructions to the prompt. Defaults to the benchmark's ``default_enable_cot`` metadata when unset (e.g. AIME defaults to True).
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution). If not set, uses the benchmark's default grader.
<br/>_Choices: [`exact_match`, `math`, `multiple_choice`, `code_execution`, `lighteval_expr`, `lighteval_latex`, `lighteval_gpqa`]_

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

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

#### `--api-port` `<int>`

AIPerf API port (enables HTTP + WebSocket endpoints).
<br/>_Constraints: ≥ 1, ≤ 65535_

#### `--api-host` `<str>`

AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set).

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of 32. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.
<br/>_Constraints: ≥ 1_

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication (message bus, dataset manager, workers) uses this host for TCP sockets.
<br/>_Default: `127.0.0.1`_

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory if not specified. Only applicable when using IPC communication backend.

#### `--zmq-dual-bind`

Select the ZMQ dual-bind communication backend (IPC + TCP). All dual-bind knobs are cluster-managed; this flag only selects the discriminator and the converter routes downstream to the default.
<br/>_Flag (no value required)_

<hr/>

## `aiperf speed-bench-report`

Assemble per-category SPEED-Bench aiperf results into a matrix report.

Run ``aiperf profile`` once per SPEED-Bench category, then point this command at the output directories to produce a matrix matching the SPEED-Bench paper format.

**Examples:**

```bash
# Scan a parent directory for per-category run subdirectories
aiperf speed-bench-report ./artifacts/

# List run directories explicitly
aiperf speed-bench-report ./artifacts/run_coding/ ./artifacts/run_math/

# Acceptance rate matrix (accepted / draft tokens)
aiperf speed-bench-report ./artifacts/ --metric accept_rate

# Throughput matrix (output tokens/sec per category)
aiperf speed-bench-report ./artifacts/ --metric throughput
```

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

## `aiperf synthesize`

Synthesize a dataset workload.

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

Validate a benchmark artifact.

#### `--target` `<str>` _(Required)_

Artifact format to validate.

#### `--input` `<str>` _(Required)_

Path to the artifact file.
