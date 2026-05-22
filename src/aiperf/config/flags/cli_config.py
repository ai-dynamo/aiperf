# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig - unified CLI-only input DTO.

This module defines the cyclopts-facing input shape for both benchmark
configuration and service-runtime knobs. It carries CLI flag metadata
(CLIParameter), field-level documentation (Field), and CLI-input parsing
helpers (BeforeValidator(parse_str_or_list)) - but NO model-level or
field-level domain validators.

CLIConfig carries logging, verbosity, ZMQ communication, worker counts, UI
type, and API host/port alongside the benchmark-shaped sections (endpoint,
input, output, tokenizer, loadgen, sweeping, accuracy).

Domain validation (e.g. "concurrency cannot exceed request_count") lives on
AIPerfConfig. The converter at aiperf.config.flags.converter translates a
populated CLIConfig into the canonical AIPerfConfig.

This file is intentionally large (~3200 LOC, ~200 fields) — every CLI flag
is a top-level field by design, so size scales linearly with field count.
The flat shape is the post-flatten architecture. Section dividers group
fields by their CLIParameter ``Groups.X``. Both the file-size and
pydantic-fields ergonomics checks have an explicit intentional exception
for this file (see ``tools/check_ergonomics.py::INTENTIONAL_FILE_SIZE_EXEMPTIONS``).

See aiperf.config.flags.__init__ for the hard rules around adding new fields,
and ``docs/dev/patterns.md`` § "Adding a New CLI Flag" for the recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from cyclopts import Parameter
from pydantic import AfterValidator, BeforeValidator, Field

from aiperf.common.enums import (
    AIPerfLogLevel,
    AudioFormat,
    ConnectionReuseStrategy,
    ConvergenceStat,
    ExportLevel,
    GPUTelemetryMode,
    ImageFormat,
    ImageSource,
    ModelSelectionStrategy,
    RequestContentType,
    ServerMetricsFormat,
    SweepMode,
    VideoAudioCodec,
    VideoFormat,
    VideoSynthType,
)
from aiperf.config.artifacts import OutputDefaults
from aiperf.config.base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, Groups
from aiperf.config.endpoint import EndpointDefaults
from aiperf.config.loader.parsing import (
    normalize_http_urls,
    parse_file,
    parse_float_or_float_list,
    parse_int_or_int_list,
    parse_str_as_numeric_dict,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list,
    parse_str_or_list_of_positive_values,
    require_turn_mean_at_least_one,
)
from aiperf.config.runtime import ServiceDefaults
from aiperf.plugin.enums import (
    AccuracyBenchmarkType,
    AccuracyGraderType,
    ArrivalPattern,
    ConvergenceCriterionType,
    CustomDatasetType,
    DatasetSamplingStrategy,
    EndpointType,
    GPUTelemetryCollectorType,
    PublicDatasetType,
    SearchPlannerType,
    TransportType,
    UIType,
    URLSelectionStrategy,
)

# Default server-metrics export formats for CLIConfig, kept self-contained here
# because aiperf.config.defaults does not carry the equivalent constant.
_DEFAULT_SERVER_METRICS_FORMATS: list[ServerMetricsFormat] = [
    ServerMetricsFormat.JSON,
    ServerMetricsFormat.CSV,
]


class CLIConfig(BaseConfig):
    """Unified CLI input (benchmark + service runtime).

    CLIConfig is a flat DTO; no nested-class forward refs remain. Validators
    are forbidden on this class - AIPerfConfig is the single validation gate.
    """

    ##############################################################################
    # Endpoint
    ##############################################################################
    model_names: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=(
                "--model-names",
                "--model",  # GenAI-Perf
                "-m",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ]

    model_selection_strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            description="When multiple models are specified, this is how a specific model should be assigned to a prompt.\n"
            "round_robin: nth prompt in the list gets assigned to n-mod len(models).\n"
            "random: assignment is uniformly random",
        ),
        CLIParameter(
            name=("--model-selection-strategy",),  # GenAI-Perf
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.MODEL_SELECTION_STRATEGY

    custom_endpoint: Annotated[
        str | None,
        Field(
            description="Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). "
            "By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. "
            "Use this option to override the default path for non-standard API implementations.",
        ),
        CLIParameter(
            name=(
                "--custom-endpoint",
                "--endpoint",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.CUSTOM_ENDPOINT

    endpoint_type: Annotated[
        EndpointType,
        Field(
            description="The API endpoint type to benchmark. Determines request/response format and supported features. "
            "Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). "
            "See enum documentation for all supported endpoint types.",
        ),
        CLIParameter(
            name=("--endpoint-type",),  # GenAI-Perf
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.TYPE

    streaming: Annotated[
        bool,
        Field(
            description="Enable streaming responses. When enabled, the server streams tokens incrementally "
            "as they are generated. Automatically disabled if the selected endpoint type does not support streaming. "
            "Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.",
        ),
        CLIParameter(
            name=("--streaming",),  # GenAI-Perf
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.STREAMING

    urls: Annotated[
        list[str],
        Field(
            description="Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing "
            "across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). "
            "The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`). "
            "URLs that do not include a scheme (no `://`) have `http://` prepended automatically.",
            min_length=1,
            # Run the validator chain on the default too — without this, a
            # bare `--wait-for-model-timeout 30` (no `--url`) would send the
            # un-normalized default to aiohttp and reproduce the original bug.
            validate_default=True,
        ),
        BeforeValidator(parse_str_or_list),
        AfterValidator(normalize_http_urls),
        CLIParameter(
            name=(
                "--url",  # GenAI-Perf
                "-u",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=Groups.ENDPOINT,
        ),
    ] = [EndpointDefaults.URL]

    url_selection_strategy: Annotated[
        URLSelectionStrategy,
        Field(
            description="Strategy for selecting URLs when multiple `--url` values are provided. "
            "'round_robin' (default): distribute requests evenly across URLs in sequential order.",
        ),
        CLIParameter(
            name=("--url-strategy",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.URL_STRATEGY

    timeout_seconds: Annotated[
        float,
        Field(
            gt=0,
            description="Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, "
            "request transmission, and response receipt. Applies to both streaming and non-streaming requests. "
            "Requests exceeding this timeout are cancelled and recorded as failures.",
        ),
        CLIParameter(
            name=("--request-timeout-seconds",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.TIMEOUT

    wait_for_model_timeout: Annotated[
        float,
        Field(
            ge=0,
            description=(
                "Seconds to wait for endpoint readiness before benchmarking "
                "(0 = skip). Sends a real inference request to verify the model "
                "is loaded and can generate output."
            ),
        ),
        CLIParameter(
            name=("--wait-for-model-timeout",),
            group=Groups.ENDPOINT,
        ),
    ] = 0.0

    wait_for_model_mode: Annotated[
        Literal["models", "inference", "both"],
        Field(
            description=(
                "How readiness probes the endpoint: 'models' checks /v1/models, "
                "'inference' sends a canned one-token inference request, and "
                "'both' runs the models check before inference."
            ),
        ),
        CLIParameter(
            name=("--wait-for-model-mode",),
            group=Groups.ENDPOINT,
        ),
    ] = "inference"

    wait_for_model_interval: Annotated[
        float,
        Field(
            gt=0.0,
            description="Seconds between endpoint readiness probe attempts.",
        ),
        CLIParameter(
            name=("--wait-for-model-interval",),
            group=Groups.ENDPOINT,
        ),
    ] = 5.0

    api_key: Annotated[
        str | None,
        Field(
            description="API authentication key for the endpoint. When provided, automatically included in request headers as "
            "`Authorization: Bearer <api_key>`.",
            repr=False,
        ),
        CLIParameter(
            name=("--api-key",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.API_KEY

    transport: Annotated[
        TransportType | None,
        Field(
            description="Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme "
            "(`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, "
            "TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.",
        ),
        CLIParameter(
            name=("--transport", "--transport-type"),
            group=Groups.ENDPOINT,
        ),
    ] = None

    use_legacy_max_tokens: Annotated[
        bool,
        Field(
            description="Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. "
            "The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.",
        ),
        CLIParameter(
            name=("--use-legacy-max-tokens",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.USE_LEGACY_MAX_TOKENS

    use_server_token_count: Annotated[
        bool,
        Field(
            description=(
                "Use server-reported token counts from API usage fields instead of "
                "client-side tokenization. When enabled, tokenizers are still loaded "
                "(needed for dataset generation) but tokenizer.encode() is not called "
                "for computing metrics. Token count fields will be None if the server "
                "does not provide usage information. For OpenAI-compatible streaming "
                "endpoints (chat/completions), stream_options.include_usage is automatically "
                "configured when this flag is enabled."
            ),
        ),
        CLIParameter(
            name=("--use-server-token-count",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.USE_SERVER_TOKEN_COUNT

    connection_reuse_strategy: Annotated[
        ConnectionReuseStrategy,
        Field(
            description=(
                "Transport connection reuse strategy. "
                "'pooled' (default): connections are pooled and reused across all requests. "
                "'never': new connection for each request, closed after response. "
                "'sticky-user-sessions': connection persists across turns of a multi-turn "
                "conversation, closed on final turn (enables sticky load balancing)."
            ),
        ),
        CLIParameter(
            name=("--connection-reuse-strategy",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.CONNECTION_REUSE_STRATEGY

    download_video_content: Annotated[
        bool,
        Field(
            description=(
                "For video generation endpoints, download the video content after generation completes. "
                "When enabled, request latency includes the video download time. "
                "When disabled (default), only generation time is measured."
            ),
        ),
        CLIParameter(
            name=("--download-video-content",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.DOWNLOAD_VIDEO_CONTENT

    request_content_type: Annotated[
        RequestContentType | None,
        Field(
            description=(
                "Content type for request body serialization. By default, requests are sent as "
                "'application/json'. Set to 'multipart/form-data' for servers that require form-encoded "
                "requests (e.g., vLLM video generation endpoints)."
            ),
        ),
        CLIParameter(
            name=("--request-content-type",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.REQUEST_CONTENT_TYPE

    session_header: Annotated[
        str | None,
        Field(
            description=(
                "HTTP header name used to carry the per-session affinity identifier. "
                "When set, replaces the default `X-Correlation-ID` header with the "
                "provided name (e.g., `--session-header X-Session-ID`)."
            ),
        ),
        CLIParameter(
            name=("--session-header",),
            group=Groups.ENDPOINT,
        ),
    ] = None

    @property
    def url(self) -> str:
        """Return the first URL for backward compatibility."""
        return self.urls[0]

    ##############################################################################
    # Tokenizer
    ##############################################################################
    tokenizer_name: Annotated[
        str | None,
        Field(
            description="HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. "
            "Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, "
            "or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). "
            "If not specified, defaults to the value of `--model-names`. "
            "If `--tokenizer` is not set and the model name looks like an obvious placeholder "
            "(e.g. `mock-model`, `test-model`, `fake-model`), AIPerf substitutes `builtin` automatically "
            "and emits a warning. Essential for accurate token-based metrics "
            "(input/output token counts, token throughput).",
        ),
        CLIParameter(
            name=("--tokenizer"),
            group=Groups.TOKENIZER,
        ),
    ] = None

    tokenizer_revision: Annotated[
        str,
        Field(
            description="Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), "
            "tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning "
            "to a specific version. Defaults to `main` branch if not specified.",
        ),
        CLIParameter(
            name=("--tokenizer-revision"),
            group=Groups.TOKENIZER,
        ),
    ] = "main"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers "
            "with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for "
            "trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.",
        ),
        CLIParameter(
            name=("--tokenizer-trust-remote-code"),
            group=Groups.TOKENIZER,
        ),
    ] = False

    ##############################################################################
    # Input
    ##############################################################################
    extra_inputs: Annotated[
        Any,
        Field(
            description="Additional input parameters to include in every API request payload. Specify as `key:value` pairs "
            "(e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{\"temperature\": 0.7}'`). "
            "These parameters are merged with request-specific inputs and sent directly to the endpoint API.",
        ),
        CLIParameter(
            name=("--extra-inputs",),
            consume_multiple=True,
            group=Groups.INPUT,
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
    ] = []

    headers: Annotated[
        Any,
        Field(
            description="Custom HTTP headers to include with every request. Specify as `Header:Value` pairs "
            "(e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. "
            "Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers "
            "(e.g., `Authorization` from `--api-key`).",
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
        CLIParameter(
            name=(
                "--header",
                "-H",
            ),
            consume_multiple=True,
            group=Groups.INPUT,
        ),
    ] = []

    input_file: Annotated[
        Any,
        Field(
            description="Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. "
            "Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL for `mooncake_trace`/`bailian_trace` (timestamped traces), "
            "directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.",
        ),
        BeforeValidator(parse_file),
        CLIParameter(
            name=("--input-file",),
            group=Groups.INPUT,
        ),
    ] = None

    public_dataset: Annotated[
        PublicDatasetType | None,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). "
            "AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. "
            "Run `aiperf plugins public_dataset_loader` to list available datasets. "
            "Use `--hf-subset` to override the HuggingFace subset/config for HF-backed datasets.",
        ),
        CLIParameter(
            name=("--public-dataset",),
            group=Groups.INPUT,
        ),
    ] = None

    hf_dataset_subset: Annotated[
        str | None,
        Field(
            description="HuggingFace dataset subset/config name to override the plugin default (e.g. `sharegpt4o`). "
            "Only applies when using `--public-dataset` with a HuggingFace-backed loader. "
            "Takes priority over the subset defined in the plugin registry.",
        ),
        CLIParameter(
            name=("--hf-subset",),
            group=Groups.INPUT,
        ),
    ] = None

    custom_dataset_type: Annotated[
        CustomDatasetType | None,
        Field(
            description="Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. "
            "Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), "
            "`mooncake_trace`/`bailian_trace` (timestamped trace files), `random_pool` (directory of reusable prompts; "
            "when using `random_pool`, `--conversation-num` defaults to 100 if not specified; "
            "batch sizes > 1 sample each modality independently from a flat pool and do not preserve "
            "per-entry associations - use `single_turn` if paired modalities must stay together). "
            "Requires `--input-file`. Mutually exclusive with `--public-dataset`.",
        ),
        CLIParameter(
            name=("--custom-dataset-type",),
            group=Groups.INPUT,
        ),
    ] = None

    dataset_sampling_strategy: Annotated[
        DatasetSamplingStrategy | None,
        Field(
            description="Strategy for selecting entries from dataset during benchmarking. "
            "`sequential`: Iterate through dataset in order, wrapping to start after end. "
            "`random`: Randomly sample with replacement (entries may repeat before all are used). "
            "`shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. "
            "Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).",
        ),
        CLIParameter(
            name=("--dataset-sampling-strategy",),
            group=Groups.INPUT,
        ),
    ] = None

    random_seed: Annotated[
        int | None,
        Field(
            ge=0,
            description="Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other "
            "random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. "
            "Initialized globally at config creation.",
        ),
        CLIParameter(
            name=("--random-seed",),
            group=Groups.INPUT,
        ),
    ] = None

    config_file: Annotated[
        Path | None,
        Field(
            default=None,
            description=(
                "Path to a YAML configuration file. "
                "CLI flags override values from the config file."
            ),
        ),
        CLIParameter(
            name=("--config", "-f"),
            group=Groups.INPUT,
        ),
    ] = None

    ##############################################################################
    # Fixed Schedule
    ##############################################################################
    fixed_schedule: Annotated[
        bool,
        Field(
            description="Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays "
            "the exact timing pattern from the dataset. This mode is automatically enabled for trace datasets."
        ),
        CLIParameter(
            name=("--fixed-schedule",),
            group=Groups.FIXED_SCHEDULE,
        ),
    ] = False

    disable_auto_fixed_schedule: Annotated[
        bool,
        Field(
            description="Suppress the automatic switch to fixed-schedule mode for "
            "trace datasets that carry per-record timestamps. By default a "
            "trace input (e.g. mooncake_trace) with timestamps in the first "
            "record auto-promotes the profiling phase to fixed_schedule. Pass "
            "--no-fixed-schedule to keep the user-selected timing mode (e.g. "
            "concurrency, request_rate) and ignore the trace timestamps.",
        ),
        CLIParameter(
            name=("--no-fixed-schedule",),
            group=Groups.FIXED_SCHEDULE,
        ),
    ] = False

    fixed_schedule_auto_offset: Annotated[
        bool,
        Field(
            description="Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. "
            "When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute "
            "offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.",
        ),
        CLIParameter(
            name=("--fixed-schedule-auto-offset",),
            group=Groups.FIXED_SCHEDULE,
        ),
    ] = False

    fixed_schedule_start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing "
            "benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. "
            "Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. "
            "Must be ≤ `--fixed-schedule-end-offset` if both specified.",
        ),
        CLIParameter(
            name=("--fixed-schedule-start-offset",),
            group=Groups.FIXED_SCHEDULE,
        ),
    ] = None

    fixed_schedule_end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing "
            "benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. "
            "Must be ≥ `--fixed-schedule-start-offset` if both specified.",
        ),
        CLIParameter(
            name=("--fixed-schedule-end-offset",),
            group=Groups.FIXED_SCHEDULE,
        ),
    ] = None

    ##############################################################################
    # Goodput
    ##############################################################################
    goodput: Annotated[
        Any | None,
        Field(
            default=None,
            description="Specify service level objectives (SLOs) for goodput as space-separated "
            "'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the "
            "metric's display unit (falls back to its base unit if no display unit is defined). "
            "Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), "
            "`output_token_throughput_per_user:600` (tokens/s).\n"
            "Only metrics applicable to the current endpoint/config are considered. "
            "For more context on the definition of goodput, "
            "refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
            "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
        ),
        BeforeValidator(parse_str_as_numeric_dict),
        CLIParameter(
            name=("--goodput",),
            group=Groups.GOODPUT,
        ),
    ] = None

    ##############################################################################
    # Conversation Input
    ##############################################################################
    conversation_num: Annotated[
        Any,
        Field(
            description="The total number of unique conversations to generate.\n"
            "Each conversation represents a single request session between client and server.\n"
            "Supported on synthetic mode and the custom random_pool dataset. The number of conversations \n"
            "will be used to determine the number of entries in both the custom random_pool and synthetic \n"
            "datasets and will be reused until benchmarking is complete. "
            "Pass a comma-separated list (e.g. `--num-conversations 50,100,200`) to sweep over "
            "session-bounded run lengths; the converter promotes the list to a sweep on "
            "phases.profiling.sessions before AIPerfConfig validation. The synthetic dataset "
            "pool is sized to max(list) so every variation has its full unique-session set.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=(
                "--conversation-num",
                "--num-conversations",
                "--num-sessions",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = None

    conversation_num_dataset_entries: Annotated[
        int,
        Field(
            ge=1,
            description="Total number of unique entries to generate for the dataset. Each entry represents one user message that can be "
            "used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. "
            "Higher values provide more diversity.",
        ),
        CLIParameter(
            name=(
                "--num-dataset-entries",
                "--num-prompts",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = 100

    conversation_turn_mean: Annotated[
        Any,
        Field(
            description="Mean number of request-response turns per conversation. Each turn consists of a user message and model response. "
            "Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. "
            "Multi-turn conversations enable testing of context retention and conversation history handling. "
            "Pass a comma-separated list (e.g. `--conversation-turn-mean 1,3,8`) to sweep over multiple "
            "turn-mean values; the converter promotes the list to a sweep on "
            "datasets.main.turns.mean before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        AfterValidator(require_turn_mean_at_least_one),
        CLIParameter(
            name=(
                "--conversation-turn-mean",
                "--session-turns-mean",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = 1

    conversation_turn_stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating "
            "diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. "
            "Set to 0 for uniform conversation lengths.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-stddev",
                "--session-turns-stddev",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = 0

    conversation_turn_delay_mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between "
            "receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). "
            "Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-mean",
                "--session-turn-delay-mean",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = 0.0

    conversation_turn_delay_stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. "
            "Delays follow normal distribution. Set to 0 for deterministic delays. "
            "Models realistic human interaction patterns with variable response times.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-stddev",
                "--session-turn-delay-stddev",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = 0.0

    conversation_turn_delay_ratio: Annotated[
        float,
        Field(
            ge=0,
            description="Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: "
            "`actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. "
            "Values < 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-ratio",
                "--session-delay-ratio",
            ),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = 1.0

    inter_turn_delay_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Clamp per-turn replay delays (read from JSONL trace "
            "files) to at most this many seconds. ``None`` disables the cap. "
            "Used by the DAG JSONL loader to keep long pre-recorded waits "
            "from stalling the benchmark; the loader reports the clamp count "
            "at end of load. Routes onto the active FileDataset's "
            "``inter_turn_delay_cap_seconds`` field at config-resolution time.",
        ),
        CLIParameter(
            name=("--inter-turn-delay-cap-seconds",),
            group=Groups.CONVERSATION_INPUT,
        ),
    ] = None

    ##############################################################################
    # Prompt
    ##############################################################################
    prompt_batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` "
            "and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. "
            "Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.",
        ),
        CLIParameter(
            name=(
                "--prompt-batch-size",
                "--batch-size-text",
                "--batch-size",
                "-b",
            ),
            group=Groups.PROMPT,
        ),
    ] = 1

    ##############################################################################
    # Prefix Prompt
    ##############################################################################
    prompt_prefix_pool_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, "
            "simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. "
            "Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.",
        ),
        CLIParameter(
            name=(
                "--prompt-prefix-pool-size",
                "--prefix-prompt-pool-size",
                "--num-prefix-prompts",
            ),
            group=Groups.PREFIX_PROMPT,
        ),
    ] = 0

    prompt_prefix_length: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "The number of tokens in each prefix prompt.\n"
                "This is only used if `--num-prefix-prompts` is greater than zero.\n"
                "Note that due to the prefix and user prompts being concatenated,\n"
                "the number of tokens in the final prompt may be off by one."
                "Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`."
            ),
        ),
        CLIParameter(
            name=(
                "--prompt-prefix-length",
                "--prefix-prompt-length",
            ),
            group=Groups.PREFIX_PROMPT,
        ),
    ] = 0

    prompt_prefix_shared_system_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Length of shared system prompt in tokens.\n"
                "This prompt is identical across all sessions and appears as a system message.\n"
                "Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`."
            ),
        ),
        CLIParameter(
            name=("--shared-system-prompt-length",),
            group=Groups.PREFIX_PROMPT,
        ),
    ] = None

    prompt_prefix_user_context_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Length of per-session user context prompt in tokens.\n"
                "Each dataset entry gets a unique user context prompt.\n"
                "Requires --num-dataset-entries to be specified.\n"
                "Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size."
            ),
        ),
        CLIParameter(
            name=("--user-context-prompt-length",),
            group=Groups.PREFIX_PROMPT,
        ),
    ] = None

    ##############################################################################
    # Input Sequence Length (ISL)
    ##############################################################################
    prompt_input_tokens_mean: Annotated[
        Any,
        Field(
            description="Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths "
            "following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, "
            "not custom or public datasets. "
            "Pass a comma-separated list (e.g. `--isl 128,512,2048`) to sweep over multiple "
            "input lengths; the converter promotes the list to a sweep on "
            "datasets.main.prompts.isl.mean before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=(
                "--prompt-input-tokens-mean",
                "--synthetic-input-tokens-mean",
                "--isl",
            ),
            group=Groups.ISL,
        ),
    ] = 550

    prompt_input_tokens_stddev: Annotated[
        Any,
        Field(
            description="Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, "
            "simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. "
            "Set to 0 for uniform prompt lengths. Applies only to synthetic data generation. "
            "Pass a comma-separated list (e.g. `--isl-stddev 10,50,200`) to sweep over multiple stddev values; "
            "the converter promotes the list to a sweep on datasets.main.prompts.isl.stddev. Pair with a "
            "zip-mode `--isl` sweep to model realistic small/medium/large traffic shapes.",
        ),
        BeforeValidator(parse_float_or_float_list),
        CLIParameter(
            name=(
                "--prompt-input-tokens-stddev",
                "--synthetic-input-tokens-stddev",
                "--isl-stddev",
            ),
            group=Groups.ISL,
        ),
    ] = 0.0

    prompt_input_tokens_block_size: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Token block size for hash-based prompt caching in trace datasets (`mooncake_trace`, `bailian_trace`). When `hash_ids` are provided in trace entries, "
            "prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation "
            "of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`. "
            "When not set, the trace loader's `default_block_size` from plugin metadata is used (e.g. 16 for `bailian_trace`, 512 for `mooncake_trace`).",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-block-size",
                "--synthetic-input-tokens-block-size",
                "--isl-block-size",
            ),
            group=Groups.ISL,
        ),
    ] = None

    prompt_sequence_distribution: Annotated[
        str | None,
        Field(
            default=None,
            description="Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. "
            "Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). "
            "Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. "
            "Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. "
            "Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.",
        ),
        CLIParameter(
            name=("--seq-dist", "--sequence-distribution"),
            group=Groups.ISL,
        ),
    ] = None

    ##############################################################################
    # Output Sequence Length (OSL)
    ##############################################################################
    prompt_output_tokens_mean: Annotated[
        Any,
        Field(
            default=None,
            description="Mean number of tokens to request in model outputs via `max_completion_tokens` field. "
            "Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit "
            "generation length. When not set, model determines output length. "
            "Pass a comma-separated list (e.g. `--osl 128,256,512`) to sweep over multiple "
            "output lengths; the converter promotes the list to a sweep on "
            "datasets.main.prompts.osl.mean before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=(
                "--prompt-output-tokens-mean",
                "--output-tokens-mean",
                "--osl",
            ),
            group=Groups.OSL,
        ),
    ] = None

    prompt_output_tokens_stddev: Annotated[
        Any,
        Field(
            default=None,
            description="Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, "
            "simulating mixed response length requirements. Lengths follow normal distribution. "
            "Only applies when `--prompt-output-tokens-mean` is set. "
            "Pass a comma-separated list (e.g. `--osl-stddev 5,25,100`) to sweep over multiple stddev values; "
            "the converter promotes the list to a sweep on datasets.main.prompts.osl.stddev. Pair with a "
            "zip-mode `--osl` sweep to model realistic output-length variance across traffic tiers.",
        ),
        BeforeValidator(parse_float_or_float_list),
        CLIParameter(
            name=(
                "--prompt-output-tokens-stddev",
                "--output-tokens-stddev",
                "--osl-stddev",
            ),
            group=Groups.OSL,
        ),
    ] = 0

    ##############################################################################
    # Audio Input
    ##############################################################################
    audio_batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The number of audio inputs to include in each request. Supported with the `chat` endpoint type for multimodal models.",
        ),
        CLIParameter(
            name=(
                "--audio-batch-size",
                "--batch-size-audio",
            ),
            group=Groups.AUDIO_INPUT,
        ),
    ] = 1

    audio_length_mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean duration in seconds for synthetically generated audio files. Audio lengths follow a normal distribution "
            "around this mean (±`--audio-length-stddev`). Used when `--audio-batch-size` > 0 for multimodal benchmarking. "
            "Generated audio is random noise with specified sample rate, bit depth, and format.",
        ),
        CLIParameter(
            name=("--audio-length-mean",),
            group=Groups.AUDIO_INPUT,
        ),
    ] = 0.0

    audio_length_stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic audio duration in seconds. Creates variability in audio lengths when > 0, "
            "simulating mixed-duration audio inputs. Durations follow normal distribution. "
            "Set to 0 for uniform audio lengths.",
        ),
        CLIParameter(
            name=("--audio-length-stddev",),
            group=Groups.AUDIO_INPUT,
        ),
    ] = 0.0

    audio_format: Annotated[
        AudioFormat,
        Field(
            description="File format for generated audio files. Supports `wav` (uncompressed PCM, larger files) and `mp3` (compressed, smaller files). "
            "Format choice affects file size in multimodal requests but not audio characteristics (sample rate, bit depth, duration).",
        ),
        CLIParameter(
            name=("--audio-format",),
            group=Groups.AUDIO_INPUT,
        ),
    ] = AudioFormat.WAV

    audio_depths: Annotated[
        list[int],
        Field(
            min_length=1,
            description="List of audio bit depths in bits to randomly select from when generating audio files. Each audio file is assigned "
            "a random depth from this list. Common values: `8` (low quality), `16` (CD quality), `24` (professional), `32` (high-end). "
            "Specify multiple values (e.g., `--audio-depths 16 24`) for mixed-quality testing.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=("--audio-depths",),
            group=Groups.AUDIO_INPUT,
        ),
    ] = [16]

    audio_sample_rates: Annotated[
        list[float],
        Field(
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.\n"
            "Common sample rates are 16, 44.1, 48, 96, etc.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=("--audio-sample-rates",),
            group=Groups.AUDIO_INPUT,
        ),
    ] = [16.0]

    audio_num_channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            description="Number of audio channels for synthetic audio generation. `1` = mono (single channel), `2` = stereo (left/right channels). "
            "Stereo doubles file size but simulates realistic audio for models supporting spatial audio processing. "
            "Most speech models use mono.",
        ),
        CLIParameter(
            name=("--audio-num-channels",),
            group=Groups.AUDIO_INPUT,
        ),
    ] = 1

    ##############################################################################
    # Image Input
    ##############################################################################
    image_width_mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean width in pixels for synthetically generated images. Image widths follow a normal distribution "
            "around this mean (±`--image-width-stddev`). Combined with `--image-height-mean` to determine image dimensions "
            "and file sizes for multimodal benchmarking.",
        ),
        CLIParameter(
            name=("--image-width-mean",),
            group=Groups.IMAGE_INPUT,
        ),
    ] = 0.0

    image_width_stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, "
            "simulating mixed-resolution image inputs. Widths follow normal distribution. "
            "Set to 0 for uniform image widths.",
        ),
        CLIParameter(
            name=("--image-width-stddev",),
            group=Groups.IMAGE_INPUT,
        ),
    ] = 0.0

    image_height_mean: Annotated[
        float,
        Field(
            ge=0,
            description="Mean height in pixels for synthetically generated images. Image heights follow a normal distribution "
            "around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. "
            "Generated images are resized from source images in `assets/source_images` directory.",
        ),
        CLIParameter(
            name=("--image-height-mean",),
            group=Groups.IMAGE_INPUT,
        ),
    ] = 0.0

    image_height_stddev: Annotated[
        float,
        Field(
            ge=0,
            description="Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, "
            "simulating mixed-resolution image inputs. Heights follow normal distribution. "
            "Set to 0 for uniform image heights.",
        ),
        CLIParameter(
            name=("--image-height-stddev",),
            group=Groups.IMAGE_INPUT,
        ),
    ] = 0.0

    image_batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. "
            "Each image is generated by randomly sampling and resizing source images from `assets/source_images` directory to specified dimensions. "
            "Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.",
        ),
        CLIParameter(
            name=(
                "--image-batch-size",
                "--batch-size-image",
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ] = 1

    image_format: Annotated[
        ImageFormat,
        Field(
            description="Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), "
            "`jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. "
            "Format affects file size in multimodal requests and encoding overhead.",
        ),
        CLIParameter(
            name=("--image-format",),
            group=Groups.IMAGE_INPUT,
        ),
    ] = ImageFormat.PNG

    image_source: Annotated[
        ImageSource | Path,
        Field(
            default=ImageSource.NOISE,
            description="Source image generation mode (default `noise`). "
            "`noise` generates random noise images on the fly at the requested dimensions — no files on disk required, "
            "and the pool is effectively unbounded so servers cannot dedupe on identical inputs. "
            "`assets` loads images from the built-in `assets/source_images` directory (ships with a small set of 4 images) "
            "and resizes them to the requested dimensions. "
            "A path to a directory loads images from the given directory (e.g. `--image-source ./source_images`). "
            "Note: random-noise images are roughly incompressible, so payload bytes are larger than equivalent natural images.",
        ),
        CLIParameter(
            name=("--image-source",),
            group=Groups.IMAGE_INPUT,
        ),
    ]

    ##############################################################################
    # Video Input
    ##############################################################################
    video_batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. "
            "Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. "
            "Higher batch sizes test multi-video understanding and significantly increase request payload size.",
        ),
        CLIParameter(
            name=(
                "--video-batch-size",
                "--batch-size-video",
            ),
            group=Groups.VIDEO_INPUT,
        ),
    ] = 1

    video_duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count "
            "(frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. "
            "Requires FFmpeg for video generation.",
        ),
        CLIParameter(
            name=("--video-duration",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = 5.0

    video_fps: Annotated[
        int,
        Field(
            ge=1,
            description="Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. "
            "Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). "
            "Total frames = `--video-duration` × FPS.",
        ),
        CLIParameter(
            name=("--video-fps",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = 4

    video_width: Annotated[
        int | None,
        Field(
            ge=1,
            description="Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution "
            "and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.",
        ),
        CLIParameter(
            name=("--video-width",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = None

    video_height: Annotated[
        int | None,
        Field(
            ge=1,
            description="Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width "
            "determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.",
        ),
        CLIParameter(
            name=("--video-height",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = None

    video_synth_type: Annotated[
        VideoSynthType,
        Field(
            description="Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. "
            "Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). "
            "Content doesn't affect semantic meaning but may impact encoding efficiency and file size.",
        ),
        CLIParameter(
            name=("--video-synth-type",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoSynthType.MOVING_SHAPES

    video_format: Annotated[
        VideoFormat,
        Field(
            description="Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). "
            "Format choice affects compatibility, file size, and encoding options. "
            "Use `webm` for open-source workflows, `mp4` for maximum compatibility.",
        ),
        CLIParameter(
            name=("--video-format",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoFormat.WEBM

    video_codec: Annotated[
        str,
        Field(
            description=(
                "The video codec to use for encoding. Common options: "
                "libvpx-vp9 (CPU, BSD-licensed, default for WebM), "
                "libx264 (CPU, GPL-licensed, widely compatible), "
                "libx265 (CPU, GPL-licensed, smaller files), "
                "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
                "Any FFmpeg-supported codec can be used."
            ),
        ),
        CLIParameter(
            name=("--video-codec",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = "libvpx-vp9"

    video_audio_sample_rate: Annotated[
        float,
        Field(
            ge=8,
            le=96000,
            description="Audio sample rate in Hz or kHz for the embedded audio track. "
            "Common values: 8/8000 (telephony), 16/16000 (speech), 44.1/44100 (CD quality), 48/48000 (professional). "
            "Higher sample rates increase audio fidelity and file size.",
        ),
        CLIParameter(
            name=("--video-audio-sample-rate",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = 44100

    video_audio_channels: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            description="Number of audio channels to embed in generated video files. "
            "0 = disabled (no audio track, default), 1 = mono, 2 = stereo. "
            "When set to 1 or 2, a Gaussian noise audio track matching the video duration "
            "is muxed into each video via FFmpeg.",
        ),
        CLIParameter(
            name=("--video-audio-num-channels",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = 0

    video_audio_codec: Annotated[
        VideoAudioCodec | None,
        Field(
            description="Audio codec for the embedded audio track. "
            "If not specified, auto-selects based on video format: "
            "aac for MP4, libvorbis for WebM. "
            "Options: aac, libvorbis, libopus.",
        ),
        CLIParameter(
            name=("--video-audio-codec",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = None

    video_audio_depth: Annotated[
        Literal[8, 16, 24, 32],
        Field(
            description="Audio bit depth for the embedded audio track. "
            "Supported values: 8, 16, 24, or 32 bits. "
            "Higher bit depths provide greater dynamic range but increase file size.",
        ),
        BeforeValidator(int),
        CLIParameter(
            name=("--video-audio-depth",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = 16

    ##############################################################################
    # Rankings
    ##############################################################################
    rankings_passages_mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query "
            "and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). "
            "Higher values test ranking at scale but increase request payload size and processing time.",
        ),
        CLIParameter(
            name=("--rankings-passages-mean",),
            group=Groups.RANKINGS,
        ),
    ] = 1

    rankings_passages_stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. "
            "Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.",
        ),
        CLIParameter(
            name=("--rankings-passages-stddev",),
            group=Groups.RANKINGS,
        ),
    ] = 0

    rankings_passages_prompt_token_mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths "
            "following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). "
            "Longer passages increase input processing demands and request size.",
        ),
        CLIParameter(
            name=("--rankings-passages-prompt-token-mean",),
            group=Groups.RANKINGS,
        ),
    ] = 550

    rankings_passages_prompt_token_stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating "
            "realistic heterogeneous document collections. Token lengths follow normal distribution. "
            "Set to 0 for uniform passage lengths.",
        ),
        CLIParameter(
            name=("--rankings-passages-prompt-token-stddev",),
            group=Groups.RANKINGS,
        ),
    ] = 0

    rankings_query_prompt_token_mean: Annotated[
        int,
        Field(
            ge=1,
            description="Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. "
            "Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`). ",
        ),
        CLIParameter(
            name=("--rankings-query-prompt-token-mean",),
            group=Groups.RANKINGS,
        ),
    ] = 550

    rankings_query_prompt_token_stddev: Annotated[
        int,
        Field(
            ge=0,
            description="Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating "
            "realistic user search patterns. Token lengths follow normal distribution. "
            "Set to 0 for uniform query lengths.",
        ),
        CLIParameter(
            name=("--rankings-query-prompt-token-stddev",),
            group=Groups.RANKINGS,
        ),
    ] = 0

    ##############################################################################
    # Synthesis
    ##############################################################################
    synthesis_speedup_ratio: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for timestamp scaling in synthesized traces",
        ),
        CLIParameter(name=("--synthesis-speedup-ratio",), group=Groups.SYNTHESIS),
    ] = 1.0

    synthesis_prefix_len_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for core prefix branch lengths in radix tree",
        ),
        CLIParameter(
            name=("--synthesis-prefix-len-multiplier",), group=Groups.SYNTHESIS
        ),
    ] = 1.0

    synthesis_prefix_root_multiplier: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Number of independent radix trees to distribute traces across",
        ),
        CLIParameter(
            name=("--synthesis-prefix-root-multiplier",), group=Groups.SYNTHESIS
        ),
    ] = 1

    synthesis_prompt_len_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for leaf path (unique prompt) lengths",
        ),
        CLIParameter(
            name=("--synthesis-prompt-len-multiplier",), group=Groups.SYNTHESIS
        ),
    ] = 1.0

    synthesis_max_isl: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.",
        ),
        CLIParameter(name=("--synthesis-max-isl",), group=Groups.SYNTHESIS),
    ] = None

    synthesis_max_osl: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.",
        ),
        CLIParameter(name=("--synthesis-max-osl",), group=Groups.SYNTHESIS),
    ] = None

    ##############################################################################
    # Load Generator
    ##############################################################################
    benchmark_duration: Annotated[
        Any,
        Field(
            description="Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, "
            "Responses received within `--benchmark-grace-period` after duration ends are included in metrics. "
            "Pass a comma-separated list (e.g. `--benchmark-duration 30,60,120`) to sweep over multiple "
            "durations; the converter promotes the list to a sweep on phases.profiling.duration before "
            "AIPerfConfig validation.",
        ),
        BeforeValidator(parse_float_or_float_list),
        CLIParameter(
            name=("--benchmark-duration",),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    benchmark_grace_period: Annotated[
        float,
        Field(
            ge=0,
            description="The grace period in seconds to wait for responses after benchmark duration ends. "
            "Only applies when --benchmark-duration is set. Responses received within this period "
            "are included in metrics. Use 'inf' to wait indefinitely for all responses.",
        ),
        CLIParameter(
            name=("--benchmark-grace-period",),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = 30.0

    concurrency: Annotated[
        Any,
        Field(
            description="Number of concurrent requests to maintain. AIPerf issues a new request immediately when one completes, "
            "maintaining this level of in-flight requests. Can be combined with `--request-rate` to control the request rate. "
            "Pass a comma-separated list (e.g. `--concurrency 10,20,30`) to sweep over multiple concurrencies; "
            "the converter promotes the list to a sweep before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=(
                "--concurrency",  # GenAI-Perf
            ),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    prefill_concurrency: Annotated[
        Any,
        Field(
            description="Max concurrent requests waiting for first token (prefill phase). "
            "Limits how many requests can be in the prefill/prompt-processing stage simultaneously. "
            "Pass a comma-separated list (e.g. `--prefill-concurrency 1,2,4`) to sweep over multiple "
            "values; the converter promotes the list to a sweep before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=("--prefill-concurrency",),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    request_rate: Annotated[
        Any,
        Field(
            description="Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` "
            "to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. "
            "Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds). "
            "Pass a comma-separated list (e.g. `--request-rate 10,20,50`) to sweep over multiple rates; "
            "the converter promotes the list to a sweep before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_float_or_float_list),
        CLIParameter(
            name=(
                "--request-rate",  # GenAI-Perf
            ),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    arrival_pattern: Annotated[
        ArrivalPattern,
        Field(
            description="Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma.\n"
            "`constant`: Generate requests at a fixed rate.\n"
            "`poisson`: Generate requests using a poisson distribution.\n"
            "`gamma`: Generate requests using a gamma distribution with tunable smoothness."
        ),
        CLIParameter(
            name=("--arrival-pattern", "--request-rate-mode"),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = ArrivalPattern.POISSON

    arrival_smoothness: Annotated[
        float | None,
        Field(
            gt=0,
            description="Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). "
            "Controls the shape of the arrival pattern:\n"
            "- 1.0: Poisson-like (exponential inter-arrivals, default)\n"
            "- <1.0: Bursty/clustered arrivals (higher variance)\n"
            "- >1.0: Smooth/regular arrivals (lower variance)\n"
            "Compatible with vLLM's --burstiness parameter (same value = same distribution).",
        ),
        CLIParameter(
            name=("--arrival-smoothness", "--vllm-burstiness"),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    request_count: Annotated[
        Any,
        Field(
            description="The maximum number of requests to send. If not set, will be automatically determined based "
            "on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`. "
            "Pass a comma-separated list (e.g. `--request-count 100,500,1000`) to sweep over multiple "
            "request counts; the converter promotes the list to a sweep on phases.profiling.requests "
            "before AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=(
                "--request-count",  # GenAI-Perf
                "--num-requests",  # GenAI-Perf
            ),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp session concurrency from 1 to target. "
            "Useful for gradual warm-up of the target system.",
        ),
        CLIParameter(
            name=("--concurrency-ramp-duration",),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    prefill_concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp prefill concurrency from 1 to target.",
        ),
        CLIParameter(
            name=("--prefill-concurrency-ramp-duration",),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    request_rate_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp request rate from a proportional minimum to target. "
            "Start rate is calculated as target * (update_interval / duration), ensuring correct "
            "behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.",
        ),
        CLIParameter(
            name=("--request-rate-ramp-duration",),
            group=Groups.LOAD_GENERATOR,
        ),
    ] = None

    ##############################################################################
    # Warmup
    ##############################################################################
    warmup_request_count: Annotated[
        int | None,
        Field(
            gt=0,
            description="The maximum number of warmup requests to send before benchmarking. "
            "If not set and no --warmup-duration is set, then no warmup phase will be used.",
        ),
        CLIParameter(
            name=(
                "--warmup-request-count",  # GenAI-Perf
                "--num-warmup-requests",  # GenAI-Perf
            ),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. "
            "If neither are set, no warmup phase will be used.",
        ),
        CLIParameter(
            name=("--warmup-duration",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_num_sessions: Annotated[
        int | None,
        Field(
            ge=1,
            description="The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.",
        ),
        CLIParameter(
            name=("--num-warmup-sessions",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            description="The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.",
        ),
        CLIParameter(
            name=("--warmup-concurrency",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_prefill_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            description="The prefill concurrency value to use for the warmup phase. "
            "If not set, it will use the `--prefill-concurrency` value.",
        ),
        CLIParameter(
            name=("--warmup-prefill-concurrency",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_request_rate: Annotated[
        float | None,
        Field(
            gt=0,
            description="The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.",
        ),
        CLIParameter(
            name=("--warmup-request-rate",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_arrival_pattern: Annotated[
        ArrivalPattern | None,
        Field(
            default=None,
            description="The arrival pattern to use for the warmup phase. "
            "If not set, it will use the `--arrival-pattern` value. "
            "Valid values: constant, poisson, gamma.",
        ),
        CLIParameter(
            name=("--warmup-arrival-pattern",),
            group=Groups.WARMUP,
            show_choices=False,
        ),
    ] = None

    warmup_grace_period: Annotated[
        float | None,
        Field(
            ge=0,
            description="The grace period in seconds to wait for responses after warmup phase ends. "
            "Only applies when warmup is enabled. Responses received within this period "
            "are included in warmup completion. If not set, waits indefinitely for all warmup responses.",
        ),
        CLIParameter(
            name=("--warmup-grace-period",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp warmup session concurrency from 1 to target. "
            "If not set, uses `--concurrency-ramp-duration` value.",
        ),
        CLIParameter(
            name=("--warmup-concurrency-ramp-duration",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_prefill_concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp warmup prefill concurrency from 1 to target. "
            "If not set, uses `--prefill-concurrency-ramp-duration` value.",
        ),
        CLIParameter(
            name=("--warmup-prefill-concurrency-ramp-duration",),
            group=Groups.WARMUP,
        ),
    ] = None

    warmup_request_rate_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp warmup request rate from a proportional minimum to target. "
            "Start rate is calculated as target * (update_interval / duration). "
            "If not set, uses `--request-rate-ramp-duration` value.",
        ),
        CLIParameter(
            name=("--warmup-request-rate-ramp-duration",),
            group=Groups.WARMUP,
        ),
    ] = None

    ##############################################################################
    # User-Centric Rate
    ##############################################################################
    user_centric_rate: Annotated[
        float | None,
        Field(
            gt=0,
            description="Enable user-centric rate limiting mode with the specified request rate (QPS). "
            "Each user has a gap = num_users / qps between turns. "
            "Users block on their previous turn (no interleaving within a user). "
            "New users are spawned on a fixed schedule to maintain steady-state throughput. "
            "Designed for KV cache benchmarking with realistic multi-user patterns. "
            "Requires --num-users to be set.",
        ),
        CLIParameter(
            name=("--user-centric-rate",),
            group=Groups.USER_CENTRIC,
        ),
    ] = None

    num_users: Annotated[
        Any,
        Field(
            description="The number of initial users to use for --user-centric-rate mode. "
            "Pass a comma-separated list (e.g. `--num-users 4,8,16`) to sweep over user counts; "
            "the converter promotes the list to a sweep on phases.profiling.users before "
            "AIPerfConfig validation.",
        ),
        BeforeValidator(parse_int_or_int_list),
        CLIParameter(
            name=("--num-users",),
            group=Groups.USER_CENTRIC,
        ),
    ] = None

    ##############################################################################
    # Request Cancellation
    ##############################################################################
    request_cancellation_rate: Annotated[
        float | None,
        Field(
            gt=0.0,
            le=100.0,
            description="Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally "
            "but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.",
        ),
        CLIParameter(
            name=("--request-cancellation-rate",),
            group=Groups.REQUEST_CANCELLATION,
        ),
    ] = None

    request_cancellation_delay: Annotated[
        float,
        Field(
            ge=0.0,
            description="Seconds to wait after the request is fully sent before cancelling. "
            "A delay of 0 means 'send the full request, then immediately disconnect'. "
            "Requires --request-cancellation-rate to be set.",
        ),
        CLIParameter(
            name=("--request-cancellation-delay",),
            group=Groups.REQUEST_CANCELLATION,
        ),
    ] = 0.0

    ##############################################################################
    # Output
    ##############################################################################
    artifact_directory: Annotated[
        Path,
        Field(
            description="Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), "
            "GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. "
            "All output file paths are constructed relative to this directory.",
        ),
        CLIParameter(
            name=(
                "--output-artifact-dir",
                "--artifact-dir",  # GenAI-Perf
            ),
            group=Groups.OUTPUT,
        ),
    ] = OutputDefaults.ARTIFACT_DIRECTORY

    profile_export_prefix: Annotated[
        Path | None,
        Field(
            description="Base filename for ALL exported files. With prefix='foo' every "
            "output becomes `foo.csv`, `foo.json`, `foo_timeslices.{csv,json}`, "
            "`foo.jsonl`, `foo_raw.jsonl`, `foo_gpu_telemetry.jsonl`, and "
            "`foo_server_metrics.{jsonl,json,csv,parquet}`. When unset (the default), "
            "historical per-file names are used: `profile_export_aiperf.{csv,json}` "
            "for the summary, `profile_export.jsonl` and `profile_export_raw.jsonl` "
            "for records, `gpu_telemetry_export.jsonl`, and `server_metrics_export.*`. "
            "Known suffixes (e.g. `_raw.jsonl`, `_timeslices.csv`, `_server_metrics.parquet`) "
            "are stripped from the supplied value.",
        ),
        CLIParameter(
            name=(
                "--profile-export-prefix",
                "--profile-export-file",  # GenAI-Perf
            ),
            group=Groups.OUTPUT,
        ),
    ] = None

    export_level: Annotated[
        ExportLevel,
        Field(
            description="Controls which output files are generated. "
            "`summary`: Only aggregate metrics files (`.csv`, `.json`). "
            "`records`: Includes per-request metrics (`.jsonl`). "
            "`raw`: Includes raw request/response data (`_raw.jsonl`).",
        ),
        CLIParameter(
            name=("--export-level", "--profile-export-level"),
            group=Groups.OUTPUT,
        ),
    ] = OutputDefaults.EXPORT_LEVEL

    slice_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length "
            "windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time "
            "(e.g., warmup effects, degradation under sustained load).",
        ),
        CLIParameter(
            name=("--slice-duration"),
            group=Groups.OUTPUT,
        ),
    ] = OutputDefaults.SLICE_DURATION

    auto_plot: Annotated[
        bool | None,
        Field(
            description=(
                "Auto-invoke `aiperf plot` against the artifact directory after the "
                "benchmark completes. None = defer to recipe default (False if no "
                "recipe). True/False = explicit override. Failures are logged but "
                "do not fail the command unless --plot-required is set."
            ),
        ),
        Parameter(
            name=("--auto-plot",),
            group=Groups.OUTPUT,
            show_env_var=False,
            negative="--no-auto-plot",
        ),
    ] = None

    plot_required: Annotated[
        bool,
        Field(
            description=(
                "Treat auto-plot failures as fatal: re-raise so `aiperf profile` exits "
                "non-zero. Only meaningful when auto-plot is on. Default False = warn "
                "and continue."
            ),
        ),
        CLIParameter(
            name=("--plot-required",),
            group=Groups.OUTPUT,
        ),
    ] = False

    ##############################################################################
    # OpenTelemetry / MLflow
    ##############################################################################
    otel_url: Annotated[
        str | None,
        Field(default=None, description="OTLP/HTTP metrics endpoint URL."),
        CLIParameter(name=("--otel-url",), group=Groups.OUTPUT),
    ] = None

    stream: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Select which AIPerf telemetry domains to stream over OTel. "
                "Valid values: 'metrics', 'timing', or 'default'. "
                "'default' streams both metrics and timing. "
                "Examples: --stream metrics | --stream timing | --stream metrics timing"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--stream",),
            consume_multiple=True,
            group=Groups.OUTPUT,
        ),
    ] = None

    otel_resource_attributes: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Custom OTel resource attributes as key=value pairs. "
                "Merged into the default resource attributes on every exported metric."
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--otel-resource-attributes",),
            consume_multiple=True,
            group=Groups.OUTPUT,
        ),
    ] = None

    gen_ai_provider: Annotated[
        str | None,
        Field(default=None, description="GenAI semantic convention provider override."),
        CLIParameter(name=("--gen-ai-provider",), group=Groups.OUTPUT),
    ] = None

    mlflow_tracking_uri: Annotated[
        str | None,
        Field(default=None, description="MLflow tracking URI."),
        CLIParameter(name=("--mlflow-tracking-uri",), group=Groups.OUTPUT),
    ] = None

    mlflow_experiment: Annotated[
        str | None,
        Field(default=None, description="MLflow experiment name."),
        CLIParameter(name=("--mlflow-experiment",), group=Groups.OUTPUT),
    ] = None

    mlflow_run_name: Annotated[
        str | None,
        Field(default=None, description="MLflow run name."),
        CLIParameter(name=("--mlflow-run-name",), group=Groups.OUTPUT),
    ] = None

    mlflow_tags: Annotated[
        list[tuple[str, Any]] | None,
        Field(
            default=None,
            description=(
                "Additional MLflow run tags to attach on upload. "
                "Specify as key:value pairs (e.g., --mlflow-tag team:perf) "
                "or as JSON string."
            ),
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
        CLIParameter(
            name=("--mlflow-tag",),
            consume_multiple=True,
            group=Groups.OUTPUT,
        ),
    ] = None

    mlflow_parent_run_id: Annotated[
        str | None,
        Field(default=None, description="Optional MLflow parent run ID."),
        CLIParameter(name=("--mlflow-parent-run-id",), group=Groups.OUTPUT),
    ] = None

    mlflow_artifact_globs: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Artifact glob overrides for MLflow upload. "
                "Can be specified multiple times or as a comma-separated list."
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--mlflow-artifact-glob",),
            consume_multiple=True,
            group=Groups.OUTPUT,
        ),
    ] = None

    ##############################################################################
    # HTTP Trace
    ##############################################################################
    export_http_trace: Annotated[
        bool,
        Field(
            description="Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. "
            "Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. "
            "See the HTTP Trace Metrics guide for details on trace data fields.",
        ),
        CLIParameter(
            name="--export-http-trace",
            group=Groups.HTTP_TRACE,
        ),
    ] = OutputDefaults.EXPORT_HTTP_TRACE

    show_trace_timing: Annotated[
        bool,
        Field(
            description="Display HTTP trace timing metrics in the console at the end of the benchmark. "
            "Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, "
            "and total duration following k6 naming conventions.",
        ),
        CLIParameter(
            name="--show-trace-timing",
            group=Groups.HTTP_TRACE,
        ),
    ] = OutputDefaults.SHOW_TRACE_TIMING

    ##############################################################################
    # Server Metrics
    ##############################################################################
    server_metrics: Annotated[
        list[str] | None,
        Field(
            description=(
                "Server metrics collection (ENABLED BY DEFAULT). "
                "Automatically collects from inference endpoint base_url + `/metrics`. "
                "Optionally specify additional custom Prometheus-compatible endpoint URLs "
                "(e.g., http://node1:8081/metrics, http://node2:9090/metrics). "
                "Use `--no-server-metrics` to disable collection. "
                "Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--server-metrics",),
            consume_multiple=True,
            group=Groups.SERVER_METRICS,
        ),
    ] = None

    no_server_metrics: Annotated[
        bool,
        Field(
            description="Disable server metrics collection entirely.",
        ),
        CLIParameter(
            name=("--no-server-metrics",),
            group=Groups.SERVER_METRICS,
        ),
    ] = False

    server_metrics_formats: Annotated[
        list[ServerMetricsFormat],
        Field(
            description=(
                "Specify which output formats to generate for server metrics. "
                "Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`)."
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--server-metrics-formats",),
            consume_multiple=True,
            group=Groups.SERVER_METRICS,
        ),
    ] = _DEFAULT_SERVER_METRICS_FORMATS

    ##############################################################################
    # GPU Telemetry
    ##############################################################################
    gpu_telemetry: Annotated[
        list[str] | None,
        Field(
            description=(
                "Enable GPU telemetry console display and optionally specify: "
                "(1) 'pynvml' or 'amdsmi' to use a local GPU library instead of DCGM HTTP endpoints, "
                "(2) 'dashboard' for realtime dashboard mode, "
                "(3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), "
                "(4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). "
                "Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. "
                "Examples: --gpu-telemetry pynvml | --gpu-telemetry amdsmi | --gpu-telemetry dashboard node1:9400"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--gpu-telemetry",),
            consume_multiple=True,
            group=Groups.GPU_TELEMETRY,
        ),
    ] = None

    no_gpu_telemetry: Annotated[
        bool,
        Field(
            description="Disable GPU telemetry collection entirely.",
        ),
        CLIParameter(
            name=("--no-gpu-telemetry",),
            group=Groups.GPU_TELEMETRY,
        ),
    ] = False

    ##############################################################################
    # UI
    ##############################################################################
    ui_type: Annotated[
        UIType,
        Field(
            description="Select the user interface type for displaying benchmark progress. "
            "`dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, "
            "`none` disables UI completely. Defaults to `dashboard` in interactive terminals, "
            "`none` when not a TTY (e.g., piped or redirected output). "
            "Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.",
        ),
        CLIParameter(
            name=("--ui-type", "--ui"),
            group=Groups.UI,
        ),
    ] = ServiceDefaults.UI_TYPE

    ##############################################################################
    # Multi-Run
    ##############################################################################
    # Upper limit of 10 runs balances statistical validity with practical considerations:
    # - Statistical: 10 samples provide reasonable confidence intervals (t-distribution)
    # - Practical: Limits total benchmark time (10 runs can take hours for long benchmarks)
    # - Diminishing returns: Confidence interval width decreases with sqrt(n), so gains
    #   beyond 10 runs are marginal compared to the additional time investment
    # - Resource efficiency: Reduces compute/GPU costs while maintaining statistical rigor
    num_profile_runs: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            description="Number of profile runs to execute for confidence reporting. "
            "Must be between 1 and 10. "
            "When set to 1 (default), runs a single benchmark. "
            "When set to >1, runs multiple benchmarks and computes aggregate statistics "
            "(mean, std, confidence intervals, coefficient of variation) across runs. "
            "Useful for quantifying variance and establishing confidence in results.",
        ),
        CLIParameter(
            name=("--num-profile-runs",),
            group=Groups.MULTI_RUN,
        ),
    ] = 1

    profile_run_cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            description="Cooldown duration in seconds between profile runs. "
            "Only applies when --num-profile-runs > 1. "
            "Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). "
            "Default is 0 (no cooldown).",
        ),
        CLIParameter(
            name=("--profile-run-cooldown-seconds",),
            group=Groups.MULTI_RUN,
        ),
    ] = 0.0

    confidence_level: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            description="Confidence level for computing confidence intervals (0-1). "
            "Only applies when --num-profile-runs > 1. "
            "Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). "
            "Higher values produce wider confidence intervals.",
        ),
        CLIParameter(
            name=("--confidence-level",),
            group=Groups.MULTI_RUN,
        ),
    ] = 0.95

    profile_run_disable_warmup_after_first: Annotated[
        bool,
        Field(
            description="Disable warmup for profile runs after the first. "
            "Only applies when --num-profile-runs > 1. "
            "When True (default), only the first run includes warmup, subsequent runs "
            "measure steady-state performance for more accurate aggregate statistics. "
            "When False, all runs include warmup (useful for long cooldown periods "
            "or when testing cold-start performance).",
        ),
        Parameter(
            name=("--profile-run-disable-warmup-after-first",),
            group=Groups.MULTI_RUN,
            show_env_var=False,
            negative="--no-profile-run-disable-warmup-after-first",
        ),
    ] = True

    set_consistent_seed: Annotated[
        bool,
        Field(
            description="Automatically set random seed for consistent workloads across runs. "
            "Only applies when --num-profile-runs > 1. "
            "When True (default), automatically sets --random-seed=42 if not specified, "
            "ensuring identical workloads across all runs for valid statistical comparison. "
            "When False, preserves None seed, resulting in different workloads per run "
            "(not recommended for confidence reporting as it produces invalid statistics). "
            "If --random-seed is explicitly set, that value is always used regardless of this setting.",
        ),
        Parameter(
            name=("--set-consistent-seed",),
            group=Groups.MULTI_RUN,
            show_env_var=False,
            negative="--no-set-consistent-seed",
        ),
    ] = True

    vary_seed_per_trial: Annotated[
        bool,
        Field(
            description="When True, derive a distinct seed for each trial of a variation "
            "via SHA-256 over (envelope_seed, variation.label, trial). "
            "When False (default), all trials of a variation share the same seed, "
            "giving pure-runtime variance for confidence "
            "intervals. Enable when you want trials to also sample different inputs "
            "(captures end-to-end variance at the cost of conflating input noise "
            "with runtime noise in the resulting confidence statistics).",
        ),
        Parameter(
            name=("--vary-seed-per-trial",),
            group=Groups.MULTI_RUN,
            show_env_var=False,
            negative="--no-vary-seed-per-trial",
        ),
    ] = False

    convergence_metric: Annotated[
        str | None,
        Field(
            description="Target metric name for adaptive convergence stopping. "
            "When set with --num-profile-runs > 1, enables adaptive mode that stops "
            "early once the metric stabilizes according to --convergence-mode. "
            "Uses --num-profile-runs as the maximum run cap. "
            "Example metrics: time_to_first_token, request_latency, inter_token_latency.",
        ),
        CLIParameter(
            name=("--convergence-metric",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    convergence_stat: Annotated[
        ConvergenceStat,
        Field(
            description="Statistic to evaluate for convergence when using ci_width or cv mode. "
            "Common values: avg, p50, p90, p95, p99. "
            "Only applies when --convergence-metric is set.",
        ),
        CLIParameter(
            name=("--convergence-stat",),
            group=Groups.MULTI_RUN,
        ),
    ] = ConvergenceStat.AVG

    convergence_threshold: Annotated[
        float | None,
        Field(
            gt=0,
            lt=1,
            description="Threshold for convergence detection. "
            "For ci_width mode: maximum CI width as a fraction of the mean. "
            "For cv mode: maximum coefficient of variation. "
            "For distribution mode: KS test p-value threshold. "
            "When unset, each mode uses its own algorithm-specific default. "
            "Only applies when --convergence-metric is set.",
        ),
        CLIParameter(
            name=("--convergence-threshold",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    convergence_mode: Annotated[
        ConvergenceCriterionType,
        Field(
            description="Statistical method for convergence detection. "
            "ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. "
            "cv: Stop when coefficient of variation (std/mean) is below threshold. "
            "distribution: Stop when KS test p-value indicates latest run matches prior runs "
            "(requires --export-level records or --export-level raw; rejected with --export-level summary). "
            "Only applies when --convergence-metric is set.",
        ),
        CLIParameter(
            name=("--convergence-mode",),
            group=Groups.MULTI_RUN,
        ),
    ] = ConvergenceCriterionType.CI_WIDTH

    parameter_sweep_cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            description="Cooldown seconds between sweep variations (e.g. between "
            "--concurrency 10 and --concurrency 20). Honored by "
            "MultiRunOrchestrator when iterating plan.configs. Default 0.",
        ),
        CLIParameter(
            name=("--parameter-sweep-cooldown-seconds",),
            group=Groups.MULTI_RUN,
        ),
    ] = 0.0

    parameter_sweep_same_seed: Annotated[
        bool,
        Field(
            description="If true, every sweep variation reuses the same random seed "
            "(correlated comparisons). If false (default), each variation derives "
            "a unique seed `base_seed + variation.index` so independent draws "
            "exercise different inputs. Requires --random-seed when true.",
        ),
        Parameter(
            name=("--parameter-sweep-same-seed",),
            group=Groups.MULTI_RUN,
            show_env_var=False,
            negative="--no-parameter-sweep-same-seed",
        ),
    ] = False

    parameter_sweep_mode: Annotated[
        SweepMode,
        Field(
            description="Execution order for sweep + multi-trial composition. "
            "'repeated' (default) iterates trials as the outer loop and "
            "variations as the inner loop, so all variations run within "
            "trial 1, then within trial 2, etc. 'independent' inverts the "
            "loops: all trials at one variation complete before the next "
            "variation starts. Both modes produce the same total runs, only "
            "the artifact-path layout and submit order differ.",
        ),
        CLIParameter(
            name=("--parameter-sweep-mode",),
            group=Groups.MULTI_RUN,
        ),
    ] = SweepMode.REPEATED

    sweep_type: Annotated[
        Literal["grid", "zip"],
        Field(
            description="Topology used when multiple CLI magic-list flags "
            "(--concurrency, --request-rate, --isl, --osl, ...) are passed together. "
            "'grid' (default) takes the Cartesian product of all lists; 'zip' pairs "
            "them element-wise (all lists must have equal length, like the YAML "
            "`sweep: {type: zip}` block). Ignored when only one magic-list flag is set "
            "or when the sweep is declared in YAML.",
        ),
        CLIParameter(
            name=("--sweep-type",),
            group=Groups.MULTI_RUN,
        ),
    ] = "grid"

    no_sweep_table: Annotated[
        bool,
        Field(
            description=(
                "Suppress the per-cell streaming sweep table during "
                "multi-variation sweeps. Auto-suppressed when stdout is "
                "non-interactive, when the dashboard UI is active, or for "
                "single-cell sweeps."
            ),
        ),
        CLIParameter(
            name=("--no-sweep-table",),
            group=Groups.MULTI_RUN,
        ),
    ] = False

    search_space: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Adaptive-search space dimensions. Repeatable. Each value is "
                "'path:lo,hi[:kind]', e.g. 'phases.profiling.concurrency:1,1000:int'. "
                "Mutually exclusive with magic-list flags (--concurrency 10,20,30) and "
                "with explicit sweep blocks. See docs/sweeping/bayesian-optimization.md."
            ),
        ),
        CLIParameter(
            name=("--search-space",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_metric: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Metric tag to optimize, e.g. 'output_token_throughput'. Required "
                "when --search-space is set. Must match a key in "
                "RunResult.summary_metrics produced by the run (NOT the flattened "
                "'_avg' / '_p99' aggregator-suffixed key)."
            ),
        ),
        CLIParameter(
            name=("--search-metric",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_stat: Annotated[
        Literal["avg", "p50", "p90", "p95", "p99"] | None,
        Field(
            default=None,
            description=(
                "Statistic on the metric: avg / p50 / p90 / p95 / p99. Defaults to "
                "'avg' when omitted (set by the CLIConfig -> AIPerfConfig converter)."
            ),
        ),
        CLIParameter(
            name=("--search-stat",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_direction: Annotated[
        Literal["maximize", "minimize"] | None,
        Field(
            default=None,
            description=(
                "Optimization direction. Required when --search-space is set."
            ),
        ),
        CLIParameter(
            name=("--search-direction",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_max_iterations: Annotated[
        int | None,
        Field(
            default=None,
            ge=2,
            le=200,
            description=(
                "Maximum number of search iterations. Each iteration runs "
                "--num-profile-runs benchmarks. Required when --search-space is set."
            ),
        ),
        CLIParameter(
            name=("--search-max-iterations",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_initial_points: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Random Sobol points before fitting the GP. Defaults to 5 "
                "when omitted. Must be < --search-max-iterations."
            ),
        ),
        CLIParameter(
            name=("--search-initial-points",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_random_seed: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "Random seed for reproducible search trajectories. When unset, "
                "the planner uses non-deterministic randomness."
            ),
        ),
        CLIParameter(
            name=("--search-random-seed",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_planner: Annotated[
        SearchPlannerType | None,
        Field(
            default=None,
            description=(
                "Outer-loop search planner plugin. Default `bayesian` is a "
                "curated Optuna preset that uses BoTorch qLogNEI/qLogNEHVI when "
                "the optional `botorch` extra is installed and otherwise falls "
                "back to Optuna TPE with a warning. `optuna` is the expert-mode "
                "alternative exposing `--optuna-sampler` (tpe / gp / botorch) "
                "and `--optuna-acquisition`. Explicit unavailable optional "
                "samplers raise. Third-party planners registered under "
                "the `search_planner` plugin category are accepted here. Only "
                "applies when --search-space is set."
            ),
        ),
        CLIParameter(
            name=("--search-planner",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    optuna_sampler: Annotated[
        Literal["gp", "tpe", "botorch"] | None,
        Field(
            default=None,
            description=(
                "Optuna sampler selection. Only consulted when "
                "--search-planner=optuna. ``botorch`` is the preferred implicit "
                "default and requires the optional ``botorch`` extra; when the "
                "implicit default is unavailable, the planner warns and falls "
                "back to ``tpe``. Explicit ``botorch`` requests raise if the "
                "optional stack is unavailable. ``tpe`` is dep-light and ships "
                "with Optuna core. ``gp`` is Optuna's native GP-EI with "
                "inequality constraints (Optuna 4.2+) but requires ``torch``."
            ),
        ),
        CLIParameter(
            name=("--optuna-sampler",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    optuna_acquisition: Annotated[
        Literal["logei", "qlogei", "qnei", "qlognei", "qehvi", "qnehvi", "qlognehvi"]
        | None,
        Field(
            default=None,
            description=(
                "Acquisition function override for the Optuna BoTorch sampler. "
                "Only consulted when --search-planner=optuna AND "
                "--optuna-sampler=botorch; rejected otherwise. ``None`` (default) "
                "lets Optuna pick (single-objective unconstrained -> LogEI per "
                "Optuna v4.x). ``logei``/``qlogei`` make that explicit. ``qnei`` "
                "selects plain noisy EI (Letham 2017). ``qlognei`` selects "
                "qLogNoisyExpectedImprovement (Ament 2023, "
                "https://arxiv.org/abs/2310.20708) -- BoTorch's strongly "
                "recommended modern noisy-EI default; requires ``botorch>=0.10``. "
                "Multi-objective variants (``qehvi``/``qnehvi``/``qlognehvi``) "
                "are accepted when ``objectives`` has length > 1; the planner "
                "rejects them on single-objective configs."
            ),
        ),
        CLIParameter(
            name=("--optuna-acquisition",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    optuna_terminator: Annotated[
        Literal["regret", "emmr", "none"] | None,
        Field(
            default=None,
            description=(
                "Optional posterior-regret stopping rule layered on top of the "
                "three-signal convergence check. Only consulted when "
                "--search-planner=optuna. ``regret`` selects Optuna's "
                "``RegretBoundEvaluator`` (Makarova et al. 2022, "
                "https://proceedings.mlr.press/v188/makarova22a.html). ``emmr`` "
                "selects ``EMMREvaluator`` (Ishibashi et al. 2023, "
                "https://proceedings.mlr.press/v206/ishibashi23a.html). Both are "
                "in the same family as Wilson 2024's PRB stopping rule and ship "
                "in Optuna core (no extra dep). ``none`` (default) disables; "
                "convergence is then driven by --search-max-iterations / "
                "--improvement-patience / --plateau-cv only."
            ),
        ),
        CLIParameter(
            name=("--optuna-terminator",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_percentile_pooling: Annotated[
        Literal["mean", "pooled"] | None,
        Field(
            default=None,
            description=(
                "Percentile aggregation strategy when --search-stat is a "
                "percentile (p50/p90/p95/p99). ``mean`` (default) computes the "
                "BO objective as the arithmetic mean of per-trial percentiles "
                "across --num-profile-runs trials. ``pooled`` walks each trial's "
                "per-request profile_export.jsonl, accumulates raw samples, and "
                "computes ``np.percentile`` over the pooled bag -- exposing "
                "more tail mass than mean-of-percentiles (correct for SLO "
                "claims; same argmax for ranking on monotone problems). "
                "``pooled`` requires --export-level records; if the JSONL is "
                "missing the planner falls back to mean with a one-time "
                "warning. Rejected when --search-stat is ``avg``."
            ),
        ),
        CLIParameter(
            name=("--search-percentile-pooling",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    bo_constraint_mode: Annotated[
        Literal["penalty", "eic"] | None,
        Field(
            default=None,
            description=(
                "Deprecated and ignored. The bayesian preset and the optuna "
                "expert mode both use Optuna's native ``constraints_func`` "
                "(Letham et al. 2019, arXiv:1706.07094), which subsumes both "
                "the soft-penalty and EIC formulations. Accepted for "
                "backwards compatibility but has no effect: the value flows "
                "through ``_converter_optionals._SWEEP_OPTIONAL_FIELDS`` to "
                "``AdaptiveSearchSweep.constraint_mode`` (see "
                "``aiperf.config.sweep.config``), and that field is not read "
                "by any planner."
            ),
        ),
        CLIParameter(
            name=("--bo-constraint-mode",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    sweep_variants: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Repeatable: each occurrence describes one sweep variation. "
                "Format: '[name:] key=value, key=value, ...'. Keys are CLI flag "
                "names with the leading '--' stripped, in either kebab-case or "
                "snake_case (isl, osl, concurrency, request-rate / request_rate, "
                "request-count / request_count, benchmark-duration / "
                "benchmark_duration, ...). Multi-occurrence emits a ScenarioSweep. "
                "Mutually exclusive with magic-list flags, --search-recipe, and "
                "YAML-declared sweeps. Single-occurrence is rejected -- use the "
                "standalone --isl / --osl / --concurrency flags for a one-off."
            ),
        ),
        CLIParameter(
            name=("--variant", "--sweep-variant"),
            group=Groups.MULTI_RUN,
        ),
    ]

    search_sla: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "SLA filter to attach to the adaptive-search or grid path. "
                "Format: 'metric_tag:stat:op:threshold'. Stat in "
                "{avg, p50, p90, p95, p99}; op in {lt, le, gt, ge}; threshold is "
                "a float. Repeatable. Example: --search-sla "
                "'time_to_first_token:p95:lt:200' --search-sla "
                "'request_error_rate:p99:lt:0.05'. Composes with recipe-named "
                "SLA flags (--ttft-sla-ms etc.); the final filter list is "
                "recipe filters first, then --search-sla filters in CLI order."
            ),
        ),
        CLIParameter(
            name=("--search-sla",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_recipe: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Named search-recipe preset that expands to an adaptive-search or "
                "sweep block. Mutually exclusive with explicit --search-* flags. "
                "Recipes are registered under the search_recipe plugin category. "
                "Example: --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200."
            ),
        ),
        CLIParameter(
            name=("--search-recipe",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    ttft_sla_ms: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description=(
                "Time-to-first-token SLA threshold in milliseconds. Required by "
                "TTFT-SLA recipes (e.g. max-throughput-ttft-sla); ignored otherwise. "
                "Must be > 0 — a 0 or negative threshold yields an unsatisfiable filter."
            ),
        ),
        CLIParameter(
            name=("--ttft-sla-ms",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    isl_osl_pairs: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Paired ISL/OSL workload shapes for the pareto-sweep recipe, "
                "e.g. '128/128,512/256,2048/512'. Each pair is '<isl>/<osl>' with "
                "positive ints; pairs are comma-separated and whitespace-tolerant. "
                "Recipe-only flag; ignored unless --search-recipe pareto-sweep is set."
            ),
        ),
        CLIParameter(
            name=("--isl-osl-pairs",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    itl_sla_ms: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description=(
                "Inter-token-latency SLA threshold in milliseconds. Required by "
                "ITL-SLA recipes (e.g. max-throughput-itl-sla); ignored otherwise. "
                "Must be > 0 — a 0 or negative threshold yields an unsatisfiable filter."
            ),
        ),
        CLIParameter(
            name=("--itl-sla-ms",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    tpot_sla_ms: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description=(
                "Time-per-output-token SLA threshold in milliseconds. Maps to the "
                "`inter_token_latency` metric tag (TPOT and ITL are equivalent in "
                "this codebase). Consumed by the max-concurrency-under-sla and "
                "max-goodput-under-slo recipes; ignored otherwise. Streaming "
                "required."
            ),
        ),
        CLIParameter(
            name=("--tpot-sla-ms",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    e2e_sla_ms: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description=(
                "End-to-end request-latency SLA threshold in milliseconds (p99). "
                "Maps to the `request_latency` metric tag. Consumed by the "
                "max-concurrency-under-sla and max-goodput-under-slo recipes; "
                "ignored otherwise. Available without streaming."
            ),
        ),
        CLIParameter(
            name=("--e2e-sla-ms",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    error_rate_sla: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            lt=1,
            description=(
                "Maximum acceptable request error rate as a fraction in (0, 1) "
                "(e.g. 0.05 = 5%). Maps to the `request_error_rate` metric tag "
                "(p99). Consumed by the max-concurrency-under-sla recipe; ignored "
                "otherwise. Available without streaming."
            ),
        ),
        CLIParameter(
            name=("--error-rate-sla",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    slo_attainment_fraction: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            le=1,
            description=(
                "Minimum fraction of requests that must satisfy ALL configured "
                "per-request SLOs (TTFT/TPOT/E2E) for a configuration to be "
                "considered feasible by the goodput recipe. Bounded in (0, 1]. "
                "Default 0.95 matches DistServe's canonical attainment-fraction "
                "convention (https://arxiv.org/pdf/2401.09670). Consumed by the "
                "max-goodput-under-slo recipe; ignored otherwise."
            ),
        ),
        CLIParameter(
            name=("--slo-attainment-fraction",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_style: Annotated[
        Literal["smooth_isotonic", "monotonic", "bo", "optuna", "grid"] | None,
        Field(
            default=None,
            description=(
                "Search strategy for the max-concurrency-under-sla recipe. "
                "'smooth_isotonic' (default) runs PAVA + PCHIP smooth-isotonic "
                "regression-based 1D SLA-saturation search. 'monotonic' runs a "
                "1D binary-search via the MonotonicSLASearchPlanner "
                "(~10-20 iterations). 'bo' runs penalty Bayesian Optimization "
                "(~30 iterations). 'optuna' runs the same penalty-BO formulation "
                "via the OptunaSearchPlanner (TPE/GP/BoTorch samplers; BoTorch "
                "requires the optional botorch extra). 'grid' runs a log-spaced 8-step sweep + "
                "sla_breach_knee post-process. Recipe-only flag; ignored unless "
                "--search-recipe max-concurrency-under-sla is set."
            ),
        ),
        CLIParameter(
            name=("--search-style",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    degradation_threshold: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            lt=1,
            description=(
                "Relative latency degradation threshold for the concurrency-ramp "
                "recipe (e.g. 0.20 = 20%). The recipe's post-process handler "
                "reports the first concurrency where p99 latency exceeds "
                "baseline * (1 + threshold). Recipe-only flag; ignored unless "
                "--search-recipe concurrency-ramp is set."
            ),
        ),
        CLIParameter(
            name=("--degradation-threshold",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    degradation_metric_tag: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "ConcurrencyRamp post-process: metric tag for knee detection "
                "(default: request_latency). Use, e.g., 'time_to_first_token' "
                "to detect the knee on TTFT instead of end-to-end request "
                "latency. Recipe-only flag; ignored unless --search-recipe "
                "concurrency-ramp is set."
            ),
        ),
        CLIParameter(
            name=("--degradation-metric-tag",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    degradation_stat: Annotated[
        Literal["avg", "p50", "p90", "p95", "p99"] | None,
        Field(
            default=None,
            description=(
                "ConcurrencyRamp post-process: statistic for knee detection "
                "(default: p99). Recipe-only flag; ignored unless "
                "--search-recipe concurrency-ramp is set."
            ),
        ),
        CLIParameter(
            name=("--degradation-stat",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    isl_min: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Minimum input-sequence-length for the prefill-ttft-curve recipe "
                "(default 256 when omitted). The recipe sweeps ISL on a log scale "
                "from --isl-min to --isl-max. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--isl-min",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    isl_max: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Maximum input-sequence-length for the prefill-ttft-curve recipe "
                "(default 32768 when omitted). The recipe sweeps ISL on a log scale "
                "from --isl-min to --isl-max. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--isl-max",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    isl_steps: Annotated[
        int | None,
        Field(
            default=None,
            ge=2,
            description=(
                "Number of log-spaced steps for the prefill-ttft-curve recipe's "
                "ISL grid (default 8 when omitted). Must be >= 2 — a single-point "
                "ramp degenerates and post-process can't compute a baseline. "
                "Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--isl-steps",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    concurrency_min: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Lower bound for the concurrency sweep axis used by concurrency-ramp "
                "and decode-itl-curve recipes (defaults: 1 for concurrency-ramp, 1 "
                "for decode-itl-curve). Must be < --concurrency-max. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--concurrency-min",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    concurrency_max: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Upper bound for the concurrency sweep axis used by concurrency-ramp "
                "and decode-itl-curve recipes (defaults: 1000 for concurrency-ramp, "
                "200 for decode-itl-curve). Must be > --concurrency-min. Recipe-only "
                "flag."
            ),
        ),
        CLIParameter(
            name=("--concurrency-max",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    concurrency_steps: Annotated[
        int | None,
        Field(
            default=None,
            ge=2,
            description=(
                "Number of log-spaced steps for the concurrency sweep axis used "
                "by concurrency-ramp (default 8) and decode-itl-curve (default 6). "
                "Must be >= 2. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--concurrency-steps",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    osl_min: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Minimum output-sequence-length for the decode-itl-curve recipe's "
                "OSL grid (default 64 when omitted). The recipe sweeps OSL on a "
                "log scale from --osl-min to --osl-max. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--osl-min",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    osl_max: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Maximum output-sequence-length for the decode-itl-curve recipe's "
                "OSL grid (default 1024 when omitted). The recipe sweeps OSL on a "
                "log scale from --osl-min to --osl-max. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--osl-max",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    osl_steps: Annotated[
        int | None,
        Field(
            default=None,
            ge=2,
            description=(
                "Number of log-spaced steps for the decode-itl-curve recipe's OSL "
                "grid (default 4 when omitted). Must be >= 2. Recipe-only flag."
            ),
        ),
        CLIParameter(
            name=("--osl-steps",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    ##############################################################################
    # Accuracy
    ##############################################################################
    accuracy_benchmark: Annotated[
        AccuracyBenchmarkType | None,
        Field(
            description="Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). "
            "When set, enables accuracy benchmarking mode alongside performance profiling.",
        ),
        CLIParameter(
            name=("--accuracy-benchmark",),
            group=Groups.ACCURACY,
        ),
    ] = None

    accuracy_tasks: Annotated[
        list[str] | None,
        BeforeValidator(parse_str_or_list),
        Field(
            description="Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). Accepts comma-separated values "
            "(e.g. abstract_algebra,anatomy) or repeated flags. If not set, all tasks are included.",
        ),
        CLIParameter(
            name=("--accuracy-tasks",),
            group=Groups.ACCURACY,
        ),
    ] = None

    accuracy_n_shots: Annotated[
        int | None,
        Field(
            ge=0,
            le=32,
            description="Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation, None uses the benchmark default (e.g. MMLU=5). Maximum 32.",
        ),
        CLIParameter(
            name=("--accuracy-n-shots",),
            group=Groups.ACCURACY,
        ),
    ] = None

    accuracy_enable_cot: Annotated[
        bool | None,
        Field(
            description="Enable chain-of-thought prompting for accuracy evaluation. "
            "Adds reasoning instructions to the prompt. Defaults to the benchmark's "
            "``default_enable_cot`` metadata when unset (e.g. AIME defaults to True).",
        ),
        CLIParameter(
            name=("--accuracy-enable-cot",),
            group=Groups.ACCURACY,
        ),
    ] = None

    accuracy_grader: Annotated[
        AccuracyGraderType | None,
        Field(
            description="Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution). "
            "If not set, uses the benchmark's default grader.",
        ),
        CLIParameter(
            name=("--accuracy-grader",),
            group=Groups.ACCURACY,
        ),
    ] = None

    accuracy_system_prompt: Annotated[
        str | None,
        Field(
            description="Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt.",
        ),
        CLIParameter(
            name=("--accuracy-system-prompt",),
            group=Groups.ACCURACY,
        ),
    ] = None

    accuracy_verbose: Annotated[
        bool,
        Field(
            description="Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details.",
        ),
        CLIParameter(
            name=("--accuracy-verbose",),
            group=Groups.ACCURACY,
        ),
    ] = False

    ##############################################################################
    # Service
    ##############################################################################
    log_level: Annotated[
        AIPerfLogLevel,
        Field(
            description="Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. "
            "Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.",
        ),
        CLIParameter(
            name=("--log-level"),
            group=Groups.SERVICE,
        ),
    ] = ServiceDefaults.LOG_LEVEL

    verbose: Annotated[
        bool,
        Field(
            description="Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. "
            "Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.",
        ),
        CLIParameter(
            name=("--verbose", "-v"),
            group=Groups.SERVICE,
        ),
    ] = ServiceDefaults.VERBOSE

    extra_verbose: Annotated[
        bool,
        Field(
            description="Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, "
            "internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.",
        ),
        CLIParameter(
            name=("--extra-verbose", "-vv"),
            group=Groups.SERVICE,
        ),
    ] = ServiceDefaults.EXTRA_VERBOSE

    record_processor_service_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of `RecordProcessor` services to spawn for parallel metric computation. "
            "Higher request rates require more processors to keep up with incoming records. "
            "If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).",
        ),
        CLIParameter(
            name=("--record-processor-service-count", "--record-processors"),
            group=Groups.SERVICE,
        ),
    ] = ServiceDefaults.RECORD_PROCESSOR_SERVICE_COUNT

    api_port: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=65535,
            description="AIPerf API port (enables HTTP + WebSocket endpoints)",
        ),
        CLIParameter(
            name="--api-port",
            group=Groups.SERVICE,
        ),
    ] = None

    api_host: Annotated[
        str | None,
        Field(
            default=None,
            description="AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set)",
        ),
        CLIParameter(
            name="--api-host",
            group=Groups.SERVICE,
        ),
    ] = None

    ##############################################################################
    # Workers
    ##############################################################################
    workers_max: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Maximum number of workers to create. If not specified, the number of"
                " workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, "
                " with a default max cap of 32. Any value provided will still be capped by"
                " the concurrency value (if specified), but not by the max cap."
            ),
        ),
        CLIParameter(
            name=("--workers-max", "--max-workers"),
            group=Groups.WORKERS,
        ),
    ] = None

    ##############################################################################
    # ZMQ Communication
    ##############################################################################
    zmq_tcp_host: Annotated[
        str,
        Field(
            description=(
                "Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for "
                "single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication "
                "(message bus, dataset manager, workers) uses this host for TCP sockets."
            ),
        ),
        CLIParameter(
            name=("--zmq-host"),
            group=Groups.ZMQ_COMMUNICATION,
        ),
    ] = "127.0.0.1"

    zmq_ipc_path: Annotated[
        Path | None,
        Field(
            description=(
                "Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, "
                "AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory "
                "if not specified. Only applicable when using IPC communication backend."
            ),
        ),
        CLIParameter(
            name=("--zmq-ipc-path"),
            group=Groups.ZMQ_COMMUNICATION,
        ),
    ] = None

    zmq_dual_bind: Annotated[
        bool,
        Field(
            description=(
                "Select the ZMQ dual-bind communication backend (IPC + TCP). All dual-bind knobs are cluster-managed; "
                "this flag only selects the discriminator and the converter routes downstream to the default."
            ),
        ),
        CLIParameter(
            name=("--zmq-dual-bind",),
            group=Groups.ZMQ_COMMUNICATION,
        ),
    ] = False

    ##############################################################################
    # Miscellaneous / Internal
    ##############################################################################
    # Internal computed fields populated by the converter, not by validators.
    # Kept as PrivateAttr-style underscore fields so the converter can stash the
    # parsed --gpu-telemetry breakdown (mode/collector/URLs/metrics file) for
    # downstream readers without re-parsing the raw list.
    _gpu_telemetry_mode: GPUTelemetryMode = GPUTelemetryMode.SUMMARY

    _gpu_telemetry_collector_type: GPUTelemetryCollectorType = (
        GPUTelemetryCollectorType.DCGM
    )

    _gpu_telemetry_urls: list[str] = []

    _gpu_telemetry_metrics_file: Path | None = None

    _server_metrics_urls: list[str] = []
