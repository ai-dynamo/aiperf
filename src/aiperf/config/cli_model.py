# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Explicit CLI model for cyclopts.

Each field uses ``annotated_type(type, flags, group, description, **kwargs)``
which returns an ``Annotated[type, Field(...), CLIParameter(...)]`` consumed
directly by cyclopts. The ``build_aiperf_config()`` converter in
``cli_converter.py`` translates a ``CLIModel`` instance into an ``AIPerfConfig``.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from aiperf.common.enums import (
    AIPerfLogLevel,
    AudioFormat,
    ConnectionReuseStrategy,
    DatasetFormat,
    ExportLevel,
    ImageFormat,
    ModelSelectionStrategy,
    ServerMetricsFormat,
    VideoFormat,
    VideoSynthType,
)
from aiperf.config.cli_parameter import (
    CLIParameter,
    DisableCLI,
    Groups,
    annotated_type,
)
from aiperf.config.parsing import (
    parse_file,
    parse_str_as_numeric_dict,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list,
    parse_str_or_list_of_positive_values,
    validate_sequence_distribution,
)
from aiperf.plugin.enums import (
    AccuracyBenchmarkType,
    AccuracyGraderType,
    ArrivalPattern,
    DatasetSamplingStrategy,
    EndpointType,
    PublicDatasetType,
    TransportType,
    UIType,
    URLSelectionStrategy,
)

__all__ = ["CLIModel", "CLIParameter", "DisableCLI"]


class CLIModel(BaseModel):
    """Flat Pydantic model consumed by cyclopts for CLI parsing.

    Every field maps 1:1 to a CLI flag. The converter (``cli_converter``)
    translates a ``CLIModel`` instance into an ``AIPerfConfig``.
    """

    model_config = ConfigDict(extra="forbid")

    # =========================================================================
    # ENDPOINT
    # =========================================================================

    model_names: annotated_type(
        list[str],
        ("--model-names", "--model", "-m"),
        Groups.ENDPOINT,
        "List of model configurations. At least one model required.",
        validators=[parse_str_or_list],
    )
    model_selection_strategy: annotated_type(
        ModelSelectionStrategy,
        "--model-selection-strategy",
        Groups.ENDPOINT,
        (
            "Strategy for selecting models when multiple are configured. "
            "round_robin cycles through models, random selects randomly, "
            "weighted uses configured weights, modality_aware routes by input type."
        ),
        default=ModelSelectionStrategy.ROUND_ROBIN,
    )
    urls: annotated_type(
        list[str],
        ("--url", "-u"),
        Groups.ENDPOINT,
        (
            "List of server URLs to benchmark. "
            "Requests distributed according to url_strategy. "
            "Example: ['http://localhost:8000/v1/chat/completions']"
        ),
        default=["localhost:8000"],
        validators=[parse_str_or_list],
        consume_multiple=True,
    )
    url_selection_strategy: annotated_type(
        URLSelectionStrategy,
        "--url-strategy",
        Groups.ENDPOINT,
        (
            "Strategy for distributing requests across multiple URLs. "
            "round_robin cycles through URLs in order."
        ),
        default=URLSelectionStrategy.ROUND_ROBIN,
    )
    endpoint_type: annotated_type(
        EndpointType,
        "--endpoint-type",
        Groups.ENDPOINT,
        (
            "API endpoint type determining request/response format. "
            "chat: OpenAI chat completions, completions: OpenAI completions, "
            "embeddings: vector embeddings, rankings: reranking, "
            "template: custom format."
        ),
        default=EndpointType.CHAT,
    )
    streaming: annotated_type(
        bool,
        "--streaming",
        Groups.ENDPOINT,
        (
            "Enable streaming (Server-Sent Events) responses. "
            "Required for accurate TTFT (time to first token) measurement. "
            "Server must support streaming for this to work."
        ),
        default=False,
    )
    custom_endpoint: annotated_type(
        str | None,
        ("--custom-endpoint", "--endpoint"),
        Groups.ENDPOINT,
        (
            "Override default endpoint path. "
            "Use for servers with non-standard API paths. "
            "Example: '/custom/v2/generate'"
        ),
    )
    api_key: annotated_type(
        str | None,
        "--api-key",
        Groups.ENDPOINT,
        (
            "API authentication key. "
            "Supports environment variable substitution: ${OPENAI_API_KEY}. "
            "Can also use ${VAR:default} syntax for defaults."
        ),
    )
    request_timeout_seconds: annotated_type(
        float,
        "--request-timeout-seconds",
        Groups.ENDPOINT,
        (
            "Request timeout in seconds (0 = no timeout). "
            "Requests exceeding this duration are marked as failed. "
            "Should exceed expected max response time."
        ),
        default=600.0,
    )
    ready_check_timeout: annotated_type(
        float,
        "--ready-check-timeout",
        Groups.ENDPOINT,
        (
            "Seconds to wait for endpoint readiness before benchmarking "
            "(0 = skip). Sends a real inference request to verify the model "
            "is loaded and can generate output."
        ),
        default=0.0,
    )
    transport_type: annotated_type(
        TransportType | None,
        ("--transport", "--transport-type"),
        Groups.ENDPOINT,
        (
            "HTTP transport protocol (http/https). "
            "Auto-detected from URL scheme if not specified. "
            "Explicit setting overrides auto-detection."
        ),
    )
    use_legacy_max_tokens: annotated_type(
        bool,
        "--use-legacy-max-tokens",
        Groups.ENDPOINT,
        (
            "Use 'max_tokens' field instead of 'max_completion_tokens'. "
            "Enable for compatibility with older OpenAI API versions."
        ),
        default=False,
    )
    use_server_token_count: annotated_type(
        bool,
        "--use-server-token-count",
        Groups.ENDPOINT,
        (
            "Use server-reported token counts from response usage field. "
            "When true, trusts usage.prompt_tokens and usage.completion_tokens. "
            "When false, counts tokens locally using configured tokenizer."
        ),
        default=False,
    )
    connection_reuse_strategy: annotated_type(
        ConnectionReuseStrategy,
        "--connection-reuse-strategy",
        Groups.ENDPOINT,
        (
            "HTTP connection management strategy. "
            "pooled: shared connection pool (fastest), "
            "never: new connection per request (includes TCP overhead), "
            "sticky_sessions: dedicated connection per session."
        ),
        default=ConnectionReuseStrategy.POOLED,
    )
    extra_inputs: annotated_type(
        list | None,
        "--extra-inputs",
        Groups.ENDPOINT,
        (
            "Additional fields to include in request body. "
            "Merged into every request. "
            "Common fields: temperature, top_p, top_k, stop."
        ),
        validators=[parse_str_or_dict_as_tuple_list],
        consume_multiple=True,
    )
    headers: annotated_type(
        list | None,
        ("--header", "-H"),
        Groups.ENDPOINT,
        (
            "Custom HTTP headers to include in all requests. "
            "Useful for authentication, tracing, or routing. "
            "Values support environment variable substitution."
        ),
        validators=[parse_str_or_dict_as_tuple_list],
        consume_multiple=True,
    )

    # =========================================================================
    # INPUT
    # =========================================================================

    input_file: annotated_type(
        Path | None,
        "--input-file",
        Groups.INPUT,
        (
            "Path to file or directory containing benchmark dataset. "
            "Can be absolute or relative. "
            "Supported formats depend on the format field: "
            "JSONL for single_turn/multi_turn, "
            "JSONL trace files for mooncake_trace, "
            "directories for random_pool."
        ),
        validators=[parse_file],
    )
    public_dataset: annotated_type(
        PublicDatasetType | None,
        "--public-dataset",
        Groups.INPUT,
        (
            "Pre-configured public dataset to download and use for benchmarking. "
            "AIPerf automatically downloads and parses these datasets."
        ),
    )
    hf_subset: annotated_type(
        str | None,
        "--hf-subset",
        Groups.INPUT,
        (
            "HuggingFace dataset subset/config name to override the plugin default "
            "(e.g. 'sharegpt4o'). Only applies when --public-dataset uses a "
            "HuggingFace-backed loader. Takes priority over the subset defined in "
            "the plugin registry."
        ),
    )
    custom_dataset_type: annotated_type(
        DatasetFormat,
        "--custom-dataset-type",
        Groups.INPUT,
        (
            "Dataset file format determining parsing logic and expected file structure. "
            "single_turn: JSONL with single prompt-response exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace: timestamped trace files for replay. "
            "random_pool: directory of reusable prompts."
        ),
        default=DatasetFormat.SINGLE_TURN,
    )
    dataset_sampling_strategy: annotated_type(
        DatasetSamplingStrategy,
        "--dataset-sampling-strategy",
        Groups.INPUT,
        (
            "Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion."
        ),
        default=DatasetSamplingStrategy.SEQUENTIAL,
    )
    random_seed: annotated_type(
        int | None,
        "--random-seed",
        Groups.INPUT,
        (
            "Global random seed for reproducibility. "
            "Can be overridden per-dataset. "
            "If not set, uses system entropy."
        ),
    )

    # =========================================================================
    # FIXED SCHEDULE
    # =========================================================================

    fixed_schedule: annotated_type(
        bool,
        "--fixed-schedule",
        Groups.FIXED_SCHEDULE,
        (
            "Run requests according to timestamps specified in the input dataset. "
            "When enabled, AIPerf replays the exact timing pattern from the dataset. "
            "This mode is automatically enabled for mooncake_trace datasets."
        ),
        default=False,
        negative=None,
    )
    fixed_schedule_auto_offset: annotated_type(
        bool,
        "--fixed-schedule-auto-offset",
        Groups.FIXED_SCHEDULE,
        (
            "Normalize trace timestamps to start at 0. "
            "Subtracts minimum timestamp from all entries."
        ),
        default=True,
    )
    fixed_schedule_start_offset: annotated_type(
        int | None,
        "--fixed-schedule-start-offset",
        Groups.FIXED_SCHEDULE,
        "Filter out trace requests before this timestamp in ms (must be >= 0).",
    )
    fixed_schedule_end_offset: annotated_type(
        int | None,
        "--fixed-schedule-end-offset",
        Groups.FIXED_SCHEDULE,
        "Filter out trace requests after this timestamp in ms (must be >= 0).",
    )

    # =========================================================================
    # GOODPUT
    # =========================================================================

    goodput: annotated_type(
        Any | None,
        "--goodput",
        Groups.GOODPUT,
        (
            "SLO (Service Level Objectives) configuration as a generic dict. "
            "Maps metric names to threshold values. "
            "A request is counted as good only if it meets ALL specified thresholds."
        ),
        validators=[parse_str_as_numeric_dict],
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    artifact_directory: annotated_type(
        Path,
        ("--output-artifact-dir", "--artifact-dir"),
        Groups.OUTPUT,
        ("Output directory for all benchmark artifacts. Created if it doesn't exist."),
        default=Path("./artifacts"),
    )
    profile_export_prefix: annotated_type(
        Path | None,
        ("--profile-export-prefix", "--profile-export-file"),
        Groups.OUTPUT,
        (
            "Filename prefix for all exported files. "
            "Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'."
        ),
    )
    export_level: annotated_type(
        ExportLevel,
        ("--export-level", "--profile-export-level"),
        Groups.OUTPUT,
        (
            "Controls which output files are generated. "
            "summary: Only aggregate metrics files. "
            "records: Includes per-request metrics. "
            "raw: Includes raw request/response data."
        ),
        default=ExportLevel.RECORDS,
    )
    slice_duration: annotated_type(
        float | None,
        "--slice-duration",
        Groups.OUTPUT,
        (
            "Time slice duration in seconds for trend analysis (must be > 0). "
            "Divides benchmark into windows for per-window statistics. "
            "Supports: 30, '30s', '5m', '2h'."
        ),
    )

    # =========================================================================
    # HTTP TRACE
    # =========================================================================

    export_http_trace: annotated_type(
        bool,
        "--export-http-trace",
        Groups.HTTP_TRACE,
        "Export HTTP trace data for debugging.",
        default=False,
    )
    export_per_chunk_data: annotated_type(
        bool,
        "--export-per-chunk-data",
        Groups.OUTPUT,
        (
            "Include per-chunk list data (e.g., inter_chunk_latency arrays) in "
            "per-record exports. These arrays contain one timing value per SSE "
            "chunk and can be very large for long responses."
        ),
        default=False,
    )
    show_trace_timing: annotated_type(
        bool,
        "--show-trace-timing",
        Groups.HTTP_TRACE,
        (
            "Display HTTP trace timing metrics in console output. "
            "Shows detailed timing breakdown: blocked, DNS, connecting, sending, "
            "waiting (TTFB), receiving, and total duration."
        ),
        default=False,
    )

    # =========================================================================
    # TOKENIZER
    # =========================================================================

    tokenizer_name: annotated_type(
        str | None,
        "--tokenizer",
        Groups.TOKENIZER,
        (
            "HuggingFace tokenizer identifier or local filesystem path. "
            "Should match the model's tokenizer for accurate token counts. "
            "Example: 'meta-llama/Llama-3.1-8B-Instruct'"
        ),
    )
    tokenizer_revision: annotated_type(
        str,
        "--tokenizer-revision",
        Groups.TOKENIZER,
        (
            "Model revision to use: branch name, tag, or commit hash. "
            "Use for version pinning to ensure reproducibility."
        ),
        default="main",
    )
    tokenizer_trust_remote_code: annotated_type(
        bool,
        "--tokenizer-trust-remote-code",
        Groups.TOKENIZER,
        (
            "Allow execution of custom tokenizer code from the repository. "
            "Required for some models but poses security risk. "
            "Only enable for trusted sources."
        ),
        default=False,
    )

    # =========================================================================
    # LOAD GENERATOR
    # =========================================================================

    benchmark_duration: annotated_type(
        float | None,
        "--benchmark-duration",
        Groups.LOAD_GENERATOR,
        "Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.",
    )
    benchmark_grace_period: annotated_type(
        float | None,
        "--benchmark-grace-period",
        Groups.LOAD_GENERATOR,
        (
            "Seconds to wait for in-flight requests after duration expires (must be >= 0). "
            "Requires 'duration' to be set. Supports: 30, '30s', '2m'."
        ),
    )
    concurrency: annotated_type(
        int | None,
        "--concurrency",
        Groups.LOAD_GENERATOR,
        (
            "Max concurrent in-flight requests (must be >= 1). "
            "Primary control for concurrency phases."
        ),
        default=1,
    )
    prefill_concurrency: annotated_type(
        int | None,
        "--prefill-concurrency",
        Groups.LOAD_GENERATOR,
        (
            "Max concurrent requests in prefill stage (must be >= 1). "
            "Limits requests before first token received."
        ),
    )
    request_rate: annotated_type(
        float | None,
        "--request-rate",
        Groups.LOAD_GENERATOR,
        "Target request rate in requests per second (must be > 0).",
    )
    arrival_pattern: annotated_type(
        ArrivalPattern,
        ("--arrival-pattern", "--request-rate-mode"),
        Groups.LOAD_GENERATOR,
        (
            "Sets the arrival pattern for the load generated by AIPerf. "
            "Valid values: constant, poisson, gamma.\n"
            "constant: Generate requests at a fixed rate.\n"
            "poisson: Generate requests using a poisson distribution.\n"
            "gamma: Generate requests using a gamma distribution with tunable smoothness."
        ),
        default=ArrivalPattern.POISSON,
    )
    arrival_smoothness: annotated_type(
        float | None,
        ("--arrival-smoothness", "--vllm-burstiness"),
        Groups.LOAD_GENERATOR,
        (
            "Gamma distribution shape parameter (must be > 0). "
            "1.0 = Poisson, <1 = bursty, >1 = regular."
        ),
    )
    request_count: annotated_type(
        int | None,
        ("--request-count", "--num-requests"),
        Groups.LOAD_GENERATOR,
        "Stop after this many requests sent (must be >= 1).",
    )
    concurrency_ramp_duration: annotated_type(
        float | None,
        "--concurrency-ramp-duration",
        Groups.LOAD_GENERATOR,
        "Seconds to ramp from start to target value.",
    )
    prefill_concurrency_ramp_duration: annotated_type(
        float | None,
        "--prefill-concurrency-ramp-duration",
        Groups.LOAD_GENERATOR,
        "Seconds to ramp from start to target value.",
    )
    request_rate_ramp_duration: annotated_type(
        float | None,
        "--request-rate-ramp-duration",
        Groups.WARMUP,
        "Seconds to ramp from start to target value.",
    )

    # =========================================================================
    # WARMUP
    # =========================================================================

    warmup_request_count: annotated_type(
        int | None,
        ("--warmup-request-count", "--num-warmup-requests"),
        Groups.WARMUP,
        (
            "Warmup phase: Stop after this many requests sent (must be >= 1). "
            "If not set, uses the --request-count value."
        ),
    )
    warmup_duration: annotated_type(
        float | None,
        "--warmup-duration",
        Groups.WARMUP,
        (
            "Warmup phase: Stop after this time elapsed (must be > 0). "
            "Supports: 300, '5m', '2h'. "
            "If not set, uses the --benchmark-duration value."
        ),
    )
    warmup_num_sessions: annotated_type(
        int | None,
        "--num-warmup-sessions",
        Groups.WARMUP,
        (
            "Warmup phase: Stop after this many sessions completed (must be >= 1). "
            "If not set, uses the --conversation-num value."
        ),
    )
    warmup_concurrency: annotated_type(
        int | None,
        "--warmup-concurrency",
        Groups.WARMUP,
        (
            "Warmup phase: Max concurrent in-flight requests (must be >= 1). "
            "Primary control for concurrency phases. "
            "If not set, uses the --concurrency value."
        ),
        default=1,
    )
    warmup_prefill_concurrency: annotated_type(
        int | None,
        "--warmup-prefill-concurrency",
        Groups.WARMUP,
        (
            "Warmup phase: Max concurrent requests in prefill stage (must be >= 1). "
            "Limits requests before first token received. "
            "If not set, uses the --prefill-concurrency value."
        ),
    )
    warmup_request_rate: annotated_type(
        float | None,
        "--warmup-request-rate",
        Groups.WARMUP,
        (
            "Warmup phase: Target request rate in requests per second (must be > 0). "
            "If not set, uses the --request-rate value."
        ),
    )
    warmup_arrival_pattern: annotated_type(
        Literal[
            "concurrency",
            "poisson",
            "gamma",
            "constant",
            "user_centric",
            "fixed_schedule",
        ]
        | None,
        "--warmup-arrival-pattern",
        Groups.WARMUP,
        (
            "Warmup phase: Concurrency-controlled immediate dispatch. "
            "If not set, uses the --arrival-pattern value."
        ),
        show_choices=False,
    )
    warmup_grace_period: annotated_type(
        float | None,
        "--warmup-grace-period",
        Groups.WARMUP,
        (
            "Warmup phase: Seconds to wait for in-flight requests after duration expires "
            "(must be >= 0). Requires 'duration' to be set. Supports: 30, '30s', '2m'. "
            "If not set, uses the --benchmark-grace-period value."
        ),
    )
    warmup_concurrency_ramp_duration: annotated_type(
        float | None,
        "--warmup-concurrency-ramp-duration",
        Groups.WARMUP,
        (
            "Warmup phase: Seconds to ramp from start to target value. "
            "If not set, uses the --concurrency-ramp-duration value."
        ),
    )
    warmup_prefill_concurrency_ramp_duration: annotated_type(
        float | None,
        "--warmup-prefill-concurrency-ramp-duration",
        Groups.WARMUP,
        (
            "Warmup phase: Seconds to ramp from start to target value. "
            "If not set, uses the --prefill-concurrency-ramp-duration value."
        ),
    )
    warmup_request_rate_ramp_duration: annotated_type(
        float | None,
        "--warmup-request-rate-ramp-duration",
        Groups.WARMUP,
        (
            "Warmup phase: Seconds to ramp from start to target value. "
            "If not set, uses the --request-rate-ramp-duration value."
        ),
    )

    # =========================================================================
    # USER-CENTRIC RATE
    # =========================================================================

    user_centric_rate: annotated_type(
        float | None,
        "--user-centric-rate",
        Groups.USER_CENTRIC,
        (
            "Enable user-centric rate limiting mode with the specified request rate (QPS). "
            "Each user has a gap = num_users / qps between turns. "
            "Designed for KV cache benchmarking with realistic multi-user patterns. "
            "Requires --num-users to be set."
        ),
    )
    num_users: annotated_type(
        int | None,
        "--num-users",
        Groups.USER_CENTRIC,
        (
            "Number of simulated concurrent users (must be >= 1). "
            "Requests distributed across users to achieve global rate."
        ),
    )

    # =========================================================================
    # REQUEST CANCELLATION
    # =========================================================================

    request_cancellation_rate: annotated_type(
        float | None,
        "--request-cancellation-rate",
        Groups.REQUEST_CANCELLATION,
        (
            "Percentage (0-100) of requests to cancel for testing cancellation handling. "
            "Cancelled requests are sent normally but aborted after "
            "--request-cancellation-delay seconds."
        ),
    )
    request_cancellation_delay: annotated_type(
        float,
        "--request-cancellation-delay",
        Groups.REQUEST_CANCELLATION,
        (
            "Seconds to wait after the request is fully sent before cancelling. "
            "A delay of 0 means send the full request, then immediately disconnect. "
            "Requires --request-cancellation-rate to be set."
        ),
        default=0.0,
    )

    # =========================================================================
    # CONVERSATION INPUT
    # =========================================================================

    turn_delay_mean: annotated_type(
        float,
        ("--conversation-turn-delay-mean", "--session-turn-delay-mean"),
        Groups.CONVERSATION_INPUT,
        (
            "Mean delay in milliseconds between consecutive turns within a multi-turn "
            "conversation. Simulates user think time. Set to 0 for back-to-back turns."
        ),
        default=0.0,
    )
    turn_delay_stddev: annotated_type(
        float,
        ("--conversation-turn-delay-stddev", "--session-turn-delay-stddev"),
        Groups.CONVERSATION_INPUT,
        "Standard deviation for turn delays in milliseconds.",
        default=0.0,
    )
    turn_delay_ratio: annotated_type(
        float,
        ("--conversation-turn-delay-ratio", "--session-delay-ratio"),
        Groups.CONVERSATION_INPUT,
        ("Multiplier for scaling all turn delays. Values < 1 speed up, > 1 slow down."),
        default=1.0,
    )
    num_turns_mean: annotated_type(
        int,
        ("--conversation-turn-mean", "--session-turns-mean"),
        Groups.CONVERSATION_INPUT,
        (
            "Mean number of request-response turns per conversation. "
            "Set to 1 for single-turn."
        ),
        default=1,
    )
    num_turns_stddev: annotated_type(
        int,
        ("--conversation-turn-stddev", "--session-turns-stddev"),
        Groups.CONVERSATION_INPUT,
        "Standard deviation for number of turns per conversation.",
        default=0,
    )
    num_sessions: annotated_type(
        int | None,
        ("--conversation-num", "--num-conversations", "--num-sessions"),
        Groups.CONVERSATION_INPUT,
        "Stop after this many sessions completed (must be >= 1).",
    )
    num_dataset_entries: annotated_type(
        int,
        ("--num-dataset-entries", "--num-prompts"),
        Groups.CONVERSATION_INPUT,
        (
            "Total number of unique entries to generate for the dataset. "
            "Each entry represents a unique prompt with sampled ISL/OSL. "
            "Entries are reused across conversations and turns according to "
            "the sampling strategy. Higher values provide more diversity."
        ),
        default=100,
    )

    # =========================================================================
    # ISL
    # =========================================================================

    isl_mean: annotated_type(
        int,
        ("--prompt-input-tokens-mean", "--synthetic-input-tokens-mean", "--isl"),
        Groups.ISL,
        "Mean value.",
        default=550,
    )
    isl_stddev: annotated_type(
        float,
        (
            "--prompt-input-tokens-stddev",
            "--synthetic-input-tokens-stddev",
            "--isl-stddev",
        ),
        Groups.ISL,
        "Standard deviation. 0 = deterministic.",
        default=0.0,
    )
    isl_block_size: annotated_type(
        int | None,
        (
            "--prompt-input-tokens-block-size",
            "--synthetic-input-tokens-block-size",
            "--isl-block-size",
        ),
        Groups.ISL,
        (
            "Token block size for hash-based prompt caching in mooncake_trace datasets. "
            "When hash_ids are provided in trace entries, prompts are divided into blocks "
            "of this size. Each hash_id maps to a cached block, enabling simulation of "
            "KV-cache sharing patterns from production workloads. "
            "Total prompt length = (num_hash_ids - 1) * block_size + final_block_size."
        ),
    )
    sequence_distribution: annotated_type(
        str | None,
        ("--seq-dist", "--sequence-distribution"),
        Groups.ISL,
        (
            "Distribution of (ISL, OSL) pairs with probabilities for mixed workload "
            "simulation. "
            "Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100)."
        ),
        validators=[validate_sequence_distribution],
    )

    # =========================================================================
    # OSL
    # =========================================================================

    osl_mean: annotated_type(
        int | None,
        ("--prompt-output-tokens-mean", "--output-tokens-mean", "--osl"),
        Groups.OSL,
        "Mean value.",
    )
    osl_stddev: annotated_type(
        float | None,
        ("--prompt-output-tokens-stddev", "--output-tokens-stddev", "--osl-stddev"),
        Groups.OSL,
        "Standard deviation. 0 = deterministic.",
        default=0.0,
    )

    # =========================================================================
    # PROMPT
    # =========================================================================

    prompt_batch_size: annotated_type(
        int,
        ("--prompt-batch-size", "--batch-size-text", "--batch-size", "-b"),
        Groups.PROMPT,
        (
            "Number of text inputs to include in each request for batch processing "
            "endpoints. Supported by embeddings and rankings endpoint types where models "
            "can process multiple inputs simultaneously. Set to 1 for single-input "
            "requests. Not applicable to chat or completions endpoints."
        ),
        default=1,
    )

    # =========================================================================
    # PREFIX PROMPT
    # =========================================================================

    num_prefix_prompts: annotated_type(
        int,
        (
            "--prompt-prefix-pool-size",
            "--prefix-prompt-pool-size",
            "--num-prefix-prompts",
        ),
        Groups.PREFIX_PROMPT,
        (
            "Number of distinct prefix prompts to generate for KV cache testing. "
            "Each prefix is prepended to user prompts, simulating cached context "
            "scenarios. Prefixes are randomly selected from pool per request. "
            "Mutually exclusive with shared_system_length/user_context_length."
        ),
        default=0,
    )
    prefix_prompt_length: annotated_type(
        int,
        ("--prompt-prefix-length", "--prefix-prompt-length"),
        Groups.PREFIX_PROMPT,
        (
            "Token length for each prefix prompt in the pool. "
            "Only used when pool_size is set. "
            "Note: due to prefix and user prompts being concatenated, "
            "the final prompt token count may be off by one. "
            "Mutually exclusive with shared_system_length/user_context_length."
        ),
        default=0,
    )
    shared_system_prompt_length: annotated_type(
        int | None,
        "--shared-system-prompt-length",
        Groups.PREFIX_PROMPT,
        (
            "Length of shared system prompt in tokens. "
            "This prompt is identical across all sessions and appears as a system message. "
            "First part of a two-part prefix structure with high cache hit rate expected. "
            "Mutually exclusive with pool_size/length."
        ),
    )
    user_context_prompt_length: annotated_type(
        int | None,
        "--user-context-prompt-length",
        Groups.PREFIX_PROMPT,
        (
            "Length of per-session user context prompt in tokens. "
            "Each dataset entry gets a unique user context prompt. "
            "Second part of two-part prefix structure with lower cache hit rate expected. "
            "Mutually exclusive with pool_size/length."
        ),
    )

    # =========================================================================
    # RANKINGS
    # =========================================================================

    passages_mean: annotated_type(
        int, "--rankings-passages-mean", Groups.RANKINGS, "Mean value.", default=10
    )
    passages_stddev: annotated_type(
        int,
        "--rankings-passages-stddev",
        Groups.RANKINGS,
        "Standard deviation. 0 = deterministic.",
        default=0,
    )
    passages_prompt_token_mean: annotated_type(
        int,
        "--rankings-passages-prompt-token-mean",
        Groups.RANKINGS,
        "Mean value.",
        default=128,
    )
    passages_prompt_token_stddev: annotated_type(
        int,
        "--rankings-passages-prompt-token-stddev",
        Groups.RANKINGS,
        "Standard deviation. 0 = deterministic.",
        default=0,
    )
    query_prompt_token_mean: annotated_type(
        int,
        "--rankings-query-prompt-token-mean",
        Groups.RANKINGS,
        "Mean value.",
        default=32,
    )
    query_prompt_token_stddev: annotated_type(
        int,
        "--rankings-query-prompt-token-stddev",
        Groups.RANKINGS,
        "Standard deviation. 0 = deterministic.",
        default=0,
    )

    # =========================================================================
    # SYNTHESIS
    # =========================================================================

    synthesis_speedup_ratio: annotated_type(
        float,
        "--synthesis-speedup-ratio",
        Groups.SYNTHESIS,
        (
            "Multiplier for timestamp scaling in synthesized traces. "
            "1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower."
        ),
        default=1.0,
    )
    synthesis_prefix_len_multiplier: annotated_type(
        float,
        "--synthesis-prefix-len-multiplier",
        Groups.SYNTHESIS,
        (
            "Multiplier for core prefix branch lengths in the radix tree. "
            "1.5 means prefix branches are 50%% longer."
        ),
        default=1.0,
    )
    synthesis_prefix_root_multiplier: annotated_type(
        int,
        "--synthesis-prefix-root-multiplier",
        Groups.SYNTHESIS,
        (
            "Number of independent radix trees to distribute traces across. "
            "Higher values increase prefix diversity."
        ),
        default=1,
    )
    synthesis_prompt_len_multiplier: annotated_type(
        float,
        "--synthesis-prompt-len-multiplier",
        Groups.SYNTHESIS,
        (
            "Multiplier for leaf path (unique prompt) lengths. "
            "2.0 means prompts are 2x longer."
        ),
        default=1.0,
    )
    synthesis_max_isl: annotated_type(
        int | None,
        "--synthesis-max-isl",
        Groups.SYNTHESIS,
        (
            "Maximum input sequence length filter. "
            "Traces with input_length > max_isl are skipped entirely."
        ),
    )
    synthesis_max_osl: annotated_type(
        int | None,
        "--synthesis-max-osl",
        Groups.SYNTHESIS,
        (
            "Maximum output sequence length cap. "
            "Traces with output_length > max_osl are capped to this value (not filtered)."
        ),
    )

    # =========================================================================
    # AUDIO INPUT
    # =========================================================================

    audio_length_mean: annotated_type(
        float,
        "--audio-length-mean",
        Groups.AUDIO_INPUT,
        "Mean value.",
        default=10.0,
    )
    audio_length_stddev: annotated_type(
        float,
        "--audio-length-stddev",
        Groups.AUDIO_INPUT,
        "Standard deviation. 0 = deterministic.",
        default=0.0,
    )
    audio_batch_size: annotated_type(
        int,
        ("--audio-batch-size", "--batch-size-audio"),
        Groups.AUDIO_INPUT,
        (
            "Number of audio inputs to include in each multimodal request. "
            "Supported with chat endpoint type for multimodal models. "
            "Set to 0 to disable audio inputs."
        ),
        default=0,
    )
    audio_format: annotated_type(
        AudioFormat,
        "--audio-format",
        Groups.AUDIO_INPUT,
        (
            "File format for generated audio files. "
            "wav: uncompressed PCM (larger files). "
            "mp3: compressed (smaller files). "
            "Format affects file size in multimodal requests but not audio characteristics."
        ),
        default=AudioFormat.WAV,
    )
    audio_depths: annotated_type(
        list[int],
        "--audio-depths",
        Groups.AUDIO_INPUT,
        (
            "List of audio bit depths in bits to randomly select from. "
            "Each audio file is assigned a random depth from this list. "
            "Common values: 8 (low quality), 16 (CD quality), 24 (professional), "
            "32 (high-end). Specify multiple values for mixed-quality testing."
        ),
        validators=[parse_str_or_list_of_positive_values],
    )
    audio_sample_rates: annotated_type(
        list[float],
        "--audio-sample-rates",
        Groups.AUDIO_INPUT,
        (
            "List of audio sample rates in kHz to randomly select from. "
            "Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), "
            "48.0 (professional). Specify multiple values for mixed-quality testing."
        ),
        validators=[parse_str_or_list_of_positive_values],
    )
    audio_num_channels: annotated_type(
        int,
        "--audio-num-channels",
        Groups.AUDIO_INPUT,
        (
            "Number of audio channels. "
            "1 = mono (single channel), 2 = stereo (left/right channels). "
            "Stereo doubles file size. Most speech models use mono."
        ),
        default=1,
    )

    # =========================================================================
    # IMAGE INPUT
    # =========================================================================

    image_height_mean: annotated_type(
        float,
        "--image-height-mean",
        Groups.IMAGE_INPUT,
        "Mean value.",
        default=512.0,
    )
    image_height_stddev: annotated_type(
        float,
        "--image-height-stddev",
        Groups.IMAGE_INPUT,
        "Standard deviation. 0 = deterministic.",
        default=0.0,
    )
    image_width_mean: annotated_type(
        float,
        "--image-width-mean",
        Groups.IMAGE_INPUT,
        "Mean value.",
        default=512.0,
    )
    image_width_stddev: annotated_type(
        float,
        "--image-width-stddev",
        Groups.IMAGE_INPUT,
        "Standard deviation. 0 = deterministic.",
        default=0.0,
    )
    image_batch_size: annotated_type(
        int,
        ("--image-batch-size", "--batch-size-image"),
        Groups.IMAGE_INPUT,
        (
            "Number of images to include in each multimodal request. "
            "Supported with chat endpoint type for vision-language models. "
            "Set to 0 to disable image inputs. "
            "Higher batch sizes test multi-image understanding and increase request "
            "payload size."
        ),
        default=0,
    )
    image_format: annotated_type(
        ImageFormat,
        "--image-format",
        Groups.IMAGE_INPUT,
        (
            "Image file format for generated images. "
            "png: lossless compression (larger files, best quality). "
            "jpeg: lossy compression (smaller files, good quality). "
            "random: randomly select between PNG and JPEG per image. "
            "Format affects file size in multimodal requests and encoding overhead."
        ),
        default=ImageFormat.JPEG,
    )

    # =========================================================================
    # VIDEO INPUT
    # =========================================================================

    video_batch_size: annotated_type(
        int,
        ("--video-batch-size", "--batch-size-video"),
        Groups.VIDEO_INPUT,
        (
            "Number of video files to include in each multimodal request. "
            "Supported with chat endpoint type for video understanding models. "
            "Set to 0 to disable video inputs. "
            "Higher batch sizes significantly increase request payload size."
        ),
        default=0,
    )
    video_duration: annotated_type(
        float,
        "--video-duration",
        Groups.VIDEO_INPUT,
        (
            "Duration in seconds for each generated video clip. "
            "Combined with fps, determines total frame count (frames = duration * fps). "
            "Longer durations increase file size and processing time. "
            "Typical values: 1-10 seconds for testing."
        ),
        default=1.0,
    )
    video_fps: annotated_type(
        int,
        "--video-fps",
        Groups.VIDEO_INPUT,
        (
            "Frames per second for generated video. "
            "Higher FPS creates smoother video but increases frame count and file size. "
            "Common values: 4 (minimal, recommended for Cosmos models), "
            "24 (cinematic), 30 (standard), 60 (high frame rate). "
            "Total frames = duration * fps."
        ),
        default=4,
    )
    video_width: annotated_type(
        int | None,
        "--video-width",
        Groups.VIDEO_INPUT,
        (
            "Video frame width in pixels. "
            "Determines video resolution and file size. "
            "Common values: 640 (SD), 1280 (HD), 1920 (Full HD). "
            "If not specified, uses codec/format defaults."
        ),
    )
    video_height: annotated_type(
        int | None,
        "--video-height",
        Groups.VIDEO_INPUT,
        (
            "Video frame height in pixels. "
            "Combined with width determines aspect ratio and total pixel count per frame. "
            "Common values: 480 (SD), 720 (HD), 1080 (Full HD). "
            "If not specified, uses codec/format defaults."
        ),
    )
    video_synth_type: annotated_type(
        VideoSynthType,
        "--video-synth-type",
        Groups.VIDEO_INPUT,
        (
            "Algorithm for generating synthetic video content. "
            "Different types produce different visual patterns for testing. "
            "Content doesn't affect semantic meaning but may impact encoding "
            "efficiency and file size."
        ),
        default=VideoSynthType.MOVING_SHAPES,
    )
    video_format: annotated_type(
        VideoFormat,
        "--video-format",
        Groups.VIDEO_INPUT,
        (
            "Container format for generated video files. "
            "webm: VP9 codec, BSD-licensed, recommended for open-source workflows. "
            "mp4: H.264/H.265, widely compatible. "
            "avi: legacy, larger files. "
            "mkv: Matroska, flexible container. "
            "Format affects compatibility, file size, and encoding options."
        ),
        default=VideoFormat.WEBM,
    )
    video_codec: annotated_type(
        str,
        "--video-codec",
        Groups.VIDEO_INPUT,
        (
            "Video codec for encoding. "
            "Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), "
            "libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), "
            "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
            "Any FFmpeg-supported codec can be used."
        ),
        default="libvpx-vp9",
    )
    video_audio_sample_rate: annotated_type(
        int,
        "--video-audio-sample-rate",
        Groups.VIDEO_INPUT,
        (
            "Audio sample rate in Hz for the embedded audio track. "
            "Common values: 8000 (telephony), 16000 (speech), 44100 (CD quality), "
            "48000 (professional). "
            "Higher sample rates increase audio fidelity and file size."
        ),
        default=44100,
    )
    video_audio_num_channels: annotated_type(
        int,
        "--video-audio-num-channels",
        Groups.VIDEO_INPUT,
        (
            "Number of audio channels to embed in generated video files. "
            "0 = disabled (no audio track, default), 1 = mono, 2 = stereo. "
            "When set to 1 or 2, a Gaussian noise audio track matching the video "
            "duration is muxed into each video via FFmpeg."
        ),
        default=0,
    )

    # =========================================================================
    # SERVICE
    # =========================================================================

    log_level: annotated_type(
        AIPerfLogLevel,
        "--log-level",
        Groups.SERVICE,
        ("Global logging verbosity level. trace: most verbose, error: least verbose."),
        default=AIPerfLogLevel.INFO,
    )
    verbose: annotated_type(
        bool,
        ("--verbose", "-v"),
        Groups.SERVICE,
        (
            "Equivalent to --log-level DEBUG. "
            "Enables detailed logging and switches UI to simple mode."
        ),
        default=False,
    )
    extra_verbose: annotated_type(
        bool,
        ("--extra-verbose", "-vv"),
        Groups.SERVICE,
        (
            "Equivalent to --log-level TRACE. "
            "Most verbose logging including ZMQ messages. "
            "Switches UI to simple mode."
        ),
        default=False,
    )
    record_processor_service_count: annotated_type(
        int | None,
        ("--record-processor-service-count", "--record-processors"),
        Groups.SERVICE,
        (
            "Number of parallel record processors. "
            "null = auto-detect based on CPU cores."
        ),
    )

    # =========================================================================
    # SERVER METRICS
    # =========================================================================

    server_metrics: annotated_type(
        list[str] | None,
        "--server-metrics",
        Groups.SERVER_METRICS,
        (
            "Server metrics collection (ENABLED BY DEFAULT). "
            "Optionally specify additional Prometheus endpoint URLs. "
            "Use --no-server-metrics to disable."
        ),
        validators=[parse_str_or_list],
        consume_multiple=True,
    )
    no_server_metrics: annotated_type(
        bool,
        "--no-server-metrics",
        Groups.SERVER_METRICS,
        "Disable server metrics collection entirely.",
        default=False,
    )
    server_metrics_formats: annotated_type(
        list[ServerMetricsFormat],
        "--server-metrics-formats",
        Groups.SERVER_METRICS,
        ("Export formats for scraped metrics. Options: json, csv, parquet, jsonl."),
        validators=[parse_str_or_list],
        consume_multiple=True,
    )

    # =========================================================================
    # GPU TELEMETRY
    # =========================================================================

    gpu_telemetry: annotated_type(
        list[str] | None,
        "--gpu-telemetry",
        Groups.GPU_TELEMETRY,
        (
            "Enable GPU telemetry and optionally specify: "
            "'dashboard' for realtime mode, "
            "custom DCGM URLs, or a metrics CSV file."
        ),
        validators=[parse_str_or_list],
        consume_multiple=True,
    )
    no_gpu_telemetry: annotated_type(
        bool,
        "--no-gpu-telemetry",
        Groups.GPU_TELEMETRY,
        "Disable GPU telemetry collection entirely.",
        default=False,
    )

    # =========================================================================
    # UI
    # =========================================================================

    ui_type: annotated_type(
        UIType,
        ("--ui-type", "--ui"),
        Groups.UI,
        (
            "User interface mode. "
            "dashboard: rich interactive UI, "
            "simple: text progress, "
            "none: silent operation."
        ),
        default=UIType.DASHBOARD,
    )

    # =========================================================================
    # WORKERS
    # =========================================================================

    workers_max: annotated_type(
        int | None,
        ("--workers-max", "--max-workers"),
        Groups.WORKERS,
        "Maximum worker processes. null = auto-detect based on CPU cores.",
    )

    # =========================================================================
    # ZMQ COMMUNICATION
    # =========================================================================

    zmq_host: annotated_type(
        str | None,
        "--zmq-host",
        Groups.ZMQ_COMMUNICATION,
        "Host address for internal ZMQ TCP communication between AIPerf services.",
    )
    zmq_ipc_path: annotated_type(
        Path | None,
        "--zmq-ipc-path",
        Groups.ZMQ_COMMUNICATION,
        "Directory path for ZMQ IPC socket files for local inter-process communication.",
    )

    # =========================================================================
    # ACCURACY
    # =========================================================================

    accuracy_benchmark: annotated_type(
        AccuracyBenchmarkType | None,
        "--accuracy-benchmark",
        Groups.ACCURACY,
        (
            "Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). "
            "When set, enables accuracy benchmarking mode alongside performance profiling."
        ),
    )
    accuracy_tasks: annotated_type(
        list[str] | None,
        "--accuracy-tasks",
        Groups.ACCURACY,
        (
            "Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). If not set, all tasks are included."
        ),
        validators=[parse_str_or_list],
        consume_multiple=True,
    )
    accuracy_n_shots: annotated_type(
        int,
        "--accuracy-n-shots",
        Groups.ACCURACY,
        (
            "Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation. Maximum 8."
        ),
        default=0,
    )
    accuracy_enable_cot: annotated_type(
        bool,
        "--accuracy-enable-cot",
        Groups.ACCURACY,
        (
            "Enable chain-of-thought prompting for accuracy evaluation. "
            "Adds reasoning instructions to the prompt."
        ),
        default=False,
    )
    accuracy_grader: annotated_type(
        AccuracyGraderType | None,
        "--accuracy-grader",
        Groups.ACCURACY,
        (
            "Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution). "
            "If not set, uses the benchmark's default grader."
        ),
    )
    accuracy_system_prompt: annotated_type(
        str | None,
        "--accuracy-system-prompt",
        Groups.ACCURACY,
        (
            "Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt."
        ),
    )
    accuracy_verbose: annotated_type(
        bool,
        "--accuracy-verbose",
        Groups.ACCURACY,
        (
            "Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details."
        ),
        default=False,
    )

    # =========================================================================
    # MULTI-RUN
    # =========================================================================

    num_profile_runs: annotated_type(
        int,
        "--num-profile-runs",
        Groups.MULTI_RUN,
        (
            "Number of profile runs to execute for confidence reporting. "
            "When 1, runs a single benchmark. "
            "When >1, computes aggregate statistics across runs."
        ),
        default=1,
    )
    profile_run_cooldown_seconds: annotated_type(
        float,
        "--profile-run-cooldown-seconds",
        Groups.MULTI_RUN,
        (
            "Cooldown duration in seconds between profile runs. "
            "Allows the system to stabilize between runs."
        ),
        default=0.0,
    )
    confidence_level: annotated_type(
        float,
        "--confidence-level",
        Groups.MULTI_RUN,
        (
            "Confidence level for computing confidence intervals (0-1). "
            "Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%)."
        ),
        default=0.95,
    )
    profile_run_disable_warmup_after_first: annotated_type(
        bool,
        "--profile-run-disable-warmup-after-first",
        Groups.MULTI_RUN,
        (
            "Disable warmup for runs after the first. "
            "When true, only the first run includes warmup for steady-state measurement."
        ),
        default=True,
        negative=None,
    )
    set_consistent_seed: annotated_type(
        bool,
        "--set-consistent-seed",
        Groups.MULTI_RUN,
        "Auto-set random seed if not specified for workload consistency.",
        default=True,
        negative=None,
    )

    # =========================================================================
    # CONFIG FILE (in INPUT group)

    # =========================================================================

    config_file: annotated_type(
        Path | None,
        ("--config", "-f"),
        Groups.INPUT,
        (
            "Path to a YAML configuration file. "
            "CLI flags override values from the config file."
        ),
    )

    # =========================================================================
    # GENERATED (disabled in CLI)

    # =========================================================================

    cli_command: annotated_type(
        str | None, None, None, "The CLI command (auto-generated).", parse=False
    )
    benchmark_id: annotated_type(
        str | None,
        None,
        None,
        "Unique benchmark run identifier (auto-generated UUID).",
        parse=False,
    )
