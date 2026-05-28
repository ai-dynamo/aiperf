# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mock server configuration."""

import importlib
import json
import logging
import os
from typing import Annotated, Any, Literal

import cyclopts


def _load_cyclopts_parameter() -> type:
    param = getattr(cyclopts, "Parameter", None)
    if param is not None:
        return param
    for module_name in (
        "parameter",
        "params",
        "param",
        "_parameter",
        "_params",
    ):
        try:
            module = importlib.import_module(f"cyclopts.{module_name}")
        except Exception:
            continue
        param = getattr(module, "Parameter", None)
        if param is not None:
            return param
    raise ImportError("cyclopts.Parameter is not available in this cyclopts version")


from pydantic import Field, model_validator  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402
from typing_extensions import Self  # noqa: E402

Parameter = _load_cyclopts_parameter()

logger = logging.getLogger(__name__)


class MockServerConfig(BaseSettings):
    """Server configuration with environment variable support."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="MOCK_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def apply_flags(self) -> Self:
        if self.verbose:
            self.log_level = "DEBUG"
        if self.fast:
            self.ttft = 0.0
            self.itl = 0.0
            self.embedding_base_latency = 0.0
            self.embedding_per_input_latency = 0.0
            self.ranking_base_latency = 0.0
            self.ranking_per_passage_latency = 0.0
            self.image_retrieval_base_latency = 0.0
            self.image_retrieval_per_image_latency = 0.0
        if self.record_requests is not None:
            if self.no_tokenizer:
                raise ValueError(
                    "--record-requests requires a tokenizer for counting ISL; "
                    "remove --no-tokenizer or omit --record-requests"
                )
            if self.workers != 1:
                logger.warning(
                    "--record-requests forces --workers=1 (was %d): the recorder "
                    "keeps per-request stats in-process, so a single uvicorn "
                    "worker is the supported producer",
                    self.workers,
                )
                self.workers = 1
        return self

    port: Annotated[
        int,
        Field(description="Port to run on", ge=1, le=65535),
        Parameter(name=("--port", "-p")),
    ] = 8000

    host: Annotated[
        str,
        Field(description="Host to bind to"),
        Parameter(name="--host"),
    ] = "127.0.0.1"

    workers: Annotated[
        int,
        Field(description="Number of workers", ge=1, le=32),
        Parameter(name=("--workers", "-w")),
    ] = 1

    ttft: Annotated[
        float,
        Field(description="Time to first token (ms)", ge=0.0),
        Parameter(name=("--ttft", "-t")),
    ] = 20.0

    itl: Annotated[
        float,
        Field(description="Inter-token latency (ms)", ge=0.0),
        Parameter(name="--itl"),
    ] = 5.0

    ttft_per_isl_token_ms: Annotated[
        float,
        Field(
            description=(
                "Per-ISL-token TTFT scaling (ms). Models prefill cost: "
                "ttft_ms = ttft + ttft_per_isl_token_ms * prompt_token_count. "
                "Set e.g. 0.05 to make TTFT scale ~50ms per 1k input tokens."
            ),
            ge=0.0,
        ),
        Parameter(name="--ttft-per-isl-token-ms"),
    ] = 0.0

    ttft_concurrency_quad_ms: Annotated[
        float,
        Field(
            description=(
                "Concurrency-quadratic TTFT penalty (ms). Models queueing: "
                "ttft_ms += ttft_concurrency_quad_ms * active_inflight^2. "
                "Set e.g. 0.001 to push TTFT past ~250ms above ~500 concurrent."
            ),
            ge=0.0,
        ),
        Parameter(name="--ttft-concurrency-quad-ms"),
    ] = 0.0

    itl_per_osl_token_ms: Annotated[
        float,
        Field(
            description=(
                "Per-OSL-token ITL scaling (ms). Captured once per request at "
                "TTFT-time (active OSL = max_tokens budget). itl_ms = itl + "
                "itl_per_osl_token_ms * osl_tokens."
            ),
            ge=0.0,
        ),
        Parameter(name="--itl-per-osl-token-ms"),
    ] = 0.0

    itl_concurrency_lin_ms: Annotated[
        float,
        Field(
            description=(
                "Concurrency-linear ITL penalty (ms). itl_ms += "
                "itl_concurrency_lin_ms * active_inflight. Set e.g. 0.05 to add "
                "~25ms of ITL at concurrency=500."
            ),
            ge=0.0,
        ),
        Parameter(name="--itl-concurrency-lin-ms"),
    ] = 0.0

    scheduler_enabled: Annotated[
        bool,
        Field(
            description=(
                "Enable the step-based batched scheduler. When true, requests "
                "compete for per-step decode and prefill slots, producing a "
                "real saturation knee. When false (default), the open-loop "
                "TTFT/ITL latency model is used."
            ),
        ),
        Parameter(name="--scheduler-enabled", negative="--no-scheduler-enabled"),
    ] = False

    scheduler_step_ms: Annotated[
        float,
        Field(
            description=(
                "Virtual decode-step cadence in milliseconds. Each step admits "
                "up to scheduler_max_batch_size decode tokens. Smaller values "
                "= finer-grained ITL but higher scheduler CPU cost."
            ),
            gt=0.0,
            le=1000.0,
        ),
        Parameter(name="--scheduler-step-ms"),
    ] = 5.0

    scheduler_max_batch_size: Annotated[
        int,
        Field(
            description=(
                "Maximum concurrent decoders served per step. Past this "
                "concurrency the per-request ITL stretches linearly. Throughput "
                "ceiling = max_batch_size / step_ms tokens/sec."
            ),
            ge=1,
        ),
        Parameter(name="--scheduler-max-batch-size"),
    ] = 256

    scheduler_max_prefill_chunks_per_step: Annotated[
        int,
        Field(
            description=(
                "Maximum prefill chunks admitted per step. Lower = prefill "
                "becomes the binding constraint, producing TTFT cliffs under "
                "concurrent prompt arrivals."
            ),
            ge=1,
        ),
        Parameter(name="--scheduler-max-prefill-chunks-per-step"),
    ] = 8

    scheduler_prefill_chunk_tokens: Annotated[
        int,
        Field(
            description=(
                "Tokens per prefill chunk. A prompt of P tokens needs "
                "ceil(P / chunk_tokens) chunks. Larger = fewer steps per "
                "prompt but coarser-grained competition."
            ),
            ge=1,
        ),
        Parameter(name="--scheduler-prefill-chunk-tokens"),
    ] = 512

    scheduler_goodput_collapse_enabled: Annotated[
        bool,
        Field(
            description=(
                "Enable goodput-collapse modeling in the scheduler. When the "
                "decode queue grows past the threshold, the per-step admit "
                "budget shrinks toward floor, so aggregate useful tok/s "
                "actually decreases past the knee instead of plateauing. "
                "Models the preemption/admission thrash real continuous-"
                "batching servers exhibit when oversubscribed."
            ),
        ),
        Parameter(
            name="--scheduler-goodput-collapse-enabled",
            negative="--no-scheduler-goodput-collapse-enabled",
        ),
    ] = False

    scheduler_goodput_collapse_threshold: Annotated[
        float,
        Field(
            description=(
                "Decode-queue overload ratio at which goodput collapse "
                "starts. ratio = decode_queue_len / max_batch_size. Below "
                "this the full batch admits per step; above, batch shrinks."
            ),
            ge=0.0,
        ),
        Parameter(name="--scheduler-goodput-collapse-threshold"),
    ] = 1.5

    scheduler_goodput_collapse_slope: Annotated[
        float,
        Field(
            description=(
                "How fast the effective batch shrinks past the threshold. "
                "shrink = (overload - threshold) * slope, capped at "
                "(1 - floor). Larger = sharper goodput cliff."
            ),
            ge=0.0,
        ),
        Parameter(name="--scheduler-goodput-collapse-slope"),
    ] = 0.5

    scheduler_goodput_collapse_floor: Annotated[
        float,
        Field(
            description=(
                "Minimum fraction of max_batch_size that still admits per "
                "step under heavy overload. Floor of 0.3 = even at 10x "
                "overload at most 70% of the batch is dropped."
            ),
            ge=0.0,
            le=1.0,
        ),
        Parameter(name="--scheduler-goodput-collapse-floor"),
    ] = 0.3

    ttft_jitter_cv: Annotated[
        float,
        Field(
            description=(
                "Lognormal jitter coefficient of variation (stddev/mean) "
                "applied to TTFT. 0.0 = deterministic. 0.2 = ~20% TTFT "
                "noise. Open-loop multiplies ttft_sec once per request; "
                "scheduler mode adds positive-only sleep on top of admit."
            ),
            ge=0.0,
        ),
        Parameter(name="--ttft-jitter-cv"),
    ] = 0.0

    itl_jitter_cv: Annotated[
        float,
        Field(
            description=(
                "Lognormal jitter coefficient of variation (stddev/mean) "
                "applied to each ITL gap. 0.0 = deterministic. 0.15 = ~15% "
                "per-token noise. Sampled fresh per token. Critical for "
                "testing whether sweep recipes converge under realistic "
                "noise."
            ),
            ge=0.0,
        ),
        Parameter(name="--itl-jitter-cv"),
    ] = 0.0

    # Embedding latency: base + per_input * num_inputs
    embedding_base_latency: Annotated[
        float,
        Field(description="Embedding base latency (ms)", ge=0.0),
        Parameter(name="--embedding-base-latency"),
    ] = 10.0

    embedding_per_input_latency: Annotated[
        float,
        Field(description="Embedding latency per input (ms)", ge=0.0),
        Parameter(name="--embedding-per-input-latency"),
    ] = 2.0

    # Ranking latency: base + per_passage * num_passages
    ranking_base_latency: Annotated[
        float,
        Field(description="Ranking base latency (ms)", ge=0.0),
        Parameter(name="--ranking-base-latency"),
    ] = 10.0

    ranking_per_passage_latency: Annotated[
        float,
        Field(description="Ranking latency per passage (ms)", ge=0.0),
        Parameter(name="--ranking-per-passage-latency"),
    ] = 1.0

    # Image retrieval latency: base + per_image * num_images
    image_retrieval_base_latency: Annotated[
        float,
        Field(description="Image retrieval base latency (ms)", ge=0.0),
        Parameter(name="--image-retrieval-base-latency"),
    ] = 10.0

    image_retrieval_per_image_latency: Annotated[
        float,
        Field(description="Image retrieval latency per image (ms)", ge=0.0),
        Parameter(name="--image-retrieval-per-image-latency"),
    ] = 5.0

    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(description="Logging level"),
        Parameter(name="--log-level"),
    ] = "INFO"

    verbose: Annotated[
        bool,
        Field(description="Verbose mode (sets log level to DEBUG)"),
        Parameter(name=("--verbose", "-v")),
    ] = False

    fast: Annotated[
        bool,
        Field(description="Fast mode (zero latency for integration testing)"),
        Parameter(name=("--fast", "-f")),
    ] = False

    access_logs: Annotated[
        bool,
        Field(description="Enable HTTP access logs"),
        Parameter(name="--access-logs"),
    ] = False

    error_rate: Annotated[
        float,
        Field(description="Error injection rate 0-100", ge=0.0, le=100.0),
        Parameter(name="--error-rate"),
    ] = 0.0

    random_seed: Annotated[
        int | None,
        Field(description="Random seed for reproducible errors"),
        Parameter(name="--random-seed"),
    ] = None

    # DCGM Faker Options (always enabled)
    dcgm_gpu_name: Annotated[
        str,
        Field(
            description="GPU model name (rtx6000, a100, h100, h100-sxm, h200, b200, gb200)"
        ),
        Parameter(name="--dcgm-gpu-name"),
    ] = "h200"

    dcgm_num_gpus: Annotated[
        int,
        Field(description="Number of GPUs to simulate", ge=1, le=8),
        Parameter(name="--dcgm-num-gpus"),
    ] = 2

    dcgm_min_throughput: Annotated[
        int,
        Field(
            description="Minimum tokens/sec baseline (auto-scales above this)",
            ge=1,
            le=100000,
        ),
        Parameter(name="--dcgm-min-throughput"),
    ] = 100

    dcgm_window_sec: Annotated[
        float,
        Field(description="Throughput sliding window in seconds", ge=0.1, le=60.0),
        Parameter(name="--dcgm-window-sec"),
    ] = 1.0

    dcgm_hostname: Annotated[
        str,
        Field(description="Hostname for DCGM metrics"),
        Parameter(name="--dcgm-hostname"),
    ] = "localhost"

    dcgm_seed: Annotated[
        int | None,
        Field(description="Random seed for DCGM metrics"),
        Parameter(name="--dcgm-seed"),
    ] = None

    dcgm_auto_load: Annotated[
        bool,
        Field(description="Auto-scale DCGM load based on token throughput"),
        Parameter(name="--dcgm-auto-load", negative="--no-dcgm-auto-load"),
    ] = True

    # Tokenizer Options (for corpus tokenization)
    tokenizer: Annotated[
        str,
        Field(
            description=(
                "Tokenizer for corpus tokenization. Default 'builtin' uses "
                "AIPerf's bundled tiktoken o200k_base encoding (zero network "
                "access). Pass any HuggingFace name or path for HF tokenizers."
            )
        ),
        Parameter(name="--tokenizer"),
    ] = "builtin"

    tokenizer_revision: Annotated[
        str,
        Field(description="Tokenizer revision (branch, tag, or commit ID)"),
        Parameter(name="--tokenizer-revision"),
    ] = "main"

    tokenizer_trust_remote_code: Annotated[
        bool,
        Field(description="Trust remote code for custom tokenizers"),
        Parameter(name="--tokenizer-trust-remote-code"),
    ] = False

    no_tokenizer: Annotated[
        bool,
        Field(
            description="Skip tokenizer loading entirely, use character-based chunking (faster startup, less realistic)."
        ),
        Parameter(name="--no-tokenizer"),
    ] = False

    # Request recording options
    record_requests: Annotated[
        str | None,
        Field(
            description="Path to a JSONL file for recording per-request ISL "
            "(tokenized input length) and requested OSL. When set, the server "
            "tokenizes each request inline (reusing the configured --tokenizer) "
            "and writes one record per request; a summary of the distributions "
            "is written to <path>.summary.json on shutdown. Requires a real "
            "tokenizer (incompatible with --no-tokenizer) and forces --workers=1."
        ),
        Parameter(name="--record-requests"),
    ] = None

    # /v1/models Options (used by readiness-probe tests)
    default_model: Annotated[
        str,
        Field(
            description="Model id returned by GET /v1/models once the models "
            "endpoint is 'ready'. Independent of which model names appear in "
            "individual inference requests (those are echoed back as-is)."
        ),
        Parameter(name="--default-model"),
    ] = "mock-model"

    models_ready_delay_seconds: Annotated[
        float,
        Field(
            description="Seconds after server start during which GET /v1/models "
            "returns an empty data list. Simulates a model server that's up "
            "but hasn't finished loading weights.",
            ge=0.0,
        ),
        Parameter(name="--models-ready-delay-seconds"),
    ] = 0.0

    disable_models_endpoint: Annotated[
        bool,
        Field(
            description="If set, GET /v1/models returns 404. Used to exercise "
            "the readiness-probe fallback to a plain base-URL GET."
        ),
        Parameter(name="--disable-models-endpoint"),
    ] = False

    inference_ready_delay_seconds: Annotated[
        float,
        Field(
            description="Seconds after server start during which the inference "
            "endpoints (/v1/chat/completions, /v1/completions, /v1/embeddings) "
            "return HTTP 503. Simulates a stack that's up on the frontend but "
            "whose workers haven't loaded weights yet. Used to exercise the "
            "inference-mode readiness probe's retry loop.",
            ge=0.0,
        ),
        Parameter(name="--inference-ready-delay-seconds"),
    ] = 0.0


server_config: MockServerConfig = MockServerConfig()


def set_server_config(config: MockServerConfig) -> None:
    """Set server configuration and propagate to environment variables."""
    global server_config
    server_config = config
    _propagate_config_to_env(config)


def _propagate_config_to_env(config: MockServerConfig) -> None:
    """Propagate configuration to environment variables for subprocess access."""
    for key, value in config.model_dump().items():
        if value is not None:
            env_key = _get_env_key(key)
            env_value = _serialize_env_value(value)
            logger.debug("Setting environment variable: %s = %s", env_key, env_value)
            os.environ[env_key] = env_value


def _get_env_key(config_key: str) -> str:
    """Convert config key to environment variable name."""
    return f"MOCK_SERVER_{config_key.upper()}"


def _serialize_env_value(value: Any) -> str:
    """Serialize value for environment variable storage."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)
