# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mock server configuration."""

import json
import logging
import os
from typing import Annotated, Any, Literal

from cyclopts import Parameter
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


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
        Field(description="HuggingFace tokenizer for corpus (name or path)"),
        Parameter(name="--tokenizer"),
    ] = "Qwen/Qwen3-0.6B"

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
            description="Skip tokenizer loading, use character-based chunking (faster startup)"
        ),
        Parameter(name="--no-tokenizer"),
    ] = False


server_config: MockServerConfig = MockServerConfig()

logger = logging.getLogger(__name__)


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
    if isinstance(value, list | dict):
        return json.dumps(value)
    return str(value)
