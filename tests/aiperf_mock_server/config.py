# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration management for the integration test server."""

from typing import Annotated, Literal

import cyclopts
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MockServerConfig(BaseSettings):
    """Unified server configuration supporting both CLI arguments and environment variables.

    Environment variables should be prefixed with MOCK_SERVER_ (e.g., MOCK_SERVER_PORT=8000).
    CLI arguments will override environment variables.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="MOCK_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server settings
    port: Annotated[
        int,
        Field(
            description="Port to run the server on",
            ge=1,
            le=65535,
        ),
        cyclopts.Parameter(
            name=("--port", "-p"),
        ),
    ] = 8000

    host: Annotated[
        str,
        Field(
            description="Host to bind the server to",
        ),
        cyclopts.Parameter(
            name=("--host", "-h"),
        ),
    ] = "0.0.0.0"

    workers: Annotated[
        int,
        Field(
            description="Number of worker processes",
            ge=1,
            le=32,
        ),
        cyclopts.Parameter(
            name=("--workers", "-w"),
        ),
    ] = 1

    # Timing settings (in milliseconds)
    ttft: Annotated[
        float,
        Field(
            description="Time to first token latency in milliseconds",
            gt=0.0,
        ),
        cyclopts.Parameter(
            name=("--ttft", "-t"),
        ),
    ] = 20.0

    itl: Annotated[
        float,
        Field(
            description="Inter-token latency in milliseconds",
            gt=0.0,
        ),
        cyclopts.Parameter(
            name=("--itl", "-i"),
        ),
    ] = 5.0

    # Logging settings
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(
            description="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
        cyclopts.Parameter(
            name=("--log-level", "-l"),
        ),
    ] = "INFO"

    # Tokenizer settings
    tokenizer_models: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="List of tokenizer models to pre-load at startup",
        ),
        cyclopts.Parameter(
            name=("--tokenizer-models", "-m"),
        ),
    ] = []

    access_logs: Annotated[
        bool,
        Field(
            description="Enable HTTP access logs printing to stdout. If logging level is DEBUG, "
            "this will be enabled by default.",
        ),
        cyclopts.Parameter(
            name=("--access-logs", "-a"),
        ),
    ] = False
