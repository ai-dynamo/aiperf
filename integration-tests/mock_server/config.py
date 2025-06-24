#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Configuration management for the integration test server."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigDefaults:
    """Centralized configuration defaults for the integration test server."""

    # Server settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    WORKERS: int = 1

    # Timing settings
    TTFT_MS: float = 50.0
    ITL_MS: float = 10.0

    # Logging settings
    LOG_LEVEL: str = "INFO"


class MockServerConfig(BaseSettings):
    """Server configuration with automatic environment variable support."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="MOCK_SERVER_",
    )

    port: int = Field(
        default=ConfigDefaults.PORT,
        description="Port to run the server on",
    )

    ttft_ms: float = Field(
        default=ConfigDefaults.TTFT_MS,
        description="Time to first token latency in milliseconds",
    )

    itl_ms: float = Field(
        default=ConfigDefaults.ITL_MS,
        description="Inter-token latency in milliseconds",
    )

    host: str = Field(
        default=ConfigDefaults.HOST,
        description="Host to bind the server to",
    )

    workers: int = Field(
        default=ConfigDefaults.WORKERS,
        description="Number of worker processes",
    )
