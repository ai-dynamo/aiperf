# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for API CLI parameter validation on RuntimeConfig."""

from aiperf.config import AIPerfConfig
from aiperf.config.models import RuntimeConfig


def _minimal_config(**runtime_kwargs) -> AIPerfConfig:
    """Build a minimal AIPerfConfig with optional runtime overrides."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32, "osl": 16},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
        runtime=runtime_kwargs if runtime_kwargs else {},
    )


class TestAPICLIParams:
    """Test API CLI parameter validation on RuntimeConfig."""

    def test_no_api_params(self) -> None:
        """Test RuntimeConfig without API parameters."""
        rt = RuntimeConfig()
        assert rt.api_port is None
        assert rt.api_host is None

    def test_port_only_leaves_host_none(self) -> None:
        """Test that setting port only leaves host as None (resolved at runtime from env)."""
        config = _minimal_config(api_port=9999)
        assert config.runtime.api_port == 9999
        assert config.runtime.api_host is None

    def test_port_and_host(self) -> None:
        """Test setting both port and host."""
        config = _minimal_config(api_port=8080, api_host="0.0.0.0")
        assert config.runtime.api_port == 8080
        assert config.runtime.api_host == "0.0.0.0"
