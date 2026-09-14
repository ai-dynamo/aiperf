# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for profile CLI command, focusing on the model autodetect path."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_commands.profile import profile
from aiperf.config.flags import CLIConfig

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_loaders() -> Generator[MagicMock, None, None]:
    """Mock the resolve_config + build_benchmark_plan + run_benchmark path."""
    with (
        patch("aiperf.config.flags.resolver.resolve_config"),
        patch("aiperf.config.loader.build_benchmark_plan"),
        patch("aiperf.cli_runner.run_benchmark") as mock_run,
    ):
        yield mock_run


@pytest.fixture
def base_cli_config() -> CLIConfig:
    """Return a minimal CLIConfig suitable for testing profile()."""
    return CLIConfig(
        model_names=["test-model"],
        urls=["http://localhost:8000"],
    )


def test_profile_with_explicit_model_calls_run_benchmark(
    base_cli_config: CLIConfig,
    mock_loaders: MagicMock,
) -> None:
    """When --model is provided, autodetect is skipped and run_benchmark is called."""
    profile(cli_config=base_cli_config)
    mock_loaders.assert_called_once()


def test_profile_autodetects_model_when_model_not_provided(
    monkeypatch: pytest.MonkeyPatch,
    mock_loaders: MagicMock,
) -> None:
    """When --model is NOT provided, autodetect_names is called and its result
    is assigned to cli_config.model_names before proceeding."""
    # Build a CLIConfig without model_names
    cli_config = CLIConfig(
        model_names=[],
        urls=["http://localhost:8000"],
    )

    async def _fake_autodetect_names(**_: object) -> list[str]:
        return ["auto-detected-model"]

    monkeypatch.setattr(
        "aiperf.common.models.model_autodetect.autodetect_names",
        _fake_autodetect_names,
    )

    def _fake_asyncio_run(coro: object) -> list[str]:
        """Synchronous wrapper that just returns the coroutine result
        without involving a real event loop."""
        import asyncio as _asyncio

        loop = _asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(
        "aiperf.cli_commands.profile.asyncio.run",
        _fake_asyncio_run,
    )

    profile(cli_config=cli_config)

    # After autodetect, model_names should be set
    assert cli_config.model_names == ["auto-detected-model"]
    mock_loaders.assert_called_once()
