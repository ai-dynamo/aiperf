# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_user_config(n_shots: int | None = None) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU, n_shots=n_shots),
    )


def _make_problem() -> BenchmarkProblem:
    return BenchmarkProblem(prompt="Q?", ground_truth="A", task="test_task")


@pytest.mark.asyncio
class TestLoadBenchmarkProblemsNShots:
    async def test_uses_explicit_n_shots_without_consulting_metadata(self) -> None:
        """When n_shots is set explicitly, plugin metadata is never consulted."""
        user_config = _make_user_config(n_shots=3)
        problem = _make_problem()

        mock_benchmark = AsyncMock()
        mock_benchmark.load_problems = AsyncMock(return_value=[problem])

        def mock_cls(**_kwargs):
            return mock_benchmark

        with (
            patch(
                "aiperf.accuracy.benchmark_loader.plugins.get_class",
                return_value=mock_cls,
            ),
            patch("aiperf.accuracy.benchmark_loader.plugins.get_metadata") as mock_meta,
        ):
            result = await load_benchmark_problems(user_config)

        mock_meta.assert_not_called()
        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=3, enable_cot=False
        )
        assert result == [problem]

    async def test_falls_back_to_default_n_shots_from_metadata(self) -> None:
        """When n_shots is None, default_n_shots from plugin metadata is used."""
        user_config = _make_user_config(n_shots=None)
        problem = _make_problem()

        mock_benchmark = AsyncMock()
        mock_benchmark.load_problems = AsyncMock(return_value=[problem])

        def mock_cls(**_kwargs):
            return mock_benchmark

        with (
            patch(
                "aiperf.accuracy.benchmark_loader.plugins.get_class",
                return_value=mock_cls,
            ),
            patch(
                "aiperf.accuracy.benchmark_loader.plugins.get_metadata",
                return_value={"default_n_shots": 5},
            ),
        ):
            result = await load_benchmark_problems(user_config)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=5, enable_cot=False
        )
        assert result == [problem]

    async def test_defaults_to_zero_when_default_n_shots_missing_from_metadata(
        self,
    ) -> None:
        """When n_shots is None and metadata has no default_n_shots, n_shots defaults to 0."""
        user_config = _make_user_config(n_shots=None)
        problem = _make_problem()

        mock_benchmark = AsyncMock()
        mock_benchmark.load_problems = AsyncMock(return_value=[problem])

        def mock_cls(**_kwargs):
            return mock_benchmark

        with (
            patch(
                "aiperf.accuracy.benchmark_loader.plugins.get_class",
                return_value=mock_cls,
            ),
            patch(
                "aiperf.accuracy.benchmark_loader.plugins.get_metadata",
                return_value={},
            ),
        ):
            result = await load_benchmark_problems(user_config)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=0, enable_cot=False
        )
        assert result == [problem]
