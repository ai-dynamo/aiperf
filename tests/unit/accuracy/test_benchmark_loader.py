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
    async def test_explicit_n_shots_passes_through_unchanged(self) -> None:
        """When ``n_shots`` is set explicitly, it's forwarded verbatim.

        The benchmark loader still reads metadata once (to resolve
        ``default_enable_cot``), so we don't assert the mock was
        un-called — we assert that metadata's ``default_n_shots`` is
        ignored when the user provides their own value.
        """
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
            patch(
                "aiperf.accuracy.benchmark_loader.plugins.get_metadata",
                # Metadata claims default_n_shots=99; user said 3, so 3 wins.
                return_value={"default_n_shots": 99},
            ),
        ):
            result = await load_benchmark_problems(user_config)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=3, enable_cot=False
        )
        assert result == [problem]

    async def test_metadata_default_enable_cot_used_when_unset(self) -> None:
        """When ``enable_cot`` is None, the benchmark's
        ``default_enable_cot`` from plugin metadata is honored."""
        user_config = _make_user_config(n_shots=0)
        # AccuracyConfig.enable_cot defaults to None now; explicitly None.
        user_config.accuracy.enable_cot = None
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
                return_value={"default_enable_cot": True},
            ),
        ):
            await load_benchmark_problems(user_config)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=0, enable_cot=True
        )

    async def test_explicit_enable_cot_overrides_metadata(self) -> None:
        """When the user explicitly sets ``enable_cot`` (True or False),
        the metadata default is ignored."""
        user_config = _make_user_config(n_shots=0)
        user_config.accuracy.enable_cot = False
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
                return_value={"default_enable_cot": True},
            ),
        ):
            await load_benchmark_problems(user_config)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=0, enable_cot=False
        )

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
