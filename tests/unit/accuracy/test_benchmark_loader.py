# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.config.flags import CLIConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_run_from_v1


def _make_run(n_shots: int | None = None):
    # v1 CLIConfig requires int for n_shots, but the loader honors None
    # as "fall back to plugin metadata". Build via v1 with a placeholder, then
    # mutate the v2 cfg to expose the None path the loader actually reads.
    cli_config = CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy_benchmark=AccuracyBenchmarkType.MMLU,
        accuracy_n_shots=n_shots if n_shots is not None else 0,
    )
    run = make_run_from_v1(cli_config)
    if n_shots is None:
        run.cfg.accuracy.n_shots = None
    return run


def _make_problem() -> BenchmarkProblem:
    return BenchmarkProblem(prompt="Q?", ground_truth="A", task="test_task")


@pytest.mark.asyncio
class TestLoadBenchmarkProblemsNShots:
    async def test_uses_explicit_n_shots_without_consulting_metadata(self) -> None:
        """When n_shots is set explicitly, plugin metadata is never consulted."""
        run = _make_run(n_shots=3)
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
            result = await load_benchmark_problems(run)

        mock_meta.assert_not_called()
        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=3, enable_cot=False
        )
        assert result == [problem]

    async def test_falls_back_to_default_n_shots_from_metadata(self) -> None:
        """When n_shots is None, default_n_shots from plugin metadata is used."""
        run = _make_run(n_shots=None)
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
            result = await load_benchmark_problems(run)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=5, enable_cot=False
        )
        assert result == [problem]

    async def test_defaults_to_zero_when_default_n_shots_missing_from_metadata(
        self,
    ) -> None:
        """When n_shots is None and metadata has no default_n_shots, n_shots defaults to 0."""
        run = _make_run(n_shots=None)
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
            result = await load_benchmark_problems(run)

        mock_benchmark.load_problems.assert_awaited_once_with(
            tasks=None, n_shots=0, enable_cot=False
        )
        assert result == [problem]
