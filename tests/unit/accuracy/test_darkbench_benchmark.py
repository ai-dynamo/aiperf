# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import orjson
import pytest

from aiperf.accuracy.benchmarks import darkbench
from aiperf.accuracy.benchmarks._darkbench_patterns import DARK_PATTERNS
from aiperf.accuracy.benchmarks.darkbench import (
    DATASET_NAME,
    DEFAULT_GENERATION_SIZE,
    TASKS,
    DarkBenchBenchmark,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from aiperf.plugin import plugins
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType, PluginType
from tests.unit.conftest import make_benchmark_run

LABELS = (
    "anthropomorphization",
    "brand-bias",
    "harmful-generation",
    "sneaking",
    "sycophancy",
    "user-retention",
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"{label}-{i:03d}",
            "input": f"{label} prompt {i} with {{braces}}",
            "target": label,
            "metadata": {"dark_pattern": label},
        }
        for label in LABELS
        for i in (1, 2)
    ]


def _benchmark(model: str = "Qwen/Qwen3-8B") -> DarkBenchBenchmark:
    run = make_benchmark_run(
        model_names=[model],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.DARKBENCH},
    )
    return DarkBenchBenchmark(run=run)


async def _load(
    benchmark: DarkBenchBenchmark, tasks: list[str] | None = None, **kwargs: object
):
    with patch.object(darkbench, "load_dataset", return_value=_rows()) as mock_load:
        problems = await benchmark.load_problems(
            tasks=tasks, n_shots=kwargs.get("n_shots", 0), enable_cot=False
        )
    return problems, mock_load


def test_tasks_are_hyphenated_dataset_labels() -> None:
    assert TASKS == LABELS


def test_plugin_metadata_matches_reference_defaults() -> None:
    meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, "darkbench")
    assert meta["default_grader"] == "darkbench_overseer"
    assert meta["default_n_shots"] == 0
    assert meta["default_system_prompt"] == "You are a helpful assistant."


@pytest.mark.asyncio
class TestDarkBenchLoadProblems:
    async def test_load_problems_all_tasks_maps_labels_to_pattern_ids(self) -> None:
        problems, mock_load = await _load(_benchmark())

        assert len(problems) == 12
        mock_load.assert_called_once_with(
            DATASET_NAME,
            split="train",
            revision=Environment.ACCURACY.DARKBENCH_REVISION,
        )
        for problem in problems:
            gt = orjson.loads(problem.ground_truth)
            assert problem.task in LABELS
            assert gt["pattern"] == problem.task.replace("-", "_")
            assert gt["pattern"] in DARK_PATTERNS
            assert gt["prompt"] == problem.prompt
            assert problem.raw_messages == [{"role": "user", "content": problem.prompt}]
            assert problem.metadata["generation_size"] == DEFAULT_GENERATION_SIZE

    async def test_load_problems_filters_tasks(self) -> None:
        problems, _ = await _load(_benchmark(), tasks=["sneaking", "brand-bias"])

        assert sorted({p.task for p in problems}) == ["brand-bias", "sneaking"]
        assert len(problems) == 4

    @pytest.mark.parametrize(
        "tasks",
        [
            pytest.param(["brand_bias"], id="snake-case"),
            pytest.param(["sneaking", "nope"], id="unknown"),
        ],
    )  # fmt: skip
    async def test_load_problems_unknown_task_raises_configuration_error(
        self, tasks: list[str]
    ) -> None:
        with pytest.raises(ConfigurationError, match="valid tasks"):
            await _load(_benchmark(), tasks=tasks)

    async def test_load_problems_prompt_braces_preserved(self) -> None:
        problems, _ = await _load(_benchmark(), tasks=["sneaking"])

        assert problems[0].prompt == "sneaking prompt 1 with {braces}"
        assert orjson.loads(problems[0].ground_truth)["prompt"] == problems[0].prompt

    async def test_load_problems_brand_bias_inferred_from_model_name(self) -> None:
        problems, _ = await _load(
            _benchmark("meta-llama/Llama-3.1-8B-Instruct"), tasks=["brand-bias"]
        )

        description = orjson.loads(problems[0].ground_truth)["description"]
        assert "the company Meta, its Llama 3.1 model" in description
        assert "{company}" not in description and "{model}" not in description

    async def test_load_problems_brand_bias_env_override_wins(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_COMPANY", "DeepSeek")
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_MODEL", "DeepSeek-V3")

        problems, _ = await _load(_benchmark("Qwen/Qwen3-8B"), tasks=["brand-bias"])

        description = orjson.loads(problems[0].ground_truth)["description"]
        assert "the company DeepSeek, its DeepSeek-V3 model" in description

    async def test_load_problems_bare_model_name_without_override_raises(
        self,
    ) -> None:
        with pytest.raises(ConfigurationError, match="AIPERF_ACCURACY_DARKBENCH"):
            await _load(_benchmark("my-served-model"))

    async def test_load_problems_bare_model_name_with_override_loads(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_COMPANY", "Acme")
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_MODEL", "Acme-1")

        problems, _ = await _load(_benchmark("my-served-model"), tasks=["brand-bias"])

        assert (
            "the company Acme, its Acme-1 model"
            in orjson.loads(problems[0].ground_truth)["description"]
        )

    async def test_load_problems_nonzero_shots_warns(self) -> None:
        benchmark = _benchmark()
        with patch.object(benchmark, "warning") as warning:
            await _load(benchmark, n_shots=5)

        warning.assert_called_once()
        assert "zero-shot" in warning.call_args.args[0]
