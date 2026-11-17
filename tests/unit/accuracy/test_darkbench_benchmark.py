# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import orjson
import pytest
from pytest import param

from aiperf.accuracy.benchmarks import darkbench
from aiperf.accuracy.benchmarks.darkbench import (
    DATASET_NAME,
    GENERATION_SIZE,
    TASKS,
    DarkBenchBenchmark,
)
from aiperf.accuracy.models import BenchmarkProblem
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
ROWS = [
    {"id": f"{label}-{i}", "input": f"{label} prompt {i} {{braces}}", "target": label}
    for label in LABELS
    for i in (1, 2)
]


def _benchmark(*models: str) -> DarkBenchBenchmark:
    run = make_benchmark_run(
        model_names=list(models) or ["Qwen/Qwen3-8B"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.DARKBENCH},
    )
    return DarkBenchBenchmark(run=run)


async def _load(
    benchmark: DarkBenchBenchmark, tasks: list[str] | None = None, n_shots: int = 0
):
    with patch.object(darkbench, "load_dataset", return_value=ROWS) as mock_load:
        problems = await benchmark.load_problems(
            tasks=tasks, n_shots=n_shots, enable_cot=False
        )
    return problems, mock_load


def test_tasks_and_plugin_metadata_match_reference() -> None:
    meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, "darkbench")
    assert TASKS == LABELS
    assert meta["default_grader"] == "darkbench_overseer"
    assert meta["default_system_prompt"] == "You are a helpful assistant."


def _description(problems: list[BenchmarkProblem]) -> str:
    return orjson.loads(problems[0].ground_truth)["description"]


@pytest.mark.asyncio
class TestDarkBenchLoadProblems:
    async def test_load_problems_maps_labels_and_pins_revision(self) -> None:
        problems, mock_load = await _load(_benchmark())

        assert len(problems) == 12
        mock_load.assert_called_once_with(
            DATASET_NAME,
            split="train",
            revision=Environment.ACCURACY.DARKBENCH_REVISION,
        )
        for problem in problems:
            gt = orjson.loads(problem.ground_truth)
            assert gt["pattern"] == problem.task.replace("-", "_")
            assert gt["prompt"] == problem.prompt and "{braces}" in problem.prompt
            assert problem.raw_messages == [{"role": "user", "content": problem.prompt}]
            assert problem.metadata["generation_size"] == GENERATION_SIZE

    async def test_load_problems_filters_tasks(self) -> None:
        problems, _ = await _load(_benchmark(), tasks=["sneaking", "brand-bias"])

        assert sorted({p.task for p in problems}) == ["brand-bias", "sneaking"]

    @pytest.mark.parametrize("tasks", [param(["brand_bias"], id="snake-case"), param(["nope"], id="unknown")])  # fmt: skip
    async def test_load_problems_unknown_task_raises(self, tasks: list[str]) -> None:
        with pytest.raises(ConfigurationError, match="valid tasks"):
            await _load(_benchmark(), tasks=tasks)

    @pytest.mark.parametrize(
        "model,company,family,expected",
        [
            param("meta-llama/Llama-3.1-8B-Instruct", None, None, "the company Meta, its Llama 3.1 model", id="inferred"),
            param("Qwen/Qwen3-8B", "DeepSeek", "DeepSeek-V3", "the company DeepSeek, its DeepSeek-V3 model", id="override-wins"),
            param("my-served-model", "Acme", "Acme-1", "the company Acme, its Acme-1 model", id="override-bare-name"),
        ],
    )  # fmt: skip
    async def test_load_problems_fills_brand_bias_subject(
        self,
        monkeypatch,
        model: str,
        company: str | None,
        family: str | None,
        expected: str,
    ) -> None:
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_COMPANY", company)
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_MODEL", family)

        problems, _ = await _load(_benchmark(model), tasks=["brand-bias"])

        assert expected in _description(problems)
        assert "{company}" not in _description(problems)

    async def test_load_problems_without_brand_bias_skips_subject(self) -> None:
        problems, _ = await _load(_benchmark("my-served-model"), tasks=["sneaking"])

        assert {p.task for p in problems} == {"sneaking"}

    @pytest.mark.parametrize(
        "models,company,match",
        [
            param(["my-served-model"], None, "AIPERF_ACCURACY_DARKBENCH", id="bare-name"),
            param(["/model"], None, "AIPERF_ACCURACY_DARKBENCH", id="empty-company"),
            param(["my-served-model"], "   ", "AIPERF_ACCURACY_DARKBENCH", id="blank-override"),
            param(["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B"], None, "single model", id="multiple-models"),
        ],
    )  # fmt: skip
    async def test_load_problems_unresolvable_brand_bias_subject_raises(
        self, monkeypatch, models: list[str], company: str | None, match: str
    ) -> None:
        monkeypatch.setattr(Environment.ACCURACY, "DARKBENCH_COMPANY", company)
        with pytest.raises(ConfigurationError, match=match):
            await _load(_benchmark(*models))

    async def test_load_problems_nonzero_shots_warns(self) -> None:
        benchmark = _benchmark()
        with patch.object(benchmark, "warning") as warning:
            await _load(benchmark, n_shots=5)

        assert "zero-shot" in warning.call_args.args[0]
