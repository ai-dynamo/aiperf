# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from unittest.mock import call, patch

import orjson
import pytest
from pytest import param

from aiperf.accuracy.benchmarks import impossible_lcb
from aiperf.accuracy.benchmarks.impossible_lcb import (
    DATASET_NAME,
    GENERATION_SIZE,
    INSTRUCTION,
    SPLITS,
    ImpossibleLCBBenchmark,
    build_prompt,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from aiperf.plugin import plugins
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType, PluginType
from tests.unit.conftest import make_benchmark_run

# sha256 of INSTRUCTION + " Modification of tests is not allowed and will be
# reverted." + INSTRUCTION_SUFFIX in safety-research/impossiblebench@d4f9a14
# src/impossiblebench/livecodebench_tasks.py.
REFERENCE_INSTRUCTION_SHA256 = (
    "35c605a6d9ffc37fefa00c2ef96e371d29e6a5f9559d2a028ab81624362f80a0"
)


def _row(task_id: str = "lcbhard_0") -> dict[str, str]:
    return {
        "task_id": task_id,
        "prompt": 'def add(a, b):\n    """Return a + b."""',
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        "original_test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        "impossible_type": "original",
        "entry_point": "add",
    }


ROWS = [_row("lcbhard_0"), _row("lcbhard_35"), _row("lcbhard_1")]


def _benchmark() -> ImpossibleLCBBenchmark:
    run = make_benchmark_run(
        model_names=["Qwen/Qwen3-0.6B"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.IMPOSSIBLE_LCB},
    )
    return ImpossibleLCBBenchmark(run=run)


async def _load(
    tasks: list[str] | None = None,
    n_shots: int = 0,
    benchmark: ImpossibleLCBBenchmark | None = None,
) -> tuple[list[BenchmarkProblem], object]:
    benchmark = benchmark or _benchmark()
    with patch.object(impossible_lcb, "load_dataset", return_value=ROWS) as mock_load:
        problems = await benchmark.load_problems(
            tasks=tasks, n_shots=n_shots, enable_cot=False
        )
    return problems, mock_load


def test_plugin_metadata_pairs_impossible_code_execution() -> None:
    meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, "impossible_lcb")
    assert meta["default_grader"] == "impossible_code_execution"
    assert meta["default_n_shots"] == 0
    assert "default_system_prompt" not in meta


def test_instruction_matches_reference_nomod_prompt() -> None:
    assert (
        hashlib.sha256(INSTRUCTION.encode()).hexdigest() == REFERENCE_INSTRUCTION_SHA256
    )


def test_build_prompt_wraps_stub_and_tests_like_reference() -> None:
    row = _row()
    assert build_prompt(row) == (
        f"{INSTRUCTION}\n\n```\n{row['prompt']}\n\n{row['test']}"
        "\n\n# Use check(add) to run tests.\n```"
    )


@pytest.mark.asyncio
class TestImpossibleLCBLoadProblems:
    async def test_load_problems_packs_grader_inputs_into_ground_truth(self) -> None:
        problems, _ = await _load(["conflicting"])

        problem = problems[0]
        assert problem.task == "conflicting"
        assert orjson.loads(problem.ground_truth) == {
            "task_id": "lcbhard_0",
            "split": "conflicting",
            "entry_point": "add",
            "prompt": ROWS[0]["prompt"],
            "test": ROWS[0]["test"],
        }
        assert problem.prompt == build_prompt(ROWS[0])
        assert problem.raw_messages == [{"role": "user", "content": problem.prompt}]
        assert problem.metadata == {
            "task_id": "lcbhard_0",
            "generation_size": GENERATION_SIZE,
        }

    async def test_load_problems_all_splits_and_pinned_revision(self) -> None:
        problems, mock_load = await _load()

        assert mock_load.call_args_list == [
            call(
                DATASET_NAME,
                split=split,
                revision=Environment.ACCURACY.IMPOSSIBLE_LCB_REVISION,
            )
            for split in SPLITS
        ]
        assert [p.task for p in problems] == [s for s in SPLITS for _ in range(2)]

    async def test_load_problems_default_exclude_applies_to_every_split(self) -> None:
        problems, _ = await _load()

        assert "lcbhard_35" in Environment.ACCURACY.IMPOSSIBLE_LCB_EXCLUDE
        assert {p.metadata["task_id"] for p in problems} == {"lcbhard_0", "lcbhard_1"}

    @pytest.mark.parametrize(
        "exclude,kept",
        [
            param("", 3, id="empty-string-keeps-all"),
            param("lcbhard_0, lcbhard_1", 1, id="csv"),
            param('["lcbhard_1"]', 2, id="json"),
        ],
    )  # fmt: skip
    async def test_load_problems_exclude_env_override(
        self, monkeypatch: pytest.MonkeyPatch, exclude: str, kept: int
    ) -> None:
        monkeypatch.setenv("AIPERF_ACCURACY_IMPOSSIBLE_LCB_EXCLUDE", exclude)
        settings = type(Environment.ACCURACY)()
        monkeypatch.setattr(
            Environment.ACCURACY,
            "IMPOSSIBLE_LCB_EXCLUDE",
            settings.IMPOSSIBLE_LCB_EXCLUDE,
        )

        problems, _ = await _load(["original"])

        assert len(problems) == kept

    async def test_load_problems_unknown_split_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown impossible_lcb task"):
            await _load(["impossible"])

    async def test_load_problems_dedups_tasks(self) -> None:
        _, mock_load = await _load(["oneoff", "oneoff"])

        assert mock_load.call_count == 1

    async def test_load_problems_nonzero_shots_warns(self) -> None:
        benchmark = _benchmark()
        with patch.object(benchmark, "warning") as warning:
            await _load(["original"], 3, benchmark)

        assert "zero-shot" in warning.call_args.args[0]
