# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``MMLUBenchmark`` CoT rejection and non-CoT prompt shape.

lighteval's MMLU reference has no chain-of-thought variant: the prompt ends
with ``"Answer:"``, ``generation_size=5`` caps the response at five tokens,
and ``MultipleChoiceGrader`` grades only the first line. Before the guard,
``--accuracy-enable-cot`` inserted "Let's think step by step." into the
prompt while the rest of the pipeline still truncated and first-line-graded
the response — silently mis-grading every problem. These tests pin the
fail-loud behavior and the unchanged non-CoT path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from aiperf.accuracy.benchmarks.mmlu import (
    GENERATION_SIZE,
    TASK_NAME,
    MMLUBenchmark,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    )


def _make_row() -> dict[str, object]:
    return {
        "question": "What is 2 + 2?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
    }


class TestEnableCotRejected:
    """``enable_cot=True`` must fail loud (validator-gate convention),
    matching the LCB loader, instead of silently mis-grading every response."""

    @pytest.mark.asyncio
    async def test_load_problems_enable_cot_raises_not_implemented(self) -> None:
        bench = MMLUBenchmark(run=_make_run())
        with pytest.raises(
            NotImplementedError, match=rf"^{TASK_NAME}: .*--accuracy-enable-cot"
        ):
            await bench.load_problems(
                tasks=["abstract_algebra"], n_shots=0, enable_cot=True
            )

    @pytest.mark.asyncio
    async def test_enable_cot_gate_fires_before_dataset_download(self) -> None:
        bench = MMLUBenchmark(run=_make_run())
        with (
            patch("aiperf.accuracy.benchmarks.mmlu.load_dataset") as mock_load,
            pytest.raises(NotImplementedError),
        ):
            await bench.load_problems(tasks=None, n_shots=5, enable_cot=True)
        mock_load.assert_not_called()


class TestNonCotPathUnchanged:
    """With ``enable_cot=False`` the loader still produces lighteval-shaped,
    first-line-gradable prompts."""

    @pytest.mark.asyncio
    async def test_load_problems_builds_answer_terminated_prompt(self) -> None:
        fake_ds = {"test": [_make_row()]}
        bench = MMLUBenchmark(run=_make_run())
        with patch(
            "aiperf.accuracy.benchmarks.mmlu.load_dataset", return_value=fake_ds
        ):
            problems = await bench.load_problems(
                tasks=["abstract_algebra"], n_shots=0, enable_cot=False
            )

        assert len(problems) == 1
        problem = problems[0]
        assert problem.prompt.endswith("\nAnswer:")
        assert "step by step" not in problem.prompt
        assert problem.ground_truth == " B"
        assert problem.metadata is not None
        assert problem.metadata["generation_size"] == GENERATION_SIZE

    @pytest.mark.asyncio
    async def test_chat_messages_final_turn_ends_with_answer_cue(self) -> None:
        fake_ds = {"test": [_make_row()]}
        bench = MMLUBenchmark(run=_make_run())
        with patch(
            "aiperf.accuracy.benchmarks.mmlu.load_dataset", return_value=fake_ds
        ):
            problems = await bench.load_problems(
                tasks=["abstract_algebra"], n_shots=0, enable_cot=False
            )

        raw_messages = problems[0].raw_messages
        assert raw_messages is not None
        final = raw_messages[-1]
        assert final["role"] == "user"
        assert final["content"].endswith("\nAnswer:")
        assert "step by step" not in final["content"]
