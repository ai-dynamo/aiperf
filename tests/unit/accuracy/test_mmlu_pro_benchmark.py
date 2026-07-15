# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.mmlu_pro import (
    GENERATION_SIZE,
    MMLUProBenchmark,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU_PRO},
    )


def _test_row(
    qid: int = 1,
    category: str = "math",
    question: str = "What is 2+2?",
    options: list[str] | None = None,
    answer: str = "B",
    cot: str = "",
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question": question,
        "options": options if options is not None else ["3", "4", "5", "N/A"],
        "answer": answer,
        "answer_index": 1,
        "category": category,
        "cot_content": cot,
    }


def _val_row(category: str = "math") -> dict[str, Any]:
    return _test_row(
        qid=99,
        category=category,
        question="1+1?",
        options=["1", "2"],
        answer="B",
        cot="We add one and one to get two. The answer is (B).",
    )


def _fake_split(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


def _fake_dataset(
    test_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]
) -> dict[str, MagicMock]:
    # load_dataset(DATASET_NAME) returns a DatasetDict-like mapping.
    dd = {"test": _fake_split(test_rows), "validation": _fake_split(val_rows)}
    return dd


async def _load(
    test_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    **kw: Any,
) -> list[BenchmarkProblem]:
    with patch(
        "aiperf.accuracy.benchmarks.mmlu_pro.load_dataset",
        return_value=_fake_dataset(test_rows, val_rows),
    ):
        bench = MMLUProBenchmark(run=_run())
        return await bench.load_problems(
            tasks=kw.get("tasks"),
            n_shots=kw.get("n_shots", 0),
            enable_cot=kw.get("enable_cot", True),
        )


@pytest.mark.asyncio
class TestMMLUProBenchmark:
    async def test_cot_zero_shot_prompt_byte_equal(self) -> None:
        problems = await _load([_test_row()], [_val_row()], n_shots=0, enable_cot=True)
        expected = (
            "The following are multiple choice questions (with answers) about "
            "math. Think step by step and then output the answer in the format "
            'of "The answer is (X)" at the end.\n\n'
            "Question: What is 2+2?\nOptions: A. 3\nB. 4\nC. 5\n"
            "Answer: Let's think step by step.\n\n"
        )
        assert problems[0].prompt == expected

    async def test_na_options_filtered(self) -> None:
        problems = await _load(
            [_test_row(options=["3", "4", "5", "N/A"])], [_val_row()], n_shots=0
        )
        assert "N/A" not in problems[0].prompt
        assert "D." not in problems[0].prompt  # only A,B,C survive

    async def test_few_shot_uses_validation_cot(self) -> None:
        problems = await _load([_test_row()], [_val_row()], n_shots=1, enable_cot=True)
        assert "The answer is (B)." in problems[0].prompt  # few-shot cot present

    async def test_non_cot_query_has_bare_answer_trailer(self) -> None:
        problems = await _load([_test_row()], [_val_row()], n_shots=0, enable_cot=False)
        assert problems[0].prompt.endswith("Answer: \n\n")
        assert "Let's think step by step." not in problems[0].prompt

    async def test_non_cot_instruction_drops_cot_directive(self) -> None:
        # --accuracy-no-enable-cot must not request chain-of-thought.
        problems = await _load([_test_row()], [_val_row()], n_shots=1, enable_cot=False)
        p = problems[0].prompt
        assert "Think step by step" not in p
        assert "The answer is (X)" not in p
        # Non-CoT few-shot answer is the bare gold letter, not "The answer is (B)."
        assert "The answer is (B)." not in p
        assert "Answer: B\n\n" in p

    async def test_ground_truth_is_gold_letter(self) -> None:
        problems = await _load([_test_row(answer="B")], [_val_row()])
        assert problems[0].ground_truth == "B"

    async def test_single_user_message(self) -> None:
        problems = await _load([_test_row()], [_val_row()])
        msgs = problems[0].raw_messages
        assert msgs and len(msgs) == 1 and msgs[0]["role"] == "user"
        assert msgs[0]["content"] == problems[0].prompt

    async def test_metadata_generation_size(self) -> None:
        problems = await _load([_test_row()], [_val_row()])
        assert problems[0].metadata["generation_size"] == GENERATION_SIZE
        assert GENERATION_SIZE == 4000

    async def test_task_is_category(self) -> None:
        problems = await _load(
            [_test_row(category="physics")], [_val_row(category="physics")]
        )
        assert problems[0].task == "physics"

    async def test_category_filter(self) -> None:
        rows = [_test_row(category="math"), _test_row(category="physics")]
        vals = [_val_row("math"), _val_row("physics")]
        problems = await _load(rows, vals, tasks=["math"])
        assert {p.task for p in problems} == {"math"}

    async def test_unknown_category_raises(self) -> None:
        with pytest.raises(ValueError, match="nonsense"):
            await _load([_test_row()], [_val_row()], tasks=["nonsense"])
