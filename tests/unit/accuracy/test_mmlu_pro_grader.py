# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.accuracy.graders.mmlu_pro import MMLUProGrader
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_grader() -> MMLUProGrader:
    return MMLUProGrader(
        run=make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.MMLU_PRO},
        )
    )


@pytest.mark.asyncio
class TestMMLUProGrade:
    async def test_tier1_clean(self) -> None:
        r = await _make_grader().grade("... The answer is (F).", "F")
        assert r.correct and not r.unparsed and r.extracted_answer == "F"

    async def test_tier2_flags_unparsed(self) -> None:
        r = await _make_grader().grade("Reasoning.\nAnswer: F", "F")
        assert r.correct and r.unparsed

    async def test_tier3_flags_unparsed(self) -> None:
        r = await _make_grader().grade("I weigh options; ultimately G", "G")
        assert r.correct and r.unparsed

    async def test_full_range_j(self) -> None:
        r = await _make_grader().grade("The answer is (J).", "J")
        assert r.correct

    async def test_wrong_letter(self) -> None:
        r = await _make_grader().grade("The answer is (A).", "B")
        assert not r.correct and not r.unparsed

    async def test_no_match_unparsed(self) -> None:
        r = await _make_grader().grade("no idea", "C")
        assert not r.correct and r.unparsed and r.extracted_answer == ""

    async def test_reasoning_plus_content_concatenated(self) -> None:
        # Simulates ReasoningResponseData.get_text() == reasoning + content:
        # a long reasoning trace precedes the final answer. This is the exact
        # shape that produced the 0/200 MMLU incident; MMLU-Pro must handle it.
        text = "We need to compute the product of roots. " * 50 + "The answer is (D)."
        r = await _make_grader().grade(text, "D")
        assert r.correct and not r.unparsed

    async def test_bare_letter_non_cot_is_clean(self) -> None:
        # --accuracy-no-enable-cot yields a bare first-line letter; clean parse.
        r = await _make_grader().grade("B", "B")
        assert r.correct and not r.unparsed and r.extracted_answer == "B"

    async def test_bare_letter_j_non_cot(self) -> None:
        r = await _make_grader().grade("J\n", "J")
        assert r.correct and not r.unparsed
