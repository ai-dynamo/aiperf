# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from TIGER-AI-Lab/MMLU-Pro
# (https://github.com/TIGER-AI-Lab/MMLU-Pro), licensed under Apache-2.0.

"""MMLU-Pro grader.

Ports TIGER-AI-Lab MMLU-Pro's answer extraction (A-J, whole-text
"answer is (X)" cascade) via the shared extract_choice_letter helper.
"""

from __future__ import annotations

from typing import Any

from aiperf.accuracy.graders._choice_extract import extract_choice_letter
from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult

LETTERS = "ABCDEFGHIJ"


class MMLUProGrader(BaseGrader):
    """Grades MMLU-Pro CoT responses by extracting the final A-J letter."""

    def _extract(self, response_text: str) -> tuple[str, bool]:
        # Non-CoT output is a bare letter on the first line; treat that as a
        # clean parse before falling to the whole-text "answer is (X)" cascade.
        first_line = response_text.split("\n", 1)[0].strip()
        if len(first_line) == 1 and first_line in LETTERS:
            return first_line, False
        letter, tier = extract_choice_letter(response_text, LETTERS)
        return letter, tier != 1  # unparsed when a fallback tier (or no match) used

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        letter, _ = self._extract(response_text)
        return letter

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        pred, unparsed = self._extract(response_text)
        gold = ground_truth.strip()
        correct = pred == gold and pred != ""
        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=1.0 if correct else 0.0,
            reasoning=(
                f"extracted '{pred}'"
                + (" (fallback)" if unparsed else "")
                + f"; ground_truth '{gold}'; match={correct}"
            ),
            extracted_answer=pred,
            ground_truth=gold,
        )
