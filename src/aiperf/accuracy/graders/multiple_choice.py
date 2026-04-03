# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig


class MultipleChoiceGrader(BaseGrader):
    """Grades multiple-choice responses using lighteval's ExactMatches approach.

    Ported from lighteval ExactMatches(strip_strings=True):
    both the gold label and model prediction are stripped, then compared
    with direct string equality. No regex extraction is performed.

    lighteval uses stop_sequence=["\\n"] for MMLU, so the model output is
    truncated at the first newline before comparison. We replicate this by
    splitting on "\\n" and taking only the first line.

    Matching:
    - Gold: choices[gold_index] e.g. " A" -> stripped to "A"
    - Pred: first line of model output e.g. " B\\n\\nQuestion:" -> "B"
    - Score: 1 if gold == pred else 0
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        super().__init__(user_config=user_config, **kwargs)

    def extract_answer(self, response_text: str, **kwargs) -> str:
        """Extract the answer: take first line (simulates stop_sequence), then strip."""
        first_line = response_text.split("\n", 1)[0]
        return first_line.strip()

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs
    ) -> GradingResult:
        pred = self.extract_answer(response_text)
        gold = ground_truth.strip()
        correct = pred == gold and pred != ""
        return GradingResult(
            correct=correct,
            confidence=1.0 if correct else 0.0,
            reasoning=f"pred='{pred}', gold='{gold}', match={correct}",
            extracted_answer=pred,
            ground_truth=gold,
        )
