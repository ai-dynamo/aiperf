# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig

# Matches a lone A-D letter not adjacent to a word character, e.g. "B.", "**B**", "(B)", "answer is B".
_LETTER_RE = re.compile(r"(?<!\w)([A-D])(?!\w)")

_VALID_CHOICES = frozenset({"A", "B", "C", "D"})


class MultipleChoiceGrader(BaseGrader):
    """Grades multiple-choice responses using lighteval's ExactMatches approach.

    Ported from lighteval ExactMatches(strip_strings=True):
    both the gold label and model prediction are stripped, then compared
    with direct string equality.

    lighteval uses stop_sequence=["\\n"] for MMLU, so the model output is
    truncated at the first newline before comparison. We replicate this by
    splitting on "\\n" and taking only the first line.

    When the first-line result is not a bare A-D letter (e.g. "The answer is B."),
    a regex fallback extracts the first lone A-D letter. Responses that required
    the fallback are flagged as unparsed in GradingResult.

    Matching:
    - Gold: choices[gold_index] e.g. " A" -> stripped to "A"
    - Pred: extracted answer e.g. " B\\n\\nQuestion:" -> "B"
    - Score: 1 if gold == pred else 0
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

    def _extract_with_flag(self, response_text: str) -> tuple[str, bool]:
        """Return (answer, unparsed). unparsed=True when regex fallback was used."""
        first_line = response_text.split("\n", 1)[0].strip()
        if first_line in _VALID_CHOICES:
            return first_line, False
        m = _LETTER_RE.search(first_line)
        if m:
            return m.group(1), True
        return first_line, True

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Extract the answer: take first line, strip; fall back to regex for non-conforming output."""
        answer, _ = self._extract_with_flag(response_text)
        return answer

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        pred, unparsed = self._extract_with_flag(response_text)
        gold = ground_truth.strip()
        correct = pred == gold and pred != ""
        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=1.0 if correct else 0.0,
            reasoning=(
                f"first-line-of-response extracted to '{pred}'"
                + (" (regex fallback)" if unparsed else "")
                + f"; ground_truth stripped to '{gold}'; match={correct}"
            ),
            extracted_answer=pred,
            ground_truth=gold,
        )
