# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from TIGER-AI-Lab/MMLU-Pro
# (https://github.com/TIGER-AI-Lab/MMLU-Pro), licensed under Apache-2.0.

"""Shared multiple-choice answer extraction.

Adapts the TIGER-AI-Lab MMLU-Pro 3-tier extraction cascade
(evaluate_from_api.py: extract_answer / extract_again / extract_final),
parameterized on the letter range so MMLU (A-D) and MMLU-Pro (A-J) share it.

Tiers 1 and 2 reproduce upstream verbatim. Tier 3 refines upstream's
``[A-J](?=[^A-J]*$)`` (the last in-range char anywhere, including mid-word) to
the last *standalone* in-range letter, avoiding spurious matches inside words
like "FLAG". Tier 3 is always flagged as an unparsed fallback, so this refinement
never affects a cleanly parsed score.
"""

from __future__ import annotations

import re
from functools import cache


@cache
def _compiled(letters: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    cls = f"[{letters}]"
    return (
        re.compile(rf"answer is \(?({cls})\)?"),
        re.compile(rf".*[aA]nswer:\s*({cls})", re.DOTALL),
        re.compile(rf"\b{cls}\b(?!.*\b{cls}\b)", re.DOTALL),
    )


def extract_choice_letter(text: str, letters: str = "ABCDEFGHIJ") -> tuple[str, int]:
    """Extract a choice letter from model output.

    Returns (letter, tier): tier 1 = clean "answer is (X)", 2 = "Answer: X"
    fallback, 3 = last lone in-range letter, 0 = no match (letter "").

    Tier 1 returns the LAST "answer is (X)" match: chain-of-thought traces may
    mention an intermediate answer ("maybe the answer is A ... actually the
    answer is B") before the final one, and the model is instructed to put the
    final answer last. Tiers 2 and 3 are already last-oriented.
    """
    tier1, tier2, tier3 = _compiled(letters)
    matches = list(tier1.finditer(text))
    if matches:
        return matches[-1].group(1), 1
    if (m := tier2.search(text)) is not None:
        return m.group(1), 2
    if (m := tier3.search(text)) is not None:
        return m.group(0), 3
    return "", 0
