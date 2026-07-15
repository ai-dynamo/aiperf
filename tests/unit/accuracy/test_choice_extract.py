# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.accuracy.graders._choice_extract import extract_choice_letter


@pytest.mark.parametrize(
    "text,expected",
    [
        param("The answer is (D).", ("D", 1), id="tier1_parens"),
        param("The answer is B", ("B", 1), id="tier1_no_parens"),
        param("Reasoning...\nAnswer: F", ("F", 2), id="tier2_answer_colon"),
        param("blah C blah then finally H", ("H", 3), id="tier3_last_lone"),
        param("no letters here", ("", 0), id="no_match"),
    ],
)  # fmt: skip
def test_extract_cascade_tiers(text, expected):
    assert extract_choice_letter(text, "ABCDEFGHIJ") == expected


def test_range_limits_to_abcd() -> None:
    # 'H' is out of A-D range; tier-1 pattern must not match it.
    assert extract_choice_letter("The answer is (H).", "ABCD") == ("", 0)


def test_range_abcd_matches_in_range() -> None:
    assert extract_choice_letter("The answer is (C).", "ABCD") == ("C", 1)


def test_tier1_returns_last_answer_is_match() -> None:
    # Intermediate musing before the final answer -> take the last one.
    text = "Maybe the answer is (A). Let me reconsider. The answer is (C)."
    assert extract_choice_letter(text, "ABCDEFGHIJ") == ("C", 1)
