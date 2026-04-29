# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aiperf.config.parsing helpers."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.parsing import parse_int_or_int_list


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        param(None, None, id="none"),
        param(10, 10, id="scalar-int"),
        param("10", 10, id="scalar-str"),
        param("10,20", [10, 20], id="csv-two"),
        param("10, 20, 30", [10, 20, 30], id="csv-three-spaces"),
        param("10,", 10, id="csv-trailing-comma-collapses"),
        param([10], 10, id="single-list-collapses"),
        param([10, 20], [10, 20], id="multi-list"),
        param([10, 20, 30], [10, 20, 30], id="multi-list-three"),
    ],
)  # fmt: skip
def test_parse_int_or_int_list_truth_table(
    value: object, expected: int | list[int] | None
) -> None:
    assert parse_int_or_int_list(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        param("not-a-number", id="gibberish"),
        param({"x": 1}, id="dict-rejected"),
        param(True, id="bool-rejected"),
    ],
)  # fmt: skip
def test_parse_int_or_int_list_invalid_raises(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_int_or_int_list(value)
