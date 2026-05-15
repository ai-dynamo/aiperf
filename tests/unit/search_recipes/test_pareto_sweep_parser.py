# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.search_recipes._pareto_sweep_parser import parse_isl_osl_pairs


@pytest.mark.parametrize(
    "raw,expected",
    [
        param("128/128", [(128, 128)], id="single_pair"),
        param("128/128,256/256", [(128, 128), (256, 256)], id="two_pairs"),
        param(" 128 / 128 , 256/256 ", [(128, 128), (256, 256)], id="whitespace_tolerant"),
        param("128/64,512/256,2048/512", [(128, 64), (512, 256), (2048, 512)], id="asymmetric_pairs"),
    ],
)  # fmt: skip
def test_parse_isl_osl_pairs_valid(raw: str, expected: list[tuple[int, int]]) -> None:
    assert parse_isl_osl_pairs(raw) == expected


@pytest.mark.parametrize(
    "raw,fragment",
    [
        param("", "at least one", id="empty_string"),
        param("128", "expected '<isl>/<osl>'", id="missing_slash"),
        param("128/128/128", "expected '<isl>/<osl>'", id="too_many_slashes"),
        param("abc/128", "must be a positive int", id="non_int_isl"),
        param("128/abc", "must be a positive int", id="non_int_osl"),
        param("0/128", "must be a positive int", id="zero_isl"),
        param("128/-1", "must be a positive int", id="negative_osl"),
        param("128/128,128/128", "duplicate", id="duplicate_pair"),
    ],
)  # fmt: skip
def test_parse_isl_osl_pairs_invalid(raw: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        parse_isl_osl_pairs(raw)
