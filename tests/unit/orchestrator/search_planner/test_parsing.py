# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the --search-space CLI parsing primitive."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.orchestrator.search_planner.parsing import parse_search_space


@pytest.mark.parametrize(
    "raw,expected",
    [
        param(
            ["phases.profiling.concurrency:1,1000:int"],
            [SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=1000, kind="int")],
            id="single_int",
        ),
        param(
            ["x:0.1,5.0"],
            [SearchSpaceDimension(path="x", lo=0.1, hi=5.0, kind="real")],
            id="default_real",
        ),
        param(
            ["a:1,10:int", "b:0,1:real"],
            [
                SearchSpaceDimension(path="a", lo=1, hi=10, kind="int"),
                SearchSpaceDimension(path="b", lo=0, hi=1, kind="real"),
            ],
            id="two_dims",
        ),
    ],
)  # fmt: skip
def test_parse_search_space_valid(raw, expected):
    assert parse_search_space(raw) == expected


@pytest.mark.parametrize(
    "raw,fragment",
    [
        param(["bad-no-colon"], "expected 'path:lo,hi", id="missing_colon"),
        param(["x:1"], "expected 'path:lo,hi", id="missing_comma"),
        param(["x:1,abc"], "could not parse bound", id="non_numeric"),
        param(["x:1,2:weird"], "kind must be 'int' or 'real'", id="bad_kind"),
        param(["x:5,1"], "hi", id="hi_below_lo"),
    ],
)  # fmt: skip
def test_parse_search_space_errors(raw, fragment):
    with pytest.raises(TypeError, match=fragment):
        parse_search_space(raw)


@pytest.mark.parametrize(
    "raw,message",
    [
        param("rate:0,10", r"lower bound must be > 0 \(got 0.0\)", id="rate"),
        param("phases.warmup.smoothness:-1,10", "lower bound must be > 0", id="warmup"),
        param("phases.measured.rate_ramp.duration:0,10", "lower bound must be > 0", id="nested"),
        param("phases.warmup.users:0,10:int", "lower bound must be >= 1", id="users-bound"),
        param("users:1,10", "users:1,50:int", id="users-kind"),
        param("rate_series:1,10", "piecewise-linear", id="rate-series"),
        param("phases.profiling.cancellation.rate:0,101", r"within \[0, 100\]", id="cancellation"),
        param("rate:nan,10", "must be finite", id="non-finite"),
        param("sweep.rate:1,10", "sweep config itself", id="path-validation"),
    ],
)  # fmt: skip
def test_parse_search_space_enforces_field_constraints(raw: str, message: str) -> None:
    with pytest.raises(TypeError, match=message) as excinfo:
        parse_search_space([raw])
    diagnostic = str(excinfo.value)
    assert f"--search-space {raw!r}" in diagnostic
    assert "validation error" not in diagnostic
    assert "input_value" not in diagnostic
    assert "errors.pydantic.dev" not in diagnostic
