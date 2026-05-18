# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the request-recorder helpers."""

from collections import Counter, defaultdict

import pytest
from aiperf_mock_server.request_recorder import _build_summary, _histogram
from pytest import param


class TestHistogram:
    @pytest.mark.parametrize(
        "values,expected",
        [
            param([], None, id="empty_returns_none"),
            param([42], {"bin_edges": [42.0, 42.0], "counts": [1]}, id="single_value"),
            param(
                [100, 100, 100],
                {"bin_edges": [100.0, 100.0], "counts": [3]},
                id="all_equal",
            ),
        ],
    )  # fmt: skip
    def test_degenerate_inputs(self, values, expected) -> None:
        assert _histogram(values) == expected

    def test_narrow_range_hits_min_bins_floor(self) -> None:
        # range 25..230 (width 205) -> ceil(205/100) = 3, but min_bins=10 wins
        values = list(range(25, 231, 5))  # 42 values spanning 25..230
        hist = _histogram(values)
        assert hist is not None
        assert len(hist["counts"]) == 10
        assert len(hist["bin_edges"]) == 11
        assert hist["bin_edges"][0] == 25.0
        assert hist["bin_edges"][-1] == 230.0
        assert sum(hist["counts"]) == len(values)

    def test_wide_range_hits_max_bin_width_cap(self) -> None:
        # range 207..1821 (width 1614) -> ceil(1614/100) = 17 bins
        values = list(range(207, 1822, 1))  # 1615 values
        hist = _histogram(values)
        assert hist is not None
        assert len(hist["counts"]) == 17
        assert len(hist["bin_edges"]) == 18
        assert hist["bin_edges"][0] == 207.0
        assert hist["bin_edges"][-1] == 1821.0
        assert sum(hist["counts"]) == len(values)

    def test_max_value_lands_in_last_bin(self) -> None:
        # Without the last-bin-closed rule, max would fall just past the last edge
        # and be lost. With it: 1000 must land in bin 9, not vanish.
        hist = _histogram([0, 1000])
        assert hist is not None
        assert len(hist["counts"]) == 10
        assert hist["counts"][0] == 1
        assert hist["counts"][-1] == 1
        assert sum(hist["counts"]) == 2

    def test_bin_widths_are_equal(self) -> None:
        hist = _histogram(list(range(0, 1001)))
        assert hist is not None
        edges = hist["bin_edges"]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        # Width tolerance accounts for float drift in `span / num_bins`; the
        # spec only promises equal-width up to representational precision.
        assert max(widths) - min(widths) < 1e-9


class TestBuildSummary:
    def test_isl_block_has_histogram_and_unique_values(self) -> None:
        isls: dict = defaultdict(list, {"/v1/chat/completions": [10, 20, 30, 10]})
        osls: dict = defaultdict(list, {"/v1/chat/completions": [100, 200, 100]})
        summary = _build_summary(
            total=4,
            isls=isls,
            osls=osls,
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        isl_stats = summary["per_endpoint"]["/v1/chat/completions"]["isl"]
        assert isl_stats["unique_values"] == 3
        assert isinstance(isl_stats["histogram"], dict)
        assert sum(isl_stats["histogram"]["counts"]) == 4

    def test_requested_osl_unique_count(self) -> None:
        osls: dict = defaultdict(list, {"/v1/chat/completions": [16, 32, 16, 64]})
        summary = _build_summary(
            total=4,
            isls=defaultdict(list, {"/v1/chat/completions": [1, 2, 3, 4]}),
            osls=osls,
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        osl_stats = summary["per_endpoint"]["/v1/chat/completions"]["requested_osl"]
        assert osl_stats["unique_values"] == 3
        assert isinstance(osl_stats["histogram"], dict)

    def test_empty_osl_block_is_none(self) -> None:
        # Mimics /v1/embeddings — requested_osl block stays `None` when no values.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/embeddings": [50, 60]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        assert summary["per_endpoint"]["/v1/embeddings"]["requested_osl"] is None
        # ISL block should still get a histogram
        isl_stats = summary["per_endpoint"]["/v1/embeddings"]["isl"]
        assert isinstance(isl_stats["histogram"], dict)
        assert isl_stats["unique_values"] == 2

    def test_min_tokens_block_unchanged(self) -> None:
        # min_tokens deliberately does NOT get the new fields.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/chat/completions": [10, 20]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list, {"/v1/chat/completions": [4, 8]}),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        mn = summary["per_endpoint"]["/v1/chat/completions"]["min_tokens"]
        assert "histogram" not in mn
        assert "unique_values" not in mn
