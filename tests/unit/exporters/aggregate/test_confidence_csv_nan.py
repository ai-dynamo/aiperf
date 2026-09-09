# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NaN handling in the confidence aggregate CSV exporter."""

from __future__ import annotations

import math

from aiperf.exporters.aggregate.aggregate_confidence_csv_exporter import (
    AggregateConfidenceCsvExporter,
)


class TestConfidenceCsvNaN:
    """NaN must be blanked, not written as the literal string `nan`.

    NaN compares equal to nothing, so it fell past both infinity branches and
    reached the float format. The sibling sweep exporter blanks non-finite
    values and its docstring names this exact defect.
    """

    def _fmt(self, value: float) -> str:
        return AggregateConfidenceCsvExporter._format_number(None, value)

    def test_nan_is_blank(self) -> None:
        assert self._fmt(float("nan")) == ""

    def test_infinities_still_render(self) -> None:
        assert self._fmt(float("inf")) == "inf"
        assert self._fmt(float("-inf")) == "-inf"

    def test_finite_values_unchanged(self) -> None:
        assert self._fmt(1.5) == "1.50"
        assert self._fmt(0.0) == "0.00"

    def test_no_output_parses_back_as_nan(self) -> None:
        for value in (float("nan"), float("inf"), 2.0, None):
            rendered = AggregateConfidenceCsvExporter._format_number(None, value)
            if rendered not in ("", "inf", "-inf"):
                assert not math.isnan(float(rendered))
