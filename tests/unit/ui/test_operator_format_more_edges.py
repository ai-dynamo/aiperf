# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for operator UI format helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

FORMAT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "format.js"
)


def _format_cases() -> dict[str, str | None]:
    script = f"""
        import {{
          fmtBytes,
          fmtInt,
          fmtNumber,
          fmtPercent,
          fmtPrecise,
          fmtThroughput,
        }} from {FORMAT_PATH.as_uri()!r};

        const cases = {{
          numberNullDefault: fmtNumber(null),
          numberUndefinedDefault: fmtNumber(undefined),
          numberCustomFallback: fmtNumber(null, 2, 'n/a'),
          numberNaN: fmtNumber(Number.NaN),
          numberInfinity: fmtNumber(Number.POSITIVE_INFINITY),
          numberNegativeInfinity: fmtNumber(Number.NEGATIVE_INFINITY),
          numberVerySmallPositive: fmtNumber(0.0001234, 2),
          numberVerySmallNegative: fmtNumber(-0.0001234, 2),
          numberSubOne: fmtNumber(0.12345, 2),
          numberLarge: fmtNumber(1234567890.125, 2),
          numberNegative: fmtNumber(-1234.56, 1),

          intNullDefault: fmtInt(null),
          intUndefinedDefault: fmtInt(undefined),
          intCustomFallback: fmtInt(null, 'n/a'),
          intNaN: fmtInt(Number.NaN),
          intInfinity: fmtInt(Number.POSITIVE_INFINITY),
          intRoundedPositive: fmtInt(1234.5),
          intRoundedNegative: fmtInt(-1234.5),

          throughputNull: fmtThroughput(null),
          throughputUndefined: fmtThroughput(undefined),
          throughputNaN: fmtThroughput(Number.NaN),
          throughputInfinity: fmtThroughput(Number.POSITIVE_INFINITY),
          throughputSmall: fmtThroughput(0.004),
          throughputLarge: fmtThroughput(1234567.89),
          throughputNegative: fmtThroughput(-12.34),

          bytesNull: fmtBytes(null),
          bytesUndefined: fmtBytes(undefined),
          bytesNaN: fmtBytes(Number.NaN),
          bytesInfinity: fmtBytes(Number.POSITIVE_INFINITY),
          bytesNegative: fmtBytes(-1),
          bytesZero: fmtBytes(0),
          bytesOne: fmtBytes(1),
          bytesJustBelowKib: fmtBytes(1023),
          bytesOneKib: fmtBytes(1024),
          bytesJustBelowMib: fmtBytes(1024 * 1024 - 1),
          bytesOneMib: fmtBytes(1024 * 1024),
          bytesLarge: fmtBytes(5 * 1024 * 1024),

          preciseNullDefault: fmtPrecise(null),
          preciseCustomFallback: fmtPrecise(null, 'n/a'),
          percentNull: fmtPercent(null),
          percentNaN: fmtPercent(Number.NaN),
          percentTiny: fmtPercent(0.004, 1),
        }};
        console.log(JSON.stringify(cases));
    """
    return json.loads(run_node(script))


def test_null_and_undefined_values_use_declared_fallbacks() -> None:
    cases = _format_cases()

    assert cases["numberNullDefault"] == "---"
    assert cases["numberUndefinedDefault"] == "---"
    assert cases["numberCustomFallback"] == "n/a"
    assert cases["intNullDefault"] == "---"
    assert cases["intUndefinedDefault"] == "---"
    assert cases["intCustomFallback"] == "n/a"
    assert cases["throughputNull"] == "---"
    assert cases["throughputUndefined"] == "---"
    assert cases["bytesNull"] == "---"
    assert cases["bytesUndefined"] == "---"
    assert cases["preciseNullDefault"] == "—"
    assert cases["preciseCustomFallback"] == "n/a"
    assert cases["percentNull"] == "---"


def test_non_finite_values_use_display_fallbacks_not_raw_nan_or_infinity() -> None:
    cases = _format_cases()

    assert cases["numberNaN"] == "---"
    assert cases["numberInfinity"] == "---"
    assert cases["numberNegativeInfinity"] == "---"
    assert cases["intNaN"] == "---"
    assert cases["intInfinity"] == "---"
    assert cases["throughputNaN"] == "---"
    assert cases["throughputInfinity"] == "---"
    assert cases["bytesNaN"] == "---"
    assert cases["bytesInfinity"] == "---"
    assert cases["percentNaN"] == "---"


def test_number_formatting_keeps_tiny_values_visible_and_large_values_grouped() -> None:
    cases = _format_cases()

    assert cases["numberVerySmallPositive"] == "0.00012"
    assert cases["numberVerySmallNegative"] == "-0.00012"
    assert cases["numberSubOne"] == "0.1235"
    assert cases["numberLarge"] == "1,234,567,890.13"
    assert cases["numberNegative"] == "-1,234.6"


def test_integer_formatting_rounds_and_groups_positive_and_negative_values() -> None:
    cases = _format_cases()

    assert cases["intRoundedPositive"] == "1,235"
    assert cases["intRoundedNegative"] == "-1,234"


def test_throughput_formatting_handles_small_large_and_negative_values() -> None:
    cases = _format_cases()

    # Rates are pinned to two decimals across the console. The one exception is
    # a value that two decimals would render as "0.00": "0.004", not "0.00400"
    # -- the widening exists to keep a tiny value visible, not to invent
    # precision, and trailing zeros after a decimal point read as significant.
    assert cases["throughputSmall"] == "0.004"
    assert cases["throughputLarge"] == "1,234,567.89"
    assert cases["throughputNegative"] == "-12.34"


def test_byte_formatting_covers_boundaries_and_rejects_negative_sizes() -> None:
    cases = _format_cases()

    assert cases["bytesNegative"] == "---"
    assert cases["bytesZero"] == "0 B"
    assert cases["bytesOne"] == "1 B"
    assert cases["bytesJustBelowKib"] == "1,023 B"
    assert cases["bytesOneKib"] == "1.0 KiB"
    assert cases["bytesJustBelowMib"] == "1,024.0 KiB"
    assert cases["bytesOneMib"] == "1.0 MiB"
    assert cases["bytesLarge"] == "5.0 MiB"


def test_percent_formatting_keeps_tiny_percentages_visible() -> None:
    cases = _format_cases()

    # Same contract as throughputSmall: stay visible, do not fabricate digits.
    assert cases["percentTiny"] == "0.004%"
