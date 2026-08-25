# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KPI_CARD = REPO / "src" / "aiperf" / "operator" / "ui" / "components" / "kpi-card.js"
REALTIME_GRID = (
    REPO / "src" / "aiperf" / "operator" / "ui" / "components" / "realtime-kpi-grid.js"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"function {name}\([^)]*\) \{{", source)
    assert match is not None, f"function {name} not found"
    depth = 1
    pos = match.end()
    while pos < len(source) and depth:
        char = source[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        pos += 1
    assert depth == 0, f"function {name} body was not balanced"
    return source[match.end() : pos - 1]


def _const_block(source: str, name: str) -> str:
    match = re.search(rf"const {name} = (\[[\s\S]*?\n\]);", source)
    assert match is not None, f"const {name} block not found"
    return match.group(1)


def _tile_tags(source: str) -> list[str]:
    block = _const_block(source, "TILES")
    return re.findall(r"tag: '([^']+)'", block)


def test_realtime_grid_null_metrics_do_not_fabricate_values_or_slo_chips() -> None:
    source = _source(REALTIME_GRID)
    pick_stat = _function_body(source, "pickStat")
    slo_status = _function_body(source, "sloStatus")

    assert "if (!metric) return { value: null, stat: null };" in pick_stat
    assert "if (value == null || !isFinite(value)) return null;" in slo_status
    assert "if (threshold == null) return null;" in slo_status
    assert "summary?.[spec.tag]" in source


def test_realtime_grid_slo_coloring_uses_metric_direction() -> None:
    source = _source(REALTIME_GRID)
    slo_status = _function_body(source, "sloStatus")

    assert "LARGER_IS_BETTER_SLO_TAGS.has(tag)" in slo_status
    assert "value >= threshold" in slo_status
    assert "value <= threshold" in slo_status
    assert "'output_token_throughput'" in source
    assert "'request_throughput'" in source


def test_realtime_grid_sparkline_matches_displayed_stat_and_filters_bad_samples() -> (
    None
):
    source = _source(REALTIME_GRID)
    pluck = _function_body(source, "pluck")

    assert "typeof v === 'number' && isFinite(v)" in pluck
    assert "pluck(series, primaryStat ?? spec.primary)" in source
    assert "goodputSeries.length > 0" in source
    assert "pluck(ts['good_request_count'], 'avg')" in source


def test_realtime_grid_exposes_fallback_metric_keys_for_throughput_tiles() -> None:
    source = _source(REALTIME_GRID)
    tiles = _const_block(source, "TILES")
    tile_tags = _tile_tags(source)

    assert "output_token_throughput" in tile_tags
    assert "request_throughput" in tile_tags
    assert re.search(r"fallbackTags:\s*\[[^\]]*e2e_output_token_throughput", tiles), (
        "Output-token KPI should fall back to e2e_output_token_throughput when the "
        "canonical output_token_throughput metric is absent."
    )


def test_realtime_grid_formats_negative_numbers_but_rejects_nan_and_infinity() -> None:
    source = _source(REALTIME_GRID)
    format_stat = _function_body(source, "formatStat")
    reliability = _function_body(source, "ReliabilityTile")

    assert "Math.abs(value)" in format_stat
    assert re.search(r"!isFinite\(value\)", format_stat), (
        "formatStat should render NaN/Infinity as the empty placeholder instead of "
        "passing them to fmtNumber/fmtInt."
    )
    assert re.search(r"!isFinite\(rate\)", reliability), (
        "Success-rate shaping should reject NaN/Infinity before fmtPercent."
    )
    assert "hasFiniteCounts" in reliability
    assert "isFinite(goodVal)" in reliability
    assert "isFinite(reqVal)" in reliability


def test_kpi_card_value_and_progress_sanitize_nan_without_hiding_negative_numbers() -> (
    None
):
    source = _source(KPI_CARD)

    assert "Math.max(0, Number(progress))" in source
    assert re.search(r"Number\.isFinite\([^)]*progress", source), (
        "KpiCard progress should clamp NaN/Infinity to a finite width, not emit "
        "style='width: NaN%'."
    )
    assert "const displayValue = (value) =>" in source
    assert re.search(r"Number\.isFinite\([^)]*value", source), (
        "KpiCard should render NaN/Infinity primary values as the placeholder while "
        "still allowing finite negative numbers to be displayed."
    )
    # Was `source.count("${displayValue(value)}") == 2` -- an inlined call in
    # each render path. The call is now hoisted to a single `shown` const so the
    # sanitized string can also drive `kpiValueFontPx`; asserting the const plus
    # two `${shown}` interpolations pins the same invariant (neither path can
    # paint a raw NaN) without pinning where the call happens to sit.
    assert "const shown = displayValue(value);" in source, (
        "KpiCard should sanitize the primary value exactly once, into `shown`."
    )
    assert source.count("${shown}") == 2, (
        "Both KpiCard render paths should render the sanitized `shown`; otherwise "
        "the rich icon path can still display NaN/Infinity."
    )
    assert "${displayValue(value)}" not in source, (
        "A second, un-hoisted sanitization path would let the two render paths drift."
    )


def test_realtime_grid_tile_labels_and_units_stay_consistent() -> None:
    source = _source(REALTIME_GRID)
    tile_tags = _tile_tags(source)
    units_block = re.search(r"const UNIT_BY_TAG = \{([\s\S]*?)\n\};", source)
    assert units_block is not None
    units = set(re.findall(r"\n\s*([a-z0-9_]+): '([^']*)'", units_block.group(1)))
    unit_by_tag = dict(units)

    for tag in tile_tags:
        assert tag in unit_by_tag, f"{tag} tile has no UNIT_BY_TAG entry"

    assert unit_by_tag["output_token_throughput"] == "tok/s"
    assert unit_by_tag["request_throughput"] == "req/s"
    assert unit_by_tag["request_latency"] == "ms"
    assert unit_by_tag["time_to_first_token"] == "ms"
    assert unit_by_tag["inter_token_latency"] == "ms"
