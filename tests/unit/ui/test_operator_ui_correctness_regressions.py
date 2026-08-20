# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for operator UI state correctness findings."""

from __future__ import annotations

import re
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
JOB_DETAIL_PATH = UI_ROOT / "pages" / "job-detail.js"
COMPARE_PATH = UI_ROOT / "pages" / "compare.js"


def _source(path: Path) -> str:
    return path.read_text()


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"function {name}\([^)]*\) \{{", source)
    assert match, name
    start = match.end()
    depth = 1
    pos = start
    while pos < len(source) and depth > 0:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    assert depth == 0, name
    return source[start : pos - 1]


def test_job_detail_redirect_effect_is_registered_before_loading_or_error_returns() -> (
    None
):
    source = _source(JOB_DETAIL_PATH)
    redirect_effect = "replaceRoute(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(resolvedEpoch)}`);"
    loading_return = "if (!job && !error) {"
    error_return = "if (error) {"

    assert source.index(redirect_effect) < source.index(loading_return)
    assert source.index(redirect_effect) < source.index(error_return)


def test_job_detail_fetches_config_for_selected_epoch() -> None:
    source = _source(JOB_DETAIL_PATH)

    assert "api.getJobConfig(namespace, name, epoch)" in source


def test_compare_selection_changes_do_not_clear_user_applied_namespace_or_model_filters() -> (
    None
):
    body = _function_body(_source(COMPARE_PATH), "clearDeepLinkContext")

    assert "setNsFilter(new Set())" not in body
    assert "setModelFilter(new Set())" not in body


def test_compare_selection_changes_clear_previous_compare_results_as_stale() -> None:
    source = _source(COMPARE_PATH)
    toggle_body = _function_body(source, "toggleJob")
    recent_body = _function_body(source, "selectRecent")

    assert "setCompareData(null)" in toggle_body
    assert "setCompareError(null)" in toggle_body
    assert "setCompareData(null)" in recent_body
    assert "setCompareError(null)" in recent_body
