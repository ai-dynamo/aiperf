# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial static/pure tests for operator UI form controls."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
JOBS_PATH = UI_DIR / "pages" / "jobs.js"
COMPARE_PATH = UI_DIR / "pages" / "compare.js"
COMPARE_FILTERS_PATH = UI_DIR / "pages" / "compare-filters.js"
LAUNCH_PATH = UI_DIR / "pages" / "launch.js"
EPOCH_SELECTOR_PATH = UI_DIR / "components" / "epoch-selector.js"
METRIC_SELECTOR_PATH = UI_DIR / "components" / "metric-selector.js"


_JOBS_LITERAL = """[
  { job_id: 'a', namespace: 'default', model: 'meta/llama-3', endpoint: '/v1/chat' },
  { job_id: 'b', namespace: 'staging', model: null, endpoint: null },
]"""


def _source(path: Path) -> str:
    return path.read_text()


def test_empty_search_values_do_not_filter_compare_job_picker() -> None:
    script = f"""
        import {{ applyJobFilters }} from {COMPARE_FILTERS_PATH.as_uri()!r};
        const jobs = {_JOBS_LITERAL};
        const cases = ['', null, undefined];
        const lengths = cases.map((search) => applyJobFilters(jobs, {{
          nsFilter: new Set(),
          modelFilter: new Set(),
          endpointFilter: new Set(),
          search,
        }}).length);
        console.log(JSON.stringify(lengths));
    """

    assert json.loads(run_node(script)) == [2, 2, 2]


def test_whitespace_only_search_is_treated_as_empty() -> None:
    source = _source(COMPARE_PATH)

    assert "const normalizedSearch = search.trim();" in source
    assert "search: normalizedSearch" in source
    assert "normalizedSearch.length > 0" in source


def test_jobs_url_backed_filter_state_and_clear_preserve_sort_preference() -> None:
    source = _source(JOBS_PATH)

    assert "const q = query.value;" in source
    assert "const phaseKey = q.phase ?? null;" in source
    assert "const ns = q.ns ?? '';" in source
    assert "const modelFilter = q.model ?? '';" in source
    assert "const endpointFilter = q.endpoint ?? '';" in source
    assert "const sort = parseSort(q.sort);" in source
    assert "const urlQ = q.q ?? '';" in source
    assert "setQuery({ q: searchText });" in source
    assert (
        "setQuery({ q: undefined, ns: undefined, phase: undefined, model: undefined, endpoint: undefined });"
        in source
    )
    assert (
        "sort: undefined"
        not in source[
            source.index("function clearFilters") : source.index(
                "function chipKeyHandler"
            )
        ]
    )


def test_jobs_enter_key_commits_pending_search_without_waiting_for_debounce() -> None:
    source = _source(JOBS_PATH)

    assert "if (e.key === 'Enter' && searchText !== urlQ)" in source
    assert "e.preventDefault(); setQuery({ q: searchText });" in source


def test_keyboard_shortcuts_cover_button_like_chips_and_launch_editor() -> None:
    jobs = _source(JOBS_PATH)
    compare = _source(COMPARE_PATH)
    launch = _source(LAUNCH_PATH)

    assert "if (e.key === 'Enter' || e.key === ' ')" in jobs
    assert "onkeydown=${chipKeyHandler(() => setQuery({ ns: undefined }))}" in jobs
    assert "onkeydown=${chipKeyHandler(() => setQuery({ model: undefined }))}" in jobs
    assert (
        "onkeydown=${chipKeyHandler(() => setQuery({ endpoint: undefined }))}" in jobs
    )

    assert "if (e.key === 'Enter' || e.key === ' ')" in compare
    assert 'data-testid="compare-clear-filters"' in compare
    assert 'data-testid="compare-chips-toggle"' in compare

    assert "if ((e.ctrlKey || e.metaKey) && e.key === 'Enter')" in launch
    assert "if (state.kind !== 'submitting') launch();" in launch


def test_metric_selector_rejects_values_that_are_not_options() -> None:
    source = _source(METRIC_SELECTOR_PATH)

    assert "METRICS.find((m) => m.value === e.target.value)" in source
    assert "STATS.find((s) => s.value === e.target.value)" in source


def test_epoch_selector_maps_latest_sentinel_and_preserves_epoch_values() -> None:
    source = _source(EPOCH_SELECTOR_PATH)

    assert "value=${current ?? '__latest__'}" in source
    assert "onPick(v === '__latest__' ? undefined : v);" in source
    assert '<option value="__latest__">' in source
    assert "<option key=${e.epoch} value=${e.epoch}>" in source


def test_launch_submit_handler_rechecks_disabled_state_inside_action() -> None:
    source = _source(LAUNCH_PATH)
    handler = source[
        source.index("async function launch") : source.index("function copyYaml")
    ]

    assert "const canSubmit = state.kind !== 'submitting'" in source
    assert "&& state.kind !== 'ok'" in source
    assert "disabled=${!canSubmit}" in source
    assert "const submitGuardRef = useRef({ canSubmit, yaml });" in source
    assert "const guard = submitGuardRef.current;" in handler
    assert "if (!guard.canSubmit) return;" in handler
    assert "const yaml = guard.yaml;" in handler
    assert "manifest = parseLaunchManifest(yaml);" in handler
    assert "submitGuardRef.current = { canSubmit: false, yaml };" in handler


def test_compare_disabled_submit_bypass_is_guarded_in_handler() -> None:
    source = _source(COMPARE_PATH)

    assert "disabled=${selectedKeys.length < 2 || comparing}" in source
    handler = source[
        source.index("async function handleCompare") : source.index(
            "const facets = useMemo"
        )
    ]
    assert "if (selectedKeys.length < 2) return;" in handler
    assert "setComparing(true);" in handler


def test_compare_clear_filters_resets_all_visible_filter_state() -> None:
    source = _source(COMPARE_PATH)
    handler = source[
        source.index("function clearFilters") : source.index(
            "function toggleFacetExpanded"
        )
    ]

    assert "setNsFilter(new Set());" in handler
    assert "setModelFilter(new Set());" in handler
    assert "setEndpointFilter(new Set());" in handler
    assert "setSearch('');" in handler
