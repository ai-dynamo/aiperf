# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for sweep-detail helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEP_DETAIL_HELPERS_PATH = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)
_SWEEP_DETAIL_PAGE_PATH = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "sweep-detail.js"
)


def test_manifest_resolution_prefers_inline_aggregate_over_fallback_sources() -> None:
    script = f"""
        import {{ resolveSweepManifest }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const arrayManifest = resolveSweepManifest({{
          detail: {{
            status: {{ aggregate: {{ children: [{{ name: 'inline-array' }}] }} }},
            children: [{{ name: 'detail-child' }}],
          }},
          archivedChildren: [{{ name: 'archived-child' }}],
        }});
        const envelopeManifest = resolveSweepManifest({{
          detail: {{
            status: {{ aggregate: {{ children: {{ children: [{{ name: 'inline-envelope' }}] }} }} }},
            children: [{{ name: 'detail-child' }}],
          }},
          archivedChildren: [{{ name: 'archived-child' }}],
        }});
        console.log(JSON.stringify({{ arrayManifest, envelopeManifest }}));
    """

    result = json.loads(run_node(script))

    assert result == {
        "arrayManifest": [{"name": "inline-array"}],
        "envelopeManifest": [{"name": "inline-envelope"}],
    }


def test_manifest_resolution_prefers_children_endpoint_before_detail_children() -> None:
    script = f"""
        import {{ resolveSweepManifest }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const manifest = resolveSweepManifest({{
          detail: {{
            status: {{ aggregate: {{ children: [] }} }},
            children: [{{ name: 'stale-detail-child' }}],
          }},
          archivedChildren: [{{ name: 'fresh-endpoint-child' }}],
        }});
        console.log(JSON.stringify(manifest));
    """

    assert json.loads(run_node(script)) == [{"name": "fresh-endpoint-child"}]


def test_variation_rollup_accepts_camel_case_and_snake_case_manifest_fields() -> None:
    script = f"""
        import {{ buildSweepVariations }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = buildSweepVariations({{
          manifest: [
            {{ name: 'sweep-v01-t0', variationIndex: 1, variationLabel: 'camel' }},
            {{ name: 'sweep-v02-t0', variation_index: 2, variation_label: 'snake' }},
          ],
          childSummaries: {{
            'sweep-v01-t0': {{ summary: {{ request_throughput: {{ avg: 10 }} }} }},
            'sweep-v02-t0': {{ summary: {{ request_throughput: {{ avg: 20 }} }} }},
          }},
          cells: {{ cells: [] }},
        }});
        console.log(JSON.stringify(variations.map(v => ({{
          index: v.variation_index,
          label: v.label,
          mean: v.perMetric['request_throughput.avg'].mean,
        }}))));
    """

    assert json.loads(run_node(script)) == [
        {"index": 1, "label": "camel", "mean": 10},
        {"index": 2, "label": "snake", "mean": 20},
    ]


def test_empty_cell_metrics_do_not_create_phantom_trials() -> None:
    script = f"""
        import {{ buildSweepVariations }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = buildSweepVariations({{
          manifest: [{{ name: 'sweep-v03-t0', variationIndex: 3, variationLabel: 'empty' }}],
          childSummaries: {{ 'sweep-v03-t0': {{ summary: null }} }},
          cells: {{ cells: [{{ variationIndex: 3, metrics: {{}} }}] }},
        }});
        console.log(JSON.stringify(variations[0]));
    """

    variation = json.loads(run_node(script))

    assert variation["n_total"] == 1
    assert variation["n_trials"] == 1
    assert variation["perMetric"]["request_throughput.avg"] == {
        "mean": None,
        "std": None,
        "cv": None,
        "n": 0,
    }


def test_archived_children_table_handles_snake_case_child_fields() -> None:
    source = _SWEEP_DETAIL_PAGE_PATH.read_text()

    assert (
        "c.variationLabel ?? c.variation_label ?? c.variationIndex ?? c.variation_index ?? '---'"
        in source
    )
    assert "c.trialIndex ?? c.trial_index ?? '---'" in source
    assert "c.childRunEpoch ?? c.child_run_epoch ?? '---'" in source


def test_historical_sweep_requests_config_and_children_at_their_epochs() -> None:
    """Pinned sweep views must not repopulate from the current sweep or jobs."""
    api_source = (
        _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "lib" / "api.js"
    ).read_text(encoding="utf-8")
    page_source = _SWEEP_DETAIL_PAGE_PATH.read_text(encoding="utf-8")

    assert "getSweepConfig(ns, name, epoch = null)" in api_source
    assert "params.set('epoch', String(epoch))" in api_source
    assert "api.getSweepConfig(namespace, name, epoch)" in page_source
    assert (
        "api.getJob(c.namespace ?? namespace, c.name, c.childRunEpoch ?? c.child_run_epoch ?? null)"
        in page_source
    )


def test_diagnostics_only_show_for_active_sweep_phases() -> None:
    script = f"""
        import {{ shouldShowSweepDiagnostics }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          pending: shouldShowSweepDiagnostics('Pending'),
          running: shouldShowSweepDiagnostics('RUNNING'),
          aggregating: shouldShowSweepDiagnostics('aggregating'),
          archived: shouldShowSweepDiagnostics('Archived'),
          partiallyFailed: shouldShowSweepDiagnostics('PartiallyFailed'),
          unknown: shouldShowSweepDiagnostics(null),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "pending": True,
        "running": True,
        "aggregating": True,
        "archived": False,
        "partiallyFailed": False,
        "unknown": False,
    }


def test_sweep_detail_uses_named_freshness_and_terminal_stop() -> None:
    source = _SWEEP_DETAIL_PAGE_PATH.read_text(encoding="utf-8")

    assert (
        "import { FreshnessPill, StaleBanner } from '../components/freshness.js';"
        in source
    )
    assert "freshness.value['sweep-detail']" in source
    assert "clearFreshnessSource('sweep-detail')" in source
    assert "source: 'sweep-detail'" in source
    assert "stopFreshness('terminal')" in source
    assert '<${StaleBanner} source=${sweepFreshness} label="Sweep detail" />' in source


def test_sweep_detail_clears_route_identity_before_first_poll() -> None:
    source = _SWEEP_DETAIL_PAGE_PATH.read_text(encoding="utf-8")
    effect_block = source.split("useEffect(() => {", 1)[1].split(
        "async function tick", 1
    )[0]

    assert "setDetail(null)" in effect_block
    assert "let firstLoadDone = false" in effect_block
    assert "let firstLoadDone = detail != null" not in source
