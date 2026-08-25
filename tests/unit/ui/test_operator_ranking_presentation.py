# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation-integrity regressions for the operator UI analytics pages.

Each test here pins a defect where the rendering asserted something the data
did not support: a ranking whose #1 was the worst run, a quadrant annotation
placed on the wrong side of its own axis, a Pareto frontier drawn across runs
that are not alternatives for one another, and a regression delta computed
between two runs that did not execute the same workload.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS_IN_TEMPLATE, run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
LEADERBOARD_PATH = UI_ROOT / "pages" / "leaderboard.js"
DASHBOARD_PATH = UI_ROOT / "pages" / "dashboard.js"
COMPARE_PATH = UI_ROOT / "pages" / "compare.js"
COMPARE_EPOCHS_PATH = UI_ROOT / "pages" / "compare-epochs.js"


def _module_source(path: Path) -> str:
    """Read a page module with its imports and exports stripped for eval."""
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"^import .*;\n", "", source, flags=re.MULTILINE)
    source = re.sub(r"^export \{[^}]*\};\n", "", source, flags=re.MULTILINE)
    return re.sub(r"^export (function|const|let) ", r"\1 ", source, flags=re.MULTILINE)


def _leaderboard_eval(expr: str) -> str:
    """Evaluate `expr` against leaderboard.js internals with stubbed deps."""
    source = _module_source(LEADERBOARD_PATH).replace(
        "export function Leaderboard", "function Leaderboard"
    )
    return run_node(
        f"""
        const palette = new Proxy({{}}, {{ get: () => '#000000' }});
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        function html(strings, ...values) {{ return {{ strings, values }}; }}
        {source}
        console.log(JSON.stringify({expr}));
        """
    )


# (job_id, value) drawn from a latency leaderboard: `slow-run` is the worst.
_LATENCY_ROWS = [
    {"job_id": "slow-run", "value": 9100.0},
    {"job_id": "mid-run", "value": 480.0},
    {"job_id": "fast-run", "value": 61.0},
]


def test_leaderboard_ranks_latency_metrics_lowest_first() -> None:
    """Rank 1 on a lower-is-better metric must be the fastest run.

    The API sorts descending unless told otherwise and the UI never sends
    `order`, so before the fix `slow-run` took the gold #1 row while the
    caption beside it read "lower = better".
    """
    ordered = json.loads(
        _leaderboard_eval(
            f"rankEntries({json.dumps(_LATENCY_ROWS)}, 'request_latency')"
            ".map((e) => e.job_id)"
        )
    )
    assert ordered == ["fast-run", "mid-run", "slow-run"]


def test_leaderboard_ranks_throughput_metrics_highest_first() -> None:
    """Higher-is-better metrics keep descending order."""
    rows = [
        {"job_id": "low", "value": 3.0},
        {"job_id": "high", "value": 91.0},
        {"job_id": "mid", "value": 40.0},
    ]
    ordered = json.loads(
        _leaderboard_eval(
            f"rankEntries({json.dumps(rows)}, 'request_throughput')"
            ".map((e) => e.job_id)"
        )
    )
    assert ordered == ["high", "mid", "low"]


def test_leaderboard_lower_is_better_covers_unlisted_latency_metrics() -> None:
    """Direction detection must not depend on a hand-maintained whitelist."""
    checks = json.loads(
        _leaderboard_eval(
            "["
            "isLowerBetter('request_latency'),"
            "isLowerBetter('inter_token_latency'),"
            "isLowerBetter('time_to_first_token'),"
            "isLowerBetter('time_to_second_token'),"
            "isLowerBetter('credit_to_start_latency'),"
            "isLowerBetter('request_throughput'),"
            "isLowerBetter('output_token_throughput_per_user'),"
            "isLowerBetter(null)"
            "]"
        )
    )
    assert checks == [True, True, True, True, True, False, False, False]


def _dashboard_quadrant_labels(plugin_options_js: str = "{ enabled: true }") -> str:
    """Run the dashboard's quadrant plugin against a fake canvas context.

    Returns the JSON list of ``{text, x, y}`` fillText calls. The chart area is
    100..500 horizontally and 0..300 vertically, so "top" is y≈0 (the MAXIMUM
    latency on a non-reversed Chart.js y-axis) and "bottom" is y≈300.
    """
    source = _module_source(DASHBOARD_PATH)
    return run_node(
        f"""
        const palette = new Proxy({{}}, {{ get: () => '#000000' }});
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        function html(strings, ...values) {{ return {{ strings, values }}; }}
        {source}
        const calls = [];
        const ctx = {{
          save() {{}}, restore() {{}},
          fillText(text, x, y) {{ calls.push({{ text, x, y }}); }},
        }};
        const chart = {{
          ctx,
          chartArea: {{ left: 100, right: 500, top: 0, bottom: 300 }},
        }};
        quadrantPlugin.afterDraw(chart, {{}}, {plugin_options_js});
        console.log(JSON.stringify(calls));
        """
    )


def test_dashboard_quadrant_labels_match_the_axis_they_annotate() -> None:
    """ "Low latency" must be annotated where latency is low: the bottom.

    Chart.js puts the y maximum at ``chartArea.top``, so the original
    placement -- "High Throughput, Low Latency" at ``top + 16`` -- pointed the
    reader at the high-throughput/HIGH-latency corner and called it good.
    """
    calls = json.loads(_dashboard_quadrant_labels())
    by_text = {c["text"].lower(): c for c in calls}

    good = next(c for t, c in by_text.items() if "low latency" in t)
    bad = next(c for t, c in by_text.items() if "high latency" in t)

    # Low latency lives near the bottom of the plot; high latency near the top.
    assert good["y"] > 150, calls
    assert bad["y"] < 150, calls
    # High throughput is the right half; low throughput the left half.
    assert good["x"] > 300, calls
    assert bad["x"] < 300, calls


def test_dashboard_quadrant_plugin_is_opt_in_per_chart() -> None:
    """A globally-registered plugin must not paint on unrelated charts.

    ``Chart.register`` is global, and no other chart in the app disables
    ``quadrantLabels``, so before the guard every leaderboard bar chart and
    compare scatter also got throughput/latency prose in its corners.
    """
    assert json.loads(_dashboard_quadrant_labels("{}")) == []
    assert json.loads(_dashboard_quadrant_labels("undefined")) == []
    assert json.loads(_dashboard_quadrant_labels("{ enabled: false }")) == []
    assert len(json.loads(_dashboard_quadrant_labels("{ enabled: true }"))) == 2


def _dashboard_eval(expr: str) -> str:
    """Evaluate `expr` against dashboard.js internals with stubbed deps."""
    source = _module_source(DASHBOARD_PATH)
    return run_node(
        f"""
        const palette = new Proxy({{}}, {{ get: () => '#000000' }});
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        function html(strings, ...values) {{ return {{ strings, values }}; }}
        {source}
        console.log(JSON.stringify({expr}));
        """
    )


# A realistic dashboard population: unrelated experiments on three models.
_JOBS = [
    {
        "name": "tiny-smoke",
        "phase": "Completed",
        "model": "org/tiny-1b",
        "throughputRps": 40.0,
        "ttftMs": 9.0,
    },
    {
        "name": "prod-70b",
        "phase": "Succeeded",
        "model": "org/big-70b",
        "throughputRps": 6.0,
        "ttftMs": 310.0,
    },
    {
        "name": "still-running",
        "phase": "Running",
        "model": "org/mid-8b",
        "throughputRps": 999.0,
        "ttftMs": 1.0,
    },
]


def test_dashboard_record_tiles_name_the_run_and_its_model() -> None:
    """A cross-run record must be attributed, not shown as a bare figure.

    "Best TTFT" over a mixed population is the smallest model on the shortest
    prompt, so the tile has to say which run and which model produced it.
    """
    record = json.loads(_dashboard_eval(f"findMin({json.dumps(_JOBS)}, 'ttftMs')"))
    assert record["name"] == "tiny-smoke"
    assert record["model"] == "org/tiny-1b"
    # The running job is excluded, so only the two completed runs are candidates.
    assert record["candidates"] == 2
    assert record["distinctModels"] == 2

    sub = json.loads(
        _dashboard_eval(f"recordSub(findMin({json.dumps(_JOBS)}, 'ttftMs'))")
    )
    assert sub == "tiny-smoke · tiny-1b"


def test_dashboard_record_tooltip_warns_when_candidates_are_incomparable() -> None:
    """Say plainly that the field of candidates was not a like-for-like set."""
    title = json.loads(
        _dashboard_eval(
            f"recordTitle(findBest({json.dumps(_JOBS)}, 'throughputRps'),"
            " 'Highest request throughput')"
        )
    )
    assert "tiny-smoke" in title
    assert "2 models" in title
    assert "not comparable" in title


def test_dashboard_record_tooltip_is_quiet_for_a_single_model() -> None:
    """No incomparability warning when every candidate ran the same model."""
    same_model = [
        {**job, "model": "org/tiny-1b"} for job in _JOBS if job["phase"] != "Running"
    ]
    title = json.loads(
        _dashboard_eval(
            f"recordTitle(findBest({json.dumps(same_model)}, 'throughputRps'), 'X')"
        )
    )
    assert "not comparable" not in title
    assert "2 completed runs" in title


def test_dashboard_record_tiles_handle_an_empty_cluster() -> None:
    """No completed runs must not render a phantom record holder."""
    record = json.loads(_dashboard_eval("findBest([], 'throughputRps')"))
    assert record["value"] is None
    assert json.loads(_dashboard_eval("recordSub(findBest([], 'throughputRps'))")) == ""
    title = json.loads(_dashboard_eval("recordTitle(findBest([], 'x'), 'Throughput')"))
    assert "No completed run" in title


def _compare_eval(expr: str) -> str:
    """Evaluate `expr` against compare.js internals with stubbed deps."""
    source = _module_source(COMPARE_PATH)
    return run_node(
        f"""
        const palette = new Proxy({{}}, {{ get: (_t, k) => '#' + String(k) }});
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        const modelColor = (m) => 'color-of-' + String(m);
        function html(strings, ...values) {{ return {{ strings, values }}; }}
        {source}
        const splitKey = (key) => {{
          const idx = key.indexOf('/');
          return idx < 0
            ? {{ ns: '', jobId: key }}
            : {{ ns: key.slice(0, idx), jobId: key.slice(idx + 1) }};
        }};
        console.log(JSON.stringify({expr}));
        """
    )


# Two namespaces x two models, per-user throughput on x and total output
# throughput on y. The 1B pair would dominate the 70B pair on both axes
# without being a substitute for it.
_MIXED_KEYS = ["prod/a-1b", "prod/b-1b", "prod/c-70b", "stage/d-70b"]
_MIXED_META = {
    "prod/a-1b": {"gpu_count": 1, "gpu_name": "NVIDIA H100", "model": "tiny-1b"},
    "prod/b-1b": {"gpu_count": 1, "gpu_name": "NVIDIA H100", "model": "tiny-1b"},
    "prod/c-70b": {"gpu_count": 8, "gpu_name": "NVIDIA H100", "model": "big-70b"},
    "stage/d-70b": {"gpu_count": 8, "gpu_name": "NVIDIA H100", "model": "big-70b"},
}
_MIXED_ENTRIES = [
    {
        "metric": "output_token_throughput_per_user",
        "stat": "avg",
        "unit": "tok/s",
        "values": {
            "prod/a-1b": 90.0,
            "prod/b-1b": 70.0,
            "prod/c-70b": 30.0,
            "stage/d-70b": 20.0,
        },
    },
    {
        "metric": "output_token_throughput",
        "stat": "avg",
        "unit": "tok/s",
        "values": {
            "prod/a-1b": 900.0,
            "prod/b-1b": 1400.0,
            "prod/c-70b": 1600.0,
            "stage/d-70b": 800.0,
        },
    },
]

_PARETO_POINTS_JS = (
    "buildScatterPoints("
    f"{json.dumps(_MIXED_ENTRIES)},"
    "{ metric: 'output_token_throughput_per_user', stat: 'avg' },"
    "{ metric: 'output_token_throughput', stat: 'avg' },"
    f"{json.dumps(_MIXED_KEYS)},"
    "splitKey,"
    f"{json.dumps(_MIXED_META)},"
    "true)"
)


def test_compare_scatter_points_carry_their_comparable_group() -> None:
    """Every plotted point must know which (namespace x model) it belongs to."""
    clusters = json.loads(
        _compare_eval(f"{_PARETO_POINTS_JS}.map((p) => p.clusterKey)")
    )
    assert clusters == [
        "prod · tiny-1b",
        "prod · tiny-1b",
        "prod · big-70b",
        "stage · big-70b",
    ]


def test_compare_pareto_frontier_never_spans_different_models() -> None:
    """One frontier per comparable group, never one across the selection.

    A single global frontier reported a model-size difference as an efficiency
    result: the 1B runs dominate the 70B runs on both axes while being no kind
    of substitute for them.
    """
    frontiers = json.loads(
        _compare_eval(
            f"clusterFrontiers({_PARETO_POINTS_JS}, true, true)"
            ".map((f) => ({ cluster: f.clusterKey, jobs: f.frontier.map((p) => p.jobName) }))"
        )
    )
    # stage/d-70b is a singleton in its own group and gets no frontier at all.
    assert [f["cluster"] for f in frontiers] == ["prod · tiny-1b"]
    assert sorted(frontiers[0]["jobs"]) == ["a-1b", "b-1b"]
    for f in frontiers:
        assert len({j[-3:] for j in f["jobs"]}) == 1, f


def test_compare_scatter_chart_draws_one_dashed_line_per_group() -> None:
    """Chart datasets must mirror the per-group frontiers, labelled by group."""
    labels = json.loads(
        _compare_eval(
            f"buildScatterChart({_PARETO_POINTS_JS},"
            f" clusterFrontiers({_PARETO_POINTS_JS}, true, true),"
            " { label: 'x', unit: 'tok/s' }, { label: 'y', unit: 'tok/s' })"
            ".datasets.map((d) => d.label)"
        )
    )
    assert labels == ["runs", "prod · tiny-1b · frontier"]


def test_compare_frontier_note_explains_why_no_line_was_drawn() -> None:
    """An absent frontier needs a reason, not silence."""
    note = json.loads(
        _compare_eval(
            f"frontierNote({_PARETO_POINTS_JS}.slice(2),"
            f" clusterFrontiers({_PARETO_POINTS_JS}.slice(2), true, true))"
        )
    )
    assert "namespace x model" in note
    assert "alternatives" in note


_TABLE_CLUSTERS = {
    "prod/a-1b": "prod · tiny-1b",
    "prod/b-1b": "prod · tiny-1b",
    "prod/c-70b": "prod · big-70b",
    "stage/d-70b": "stage · big-70b",
}


def test_compare_table_best_marker_is_scoped_to_one_model() -> None:
    """The green "best" cell must not crown a small model over a large one.

    Marking the 1B run best on throughput reports its parameter count, not its
    quality; the 70B runs are not alternatives to it.
    """
    values = {
        "prod/a-1b": 900.0,
        "prod/b-1b": 1400.0,
        "prod/c-70b": 120.0,
        "stage/d-70b": 80.0,
    }
    best = json.loads(
        _compare_eval(
            "Object.fromEntries(bestValuePerCluster('request_throughput',"
            f" {json.dumps(values)}, {json.dumps(_TABLE_CLUSTERS)}))"
        )
    )
    # Only the two-run group is rankable; the two singleton groups are not.
    assert best == {"prod · tiny-1b": 1400.0}


def test_compare_table_best_marker_follows_metric_direction() -> None:
    """Within a group, latency metrics pick the minimum."""
    values = {"prod/a-1b": 900.0, "prod/b-1b": 140.0}
    best = json.loads(
        _compare_eval(
            "Object.fromEntries(bestValuePerCluster('request_latency',"
            f" {json.dumps(values)}, {json.dumps(_TABLE_CLUSTERS)}))"
        )
    )
    assert best == {"prod · tiny-1b": 140.0}


def test_compare_table_scope_note_states_what_is_being_ranked() -> None:
    """The reader must be told what the colour ranks against."""
    note = json.loads(_compare_eval(f"tableScopeNote({json.dumps(_TABLE_CLUSTERS)})"))
    assert "namespace x model" in note
    assert "never ranked against each other" in note

    singletons = json.loads(
        _compare_eval(
            "tableScopeNote({ 'a/1': 'a · x', 'b/2': 'b · y', 'c/3': 'c · z' })"
        )
    )
    assert "nothing is marked best" in singletons


def test_compare_frontier_note_flags_multiple_frontiers() -> None:
    """When several lines are drawn, say they are not one frontier."""
    note = json.loads(
        _compare_eval(
            "frontierNote("
            "[{ clusterKey: 'a' }, { clusterKey: 'b' }],"
            "[{ clusterKey: 'a' }, { clusterKey: 'b' }])"
        )
    )
    assert note.startswith("2 frontiers")
    assert "not substitutes" in note


def _compare_epochs_eval(expr: str) -> str:
    """Evaluate `expr` against compare-epochs.js internals with stubbed deps."""
    source = _module_source(COMPARE_EPOCHS_PATH)
    return run_node(
        f"""
        const palette = new Proxy({{}}, {{ get: (_t, k) => '#' + String(k) }});
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        function html(strings, ...values) {{ return {{ strings, values }}; }}
        {source}
        console.log(JSON.stringify({expr}));
        """
    )


def _summary(*, concurrency: int = 2, model: str = "mock-model") -> dict:
    """A minimal profile export shaped like the real exporter writes it."""
    return {
        "request_throughput": {"avg": 10.0},
        "input_config": {
            "models": {"items": [{"name": model}]},
            "endpoint": {
                "urls": ["http://server:8000/v1"],
                "type": "chat",
                "streaming": False,
            },
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": {"mean": 75.0}, "osl": {"mean": 30.0}},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "kind": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": concurrency,
                }
            ],
        },
    }


def test_compare_epochs_identical_config_is_comparable() -> None:
    """Two epochs of the same workload produce no drift."""
    a = json.dumps(_summary())
    b = json.dumps(_summary())
    assert json.loads(_compare_epochs_eval(f"configDrift({a}, {b})")) == []


def test_compare_epochs_detects_changed_load_parameters() -> None:
    """A concurrency change between epochs must be surfaced, not absorbed.

    Same job name, new epoch, edited spec: the delta then measures the config
    change, not a server regression.
    """
    drift = json.loads(
        _compare_epochs_eval(
            f"configDrift({json.dumps(_summary(concurrency=2))},"
            f" {json.dumps(_summary(concurrency=64))})"
        )
    )
    assert drift == [{"field": "profiling.concurrency", "a": "2", "b": "64"}]


def test_compare_epochs_detects_changed_model() -> None:
    """A different model under the same job name is not the same experiment."""
    drift = json.loads(
        _compare_epochs_eval(
            f"configDrift({json.dumps(_summary(model='llama-8b'))},"
            f" {json.dumps(_summary(model='llama-70b'))})"
        )
    )
    assert [d["field"] for d in drift] == ["model"]


def test_compare_epochs_missing_config_is_unknown_not_comparable() -> None:
    """No config means no comparability claim -- null, never an empty diff."""
    assert (
        json.loads(_compare_epochs_eval(f"configDrift({json.dumps(_summary())}, null)"))
        is None
    )
    assert json.loads(_compare_epochs_eval("configDrift(null, null)")) is None
    assert json.loads(_compare_epochs_eval("loadFingerprint({})")) is None


def test_compare_epochs_fingerprint_ignores_non_load_config() -> None:
    """Incidental config must not trip the warning on every diff."""
    payload = _summary()
    payload["input_config"]["artifacts"] = {"directory": "/results/epoch-1"}
    other = _summary()
    other["input_config"]["artifacts"] = {"directory": "/results/epoch-2"}
    drift = json.loads(
        _compare_epochs_eval(f"configDrift({json.dumps(payload)}, {json.dumps(other)})")
    )
    assert drift == []


def test_compare_epochs_delta_is_uncoloured_when_workloads_differ() -> None:
    """Green/red is a regression verdict; withhold it when inputs changed."""
    verdicts = json.loads(
        _compare_epochs_eval(
            "["
            "deltaClass(25, 'higher'),"
            "deltaClass(-25, 'higher'),"
            "deltaClass(0.2, 'higher')"
            "]"
        )
    )
    assert verdicts == ["better", "worse", "neutral"]

    # And the Row wiring must route `comparable === false` to the neutral color.
    source = COMPARE_EPOCHS_PATH.read_text()
    assert "comparable === false ? 'neutral' : deltaClass(pct, metric.better)" in source
    assert "comparable=${comparable}" in source
    assert "const comparable = drift != null && drift.length === 0;" in source


def test_leaderboard_rank_direction_is_independent_of_input_order() -> None:
    """Sorting must not rely on the order the API happened to return."""
    shuffled = list(reversed(_LATENCY_ROWS))
    ordered = json.loads(
        _leaderboard_eval(
            f"rankEntries({json.dumps(shuffled)}, 'time_to_first_token')"
            ".map((e) => e.job_id)"
        )
    )
    assert ordered == ["fast-run", "mid-run", "slow-run"]
