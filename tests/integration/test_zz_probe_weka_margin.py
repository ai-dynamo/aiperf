# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DO NOT MERGE: repeat the weka warmup-handoff scenario to catch the SPAWN_JOIN flake on CI.

Each iteration runs the exact ``test_cache_warmup_handoff_preserves_order_and_spawn_joins[1]``
scenario with controller-side join instrumentation (``AIPERF_DBG_JOIN_LOG``) enabled, then
checks every join two ways:

* client: gated root turn ``request_start_ns`` vs the child's last ``request_end_ns``
  (what the real test asserts; timestamps come from two worker processes);
* controller: gated credit ``issued_at_ns`` vs the time the controller processed the
  child's final credit return (same process, same clock).

A client violation with a clean controller order points at timestamp skew between
workers; a controller violation is a real early release.
"""

from __future__ import annotations

import bisect
import warnings
from pathlib import Path

import orjson
import pytest

from tests.integration.test_weka_flat_split_e2e import (
    FANOUT_EXPECTED,
    MockServerFactory,
    _assert_success,
    _child_suffix,
    _collect_plays,
    _is_complete_play,
    _run_weka_profile,
    _write_fanout_corpus,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

ITERATIONS = 15
JOINS = {
    "trace_alpha": {1: ("fa:001",), 2: ("fa:000",)},
    "trace_beta": {1: ("fa:000", "fa:001")},
    "trace_gamma": {1: ("fa:000",)},
}


def _controller_view(
    log_path: Path,
) -> tuple[dict, list[str], float, list[dict], float]:
    """Controller-side join order in monotonic time, plus wall-clock drift.

    Margins use ``perf_counter_ns`` captured in one process, so they are
    immune to wall-clock adjustments. ``drift_ms`` is how far
    ``time.time_ns() - perf_counter_ns()`` moved during the run: a non-zero
    value means the system wall clock was stepped or slewed mid-benchmark,
    which shifts worker timestamps that mix fresh ``time.time_ns()`` reads
    with ``perf_counter`` deltas.
    """
    events = [orjson.loads(line) for line in log_path.read_bytes().splitlines() if line]
    rets = [e for e in events if e["ev"] == "RET"]
    issues = {(e["corr"], e["turn"]): e for e in events if e["ev"] == "ISSUE"}
    child_final = {
        (r["parent"], _child_suffix(r["conv"])): r
        for r in rets
        if r["depth"] > 0 and r["final"]
    }
    gated: dict = {}
    violations: list[str] = []
    min_margin = float("inf")
    for r in rets:
        if r["depth"] != 0 or r["conv"] not in JOINS:
            continue
        issue = issues.get((r["corr"], r["turn"]))
        if issue is None:
            continue
        for suffix in JOINS[r["conv"]].get(r["turn"], ()):
            cf = child_final.get((r["corr"], suffix))
            if cf is None:
                continue
            margin_ms = (issue["m_ns"] - cf["m_ns"]) / 1e6
            gated[(r["corr"], r["turn"], suffix)] = (margin_ms, issue, cf)
            min_margin = min(min_margin, margin_ms)
            if margin_ms < 0:
                violations.append(
                    f"CONTROLLER {r['conv']} turn {r['turn']} vs {suffix}: gated "
                    f"issued {margin_ms:.3f}ms (monotonic) after child final return "
                    f"processed (phase {r['phase']}/{cf['phase']}, root {r['corr']})"
                )
    offsets = [e["t_ns"] - e["m_ns"] for e in events]
    drift_ms = (max(offsets) - min(offsets)) / 1e6 if offsets else 0.0
    stops = [e for e in events if e["ev"] == "STOP"]
    return gated, violations, min_margin, stops, drift_ms


def _start_skews(log_path: Path) -> dict[tuple[str, int], float]:
    """Per-request start skew in ms, keyed by (x_correlation_id, turn_index).

    A record's start is two clock reads: ``start_perf_ns`` (perf_counter) and
    ``timestamp_ns`` (wall). ``timestamp_ns - start_perf_ns`` should equal the
    system's wall-minus-monotonic offset at that moment; both clocks are
    system-wide, so any instrumented event (``t_ns - m_ns``) nearest in
    monotonic time is the reference. A positive skew means the wall read was
    taken that much later than the perf read, shifting the exported start and
    end later by the same amount.
    """
    events = [orjson.loads(line) for line in log_path.read_bytes().splitlines() if line]
    refs = sorted(
        (e["m_ns"], e["t_ns"] - e["m_ns"]) for e in events if e["ev"] != "REC"
    )
    ref_m = [m for m, _ in refs]
    skews: dict[tuple[str, int], float] = {}
    for e in events:
        if e["ev"] != "REC" or not refs:
            continue
        i = bisect.bisect_left(ref_m, e["start_perf_ns"])
        nearby = [refs[j] for j in (i - 1, i) if 0 <= j < len(refs)]
        _, offset = min(nearby, key=lambda r: abs(r[0] - e["start_perf_ns"]))
        skews[(e["corr"], e["turn"])] = (e["ts_ns"] - e["start_perf_ns"] - offset) / 1e6
    return skews


@pytest.mark.parametrize("iteration", range(ITERATIONS))
async def test_zz_weka_spawn_join_flake_probe(
    tmp_path: Path, mock_server_factory: MockServerFactory, iteration: int
) -> None:
    log_path = tmp_path / "join_events.jsonl"
    corpus = _write_fanout_corpus(tmp_path / "traces")
    async with mock_server_factory(fast=True, workers=4) as server:
        result = await _run_weka_profile(
            input_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            url=server.url,
            duration=3.0,
            concurrency=3,
            extra_args=[
                "--scenario",
                "inferencex-agentx-mvp",
                "--unsafe-override",
                "--ignore-trace-delays",
                "--warmup-requests-per-lane",
                "1",
            ],
            extra_env={"AIPERF_DBG_JOIN_LOG": str(log_path)},
            timeout=240.0,
        )
    _assert_success(result, f"flake probe iteration {iteration}")

    gated, problems, controller_min, stops, drift_ms = _controller_view(log_path)
    skew_of = _start_skews(log_path)
    max_skew_ms = max(skew_of.values(), default=0.0)
    client_min = float("inf")
    corrected_min = float("inf")
    client_checks = 0
    for play in _collect_plays(result):
        if not _is_complete_play(play, FANOUT_EXPECTED):
            continue
        kids = {_child_suffix(cid): recs for cid, recs in play.children.items()}
        for turn, suffixes in JOINS[play.trace_id].items():
            g = play.root[turn]
            g_skew = skew_of.get((g.x_correlation_id, g.turn_index))
            for suffix in suffixes:
                last = kids[suffix][-1]
                margin_ms = (g.request_start_ns - last.request_end_ns) / 1e6
                c_skew = skew_of.get((last.x_correlation_id, last.turn_index))
                corrected_ms = (
                    None
                    if g_skew is None or c_skew is None
                    else margin_ms - g_skew + c_skew
                )
                client_checks += 1
                client_min = min(client_min, margin_ms)
                if corrected_ms is not None:
                    corrected_min = min(corrected_min, corrected_ms)
                if margin_ms >= 0:
                    continue
                ctl = gated.get((play.root_corr, turn, suffix))
                ctl_text = (
                    "no controller events"
                    if ctl is None
                    else f"controller monotonic margin {ctl[0]:.3f}ms"
                )
                skew_text = (
                    "start skew unavailable"
                    if corrected_ms is None
                    else f"start skew gated {g_skew:.3f}ms, child {c_skew:.3f}ms; "
                    f"skew-corrected margin {corrected_ms:.3f}ms"
                )
                problems.append(
                    f"CLIENT {play.trace_id} turn {turn} vs {suffix}: margin "
                    f"{margin_ms:.3f}ms; gated start {g.request_start_ns} on "
                    f"{g.worker_id}, child end {last.request_end_ns} on "
                    f"{last.worker_id}; {ctl_text}; {skew_text}; root {play.root_corr}"
                )
    stop_lines = [
        f"STOP {s['corr']} has_entries={s['has_entries']} caller={s['caller']}"
        for s in stops
    ]
    summary = (
        f"iteration {iteration}: client checks={client_checks} min={client_min:.3f}ms; "
        f"skew-corrected min={corrected_min:.3f}ms; max start skew={max_skew_ms:.3f}ms "
        f"over {len(skew_of)} records; "
        f"controller checks={len(gated)} min={controller_min:.3f}ms; stops={len(stops)}; "
        f"wall-vs-monotonic drift={drift_ms:.3f}ms"
    )
    print(summary)
    # Surfaces in pytest's warnings summary on green runs too, so CI logs
    # carry the margins and drift for every iteration.
    warnings.warn(summary, stacklevel=1)
    assert not problems, "\n".join([summary, *problems, *stop_lines])
