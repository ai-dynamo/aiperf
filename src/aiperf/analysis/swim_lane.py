# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session swim-lane plot for an AIPerf run.

Renders one horizontal lane per concurrent session slot (greedy reuse once a
session retires) plus a concurrency line chart driven by the same
``aiperf.analysis.sweepline`` primitives the metrics accumulator uses.
Subagent sessions (``agent_depth > 0``) do not get their own slots: they nest
as child tiers below their root session's lane. Lanes are keyed on
``root_correlation_id`` -- the depth-0 root id every node of a session TREE
shares -- so each agentic session (root + all its subagents, at any depth)
maps to exactly one lane; a subagent whose root made no profiled request (root
all before t*, or a rootless lane) nests under a zero-row phantom root for that
shared id. Because the runtime holds exactly one session slot per tree until
the whole tree drains, the rendered slot count equals the configured
concurrency. Exports predating ``root_correlation_id`` fall back to a
``parent_correlation_id`` walk (one phantom lane per absent root).

Inputs per run directory:
  - ``profile_export.jsonl``        (required)  per-record AIPerf export
  - ``profile_export_aiperf.json``  (optional)  used for ramp/benchmark axvlines

CLI: ``aiperf analyze swim-lane <run_dir> [<run_dir>...] [-o OUT]``
"""

from __future__ import annotations

import base64
import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import orjson
from numpy.typing import NDArray

from aiperf.analysis.sweepline import (
    add_step_functions,
    concurrency_sweep_line,
    prefill_throughput_sweep_line,
    throughput_sweep_line,
    tokens_in_flight_sweep_line,
    total_throughput_sweep_line,
    weighted_concurrency_sweep_line,
)
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import NANOS_PER_MILLIS, NANOS_PER_SECOND

PROFILE_JSONL = OutputDefaults.PROFILE_EXPORT_JSONL_FILE.name
PROFILE_JSON = OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE.name
VIEWER_TEMPLATE = Path(__file__).with_name("swim_lane_viewer.html")


class SwimLaneError(RuntimeError):
    """Raised when a swim-lane plot cannot be produced for a run directory."""


def _load_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with open(jsonl_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = orjson.loads(line)
            meta = r.get("metadata") or {}
            if meta.get("was_cancelled"):
                continue
            if (
                meta.get("request_start_ns") is None
                or meta.get("request_end_ns") is None
            ):
                continue
            records.append(r)
    return records


def _group_into_sessions(records: list[dict]) -> dict[str, list[dict]]:
    """Group records into benchmark session instances by x_correlation_id.

    The same source conversation can be replayed concurrently by several
    sessions, all sharing its conversation_id; x_correlation_id identifies the
    actual session instance. Falls back to conversation_id for exports that
    predate correlation ids.
    """
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        meta = r["metadata"]
        sessions[meta.get("x_correlation_id") or meta["conversation_id"]].append(r)
    for turns in sessions.values():
        turns.sort(key=lambda r: r["metadata"]["turn_index"])
    return dict(sessions)


def _session_span_ns(turns: list[dict]) -> tuple[int, int]:
    """True session time span. Turns are sorted by turn_index, and with parallel
    subagent requests the last turn by index is not necessarily the latest-ending."""
    return (
        min(t["metadata"]["request_start_ns"] for t in turns),
        max(t["metadata"]["request_end_ns"] for t in turns),
    )


def _merged_active_ns(turns: list[dict]) -> list[tuple[int, int]]:
    """Union of the session's request intervals; gaps between them are true idle."""
    intervals = sorted(
        (t["metadata"]["request_start_ns"], t["metadata"]["request_end_ns"])
        for t in turns
    )
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _turn_depths(turns: list[dict]) -> tuple[list[int], int]:
    """Greedy sub-row packing of possibly-parallel turns within one session.

    Returns (depth per turn, row count) so overlapping requests render stacked
    instead of over-painting each other in the lane.
    """
    order = sorted(
        range(len(turns)), key=lambda i: turns[i]["metadata"]["request_start_ns"]
    )
    row_ends: list[int] = []
    depths = [0] * len(turns)
    for i in order:
        start = turns[i]["metadata"]["request_start_ns"]
        end = turns[i]["metadata"]["request_end_ns"]
        for d, row_end in enumerate(row_ends):
            if start >= row_end:
                row_ends[d] = end
                depths[i] = d
                break
        else:
            row_ends.append(end)
            depths[i] = len(row_ends) - 1
    return depths, len(row_ends)


def _assign_slots(spans: dict[str, tuple[int, int]]) -> list[tuple[str, int]]:
    """Greedy slot packing: reuse a slot once its previous occupant retired."""
    ordered = sorted(spans, key=lambda key: spans[key][0])
    slot_ends: list[int] = []
    assignments: list[tuple[str, int]] = []
    for key in ordered:
        start, end = spans[key]
        for i, slot_end in enumerate(slot_ends):
            if start >= slot_end:
                slot_ends[i] = end
                assignments.append((key, i))
                break
        else:
            slot_ends.append(end)
            assignments.append((key, len(slot_ends) - 1))
    return assignments


def _is_sub_session(turns: list[dict]) -> bool:
    """True when any record in the session was produced at agent_depth > 0."""
    return any((t["metadata"].get("agent_depth") or 0) > 0 for t in turns)


def _root_map(sessions: dict[str, list[dict]]) -> dict[str, str]:
    """Resolve each session to its lane key (its session-tree root).

    Authoritative path: when the export carries ``root_correlation_id`` (every
    node of a session tree -- root + every descendant subagent at any depth --
    shares its depth-0 root's id, written by the runtime's per-tree
    accounting), group directly on it. No parent-walk, no phantom-root guessing:
    a root keys on its own id; a subagent keys on its root's id; a subagent
    whose root made no profiled request (root all before t*, or rootless lane)
    keys on the shared synthetic root id, so each concurrency lane is exactly
    one swim lane. Because exactly N trees are live at once, the resulting slot
    assignment is exactly the configured concurrency.

    Legacy heuristic (exports predating ``root_correlation_id``): walk
    ``parent_correlation_id`` to the top, or the session's own id when it has no
    parent. A subagent whose parent is ABSENT roots at that absent
    ``parent_correlation_id`` itself (one phantom lane per concurrency lane).
    """
    if any(
        t["metadata"].get("root_correlation_id")
        for turns in sessions.values()
        for t in turns
    ):
        return {
            sid: next(
                (
                    t["metadata"]["root_correlation_id"]
                    for t in turns
                    if t["metadata"].get("root_correlation_id")
                ),
                sid,
            )
            for sid, turns in sessions.items()
        }

    parent: dict[str, str | None] = {}
    for sid, turns in sessions.items():
        parent[sid] = next(
            (
                t["metadata"].get("parent_correlation_id")
                for t in turns
                if t["metadata"].get("parent_correlation_id")
            ),
            None,
        )
    roots: dict[str, str] = {}
    for sid in sessions:
        seen = {sid}
        cur = sid
        # Walk up through parents that are themselves present in the export.
        while (
            (nxt := parent.get(cur)) is not None
            and nxt != cur
            and nxt in sessions
            and nxt not in seen
        ):
            cur = nxt
            seen.add(cur)
        top = parent.get(cur)
        if top is None:
            roots[sid] = cur  # genuine root reached; key on it (its x_corr)
        elif top in seen:
            roots[sid] = sid  # parent cycle (malformed export) -> detach to self
        else:
            # cur's parent is absent from the export -> key on that parent_corr,
            # the lane's phantom root shared by all its orphan subagents.
            roots[sid] = top
    return roots


@dataclass(slots=True)
class _SessionLayout:
    """Row placement of one session inside its lane group."""

    sid: str
    """Session instance id (x_correlation_id, or conversation_id fallback)."""
    is_sub: bool
    """True when the session ran at agent_depth > 0 (a subagent)."""
    row0: int
    """First row the session occupies within the lane group."""
    rows: int
    """Rows occupied by the session's own (possibly parallel) turns."""
    depths: list[int]
    """Per-turn sub-row within the session, aligned with the turn list."""


@dataclass(slots=True)
class _LaneGroup:
    """One swim lane: a root session plus its nested subagent sessions."""

    root_sid: str
    """Root session id of the lane."""
    slot: int
    """Assigned swim-lane slot."""
    rows: int
    """Total rows of the lane (root rows + packed subagent child rows)."""
    span_ns: tuple[int, int]
    """(first request start, last request end) over all member sessions."""
    members: list[_SessionLayout]
    """Root session first, then subagent sessions ordered by start time."""


def _layout_groups(sessions: dict[str, list[dict]]) -> list[_LaneGroup]:
    """Nest subagent sessions as child tiers below their root session's lane.

    The root session keeps the top rows of the lane (one per parallel turn);
    each subagent session is greedy-packed into child rows underneath, reusing
    a row once the previous subagent on it retired. Subagents whose root is
    absent from the export nest under a zero-row phantom root keyed by their
    shared ``parent_correlation_id`` (one lane per concurrency lane). Lane
    groups are then slot-packed exactly like plain sessions, ordered by group
    start time.
    """
    root_of = _root_map(sessions)
    children: dict[str, list[str]] = defaultdict(list)
    for sid, root in root_of.items():
        if sid != root:
            children[root].append(sid)
    spans: dict[str, tuple[int, int]] = {}
    for root in set(root_of.values()):
        member_spans = [
            _session_span_ns(sessions[m])
            for m in (root, *children.get(root, ()))
            if m in sessions
        ]
        spans[root] = (
            min(s for s, _ in member_spans),
            max(e for _, e in member_spans),
        )

    groups: list[_LaneGroup] = []
    for root_sid, slot in _assign_slots(spans):
        if root_sid in sessions:
            depths, root_rows = _turn_depths(sessions[root_sid])
            members = [
                _SessionLayout(
                    root_sid, _is_sub_session(sessions[root_sid]), 0, root_rows, depths
                )
            ]
        else:
            # phantom root: no records of its own, subs start at row 0
            root_rows = 0
            members = []
        kids = sorted(
            children.get(root_sid, ()),
            key=lambda k: _session_span_ns(sessions[k])[0],
        )
        row_ends: list[int] = []
        for kid in kids:
            kid_depths, kid_rows = _turn_depths(sessions[kid])
            start, end = _session_span_ns(sessions[kid])
            # first-fit: lowest child row window where every row retired before
            # this subagent starts (rows past the end of the list are free)
            for row in range(len(row_ends) + 1):
                if all(
                    row_ends[row + j] <= start
                    for j in range(kid_rows)
                    if row + j < len(row_ends)
                ):
                    break
            row_ends.extend([0] * (row + kid_rows - len(row_ends)))
            for j in range(kid_rows):
                row_ends[row + j] = end
            members.append(
                _SessionLayout(kid, True, root_rows + row, kid_rows, kid_depths)
            )
        groups.append(
            _LaneGroup(
                root_sid, slot, root_rows + len(row_ends), spans[root_sid], members
            )
        )
    return groups


def _load_bench_config(run_dir: Path) -> tuple[float | None, float | None]:
    """Return (concurrency_ramp_duration_s, benchmark_duration_s) when available."""
    path = run_dir / PROFILE_JSON
    if not path.is_file():
        return None, None
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    loadgen = data.get("input_config", {}).get("loadgen", {}) or {}
    ramp = loadgen.get("concurrency_ramp_duration")
    # benchmark_duration lives at top-level on the export, but fall back to loadgen.
    bench = data.get("benchmark_duration") or loadgen.get("benchmark_duration")
    bench_value = bench.get("avg") if isinstance(bench, dict) else bench
    return (
        float(ramp) if ramp is not None else None,
        float(bench_value) if bench_value is not None else None,
    )


def _per_session_spans_ns(
    sessions: dict[str, list[dict]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-session (first_start_ns, last_end_ns) arrays for alive-session sweep."""
    if not sessions:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    spans = [_session_span_ns(turns) for turns in sessions.values()]
    starts = np.fromiter(
        (span[0] for span in spans), dtype=np.float64, count=len(spans)
    )
    ends = np.fromiter((span[1] for span in spans), dtype=np.float64, count=len(spans))
    return starts, ends


def _concurrency_curves(
    sessions: dict[str, list[dict]], n_records: int, t0_ns: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return (active_ts, active_vals, idle_ts, idle_vals) in seconds from t0.

    Sweep-line curves reuse the accumulator's primitive. Work in seconds-from-t0
    so float64 has enough precision (absolute ns timestamps are ~1.7e18, eating
    ~16 sig figs and collapsing sub-second resolution).
    """
    starts_ns = np.fromiter(
        (
            r["metadata"]["request_start_ns"]
            for turns in sessions.values()
            for r in turns
        ),
        dtype=np.float64,
        count=n_records,
    )
    ends_ns = np.fromiter(
        (r["metadata"]["request_end_ns"] for turns in sessions.values() for r in turns),
        dtype=np.float64,
        count=n_records,
    )
    starts_s = (starts_ns - t0_ns) / NANOS_PER_SECOND
    ends_s = (ends_ns - t0_ns) / NANOS_PER_SECOND
    active_ts, active_vals = concurrency_sweep_line(starts_s, ends_s)
    alive_starts_ns, alive_ends_ns = _per_session_spans_ns(sessions)
    alive_starts_s = (alive_starts_ns - t0_ns) / NANOS_PER_SECOND
    alive_ends_s = (alive_ends_ns - t0_ns) / NANOS_PER_SECOND
    alive_ts, alive_vals = concurrency_sweep_line(alive_starts_s, alive_ends_s)
    # idle = alive - active. ``add_step_functions`` aligns onto the merged event grid.
    idle_ts, idle_vals = add_step_functions(
        alive_ts, alive_vals, active_ts, -active_vals
    )
    idle_vals = np.maximum(idle_vals, 0.0)
    return active_ts, active_vals, idle_ts, idle_vals


def _new_isl_curve(
    sessions: dict[str, list[dict]],
    new_isl_map: dict[str, list[float]],
    t0_ns: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sweep-line of "new" input tokens being prefilled, summed across requests.

    Each turn's ``new_isl`` (see ``_session_new_isls``) is in flight only during
    prefill -- the window ``[start_ns, generation_start_ns)`` where
    ``generation_start_ns = start_ns + TTFT`` -- after which generation begins
    and the prompt is no longer "new". This curve sums those weights at every
    event boundary, the token-weighted prefill analogue of the concurrency
    curve. Returns (timestamps, values) in seconds-from-t0. Zero-weight turns
    are dropped since they add no events; turns missing TTFT fall back to the
    request end.
    """
    starts: list[float] = []
    ends: list[float] = []
    weights: list[float] = []
    for sid, turns in sessions.items():
        for r, w in zip(turns, new_isl_map[sid], strict=True):
            if w <= 0.0:
                continue
            start_ns = r["metadata"]["request_start_ns"]
            end_ns = r["metadata"]["request_end_ns"]
            ttft_ms = _metric_value(r, "time_to_first_token")
            generation_start_ns = (
                min(start_ns + ttft_ms * NANOS_PER_MILLIS, end_ns)
                if ttft_ms is not None
                else end_ns
            )
            starts.append(start_ns)
            ends.append(generation_start_ns)
            weights.append(w)
    if not starts:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    starts_s = (np.array(starts, dtype=np.float64) - t0_ns) / NANOS_PER_SECOND
    ends_s = (np.array(ends, dtype=np.float64) - t0_ns) / NANOS_PER_SECOND
    return weighted_concurrency_sweep_line(
        starts_s, ends_s, np.array(weights, dtype=np.float64)
    )


def _throughput_curves(
    sessions: dict[str, list[dict]], t0_ns: float
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """Prefill, decode, and total throughput sweep curves (tokens/sec over time).

    Reuses the same sweep-line primitives the metrics accumulator uses, but fed
    from the exported records: prefill runs over ``[start, generation_start)``
    weighted by ISL, decode over ``[generation_start, end)`` weighted by
    ``OSL - 1``, and total combines both. ``generation_start`` is derived from
    TTFT (this is the post-hoc jsonl path). Timestamps are seconds-from-t0 so
    rates come out directly in tokens/sec. Returns ((pf_ts, pf), (dec_ts, dec),
    (tot_ts, tot)); each value array is clamped at 0.
    """
    start_s, gen_s, end_s, isl_a, osl_a = _record_arrays(sessions, t0_ns)
    pf_ts, pf = prefill_throughput_sweep_line(start_s, gen_s, isl_a)
    dec_ts, dec = throughput_sweep_line(gen_s, end_s, osl_a)
    tot_ts, tot = total_throughput_sweep_line(
        start_s, gen_s, end_s, isl_a, output_tokens=osl_a
    )
    return (
        (pf_ts, np.maximum(pf, 0.0)),
        (dec_ts, np.maximum(dec, 0.0)),
        (tot_ts, np.maximum(tot, 0.0)),
    )


def _record_arrays(
    sessions: dict[str, list[dict]], t0_ns: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Per-record (start, generation_start, end) in seconds-from-t0 plus ISL/OSL.

    ``generation_start = start + TTFT`` (the post-hoc jsonl path); entries are
    NaN where the metric is missing so the sweep primitives skip them.
    """
    starts: list[float] = []
    gens: list[float] = []
    ends: list[float] = []
    isls: list[float] = []
    osls: list[float] = []
    nan = float("nan")
    for turns in sessions.values():
        for r in turns:
            meta = r["metadata"]
            s = meta["request_start_ns"]
            ttft_ms = _metric_value(r, "time_to_first_token")
            isl = _metric_value(r, "input_sequence_length")
            osl = _metric_value(r, "output_sequence_length")
            starts.append((s - t0_ns) / NANOS_PER_SECOND)
            ends.append((meta["request_end_ns"] - t0_ns) / NANOS_PER_SECOND)
            gens.append(
                (s + ttft_ms * NANOS_PER_MILLIS - t0_ns) / NANOS_PER_SECOND
                if ttft_ms is not None
                else nan
            )
            isls.append(float(isl) if isl is not None else nan)
            osls.append(float(osl) if osl is not None else nan)
    return (
        np.array(starts, dtype=np.float64),
        np.array(gens, dtype=np.float64),
        np.array(ends, dtype=np.float64),
        np.array(isls, dtype=np.float64),
        np.array(osls, dtype=np.float64),
    )


def _tokens_in_flight_curve(
    sessions: dict[str, list[dict]], t0_ns: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """KV-cache token load over time (the GPU-memory-pressure view).

    Input tokens are held for the whole request lifetime and output tokens
    accumulate during generation: ISL over ``[start, generation_start)`` then
    ISL+OSL over ``[generation_start, end)``, summed across in-flight requests.
    Returns (timestamps seconds-from-t0, tokens), clamped at 0.
    """
    start_s, gen_s, end_s, isl_a, osl_a = _record_arrays(sessions, t0_ns)
    ts, vals = tokens_in_flight_sweep_line(
        start_s, gen_s, end_s, isl_a, output_tokens=osl_a
    )
    return ts, np.maximum(vals, 0.0)


def _latency_band(
    sessions: dict[str, list[dict]], t0_ns: float, n_bins: int = 240
) -> tuple[list[float], list[float], list[float]]:
    """p50 / p99 TTFT (ms) over time bins keyed by request start.

    Bins the run into ``n_bins`` equal time slices and reports the median and
    99th-percentile TTFT of requests starting in each non-empty bin, so a
    shaded band shows latency drift as load ramps. Returns (centers_s, p50, p99)
    for non-empty bins only.
    """
    starts: list[float] = []
    ttfts: list[float] = []
    for turns in sessions.values():
        for r in turns:
            ttft = _metric_value(r, "time_to_first_token")
            if ttft is None:
                continue
            starts.append(
                (r["metadata"]["request_start_ns"] - t0_ns) / NANOS_PER_SECOND
            )
            ttfts.append(ttft)
    if not starts:
        return [], [], []
    starts_a = np.array(starts, dtype=np.float64)
    ttfts_a = np.array(ttfts, dtype=np.float64)
    t_max = float(starts_a.max())
    if t_max <= 0:
        return (
            [0.0],
            [round(float(np.percentile(ttfts_a, 50)), 1)],
            [round(float(np.percentile(ttfts_a, 99)), 1)],
        )
    edges = np.linspace(0.0, t_max, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, starts_a, side="right") - 1, 0, n_bins - 1)
    centers: list[float] = []
    p50: list[float] = []
    p99: list[float] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        vals = ttfts_a[mask]
        centers.append(round(float((edges[b] + edges[b + 1]) / 2.0), 3))
        p50.append(round(float(np.percentile(vals, 50)), 1))
        p99.append(round(float(np.percentile(vals, 99)), 1))
    return centers, p50, p99


def plot_swim_lane(
    run_dir: Path,
    out: Path | None = None,
    concurrency: int | None = None,
    ramp: float | None = None,
) -> Path:
    """Render a swim-lane PNG for ``run_dir``. Returns the output path.

    Subagent sessions render inline as child tiers below their root session's
    lane rather than occupying their own slots. When ``concurrency`` is given,
    the bottom panel draws a horizontal target-concurrency reference line at
    that value. ``ramp`` overrides the ramp duration read from
    ``profile_export_aiperf.json``.
    """
    jsonl_path = run_dir / PROFILE_JSONL
    if not jsonl_path.is_file():
        raise SwimLaneError(f"{jsonl_path} not found")

    records = _load_records(jsonl_path)
    if not records:
        raise SwimLaneError(f"no valid records in {jsonl_path}")

    sessions = _group_into_sessions(records)
    groups = _layout_groups(sessions)
    ramp_dur_s, bench_dur_s = _load_bench_config(run_dir)
    if ramp is not None:
        ramp_dur_s = ramp

    t0_ns = float(
        min(
            t["metadata"]["request_start_ns"]
            for turns in sessions.values()
            for t in turns
        )
    )
    t_max_s = float(
        max(
            (t["metadata"]["request_end_ns"] - t0_ns) / NANOS_PER_SECOND
            for turns in sessions.values()
            for t in turns
        )
    )
    n_slots = max(g.slot for g in groups) + 1

    cmap = plt.colormaps["tab20"]
    lane_color = {g.root_sid: cmap(i % 20) for i, g in enumerate(groups)}

    slot_lane_starts: dict[int, list[float]] = defaultdict(list)
    for g in groups:
        slot_lane_starts[g.slot].append((g.span_ns[0] - t0_ns) / NANOS_PER_SECOND)
    reuse_markers = [
        (slot, s)
        for slot, starts in slot_lane_starts.items()
        for s in sorted(starts)[1:]
    ]

    active_ts, active_vals, idle_ts, idle_vals = _concurrency_curves(
        sessions, len(records), t0_ns
    )

    # each slot expands to one full-size sub-lane per concurrently-running turn,
    # plus one child tier per concurrently-running subagent session
    slot_rows = [1] * n_slots
    for g in groups:
        slot_rows[g.slot] = max(slot_rows[g.slot], g.rows)
    slot_offset = [0] * n_slots
    for s in range(1, n_slots):
        slot_offset[s] = slot_offset[s - 1] + slot_rows[s - 1]
    total_rows = slot_offset[-1] + slot_rows[-1]

    bar_height = 0.7
    row_inches = min(0.45, 58.0 / total_rows)
    swim_height = max(4.0, total_rows * row_inches + 1.0)
    fig, (ax_swim, ax_conc) = plt.subplots(
        2,
        1,
        figsize=(20, swim_height + 2.5),
        gridspec_kw={"height_ratios": [swim_height, 2.5]},
        sharex=True,
    )

    slot_spans: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for g in groups:
        s = (g.span_ns[0] - t0_ns) / NANOS_PER_SECOND
        e = (g.span_ns[1] - t0_ns) / NANOS_PER_SECOND
        slot_spans[g.slot].append((s, e))

    for slot in range(n_slots):
        slot_center = slot_offset[slot] + (slot_rows[slot] - 1) / 2
        slot_height = slot_rows[slot] - (1 - bar_height)
        spans = sorted(slot_spans.get(slot, []))
        gaps: list[tuple[float, float]] = []
        if not spans:
            gaps.append((0.0, t_max_s))
        else:
            if spans[0][0] > 0:
                gaps.append((0.0, spans[0][0]))
            for i in range(1, len(spans)):
                if spans[i][0] > spans[i - 1][1]:
                    gaps.append((spans[i - 1][1], spans[i][0]))
            if spans[-1][1] < t_max_s:
                gaps.append((spans[-1][1], t_max_s))
        for left, right in gaps:
            ax_swim.barh(
                slot_center,
                right - left,
                left=left,
                height=slot_height,
                color="#f0f0f0",
                edgecolor="none",
                linewidth=0,
            )

    for g in groups:
        color = lane_color[g.root_sid]
        for m in g.members:
            turns = sessions[m.sid]
            row_base = slot_offset[g.slot] + m.row0
            idle_color = (*color[:3], 0.15 if m.is_sub else 0.25)
            merged = _merged_active_ns(turns)
            # idle bands span only this session's own rows within the lane
            for (_, prev_end), (start, _) in zip(merged, merged[1:], strict=False):
                ax_swim.barh(
                    row_base + (m.rows - 1) / 2,
                    (start - prev_end) / NANOS_PER_SECOND,
                    left=(prev_end - t0_ns) / NANOS_PER_SECOND,
                    height=m.rows - (1 - bar_height),
                    color=idle_color,
                    edgecolor="none",
                    linewidth=0,
                )
            # parallel turns occupy their own full-size sub-lane within the slot;
            # subagent turns render lighter in their child tier
            bar_color = (*color[:3], 0.55) if m.is_sub else color
            for turn, depth in zip(turns, m.depths, strict=True):
                rs = (turn["metadata"]["request_start_ns"] - t0_ns) / NANOS_PER_SECOND
                re = (turn["metadata"]["request_end_ns"] - t0_ns) / NANOS_PER_SECOND
                ax_swim.barh(
                    row_base + depth,
                    re - rs,
                    left=rs,
                    height=bar_height,
                    color=bar_color,
                    edgecolor="none",
                    linewidth=0,
                )

    for slot, x in reuse_markers:
        ax_swim.plot(
            [x, x],
            [
                slot_offset[slot] - bar_height / 2,
                slot_offset[slot] + slot_rows[slot] - 1 + bar_height / 2,
            ],
            color="black",
            linewidth=1.5,
            solid_capstyle="butt",
        )

    if total_rows > n_slots:
        for slot in range(1, n_slots):
            ax_swim.axhline(
                slot_offset[slot] - 0.5, color="#999999", linewidth=0.4, alpha=0.6
            )

    n_sub = sum(1 for turns in sessions.values() if _is_sub_session(turns))
    session_counts = (
        f"{len(sessions) - n_sub} sessions + {n_sub} subagents"
        if n_sub
        else f"{len(sessions)} sessions"
    )
    ax_swim.set_xlim(0, max(t_max_s * 1.01, 1.0))
    ax_swim.set_ylim(-0.5, total_rows - 0.5)
    ax_swim.set_ylabel("Session Slot", fontsize=12)
    ax_swim.set_title(
        f"{run_dir.name}  —  {session_counts} across {n_slots} slots",
        fontsize=13,
        fontweight="bold",
        pad=30,
    )
    stride = 1 if n_slots <= 30 else 2 if n_slots <= 60 else 5
    tick_slots = list(range(0, n_slots, stride))
    ax_swim.set_yticks([slot_offset[s] + (slot_rows[s] - 1) / 2 for s in tick_slots])
    ax_swim.set_yticklabels([str(s) for s in tick_slots])
    ax_swim.invert_yaxis()
    ax_swim.grid(axis="x", alpha=0.3, linewidth=0.5)

    legend_color = cmap(0)
    legend_handles = [
        mpatches.Patch(color=legend_color, label="Active request"),
        *(
            [mpatches.Patch(color=(*legend_color[:3], 0.55), label="Subagent request")]
            if n_sub
            else []
        ),
        mpatches.Patch(color=(*legend_color[:3], 0.25), label="Inter-turn delay"),
        plt.Line2D([0], [0], color="black", linewidth=1.5, label="New session in slot"),
    ]
    ax_swim.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9,
        ncol=len(legend_handles),
        frameon=False,
    )

    if ramp_dur_s is not None:
        ax_swim.axvline(
            ramp_dur_s, color="orange", linestyle="--", linewidth=1.5, alpha=0.8
        )
    if bench_dur_s is not None:
        ax_swim.axvline(
            bench_dur_s, color="red", linestyle="--", linewidth=1.5, alpha=0.8
        )

    peak_active = int(active_vals.max()) if active_vals.size else 0
    ax_conc.fill_between(
        active_ts, active_vals, step="post", alpha=0.3, color="#2196F3"
    )
    ax_conc.step(
        active_ts,
        active_vals,
        where="post",
        color="#2196F3",
        linewidth=0.8,
        label=f"Active requests (peak {peak_active})",
    )
    ax_conc.step(
        idle_ts,
        idle_vals,
        where="post",
        color="#999999",
        linewidth=0.8,
        alpha=0.8,
        label="Idle sessions (between turns)",
    )
    ax_conc.axhline(
        n_slots,
        color="red",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label=f"Slot count ({n_slots})",
    )
    if concurrency is not None:
        ax_conc.axhline(
            concurrency,
            color="green",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"Target concurrency ({concurrency})",
        )
    if ramp_dur_s is not None:
        ax_conc.axvline(
            ramp_dur_s,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Ramp done ({ramp_dur_s:.0f}s)",
        )
    if bench_dur_s is not None:
        ax_conc.axvline(
            bench_dur_s,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Benchmark end ({bench_dur_s:.0f}s)",
        )
    ax_conc.set_xlabel("Time (seconds)", fontsize=12)
    ax_conc.set_ylabel("Sessions", fontsize=11)
    ax_conc.set_ylim(
        0, max(peak_active * 1.15, n_slots * 1.15, (concurrency or 0) * 1.15, 1.0)
    )
    handles, _ = ax_conc.get_legend_handles_labels()
    ax_conc.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        fontsize=9,
        ncol=len(handles),
        frameon=False,
    )
    ax_conc.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_path = out or (run_dir / "swim_lane.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _metric_value(record: dict, name: str) -> float | None:
    metric = record.get("metrics", {}).get(name)
    return metric.get("value") if isinstance(metric, dict) else None


CACHE_READ_METRIC = "usage_prompt_cache_read_tokens"


def _session_new_isls(turns: list[dict]) -> list[float]:
    """Per-turn "new" input tokens: this turn's ISL minus the cached prefix
    carried over from the conversation history.

    The cached prefix is the server-reported cache-read count when the endpoint
    exposes it (``usage_prompt_cache_read_tokens``), else it falls back to the
    previous turn's ISL. The first turn (no predecessor) and any turn missing
    ISL contribute their full ISL. ``turns`` must be turn_index-ordered.
    Values are clamped at 0 -- "new" tokens are never negative.
    """
    out: list[float] = []
    prev_isl: float | None = None
    for r in turns:
        isl = _metric_value(r, "input_sequence_length")
        if isl is None:
            out.append(0.0)
            continue
        cached = _metric_value(r, CACHE_READ_METRIC)
        if cached is not None:
            new = isl - cached
        elif prev_isl is not None:
            new = isl - prev_isl
        else:
            new = isl
        out.append(max(new, 0.0))
        prev_isl = isl
    return out


def _peak_concurrent(intervals: list[tuple[int, int]]) -> int:
    """Max number of simultaneously-open (start_ns, end_ns) intervals."""
    if not intervals:
        return 0
    events = sorted(
        [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
        key=lambda ev: (ev[0], ev[1]),
    )
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _agent_peaks(sessions: dict[str, list[dict]]) -> dict[str, int]:
    """Peak concurrency split by main agents (agent_depth 0) vs subagents.

    "Active" counts in-flight requests; "alive" counts whole session spans,
    inter-turn idle included.
    """
    turn_ivs: dict[bool, list[tuple[int, int]]] = {False: [], True: []}
    span_ivs: dict[bool, list[tuple[int, int]]] = {False: [], True: []}
    for turns in sessions.values():
        is_sub = _is_sub_session(turns)
        span_ivs[is_sub].append(_session_span_ns(turns))
        for t in turns:
            turn_ivs[is_sub].append(
                (t["metadata"]["request_start_ns"], t["metadata"]["request_end_ns"])
            )
    return {
        "mainActive": _peak_concurrent(turn_ivs[False]),
        "mainAlive": _peak_concurrent(span_ivs[False]),
        "subActive": _peak_concurrent(turn_ivs[True]),
        "subAlive": _peak_concurrent(span_ivs[True]),
    }


def write_swim_lane_html(
    run_dir: Path,
    out: Path | None = None,
    concurrency: int | None = None,
    ramp: float | None = None,
) -> Path:
    """Write an interactive single-file HTML trace viewer for ``run_dir``.

    Embeds a compact JSON payload (sessions, per-turn metrics, concurrency
    sweep curves) into the canvas-based viewer template next to this module.
    Subagent sessions tier inline below their root session's lane.
    ``ramp`` overrides the ramp duration read from ``profile_export_aiperf.json``.
    Returns the output path.
    """
    jsonl_path = run_dir / PROFILE_JSONL
    if not jsonl_path.is_file():
        raise SwimLaneError(f"{jsonl_path} not found")
    records = _load_records(jsonl_path)
    if not records:
        raise SwimLaneError(f"no valid records in {jsonl_path}")

    sessions = _group_into_sessions(records)
    groups = _layout_groups(sessions)
    ramp_dur_s, bench_dur_s = _load_bench_config(run_dir)
    if ramp is not None:
        ramp_dur_s = ramp
    t0_ns = float(
        min(
            t["metadata"]["request_start_ns"]
            for turns in sessions.values()
            for t in turns
        )
    )
    active_ts, active_vals, idle_ts, idle_vals = _concurrency_curves(
        sessions, len(records), t0_ns
    )
    peak_active = int(active_vals.max()) if active_vals.size else 0
    # peak of the active+idle curve at each instant (NOT the sum of their
    # separate peaks): the most sessions concurrently in flight, idle included
    _, total_vals = add_step_functions(active_ts, active_vals, idle_ts, idle_vals)
    peak_total = int(total_vals.max()) if total_vals.size else 0
    new_isl_map = {sid: _session_new_isls(turns) for sid, turns in sessions.items()}
    new_isl_ts, new_isl_vals = _new_isl_curve(sessions, new_isl_map, t0_ns)
    (pf_ts, pf_vals), (dec_ts, dec_vals), (tot_ts, tot_vals) = _throughput_curves(
        sessions, t0_ns
    )
    kv_ts, kv_vals = _tokens_in_flight_curve(sessions, t0_ns)
    lat_t, lat_p50, lat_p99 = _latency_band(sessions, t0_ns)

    session_payload = []
    for ci, g in enumerate(groups):
        for m in g.members:
            session = sessions[m.sid]
            turns = []
            for r, depth, new_isl in zip(
                session, m.depths, new_isl_map[m.sid], strict=True
            ):
                start_s = (r["metadata"]["request_start_ns"] - t0_ns) / NANOS_PER_SECOND
                end_s = (r["metadata"]["request_end_ns"] - t0_ns) / NANOS_PER_SECOND
                ttft = _metric_value(r, "time_to_first_token")
                latency = _metric_value(r, "request_latency")
                isl = _metric_value(r, "input_sequence_length")
                osl = _metric_value(r, "output_sequence_length")
                # status: 1 = errored, 2 = context-overflow skip, 0 = ok
                # (cancelled records are dropped upstream in _load_records)
                status = (
                    1
                    if r.get("error")
                    else 2
                    if r["metadata"].get("context_overflow_skip")
                    else 0
                )
                turns.append(
                    [
                        round(start_s, 4),
                        round(end_s, 4),
                        round(ttft, 1) if ttft is not None else None,
                        round(latency, 1) if latency is not None else None,
                        int(isl) if isl is not None else None,
                        int(osl) if osl is not None else None,
                        depth,
                        r["metadata"].get("agent_depth"),
                        int(round(new_isl)),
                        status,
                    ]
                )
            merged = _merged_active_ns(session)
            gaps = [
                [
                    round((prev_end - t0_ns) / NANOS_PER_SECOND, 4),
                    round((start - t0_ns) / NANOS_PER_SECOND, 4),
                ]
                for (_, prev_end), (start, _) in zip(merged, merged[1:], strict=False)
            ]
            entry = {
                "id": m.sid,
                "conv": session[0]["metadata"]["conversation_id"],
                "slot": g.slot,
                "ci": ci,
                "row0": m.row0,
                "rows": m.rows,
                "turns": turns,
                "gaps": gaps,
            }
            if m.is_sub:
                entry["sub"] = True
            if ":aux:" in entry["conv"]:  # ::aux: (flat) or ::sa:..:aux:NNN (subagent)
                entry["aux"] = True
            if ":wg:" in entry["conv"]:  # parallel worker-group fan-out member
                entry["wg"] = True
            if m.sid != g.root_sid:
                entry["root"] = g.root_sid
            session_payload.append(entry)

    t_max_s = max(turn[1] for s in session_payload for turn in s["turns"])
    payload = {
        "run": run_dir.name,
        "tMax": round(t_max_s, 4),
        "nSlots": max(g.slot for g in groups) + 1,
        "target": concurrency,
        "rampS": ramp_dur_s,
        "benchEnd": bench_dur_s,
        "peakActive": peak_active,
        "peakTotal": peak_total,
        "peakNewIsl": int(new_isl_vals.max()) if new_isl_vals.size else 0,
        "peaks": _agent_peaks(sessions),
        "sessions": session_payload,
        "active": {
            "t": np.round(active_ts, 3).tolist(),
            "v": active_vals.astype(int).tolist(),
        },
        "idle": {
            "t": np.round(idle_ts, 3).tolist(),
            "v": idle_vals.astype(int).tolist(),
        },
        "newIsl": {
            "t": np.round(new_isl_ts, 3).tolist(),
            "v": np.round(new_isl_vals).astype(int).tolist(),
        },
        "peakTput": int(tot_vals.max()) if tot_vals.size else 0,
        "prefillTput": {
            "t": np.round(pf_ts, 3).tolist(),
            "v": np.round(pf_vals).astype(int).tolist(),
        },
        "decodeTput": {
            "t": np.round(dec_ts, 3).tolist(),
            "v": np.round(dec_vals).astype(int).tolist(),
        },
        "totalTput": {
            "t": np.round(tot_ts, 3).tolist(),
            "v": np.round(tot_vals).astype(int).tolist(),
        },
        "peakKv": int(kv_vals.max()) if kv_vals.size else 0,
        "kvCache": {
            "t": np.round(kv_ts, 3).tolist(),
            "v": np.round(kv_vals).astype(int).tolist(),
        },
        "peakLat": max(lat_p99) if lat_p99 else 0,
        "latencyBand": {"t": lat_t, "p50": lat_p50, "p99": lat_p99},
    }
    # gzip + base64 the payload so the single-file viewer stays small (the JSON
    # is mostly dense numeric arrays; gzip ~3x, base64 gives back ~33%, ~2.3x net
    # on large runs). The template inflates it client-side via the native
    # DecompressionStream; base64 is the safe way to carry the binary blob inside
    # a JS string literal (no "</", quote, or non-UTF8 hazards).
    payload_b64 = base64.b64encode(gzip.compress(orjson.dumps(payload), 6)).decode()
    html = (
        VIEWER_TEMPLATE.read_text()
        .replace("__AIPERF_TITLE__", run_dir.name)
        .replace("__AIPERF_PAYLOAD_B64__", payload_b64)
    )
    out_path = out or (run_dir / "swim_lane.html")
    out_path.write_text(html)
    return out_path
