# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session swim-lane plot for an AIPerf run.

Renders one horizontal lane per concurrent session slot (greedy reuse once a
session retires) plus a concurrency line chart driven by the same
``aiperf.analysis.sweepline`` primitives the metrics accumulator uses.

Inputs per run directory:
  - ``profile_export.jsonl``        (required)  per-record AIPerf export
  - ``profile_export_aiperf.json``  (optional)  used for ramp/benchmark axvlines

CLI: ``aiperf analyze swim-lane <run_dir> [<run_dir>...] [-o OUT]``
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import orjson
from numpy.typing import NDArray

from aiperf.analysis.sweepline import add_step_functions, concurrency_sweep_line
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.config.artifacts import OutputDefaults

PROFILE_JSONL = OutputDefaults.PROFILE_EXPORT_JSONL_FILE.name
PROFILE_JSON = OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE.name


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
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sessions[r["metadata"]["conversation_id"]].append(r)
    for turns in sessions.values():
        turns.sort(key=lambda r: r["metadata"]["turn_index"])
    return dict(sessions)


def _assign_slots(sessions: dict[str, list[dict]]) -> list[tuple[str, int]]:
    """Greedy slot packing: reuse a slot once its previous session retired."""
    ordered = sorted(
        sessions.items(),
        key=lambda kv: kv[1][0]["metadata"]["request_start_ns"],
    )
    slot_ends: list[int] = []
    assignments: list[tuple[str, int]] = []
    for conv_id, turns in ordered:
        start = turns[0]["metadata"]["request_start_ns"]
        end = turns[-1]["metadata"]["request_end_ns"]
        for i, slot_end in enumerate(slot_ends):
            if start >= slot_end:
                slot_ends[i] = end
                assignments.append((conv_id, i))
                break
        else:
            slot_ends.append(end)
            assignments.append((conv_id, len(slot_ends) - 1))
    return assignments


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
    starts = np.fromiter(
        (turns[0]["metadata"]["request_start_ns"] for turns in sessions.values()),
        dtype=np.float64,
        count=len(sessions),
    )
    ends = np.fromiter(
        (turns[-1]["metadata"]["request_end_ns"] for turns in sessions.values()),
        dtype=np.float64,
        count=len(sessions),
    )
    return starts, ends


def plot_swim_lane(run_dir: Path, out: Path | None = None) -> Path:
    """Render a swim-lane PNG for ``run_dir``. Returns the output path."""
    jsonl_path = run_dir / PROFILE_JSONL
    if not jsonl_path.is_file():
        raise SwimLaneError(f"{jsonl_path} not found")

    records = _load_records(jsonl_path)
    if not records:
        raise SwimLaneError(f"no valid records in {jsonl_path}")

    sessions = _group_into_sessions(records)
    assignments = _assign_slots(sessions)
    ramp_dur_s, bench_dur_s = _load_bench_config(run_dir)

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
    n_slots = max(slot for _, slot in assignments) + 1

    cmap = plt.colormaps["tab20"]
    session_color = {
        conv_id: cmap(i % 20) for i, (conv_id, _) in enumerate(assignments)
    }

    bars: list[
        tuple[str, int, list[tuple[float, float]], list[tuple[float, float]]]
    ] = []
    for conv_id, slot in assignments:
        turns = sessions[conv_id]
        active: list[tuple[float, float]] = []
        idle: list[tuple[float, float]] = []
        for i, turn in enumerate(turns):
            rs = (turn["metadata"]["request_start_ns"] - t0_ns) / NANOS_PER_SECOND
            re = (turn["metadata"]["request_end_ns"] - t0_ns) / NANOS_PER_SECOND
            active.append((rs, re - rs))
            if i > 0:
                prev_end = (
                    turns[i - 1]["metadata"]["request_end_ns"] - t0_ns
                ) / NANOS_PER_SECOND
                idle.append((prev_end, rs - prev_end))
        bars.append((conv_id, slot, active, idle))

    slot_session_starts: dict[int, list[float]] = defaultdict(list)
    for conv_id, slot in assignments:
        first = (
            sessions[conv_id][0]["metadata"]["request_start_ns"] - t0_ns
        ) / NANOS_PER_SECOND
        slot_session_starts[slot].append(first)
    reuse_markers = [
        (slot, s) for slot, starts in slot_session_starts.items() for s in starts[1:]
    ]

    # Sweep-line curves — reuse the accumulator's primitive. Work in
    # seconds-from-t0 so float64 has enough precision (absolute ns timestamps
    # are ~1.7e18, eating ~16 sig figs and collapsing sub-second resolution).
    starts_ns = np.fromiter(
        (
            r["metadata"]["request_start_ns"]
            for turns in sessions.values()
            for r in turns
        ),
        dtype=np.float64,
        count=len(records),
    )
    ends_ns = np.fromiter(
        (r["metadata"]["request_end_ns"] for turns in sessions.values() for r in turns),
        dtype=np.float64,
        count=len(records),
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

    bar_height = 0.7
    swim_height = max(4.0, n_slots * 0.45 + 1.0)
    fig, (ax_swim, ax_conc) = plt.subplots(
        2,
        1,
        figsize=(20, swim_height + 2.5),
        gridspec_kw={"height_ratios": [swim_height, 2.5]},
        sharex=True,
    )

    slot_spans: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for conv_id, slot in assignments:
        turns = sessions[conv_id]
        s = (turns[0]["metadata"]["request_start_ns"] - t0_ns) / NANOS_PER_SECOND
        e = (turns[-1]["metadata"]["request_end_ns"] - t0_ns) / NANOS_PER_SECOND
        slot_spans[slot].append((s, e))

    for slot in range(n_slots):
        y = float(slot)
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
                y,
                right - left,
                left=left,
                height=bar_height,
                color="#f0f0f0",
                edgecolor="none",
                linewidth=0,
            )

    for conv_id, slot, active, idle in bars:
        y = float(slot)
        color = session_color[conv_id]
        idle_color = (*color[:3], 0.25)
        for left, width in idle:
            ax_swim.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=idle_color,
                edgecolor="none",
                linewidth=0,
            )
        for left, width in active:
            ax_swim.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="none",
                linewidth=0,
            )

    for slot, x in reuse_markers:
        ax_swim.plot(
            [x, x],
            [slot - bar_height / 2, slot + bar_height / 2],
            color="black",
            linewidth=1.5,
            solid_capstyle="butt",
        )

    ax_swim.set_xlim(0, max(t_max_s * 1.01, 1.0))
    ax_swim.set_ylim(-0.5, n_slots - 0.5)
    ax_swim.set_ylabel("Session Slot", fontsize=12)
    ax_swim.set_title(
        f"{run_dir.name}  —  {len(sessions)} sessions across {n_slots} slots",
        fontsize=13,
        fontweight="bold",
    )
    yticks = list(range(n_slots))
    stride = 1 if n_slots <= 30 else 2 if n_slots <= 60 else 5
    ax_swim.set_yticks(yticks[::stride])
    ax_swim.set_yticklabels([str(i) for i in yticks[::stride]])
    ax_swim.invert_yaxis()
    ax_swim.grid(axis="x", alpha=0.3, linewidth=0.5)

    legend_color = cmap(0)
    ax_swim.legend(
        handles=[
            mpatches.Patch(color=legend_color, label="Active request"),
            mpatches.Patch(color=(*legend_color[:3], 0.25), label="Inter-turn delay"),
            plt.Line2D(
                [0], [0], color="black", linewidth=1.5, label="New session in slot"
            ),
        ],
        loc="lower left",
        fontsize=9,
        ncol=2,
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
    ax_conc.set_ylim(0, max(n_slots * 1.15, 1.0))
    ax_conc.legend(loc="upper left", fontsize=9)
    ax_conc.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_path = out or (run_dir / "swim_lane.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
