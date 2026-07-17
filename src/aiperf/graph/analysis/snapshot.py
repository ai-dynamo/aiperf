# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot-at-t* partition of a trace timeline (native, over ``TraceTimeline``).

Reproduces the agentx "snapshot at t*" liveness partition (see
``aiperf/timing/trajectory_source.py::_snapshot_for``) natively over the firing
timeline produced by :func:`elaborate_trace`.

A sampling instant ``t*`` splits a trace's firings into a warmup prefix
(``arrival_offset_us < t*`` -- cache-priming history dispatched immediately) and
a profiled set (``arrival_offset_us >= t*`` -- dispatched at its offset from
``t*``). Firings carrying no offset are profiled at dispatch 0 (agentx's
no-timestamp fallback).

The flat IR emits a single-stream timeline (every firing on the trace's own
stream), so the partition is a plain ``arrival_offset_us`` vs ``t*`` test per
firing.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.dataset.graph.models import ParsedGraph, TraceRecord
from aiperf.graph.analysis.timeline import (
    Firing,
    TraceTimeline,
    elaborate_trace,
)


@dataclass(slots=True, frozen=True)
class SnapshotFiring:
    """One timeline firing located in the snapshot, with its dispatch timing.

    Attributes:
        firing: The underlying :class:`Firing` from the trace timeline.
        dispatch_offset_us: ``0`` for warmup firings; ``max(0, arrival - t*)``
            for profiled firings (a firing with no ``arrival_offset_us`` is
            profiled at ``0``).
    """

    firing: Firing
    dispatch_offset_us: int


@dataclass(slots=True, frozen=True)
class Snapshot:
    """A trace timeline partitioned at a sampling instant ``t*``.

    Attributes:
        t_star_us: The sampling instant in microseconds (pinned to ``0`` for a
            ``full_replay`` snapshot).
        warmup: Firings whose ``arrival_offset_us < t*`` (cache-priming history;
            dispatch offset ``0``), in timeline order.
        profiled: Firings at/after ``t*`` (dispatched during PROFILING at their
            offset from ``t*``), sorted by ``(dispatch_offset_us, cohort)``.
    """

    t_star_us: int
    warmup: tuple[SnapshotFiring, ...]
    profiled: tuple[SnapshotFiring, ...]


def trace_duration_us(parsed: ParsedGraph, trace: TraceRecord) -> int:
    """Return the trace's intrinsic wall-clock span in microseconds.

    The largest ``arrival_offset_us`` across the trace's firings. A trace where
    no node carries timing returns ``0``. Snapshot-at-``t*`` lane construction
    uses this to choose a sampling instant as a fraction of the trace's duration.
    """
    return elaborate_trace(parsed, trace).duration_us()


def compute_snapshot(
    parsed: ParsedGraph,
    trace: TraceRecord,
    *,
    t_star_us: int,
    full_replay: bool = False,
) -> Snapshot:
    """Partition ``trace``'s firing timeline at the sampling instant ``t_star_us``.

    ``arrival_offset_us < t*`` firings go to warmup (dispatch offset ``0``); the
    rest go to profiled at ``max(0, arrival - t*)`` (a firing with no offset is
    profiled at ``t*`` -> dispatch ``0``).

    ``full_replay`` (recycle path) pins ``t*`` to ``0`` and places every firing
    in profiled at its own ``arrival_offset_us`` -- byte parity with agentx's
    turn-0 recycle.
    """
    if full_replay:
        t_star_us = 0

    timeline: TraceTimeline = elaborate_trace(parsed, trace)

    warmup: list[SnapshotFiring] = []
    profiled: list[SnapshotFiring] = []
    for firing in timeline.firings:
        off = firing.arrival_offset_us
        if not full_replay and off is not None and off < t_star_us:
            warmup.append(SnapshotFiring(firing=firing, dispatch_offset_us=0))
        else:
            absolute = off if off is not None else t_star_us
            profiled.append(
                SnapshotFiring(
                    firing=firing,
                    dispatch_offset_us=max(0, absolute - t_star_us),
                )
            )

    profiled.sort(key=lambda s: (s.dispatch_offset_us, s.firing.cohort))
    return Snapshot(
        t_star_us=t_star_us,
        warmup=tuple(warmup),
        profiled=tuple(profiled),
    )
