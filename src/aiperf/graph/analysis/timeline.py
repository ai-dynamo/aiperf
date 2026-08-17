# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-engine static elaboration of a trace into an ordered firing timeline.

``elaborate_trace`` performs an in-engine dataflow dry-run: it seeds the readiness
frontier from the dataflow ``Scheduler.entry_nodes()``, advances a lightweight
static arrival tracker on each fired node's ``write_channels``, and computes
the next frontier from every scheduled successor -- ``successors_after``
(completion edges) AND ``start_anchored_successors`` (dispatch-anchored edges,
which the runtime schedules at the predecessor's dispatch) -- with fan-in
satisfaction against ``channels.producers_per_channel``.

There are no lockstep rounds and no channel reduction: the dry-run only needs
arrival counts (how many producers have written each channel) to resolve
AND-fan-in joins, never the channel values themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.dataset.graph.models import (
    GraphRecord,
    NodeKind,
    NodeUnion,
    ParsedGraph,
    TraceRecord,
    resolve_trace_graph,
)
from aiperf.graph.channels import producers_per_channel
from aiperf.graph.scheduler import Scheduler


class GraphCycleError(RuntimeError):
    """Raised when ``elaborate_trace(..., depth_cap=N)`` exceeds the firing cap.

    In-engine replacement for the firing-plan cycle guard: a trace whose readiness
    walk emits more than the cap likely traverses a cycle. Runtime callers pass
    ``depth_cap=None`` (unbounded) on already-validated graphs; validators pass
    a finite cap so a cyclic graph fails loudly instead of looping forever.

    Also raised, independently of ``depth_cap``, when the frontier drains with
    nodes whose inputs can never arrive (fan-in deadlock).
    """


@dataclass(slots=True, frozen=True)
class Firing:
    """One node firing in a trace's static elaboration."""

    node_id: str
    kind: NodeKind
    arrival_offset_us: int | None
    cohort: int


@dataclass(slots=True, frozen=True)
class TraceTimeline:
    """A trace's full firing sequence in parallel-readiness order.

    ``cohort`` indexes the parallel-readiness frontier; it is a derived view
    for parallel-TTFT cohorting and visualization, NOT an execution barrier.
    """

    firings: tuple[Firing, ...]

    def duration_us(self) -> int:
        """Return the max ``arrival_offset_us`` over the timeline (0 default).

        Firings carrying no offset contribute nothing; a trace where no node
        carries timing returns 0.
        """
        return max(
            (
                f.arrival_offset_us
                for f in self.firings
                if f.arrival_offset_us is not None
            ),
            default=0,
        )


def _inputs_satisfied(
    node: NodeUnion,
    arrivals: dict[str, int],
    all_counts: dict[str, int],
) -> bool:
    """Return True iff every ``ChannelRequirement`` on ``node`` is satisfied.

    Pure static analogue of the dataflow channel store's ``"all"`` fan-in gate:
    ``count == "all"`` resolves to the channel's static producer count
    (``producers_per_channel``); a finite ``count`` requires that many arrivals.
    A node with no ``inputs`` always passes (OR-fan-in successor-walk default).
    """
    inputs = node.inputs
    for req in inputs:
        required = all_counts.get(req.channel, 0) if req.count == "all" else req.count
        if arrivals.get(req.channel, 0) < required:
            return False
    return True


def _elaborate_graph(
    graph: GraphRecord,
    scheduler: Scheduler,
    *,
    depth_cap: int | None,
    emitted: list[Firing],
) -> None:
    """Walk the graph's readiness frontier, appending a Firing per fired node.

    Static mirror of the executor's scheduling: seed from ``entry_nodes()``,
    emit firings per frontier in lexical node order, advance the arrival
    tracker on each fired node's ``write_channels``, then schedule successors.
    Both anchor kinds are followed -- ``successors_after`` (completion edges)
    AND ``start_anchored_successors`` (the runtime schedules those at the
    predecessor's DISPATCH, which collapses onto the same firing step here) --
    so start-anchored subtrees elaborate exactly as they fire live. A scheduled
    node whose fan-in is not yet satisfied stays ``pending`` and re-enters the
    frontier once later arrivals satisfy it, mirroring the runtime's parked
    ``await_inputs`` (a runtime node is scheduled once, then blocks). ``cohort``
    is the monotone frontier index.
    """
    all_counts = producers_per_channel(graph)
    arrivals: dict[str, int] = {}
    scheduled: set[str] = set(scheduler.entry_nodes())
    pending: set[str] = set(scheduled)

    def _satisfied() -> list[str]:
        return [
            nid
            for nid in pending
            if _inputs_satisfied(graph.nodes[nid], arrivals, all_counts)
        ]

    frontier = _satisfied()
    cohort = 0
    fired_count = 0
    while frontier:
        pending.difference_update(frontier)
        for node_id in sorted(frontier):
            node = graph.nodes[node_id]
            emitted.append(
                Firing(
                    node_id=node_id,
                    kind=node.node_type,
                    arrival_offset_us=node.arrival_offset_us,
                    cohort=cohort,
                )
            )
            fired_count += 1
            if depth_cap is not None and fired_count > depth_cap:
                raise GraphCycleError(
                    f"trace elaboration exceeded {depth_cap} firings; "
                    "trace likely traverses a cycle"
                )
            for ch in node.write_channels:
                arrivals[ch] = arrivals.get(ch, 0) + 1
        for node_id in frontier:
            for succ in (
                *scheduler.successors_after(node_id),
                *scheduler.start_anchored_successors(node_id),
            ):
                if succ in scheduled:
                    continue
                scheduled.add(succ)
                pending.add(succ)
        frontier = _satisfied()
        cohort += 1

    if pending:
        # The frontier drained with nodes still unsatisfied: every reachable
        # producer already fired, so their requirement counts exceed the arrivals
        # anything can still supply. This is a fan-in deadlock, NOT a cycle -- a
        # true A->B->A off START fires both nodes once and leaves ``pending``
        # empty (cycles are caught upstream by the adapter and the executor's
        # ``_schedule`` guard). ``fired_count`` cannot trip ``depth_cap`` here
        # (each node fires at most once), so the stall has to be detected on exit.
        raise GraphCycleError(
            f"trace elaboration stalled with {len(pending)} node(s) whose inputs "
            f"can never arrive ({', '.join(sorted(pending)[:5])}); "
            "fan-in deadlock (unsatisfiable AND-join)"
        )


def elaborate_trace(
    parsed: ParsedGraph,
    trace: TraceRecord,
    *,
    depth_cap: int | None = None,
) -> TraceTimeline:
    """In-engine dataflow dry-run of one trace into a full firing timeline.

    Walks the trace's top-level graph (``resolve_trace_graph``) frontier.
    Fan-in reuses the dataflow scheduler and channel model; no lockstep-round
    state, no channel reduction.

    ``depth_cap`` bounds the total firings emitted (raises ``GraphCycleError``
    on overflow) for validators that may be called on cyclic graphs; ``None``
    (default) is unbounded for runtime callers on validated graphs.
    """
    graph = resolve_trace_graph(parsed, trace)
    scheduler = Scheduler(graph)
    emitted: list[Firing] = []
    _elaborate_graph(
        graph,
        scheduler,
        depth_cap=depth_cap,
        emitted=emitted,
    )
    return TraceTimeline(firings=tuple(emitted))
