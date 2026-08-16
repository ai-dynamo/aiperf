# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace t* planning, lane salting, and the dataset draw.

``t*`` is the SNAPSHOT INSTANT: the point in a trace's own recorded timeline at
which this replay instance starts, in MICROSECONDS from that trace's start (a
float -- no integer truncation, for AgentX parity). ``t*=0`` replays the trace
in full. ``t*>0`` chops the trie
(:func:`~aiperf.timing.snapshot_chop.chop_trie_at_tstar`) so the instance
resumes mid-conversation: turns recorded before ``t*`` are dropped from the
profiled graph (they were dispatched during warmup, so the server holds their
KV) while each survivor keeps its full prompt prefix and is re-rooted from
``START`` at ``arrival_offset_us - t*``. The per-trace value is drawn
uniformly over ``[start_min_ratio, start_max_ratio] * trace_duration_us`` under
a per-(trace, lane)-salted RNG, so it is deterministic given the run seed and
decorrelated across traces and lanes.

Answers "which trace, at which t\\*, for which lane" on behalf of
``AgentGraphReplayStrategy``. The strategy owns admission and dispatch; everything
about WHICH template a freed lane serves and WHERE in its recorded timeline
that instance resumes lives here.

The default ``[0, 0]`` window collapses every plan to ``t*=0`` (full recorded
replay, identity rewrite), so the profiling path is byte-identical unless a
caller supplies a positive window.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import msgspec

from aiperf.dataset.graph.graph_path_catalog import (
    CatalogContext,
    build_catalog_context,
)
from aiperf.graph.scheduler import collapse_leading_start_offsets
from aiperf.timing.agent_graph_source import AgentGraphConversationSource, GraphTrace
from aiperf.timing.agent_graph_trace_view import parsed_for_trace
from aiperf.timing.snapshot_chop import chop_trie_at_tstar
from aiperf.timing.strategies.graph_warmup import GraphWarmupKind, rewrite_for_warmup

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import ParsedGraph, TraceRecord
    from aiperf.plugin.enums import DatasetSamplingStrategy

__all__ = ["GraphTracePlanner", "seed_for_draw_pass"]


def seed_for_draw_pass(base_seed: int, pass_index: int) -> int:
    """Derive a per-pass RNG seed for the shuffle draw (matches the t* salt).

    Mirrors :func:`aiperf.timing.agent_graph_source._seed_for_trace_lane`: SHA-256
    over ``f"{base_seed}:dataset-draw:{pass_index}"`` and take the low 8 bytes,
    so each recycle pass re-permutes under a distinct-yet-deterministic seed
    derived from the run's ``t_star_random_seed``. Same base seed + pass index
    always yields the same permutation (cross-run reproducibility), while
    different passes decorrelate -- the same order-independent SHA-256
    derivation the conversation-plane samplers use via ``rng.derive``.
    """
    digest = hashlib.sha256(f"{base_seed}:dataset-draw:{pass_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class GraphTracePlanner:
    """Resolves per-(trace, lane) t* plans and the cross-trace dataset draw.

    Invariants:
        * **Fully synchronous.** Every method is a plain ``def``, so the event
          loop cannot interleave a read and a write of the caches below and none
          of them needs a lock. Keep it that way: making any method ``async``
          would silently introduce a race on ``_lane_plans`` / ``_lane_sources``
          / ``_draw_perm_cache``.
        * **Lane 0 is the prebuilt plan.** ``plan_for_lane(trace, 0)`` returns
          the very object in ``plans[trace.id]``, so the default single-pass path
          stays byte-identical to the pre-lane-fan-out behavior, and
          ``lane_salted_t_star(trace, 0)`` reads that same plan's ``t_star_us``.
        * **Caches are unbounded**, sized by (corpus x lanes). That is bounded in
          practice because the planner is constructed per-phase by
          ``AgentGraphReplayStrategy`` and discarded with it.
    """

    def __init__(
        self,
        *,
        parsed: ParsedGraph,
        start_min_ratio: float,
        start_max_ratio: float,
        t_star_random_seed: int,
        dataset_sampling_strategy: DatasetSamplingStrategy | None = None,
    ) -> None:
        """Build the t* snapshot source, per-trace t* plans, and node catalog.

        Args:
            parsed: The built ``ParsedGraph`` whose traces the phase replays.
            start_min_ratio: Lower bound (fraction of duration) of the t* window.
            start_max_ratio: Upper bound of the t* window. Together with
                ``start_min_ratio=0.0`` a ``0.0`` here selects full recorded replay.
            t_star_random_seed: Base seed for per-trace t* sampling (trace-salted).
            dataset_sampling_strategy: Resolved run-level strategy consumed by
                :meth:`draw_index`. SHUFFLE/RANDOM remap the draw through a
                seeded permutation; SEQUENTIAL/None keep the byte-identical
                cursor.
        """
        self._parsed = parsed
        self._start_min_ratio = start_min_ratio
        self._start_max_ratio = start_max_ratio
        self._t_star_random_seed = t_star_random_seed
        self._dataset_sampling_strategy = dataset_sampling_strategy

        self._source = AgentGraphConversationSource(
            parsed=parsed,
            start_min_ratio=start_min_ratio,
            start_max_ratio=start_max_ratio,
            random_seed=t_star_random_seed,
        )
        # ``{trace_id: GraphTrace}`` -- the lane-0 per-trace t* plan (the default
        # single-pass disposition). Lanes > 0 and recycle passes resolve their own
        # lane-salted plan lazily via :meth:`plan_for_lane`.
        self._plans: dict[str, GraphTrace] = {
            gt.trace_id: gt for gt in self._source.iter_traces()
        }
        self._catalog = build_catalog_context(parsed)
        # Per-(template_trace_id, lane_index) t* plan cache. AgentX seeds t* per
        # ``(trace_id, lane)`` so the same template recurring across lanes (or
        # recycled onto one lane) resumes at a DIFFERENT t*. Built lazily so a
        # large concurrency over a small corpus only plans the lanes it runs.
        self._lane_plans: dict[tuple[str, int], GraphTrace] = {}
        # Per-lane t* source cache (catalog/namespace map are lane-independent),
        # so a fan-out reuses one ``AgentGraphConversationSource`` per lane instead
        # of rebuilding the corpus catalog on every per-trace plan.
        self._lane_sources: dict[int, AgentGraphConversationSource] = {}
        # Per-(total, pass) shuffled trace-index permutation cache for the
        # ``--dataset-sampling-strategy`` draw (:meth:`draw_index`). Built lazily
        # once per pass and reused for every draw in that pass so repeated draws
        # are cheap and consistent. Empty / unused under the default SEQUENTIAL
        # draw. Locking: see the fully-synchronous class invariant.
        self._draw_perm_cache: dict[tuple[int, int], list[int]] = {}

    @property
    def plans(self) -> dict[str, GraphTrace]:
        """``{trace_id: GraphTrace}`` -- the lane-0 per-trace t* plans."""
        return self._plans

    @property
    def catalog(self) -> CatalogContext:
        """The build-time node catalog context for the whole corpus."""
        return self._catalog

    @property
    def source(self) -> AgentGraphConversationSource:
        """The lane-0 t* snapshot source."""
        return self._source

    def plan_for_lane(self, trace: TraceRecord, lane_index: int) -> GraphTrace | None:
        """Resolve the t* plan for ``trace`` on ``lane_index`` (AgentX lane salt).

        Lane ``0`` reuses the prebuilt ``plans`` entry (byte-identical to the
        single-pass path). Higher lanes / recycle passes draw a DISTINCT
        lane-salted t* (``sha256(seed:trace_id:lane)``) so the same template
        recurring across lanes resumes at a different snapshot instant, exactly
        as AgentX's ``_build_trajectory_for_lane`` seeds per absolute lane. The
        per-lane source is cached by ``(trace_id, lane_index)`` so repeated
        recycle passes onto the same lane re-plan only once. With the default
        ``[0, 0]`` window every lane collapses to ``t*=0`` (identity), so lane
        fan-out adds no t* divergence on the working profiling path.
        """
        key = (trace.id, lane_index)
        cached = self._lane_plans.get(key)
        if cached is not None:
            return cached
        if lane_index == 0:
            plan = self._plans.get(trace.id)
        else:
            # Plan ONLY this trace on the lane-salted source. ``iter_traces`` would
            # plan EVERY corpus trace just to find one (O(lanes * traces) snapshot
            # elaborations across a fan-out), so call ``_plan_trace`` directly. The
            # per-lane source is cached by ``lane_index`` (catalog / namespace map
            # are lane-independent) so a fan-out reuses one source per lane.
            plan = self._lane_source(lane_index)._plan_trace(trace)
        if plan is not None:
            self._lane_plans[key] = plan
        return plan

    def _lane_source(self, lane_index: int) -> AgentGraphConversationSource:
        """Return the (cached) lane-salted t* source for ``lane_index``.

        The source's catalog + namespace map are lane-independent, so one source
        per lane is reused across every trace planned on that lane (avoids the
        O(lanes * traces) catalog rebuild a per-call construction would incur).
        """
        source = self._lane_sources.get(lane_index)
        if source is None:
            source = AgentGraphConversationSource(
                parsed=self._parsed,
                start_min_ratio=self._start_min_ratio,
                start_max_ratio=self._start_max_ratio,
                random_seed=self._t_star_random_seed,
                lane_index=lane_index,
            )
            self._lane_sources[lane_index] = source
        return source

    def select_corpus(
        self, traces: list[TraceRecord], limit: int | None
    ) -> list[TraceRecord]:
        """Select ``limit`` traces from ``traces`` through the dataset draw.

        This is how ``--num-conversations N`` bounds a graph replay: a
        SELECTION-time bound on WHICH templates the phase enumerates, never a
        mid-trace stop condition (a fan-out DAG must always run to completion --
        see ``issue_graph_credit``'s deliberate ``can_send_dag_child_turn``).

        Selection obeys ``--dataset-sampling-strategy``, and its DEFAULT
        (``sequential``) takes the N EARLIEST traces by recorded start
        (:meth:`_temporal_order`). Bounding a recorded corpus is a TEMPORAL
        subsample -- "which slice of the captured traffic" -- so a scattered
        draw would destroy the arrival process: 3 traces drawn from across a
        500-trace, hour-long capture are three sparse arrivals separated by
        large idle gaps, a load shape nothing like the recording. Ordering by
        recorded start keeps the selection contiguous and deterministic.

        CORPUS order is deliberately NOT the ordering: the dynamo adapter emits
        its traces id-sorted, so a corpus-order prefix is a LEXICOGRAPHIC slice
        and, with unordered session ids, produces exactly the scattered shape
        this is meant to avoid.

        A shuffle is right for a CONTENT subsample ("which prompts"), so an
        explicit ``shuffle``/``random`` still shuffles.

        The unbounded path is untouched under every strategy: this returns
        ``traces`` itself, so sequential stays byte-identical wherever it
        currently matters (unbounded replay and lane recycle).

        ``limit`` of ``None`` / ``<= 0`` / ``>= len(traces)`` returns the corpus
        unchanged (no cloning, no wrap): asking for more than exists replays
        everything exactly once. The draw is a single pass, so it is inherently
        without replacement.

        Args:
            traces: The loaded corpus, in corpus order.
            limit: Requested trace count, or ``None`` for no bound.

        Returns:
            The selected traces, in draw order.
        """
        total = len(traces)
        if limit is None or limit <= 0 or limit >= total:
            return traces
        if self._draw_is_shuffled():
            # Pass 0 of the SAME seeded permutation ``draw_index`` uses, so an
            # explicit shuffle agrees with the lane draw for a given seed.
            order = self._draw_permutation(0, total)
        else:
            order = self._temporal_order(traces)
        return [traces[index] for index in order[:limit]]

    def _temporal_order(self, traces: list[TraceRecord]) -> list[int]:
        """Trace indices ordered by RECORDED START -- the timeline slice.

        Corpus order cannot serve as the temporal order: the dynamo adapter
        emits its traces id-sorted, so slicing corpus order yields the N
        lexicographically-smallest session ids. With unordered ids that is a
        scattered sample across the whole capture -- precisely the "sparse
        arrivals separated by large idle gaps" shape :meth:`select_corpus`
        exists to avoid. Sorting by recorded start makes the bound the
        contiguous head of the capture it is documented to be.

        Ties (and a wholly untimestamped corpus) fall back to corpus order via
        the index tie-break, so a hand-authored corpus with no timestamps
        anywhere keeps the byte-identical prefix it had before.

        Untimestamped traces sort LAST: they cannot be paced, and
        ``_validate_recorded_starts`` explicitly treats bounding onto a fully
        timestamped subset as a legitimate run -- so preferring timestamped
        traces turns a corpus that would otherwise be refused into one that
        replays faithfully.
        """
        from aiperf.dataset.graph.models import trace_recorded_start_ms

        def sort_key(indexed: tuple[int, TraceRecord]) -> tuple[int, int, int]:
            index, trace = indexed
            start = trace_recorded_start_ms(self._parsed, trace)
            # (untimestamped-last, recorded start, corpus position)
            return (1, 0, index) if start is None else (0, start, index)

        return [index for index, _ in sorted(enumerate(traces), key=sort_key)]

    def draw_index(self, x: int, total: int) -> int:
        """Remap a monotonic draw counter ``x`` to a trace index in ``[0, total)``.

        This is the single choke point every cross-trace draw in the lane
        fan-out / recycle loop routes through, so ``--dataset-sampling-strategy``
        governs WHICH template a freed lane serves without changing the draw
        counters.

        * ``sequential`` (or ``None``): return ``x % total`` -- byte-for-byte the
          historical cursor-with-wrap draw. Sequential must be unchanged.
        * ``shuffle``: map ``x`` to ``perm[pass][x % total]`` where
          ``pass = x // total``, drawing each pass's permutation from a
          pass-salted seed (:func:`seed_for_draw_pass`). Each pass of ``total``
          draws covers every index exactly ONCE (without replacement), then a
          fresh seeded permutation begins -- the same music-shuffle contract the
          conversation-plane ``ShuffleSampler`` provides.
        * ``random``: coerced to ``shuffle`` (without-replacement) semantics.
          Each lane recycle here is a single corpus pass, so with-replacement
          ``random`` would duplicate/omit templates within a pass; coercing to
          shuffle keeps coverage exact. random == shuffle in this context.
        """
        if total <= 0:
            return 0
        if not self._draw_is_shuffled():
            return x % total
        pass_index, offset = divmod(x, total)
        return self._draw_permutation(pass_index, total)[offset]

    def _draw_is_shuffled(self) -> bool:
        """True iff the resolved sampling strategy permutes (shuffle / random).

        ``None`` and ``sequential`` take the byte-identical ``x % total`` draw;
        ``shuffle`` and ``random`` (coerced to without-replacement) permute.
        """
        strategy = self._dataset_sampling_strategy
        if strategy is None:
            return False
        from aiperf.plugin.enums import DatasetSamplingStrategy

        return strategy in (
            DatasetSamplingStrategy.SHUFFLE,
            DatasetSamplingStrategy.RANDOM,
        )

    def _draw_permutation(self, pass_index: int, total: int) -> list[int]:
        """Return the cached seeded permutation of ``range(total)`` for a pass.

        Built once per ``(total, pass_index)`` from a pass-salted numpy RNG
        (:func:`seed_for_draw_pass` -> ``np.random.default_rng`` -> in-place
        Fisher-Yates ``shuffle``, matching ``ShuffleSampler``'s numpy shuffle),
        then reused for every draw in that pass so draws are cheap + consistent.
        """
        key = (total, pass_index)
        cached = self._draw_perm_cache.get(key)
        if cached is not None:
            return cached
        import numpy as np

        rng = np.random.default_rng(
            seed_for_draw_pass(self._t_star_random_seed, pass_index)
        )
        perm = list(range(total))
        rng.shuffle(perm)
        self._draw_perm_cache[key] = perm
        return perm

    def lane_salted_t_star(self, trace: TraceRecord, lane_index: int) -> float:
        """Compute ``trace``'s lane-salted t* (us) WITHOUT building a t* source.

        Reuses the prebuilt lane-0 plan when ``lane_index == 0`` (the common
        single-pass case). For higher lanes it inlines ``_sample_t_star``'s seed +
        duration math (``_seed_for_trace_lane`` -> ``rng.uniform(lo, hi)`` over the
        ratio window) so the spawnability scan never constructs a
        ``AgentGraphConversationSource`` (whose ``__init__`` rebuilds the whole-corpus
        node catalog -- an O(lanes * traces) cost across a fan-out the scan does
        not need). The value is byte-identical to :meth:`plan_for_lane`'s
        ``t_star_us`` because it is the SAME computation on the SAME seed.
        """
        if lane_index == 0:
            plan = self._plans.get(trace.id)
            return plan.t_star_us if plan is not None else 0.0
        import numpy as np

        from aiperf.graph.analysis import trace_duration_us
        from aiperf.timing.agent_graph_source import _seed_for_trace_lane

        # ``trace_duration_us`` resolves the trace's OWN graph internally
        # (``elaborate_trace`` -> ``resolve_trace_graph``), so passing the full
        # multi-graph parse matches ``AgentGraphConversationSource._sample_t_star``
        # byte-for-byte (it too uses the source's raw ``self._parsed``).
        duration_us = trace_duration_us(self._parsed, trace)
        if duration_us <= 0:
            return 0.0
        lo = self._start_min_ratio * duration_us
        hi = self._start_max_ratio * duration_us
        if hi <= lo:
            return float(lo)
        rng = np.random.default_rng(
            _seed_for_trace_lane(self._t_star_random_seed, trace.id, lane_index)
        )
        return float(rng.uniform(lo, hi))

    @staticmethod
    def is_trie_graph(parsed: ParsedGraph) -> bool:
        """True iff ``parsed`` is a segment-trie graph (the flat ``LlmNode`` form
        the dynamo adapter emits).

        The trie builder stamps ``metadata["trie"]`` on every emitted ``LlmNode``.
        Detecting the trie marker on any top-level node confirms the trie path; a
        non-trie parse reaching ``t*>0`` is a lowering bug (raises in
        :meth:`graph_at_t_star`).
        """
        return any("trie" in node.metadata for node in parsed.graph.nodes.values())

    def graph_at_t_star(
        self,
        trace: TraceRecord,
        plan: GraphTrace | None,
        *,
        warmup: GraphWarmupKind | None,
        burst_phase_starts: bool,
    ) -> tuple[ParsedGraph, TraceRecord]:
        """Reconstruct the per-trace graph + trace at this instance's t* disposition.

        ``plan`` is the lane-resolved t* plan for this instance (lane-salted on
        pass 0, ``None`` == full t*=0 replay for recycle passes). Returns
        ``(parsed_to_run, trace_to_run)`` for the ``TraceExecutor``.

        PROFILING: ``t*==0`` (default full-replay window, or any recycle pass)
        => IDENTITY (byte-identical to the original) unless
        ``burst_phase_starts`` collapses the leading offsets. ``t*>0`` =>
        ``chop_trie_at_tstar`` (a frontier chop re-rooting each live chain at
        the ``t*`` frontier) for a trie graph. Surviving nodes keep their ids so
        the adapter resolves the unmodified catalog ordinal.

        ``burst_phase_starts`` applies at BOTH dispositions. A t*=0 trie graph
        already carries leading START offsets -- ``interval_order`` roots every
        gap-started chain at START with its warped arrival offset -- so skipping
        the collapse here made ``--burst-phase-starts`` a silent no-op in the
        DEFAULT full-replay configuration, which is the only configuration most
        runs use.

        WARMUP (BOUNDARY_SNAPSHOT): :func:`rewrite_for_warmup` -- the flat
        boundary-priming graph (one boundary turn per chain live at t*,
        START-rooted, zero leading offsets). ``t*<=0`` yields an EMPTY warmup
        graph so the instance finalizes immediately (the ``timing.config``
        auto-warmup contract).

        WARMUP (RECORDED): the corpus authored this graph AS warmup
        (``ParsedGraph.warmup_traces``). It is already exactly what must go on
        the wire; ``rewrite_for_warmup`` is the SNAPSHOT transform (derive
        priming turns from a profiled trace) and would correctly-but-uselessly
        reduce it to nothing, so it dispatches verbatim. ``plan`` is ignored
        by design -- a recorded warmup graph has no recorded timeline to resume
        into.

        A non-trie graph at ``t*>0`` is a lowering bug (raises). Multi-graph
        workloads project onto each trace's OWN graph via ``parsed_for_trace``
        first (else a non-first trace runs the first file's topology).

        ``warmup`` / ``burst_phase_starts`` are passed in rather than read
        from a config: the planner is phase-agnostic, and the owning strategy
        is the thing that knows which phase it is running.
        """
        parsed = parsed_for_trace(self._parsed, trace)
        if warmup is GraphWarmupKind.RECORDED:
            # Corpus-authored warmup graph: dispatch verbatim. burst_phase_starts
            # collapses leading offsets for consistency with the other warmup arm.
            return (
                burst_collapse_leading_offsets(parsed) if burst_phase_starts else parsed
            ), trace
        t_star_us = plan.t_star_us if plan is not None else 0
        if t_star_us <= 0:
            if warmup is GraphWarmupKind.BOUNDARY_SNAPSHOT:
                return rewrite_for_warmup(parsed, 0), trace
            if burst_phase_starts:
                return burst_collapse_leading_offsets(parsed), trace
            return parsed, trace
        if not self.is_trie_graph(parsed):
            raise RuntimeError(
                f"graph_at_t_star: t*={t_star_us:.0f}µs but graph has no trie metadata "
                f"on any node -- this is a lowering bug, not a runtime error; "
                f"trace={trace.id!r}"
            )
        if warmup is GraphWarmupKind.BOUNDARY_SNAPSHOT:
            return rewrite_for_warmup(parsed, t_star_us), trace
        rewritten = chop_trie_at_tstar(parsed, t_star_us)
        if burst_phase_starts:
            rewritten = burst_collapse_leading_offsets(rewritten)
        return rewritten, trace


def burst_collapse_leading_offsets(rewritten: ParsedGraph) -> ParsedGraph:
    """Collapse leading phase-start offsets (AgentX ``--burst-phase-starts``).

    AgentX burst collapses the phase START into a synchronized burst: every
    trace's earliest profiling resume fires at once, IGNORING the per-stream
    leading offset from t*. On a chopped trie graph that leading offset lives on
    each re-rooted node's START in-edge ``min_start_delay_us`` (stamped by
    ``snapshot_chop._chop_edges``; the node-level field is never stamped by the
    trie producers but is collapsed too wherever it is present) -- delegate to
    the pure :func:`aiperf.graph.scheduler.collapse_leading_start_offsets`. The
    inter-turn ``StaticEdge.delay_after_predecessor_us`` end-to-start gaps are
    UNTOUCHED -- burst governs only the phase start, not the faithful inter-turn
    pacing. Warmup builds its boundary graph offset-free
    (:func:`rewrite_for_warmup`), keeping spread/burst warmup identical.
    """
    return msgspec.structs.replace(
        rewritten, graph=collapse_leading_start_offsets(rewritten.graph)
    )
