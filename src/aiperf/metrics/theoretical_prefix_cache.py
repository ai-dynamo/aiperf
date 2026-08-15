# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-overhead theoretical prefix-cache hit accounting for trace replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase, GenericMetricUnit
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricResult
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.common.models import DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun


THEORETICAL_PREFIX_CACHE_HIT_TAG = "theoretical_prefix_cache_hit"


def _graph_node_identity(
    conversation_id: str | None, turn_index: int | None
) -> tuple[str, str] | None:
    """Recover ``(template_trace_id, node_id)`` from a record's own fields.

    Graph records carry LEGACY-shaped identity: ``conversation_id`` is the
    trajectory TEMPLATE id (``{trace}`` for the root scope,
    ``{trace}::{scope}`` for children) and ``turn_index`` is the node's
    0-based turn within its trajectory -- so the ``{scope}:{turn}`` node id is
    a pure function of the two record fields, no correlation-id parsing.
    """
    if not conversation_id or turn_index is None:
        return None
    trace, sep, scope = conversation_id.partition("::")
    return trace, f"{scope if sep else trace}:{turn_index}"


class TheoreticalPrefixCacheAccumulator(BaseMetricsProcessor):
    """Track infinite-cache prefix hits from loader-provided counts.

    Two join keys, one accumulator:

    * Agent-graph replays stamp per-node ``hit_blocks`` / ``total_blocks`` during
      the shared segment trie build, surfaced as the graph facet
      ``DatasetMetadata.graph.prefix_cache_by_trace``
      (``{trace_id: {node_id: [hit, total]}}``).
    * Linear trace loaders stamp each turn with
      ``theoretical_prefix_cache_hit_blocks`` /
      ``theoretical_prefix_cache_total_blocks``.

    Runtime accounting therefore avoids carrying hash_ids or re-tokenizing
    prompts; each completed record is only a metadata lookup plus two integer
    additions.

    Phase scoping mirrors ``AccuracyAccumulator``: ``export_results(ctx)``
    filters to ``ctx.phase`` so warmup blocks never leak into the profiling
    summary (and vice versa).
    """

    # RecordsManager routes phase-scoped-export accumulators through
    # export_results(ctx) instead of the unscoped summarize().
    supports_phase_scoped_export = True

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(run=run, **kwargs)
        # Per-node counts keyed by (trace_id, node_id) for the graph dispatch
        # path. The trace_id qualifier keeps the bare per-trace node ids
        # (``parent_0`` repeats in every trace of a corpus) disjoint.
        self._blocks_by_node: dict[tuple[str, str], tuple[int, int]] = {}
        self._turn_blocks_by_conversation: dict[
            str, tuple[tuple[int, int] | None, ...]
        ] = {}
        self._hit_blocks_by_phase: dict[CreditPhase, int] = {
            CreditPhase.WARMUP: 0,
            CreditPhase.PROFILING: 0,
        }
        self._total_blocks_by_phase: dict[CreditPhase, int] = {
            CreditPhase.WARMUP: 0,
            CreditPhase.PROFILING: 0,
        }
        self._enabled = False

    def on_dataset_configured(self, metadata: DatasetMetadata) -> None:
        """Receive per-node (graph facet) / per-turn prefix-cache metadata."""
        # Graph facet: per-trace {node_id: [hit, total]}, keyed by the BASE
        # template trace id -- the same template identity records carry in
        # conversation_id (instance identity rides x_correlation_id).
        by_node: dict[tuple[str, str], tuple[int, int]] = {}
        graph = metadata.graph
        if graph is not None and graph.prefix_cache_by_trace:
            for trace_id, node_map in graph.prefix_cache_by_trace.items():
                for node_id, counts in node_map.items():
                    if len(counts) >= 2:
                        by_node[(trace_id, node_id)] = (int(counts[0]), int(counts[1]))
        lookup: dict[str, tuple[tuple[int, int] | None, ...]] = {}
        for conv in metadata.conversations:
            per_turn: list[tuple[int, int] | None] = []
            has_prefix_metadata = False
            for turn in conv.turns:
                hit_blocks = turn.theoretical_prefix_cache_hit_blocks
                total_blocks = turn.theoretical_prefix_cache_total_blocks
                if hit_blocks is None or total_blocks is None:
                    per_turn.append(None)
                    continue
                has_prefix_metadata = True
                per_turn.append((hit_blocks, total_blocks))
            if has_prefix_metadata:
                lookup[conv.conversation_id] = tuple(per_turn)
        self._blocks_by_node = by_node
        self._turn_blocks_by_conversation = lookup
        self._enabled = bool(by_node) or bool(lookup)

    def _lookup_counts(
        self, conversation_id: str | None, turn_index: int | None
    ) -> tuple[int, int] | None:
        """Resolve (hit_blocks, total_blocks) for one record, node map first.

        The graph node map keys by ``(template_trace_id, node_id)``; both are
        derived from the record's template-level ``(conversation_id,
        turn_index)`` -- the SAME join key the linear per-turn fallback uses,
        so recycled instances of one template correctly re-apply the template
        counts (duplication is desired, exactly like the linear path).
        """
        identity = _graph_node_identity(conversation_id, turn_index)
        if identity is not None:
            counts = self._blocks_by_node.get(identity)
            if counts is not None:
                return counts
        if conversation_id is None or turn_index is None:
            return None
        per_turn = self._turn_blocks_by_conversation.get(conversation_id)
        if per_turn is None or turn_index < 0 or turn_index >= len(per_turn):
            return None
        return per_turn[turn_index]

    async def process_record(self, record: MetricRecordsData) -> None:
        """Accumulate block counts for one successful profiling request."""
        if not self._enabled or not record.valid:
            return
        metadata = record.metadata
        # A context-overflow skip record reaches this accumulator as a
        # trimmed, error-free carrier for the overflow count (see
        # RecordsManager._send_overflow_count_only): the request never ran to
        # completion, so counting its planned blocks would pollute the hit rate.
        if metadata.context_overflow_skip:
            return
        counts = self._lookup_counts(metadata.conversation_id, metadata.turn_index)
        if counts is None:
            return
        hit_blocks, total_blocks = counts
        if total_blocks <= 0:
            return
        # Clamp the hit count into [0, total_blocks]: a loader miscount must not
        # drive the cumulative hit rate above 100% (or below 0%).
        hit_blocks = max(0, min(hit_blocks, total_blocks))
        phase = metadata.benchmark_phase
        self._hit_blocks_by_phase[phase] = (
            self._hit_blocks_by_phase.get(phase, 0) + hit_blocks
        )
        self._total_blocks_by_phase[phase] = (
            self._total_blocks_by_phase.get(phase, 0) + total_blocks
        )

    async def summarize(self, ctx: SummaryContext | None = None) -> list[MetricResult]:
        """Return the phase-agnostic (all-phase) theoretical prefix-cache hit rate."""
        return self._summarize_phase(None)

    async def export_results(self, ctx: ExportContext) -> list[MetricResult]:
        """Return prefix-cache hit rate scoped to ``ctx.phase`` (all if None)."""
        return self._summarize_phase(ctx.phase)

    def _summarize_phase(self, phase: CreditPhase | None) -> list[MetricResult]:
        if phase is None:
            hit_blocks = sum(self._hit_blocks_by_phase.values())
            total_blocks = sum(self._total_blocks_by_phase.values())
        else:
            hit_blocks = self._hit_blocks_by_phase.get(phase, 0)
            total_blocks = self._total_blocks_by_phase.get(phase, 0)

        if total_blocks <= 0:
            return []
        hit_rate_pct = 100.0 * hit_blocks / total_blocks
        return [
            MetricResult(
                tag=THEORETICAL_PREFIX_CACHE_HIT_TAG,
                header="Theoretical Prefix Cache Hit",
                unit=str(GenericMetricUnit.PERCENT),
                count=total_blocks,
                current=hit_rate_pct,
                avg=hit_rate_pct,
                sum=hit_blocks,
            )
        ]
