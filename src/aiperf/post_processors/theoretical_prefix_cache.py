# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theoretical prefix-cache hit accounting for trace replay.

A metric-record accumulator (``on_dataset_configured`` / ``process_record`` /
``summarize``) that the ``RecordsManager`` drives through record-type routing,
summing loader-provided per-turn hit/total block counts into one
ratio-of-sums PERCENT metric (``theoretical_prefix_cache_hit``).

Lookup key
----------
ONE join key for both planes: ``(conversation_id, turn_index)``. Non-graph
trace loaders stamp per-``TurnMetadata`` counts keyed by it directly. Graph
records carry the SAME legacy-shaped identity -- ``conversation_id`` is the
trajectory TEMPLATE id (``{trace}`` / ``{trace}::{scope}``) and
``turn_index`` the node's 0-based turn -- so the ``{scope}:{turn}`` node id
into the graph facet ``DatasetMetadata.graph.prefix_cache_by_trace``
(``{trace_id: {node_id: [hit, total]}}``) is a pure function of the two
record fields. No correlation-id parsing anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.models import MetricResult
from aiperf.metrics.types.theoretical_prefix_cache_metric import (
    TheoreticalPrefixCacheHitMetric,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.common.messages.inference_messages import MetricRecordsData
    from aiperf.common.models.dataset_models import DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun


# Back-compat alias: the display metadata (header, unit, console group) lives on
# the registered TheoreticalPrefixCacheHitMetric class so the realtime dashboard
# and console exporter can resolve the tag from the MetricRegistry.
THEORETICAL_PREFIX_CACHE_HIT_TAG = TheoreticalPrefixCacheHitMetric.tag


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
    """Track infinite-cache prefix hits from loader-provided per-turn counts.

    Consumes leading prefix-cache ``hit_blocks`` and ``total_blocks`` integers
    stamped per node on the native ``LlmNode.theoretical_prefix_cache_hit_blocks``
    / ``_total_blocks`` fields by the shared trie build
    (``segment_ir.prefix_cache.stamp_theoretical_prefix_cache``, weka and
    dynamo), or per turn on ``Turn`` (non-graph trace loaders). Runtime accounting
    therefore avoids carrying hash_ids or re-tokenizing prompts; each completed
    record is only a metadata lookup plus two integer additions.

    Emits the cumulative ``theoretical_prefix_cache_hit`` (percent) = ``100 *
    sum(hit_blocks) / sum(total_blocks)`` over valid profiling records. The
    ``RecordsManager`` only forwards profiling-phase records to results
    processors, so no explicit phase gate is needed here.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(run=run, **kwargs)
        # Per-node counts keyed by (trace_id, node_id) for the graph dispatch
        # path. The trace_id qualifier keeps the bare per-trace node ids
        # (``parent_0`` repeats in every trace of a corpus) disjoint.
        self._blocks_by_node: dict[tuple[str, str], tuple[int, int]] = {}
        # Per-turn counts keyed by (conversation_id, turn_index), for non-graph
        # trace loaders that stamp the TurnMetadata fields.
        self._turn_blocks_by_conversation: dict[
            str, tuple[tuple[int, int] | None, ...]
        ] = {}
        self._hit_blocks = 0
        self._total_blocks = 0
        self._enabled = False

    def on_dataset_configured(self, metadata: DatasetMetadata) -> None:
        """Receive per-node (graph facet) / per-turn prefix-cache metadata."""
        by_node: dict[tuple[str, str], tuple[int, int]] = {}
        # Graph facet: per-trace {node_id: [hit, total]}, keyed by the BASE
        # template trace id -- the same template identity records carry in
        # conversation_id (instance identity rides x_correlation_id).
        graph = metadata.graph
        if graph is not None and graph.prefix_cache_by_trace:
            for trace_id, node_map in graph.prefix_cache_by_trace.items():
                for node_id, counts in node_map.items():
                    if len(counts) >= 2:
                        by_node[(trace_id, node_id)] = (
                            int(counts[0]),
                            int(counts[1]),
                        )
        # Per-turn counts for non-graph trace loaders that stamp the
        # TurnMetadata fields.
        by_conv: dict[str, tuple[tuple[int, int] | None, ...]] = {}
        for conv in metadata.conversations:
            per_turn: list[tuple[int, int] | None] = []
            has_turn_metadata = False
            for turn in conv.turns:
                hit_blocks = turn.theoretical_prefix_cache_hit_blocks
                total_blocks = turn.theoretical_prefix_cache_total_blocks
                if hit_blocks is None or total_blocks is None:
                    per_turn.append(None)
                    continue
                has_turn_metadata = True
                per_turn.append((hit_blocks, total_blocks))
            if has_turn_metadata:
                by_conv[conv.conversation_id] = tuple(per_turn)
        self._blocks_by_node = by_node
        self._turn_blocks_by_conversation = by_conv
        self._enabled = bool(by_node) or bool(by_conv)

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

    async def process_record(self, record_data: MetricRecordsData) -> None:
        """Accumulate block counts for one successful profiling request."""
        if not self._enabled or not record_data.valid:
            return
        metadata = record_data.metadata
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
        self._hit_blocks += hit_blocks
        self._total_blocks += total_blocks

    async def export_results(self, ctx: object) -> list[MetricResult]:
        """Export final theoretical prefix-cache metrics."""
        return await self.summarize()

    async def summarize(self) -> list[MetricResult]:
        """Return the current cumulative theoretical prefix-cache hit rate."""
        if self._total_blocks <= 0:
            return []
        hit_rate_pct = 100.0 * self._hit_blocks / self._total_blocks
        return [
            MetricResult(
                tag=TheoreticalPrefixCacheHitMetric.tag,
                header=TheoreticalPrefixCacheHitMetric.header,
                unit=str(TheoreticalPrefixCacheHitMetric.unit),
                count=self._total_blocks,
                current=hit_rate_pct,
                avg=hit_rate_pct,
                sum=self._hit_blocks,
            )
        ]
