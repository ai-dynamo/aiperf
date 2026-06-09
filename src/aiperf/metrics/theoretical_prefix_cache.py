# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-overhead theoretical prefix-cache hit accounting for trace replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.enums import GenericMetricUnit
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricResult
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.common.config import UserConfig
    from aiperf.common.models import DatasetMetadata


THEORETICAL_PREFIX_CACHE_HIT_TAG = "theoretical_prefix_cache_hit"


class TheoreticalPrefixCacheAccumulator(BaseMetricsProcessor):
    """Track infinite-cache prefix hits from loader-provided per-turn counts.

    The WEKA loader already walks every request's hash_ids while reconstructing
    prompts. It stamps each turn with two small integers:
    ``theoretical_prefix_cache_hit_blocks`` and
    ``theoretical_prefix_cache_total_blocks``. Runtime accounting therefore
    avoids carrying hash_ids or re-tokenizing prompts; each completed record is
    only a metadata lookup plus two integer additions.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)
        self._turn_blocks_by_conversation: dict[
            str, tuple[tuple[int, int] | None, ...]
        ] = {}
        self._hit_blocks = 0
        self._total_blocks = 0
        self._enabled = False

    def on_dataset_configured(self, metadata: DatasetMetadata) -> None:
        """Receive per-turn theoretical prefix-cache metadata from the loader."""
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
        self._turn_blocks_by_conversation = lookup
        self._enabled = bool(lookup)

    async def process_record(self, record: MetricRecordsData) -> None:
        """Accumulate block counts for one successful profiling request."""
        if not self._enabled or not record.valid:
            return
        metadata = record.metadata
        conversation_id = metadata.conversation_id
        turn_index = metadata.turn_index
        if conversation_id is None or turn_index is None:
            return
        per_turn = self._turn_blocks_by_conversation.get(conversation_id)
        if per_turn is None or turn_index < 0 or turn_index >= len(per_turn):
            return
        counts = per_turn[turn_index]
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

    async def summarize(self, ctx: SummaryContext | None = None) -> list[MetricResult]:
        """Return the current cumulative theoretical prefix-cache hit rate."""
        if self._total_blocks <= 0:
            return []
        hit_rate_pct = 100.0 * self._hit_blocks / self._total_blocks
        return [
            MetricResult(
                tag=THEORETICAL_PREFIX_CACHE_HIT_TAG,
                header="Theoretical Prefix Cache Hit",
                unit=str(GenericMetricUnit.PERCENT),
                count=self._total_blocks,
                current=hit_rate_pct,
                avg=hit_rate_pct,
                sum=self._hit_blocks,
            )
        ]
