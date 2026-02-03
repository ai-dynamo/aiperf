# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyzer for extracting prefix statistics from traces."""

import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import orjson

from aiperf.common.config.config_defaults import InputTokensDefaults
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.dataset.synthesis.models import AnalysisStats, MetricStats
from aiperf.dataset.synthesis.radix_tree import RadixTree


class PrefixAnalyzer(AIPerfLoggerMixin):
    """Analyzes traces to extract ISL/OSL statistics and prefix patterns.

    Computes:
    - Input/output sequence length distributions
    - Unique prefix patterns
    - Theoretical cache hit rates
    - Prefix reuse ratios
    """

    def __init__(self, block_size: int = InputTokensDefaults.BLOCK_SIZE) -> None:
        """Initialize the analyzer.

        Args:
            block_size: Number of tokens per block for analysis.
        """
        super().__init__(config=None, tokenizer=None)
        self.block_size = block_size
        self._reset()

    def analyze_file(self, trace_file: Path | str) -> AnalysisStats:
        """Analyze a mooncake trace file.

        Args:
            trace_file: Path to JSONL trace file.

        Returns:
            AnalysisStats with computed statistics.
        """
        self._reset()
        trace_file = Path(trace_file)

        # First pass: collect all data
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    data = orjson.loads(line)
                    self._process_trace_first_pass(data)

        # Second pass: compute context lengths
        self._compute_context_lengths()

        return self._compute_stats()

    def analyze_traces(self, traces: list[dict]) -> AnalysisStats:
        """Analyze a list of trace dictionaries.

        Args:
            traces: List of trace dictionaries.

        Returns:
            AnalysisStats with computed statistics.
        """
        self._reset()
        # First pass
        for trace in traces:
            self._process_trace_first_pass(trace)
        # Second pass
        self._compute_context_lengths()
        return self._compute_stats()

    def _reset(self) -> None:
        """Reset internal state."""
        self.isls: list[int] = []
        self.osls: list[int] = []
        self.context_lengths: list[int] = []
        self.unique_prompt_lengths: list[int] = []
        self.hash_ids_per_trace: list[list[int]] = []
        self._prefix_tree = RadixTree()
        self._prefix_counter: Counter[tuple[int, ...]] = Counter()
        self._hash_position_counter: Counter[tuple[int, int]] = Counter()

    def _process_trace_first_pass(self, trace: dict) -> None:
        """First pass: collect basic data and build hash position counter.

        Args:
            trace: Dictionary with 'input_length', 'output_length', and optional 'hash_ids'.
        """
        isl = trace.get("input_length", 0)
        osl = trace.get("output_length", 0)
        hash_ids = trace.get("hash_ids", [])

        self.isls.append(isl)
        self.osls.append(osl)

        if hash_ids:
            self.hash_ids_per_trace.append(hash_ids)
            # Add path to tree
            self._prefix_tree.add_path(hash_ids)
            # Track prefix patterns
            for i in range(1, len(hash_ids) + 1):
                prefix = tuple(hash_ids[:i])
                self._prefix_counter[prefix] += 1
            # Track (position, hash_id) pairs for context length calculation
            for pos, hash_id in enumerate(hash_ids):
                self._hash_position_counter[(pos, hash_id)] += 1
        else:
            self.hash_ids_per_trace.append([])

    def _compute_context_lengths(self) -> None:
        """Second pass: compute context and unique prompt lengths."""
        # Find repeated (position, hash_id) pairs
        repeated_hash_ids = {
            (pos, hash_id)
            for (pos, hash_id), count in self._hash_position_counter.items()
            if count > 1
        }

        for isl, hash_ids in zip(self.isls, self.hash_ids_per_trace, strict=True):
            if not hash_ids:
                self.context_lengths.append(0)
                self.unique_prompt_lengths.append(isl)
                continue

            # Check if all (position, hash_id) pairs are repeated
            if all(
                (pos, hash_id) in repeated_hash_ids
                for pos, hash_id in enumerate(hash_ids)
            ):
                context_len = isl
                unique_prompt_len = 0
            else:
                # Count repeated (position, hash_id) pairs
                repeated_count = sum(
                    1
                    for pos, hash_id in enumerate(hash_ids)
                    if (pos, hash_id) in repeated_hash_ids
                )
                context_len = repeated_count * self.block_size
                unique_prompt_len = isl - context_len

            self.context_lengths.append(context_len)
            self.unique_prompt_lengths.append(unique_prompt_len)

    def _compute_metric_stats(
        self, values: Sequence[float | int]
    ) -> MetricStats | None:
        """Compute full statistics for a list of values.

        Args:
            values: List of numeric values.

        Returns:
            MetricStats with mean, std_dev, min, percentiles, max, or None if empty.
        """
        if not values:
            return None

        arr = np.asarray(values)

        return MetricStats(
            mean=float(np.mean(arr)),
            std_dev=float(np.std(arr)),
            min=float(np.min(arr)),
            p25=float(np.percentile(arr, 25)),
            median=float(np.median(arr)),
            p75=float(np.percentile(arr, 75)),
            max=float(np.max(arr)),
        )

    def _compute_stats(self) -> AnalysisStats:
        """Compute final statistics.

        Returns:
            AnalysisStats with all computed metrics.
        """
        total = len(self.isls)
        per_request_hit_rates = self._compute_per_request_hit_rates()
        cache_hit_rate = (
            statistics.mean(per_request_hit_rates) if per_request_hit_rates else 0.0
        )
        prefix_reuse = self._compute_prefix_reuse()

        return AnalysisStats(
            total_requests=total,
            unique_prefixes=len(self._prefix_counter),
            num_prefix_groups=self._compute_num_prefix_groups(),
            cache_hit_rate=cache_hit_rate,
            min_isl=min(self.isls) if self.isls else 0,
            max_isl=max(self.isls) if self.isls else 0,
            avg_isl=sum(self.isls) / len(self.isls) if self.isls else 0.0,
            min_osl=min(self.osls) if self.osls else 0,
            max_osl=max(self.osls) if self.osls else 0,
            avg_osl=sum(self.osls) / len(self.osls) if self.osls else 0.0,
            prefix_reuse_ratio=prefix_reuse,
            # Extended statistics
            isl_stats=self._compute_metric_stats(self.isls),
            osl_stats=self._compute_metric_stats(self.osls),
            context_length_stats=self._compute_metric_stats(self.context_lengths),
            unique_prompt_length_stats=self._compute_metric_stats(
                self.unique_prompt_lengths
            ),
            hit_rate_stats=self._compute_metric_stats(per_request_hit_rates),
        )

    def _compute_per_request_hit_rates(self) -> list[float]:
        """Compute per-request cache hit rates assuming infinite cache.

        For each request, computes the fraction of hash_ids that were already
        in cache when the request arrived. Uses the dynamo algorithm: finds
        the first unseen hash_id position to determine hit rate.

        Returns:
            List of cache hit rates (0.0 to 1.0) for each request.
        """
        if not self.hash_ids_per_trace:
            return []

        seen_hash_ids: set[int] = set()
        hit_rates: list[float] = []

        for hash_ids in self.hash_ids_per_trace:
            if not hash_ids:
                continue

            # Find first index where hash_id hasn't been seen
            first_unseen_idx = len(hash_ids)
            for idx, hash_id in enumerate(hash_ids):
                if hash_id not in seen_hash_ids:
                    first_unseen_idx = idx
                    break

            hit_rate = first_unseen_idx / len(hash_ids)
            hit_rates.append(hit_rate)

            # Add all hash_ids to seen set
            seen_hash_ids.update(hash_ids)

        return hit_rates

    def _compute_num_prefix_groups(self) -> int:
        """Compute number of prefix groups (shared first blocks).

        Counts distinct first blocks that appear in 2+ sequences.
        A first block is "shared" if multiple sequences start with it.

        Returns:
            Count of unique shared first blocks.
        """
        first_blocks = [h[0] for h in self.hash_ids_per_trace if h]
        if not first_blocks:
            return 0

        first_block_counts = Counter(first_blocks)
        return sum(1 for count in first_block_counts.values() if count > 1)

    def _compute_prefix_reuse(self) -> float:
        """Compute ratio of reused prefixes to total prefixes.

        Returns:
            Reuse ratio as a fraction (0.0 to 1.0).
        """
        if not self._prefix_counter:
            return 0.0

        reused = sum(count for count in self._prefix_counter.values() if count > 1)
        total = sum(self._prefix_counter.values())

        return reused / total if total > 0 else 0.0
