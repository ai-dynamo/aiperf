# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyzer for extracting prefix statistics from traces."""

import json
from collections import Counter
from pathlib import Path

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.dataset.synthesis.models import AnalysisStats
from aiperf.dataset.synthesis.radix_tree import RadixTree


class PrefixAnalyzer(AIPerfLoggerMixin):
    """Analyzes traces to extract ISL/OSL statistics and prefix patterns.

    Computes:
    - Input/output sequence length distributions
    - Unique prefix patterns
    - Theoretical cache hit rates
    - Prefix reuse ratios
    """

    def __init__(self, block_size: int = 512) -> None:
        """Initialize the analyzer.

        Args:
            block_size: Number of tokens per block for analysis (default: 512).
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

        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._process_trace(data)

        return self._compute_stats()

    def analyze_traces(self, traces: list[dict]) -> AnalysisStats:
        """Analyze a list of trace dictionaries.

        Args:
            traces: List of trace dictionaries.

        Returns:
            AnalysisStats with computed statistics.
        """
        self._reset()
        for trace in traces:
            self._process_trace(trace)
        return self._compute_stats()

    def _reset(self) -> None:
        """Reset internal state."""
        self.isls: list[int] = []
        self.osls: list[int] = []
        self.hash_ids_per_trace: list[list[int]] = []
        self._prefix_tree = RadixTree()
        self._prefix_counter: Counter[tuple[int, ...]] = Counter()

    def _process_trace(self, trace: dict) -> None:
        """Process a single trace entry.

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

    def _compute_stats(self) -> AnalysisStats:
        """Compute final statistics.

        Returns:
            AnalysisStats with all computed metrics.
        """
        total = len(self.isls)
        cache_hit_rate = self._compute_cache_hit_rate()
        prefix_reuse = self._compute_prefix_reuse()

        return AnalysisStats(
            total_requests=total,
            unique_prefixes=len(self._prefix_counter),
            cache_hit_rate=cache_hit_rate,
            min_isl=min(self.isls) if self.isls else 0,
            max_isl=max(self.isls) if self.isls else 0,
            avg_isl=sum(self.isls) / len(self.isls) if self.isls else 0.0,
            min_osl=min(self.osls) if self.osls else 0,
            max_osl=max(self.osls) if self.osls else 0,
            avg_osl=sum(self.osls) / len(self.osls) if self.osls else 0.0,
            prefix_reuse_ratio=prefix_reuse,
        )

    def _compute_cache_hit_rate(self) -> float:
        """Compute theoretical cache hit rate assuming infinite cache.

        Returns:
            Cache hit rate as a fraction (0.0 to 1.0).
        """
        if not self.hash_ids_per_trace:
            return 0.0

        total_blocks = 0
        reused_blocks = 0

        seen_blocks: set[int] = set()

        for hash_ids in self.hash_ids_per_trace:
            for hash_id in hash_ids:
                total_blocks += 1
                if hash_id in seen_blocks:
                    reused_blocks += 1
                else:
                    seen_blocks.add(hash_id)

        return reused_blocks / total_blocks if total_blocks > 0 else 0.0

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
