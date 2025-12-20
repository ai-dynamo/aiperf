# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesizer for generating synthetic traces with prefix patterns."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.dataset.synthesis.empirical_sampler import EmpiricalSampler
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.prefix_analyzer import PrefixAnalyzer
from aiperf.dataset.synthesis.radix_tree import RadixTree


class Synthesizer(AIPerfLoggerMixin):
    """Generates synthetic traces preserving prefix-sharing patterns.

    Reads an input trace file, builds a radix tree of prefix patterns,
    and generates new synthetic traces with configurable multipliers
    for controlling prefix reuse characteristics.
    """

    def __init__(self, params: SynthesisParams | None = None) -> None:
        """Initialize the synthesizer.

        Args:
            params: SynthesisParams with generation configuration.
                   If None, uses defaults.
        """
        super().__init__(config=None, tokenizer=None)
        self.params = params or SynthesisParams()
        self._tree = RadixTree()
        self._isl_sampler: EmpiricalSampler | None = None
        self._osl_sampler: EmpiricalSampler | None = None
        self._rng = np.random.default_rng(42)

    def synthesize_from_file(self, trace_file: Path | str) -> list[dict]:
        """Synthesize traces from an input trace file.

        Args:
            trace_file: Path to input JSONL trace file.

        Returns:
            List of synthetic trace dictionaries in mooncake format.
        """
        trace_file = Path(trace_file)

        traces = []
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

        return self.synthesize_traces(traces)

    def synthesize_traces(self, traces: list[dict]) -> list[dict]:
        """Synthesize traces from a list of trace dictionaries.

        Args:
            traces: List of input trace dictionaries.

        Returns:
            List of synthetic trace dictionaries.
        """
        # First pass: analyze input traces and build tree
        analyzer = PrefixAnalyzer(block_size=self.params.block_size)
        stats = analyzer.analyze_traces(traces)

        self.info(f"Input trace statistics: {stats.total_requests} traces")
        self.info(
            f"ISL range: {stats.min_isl}-{stats.max_isl}, avg: {stats.avg_isl:.1f}"
        )
        self.info(f"Cache hit rate: {stats.cache_hit_rate:.2%}")

        # Build samplers from input distributions
        isls = [t.get("input_length", 0) for t in traces if "input_length" in t]
        osls = [t.get("output_length", 0) for t in traces if "output_length" in t]

        self._isl_sampler = EmpiricalSampler(isls) if isls else None
        self._osl_sampler = EmpiricalSampler(osls) if osls else None

        # Build radix tree from input traces
        for trace in traces:
            hash_ids = trace.get("hash_ids", [])
            if hash_ids:
                self._tree.add_path(hash_ids)

        # Second pass: generate synthetic traces
        synthetic_traces = []
        for trace in traces:
            hash_ids = trace.get("hash_ids", [])

            # Generate new hash_ids based on parameters
            new_hash_ids = self._apply_multipliers(hash_ids) if hash_ids else []

            # Sample ISL/OSL from distributions
            isl = self._sample_isl()
            osl = self._sample_osl()

            # Apply max_isl filter
            if self.params.max_isl and isl > self.params.max_isl:
                isl = self.params.max_isl

            # Apply timestamp scaling if present
            timestamp = trace.get("timestamp")
            if timestamp is not None:
                timestamp = int(timestamp / self.params.speedup_ratio)

            synthetic_trace = {
                "input_length": isl,
                "output_length": osl,
            }

            if new_hash_ids:
                synthetic_trace["hash_ids"] = new_hash_ids

            if timestamp is not None:
                synthetic_trace["timestamp"] = timestamp

            # Preserve session_id and delay if present
            if "session_id" in trace:
                synthetic_trace["session_id"] = trace["session_id"]
            if "delay" in trace:
                synthetic_trace["delay"] = trace["delay"]

            synthetic_traces.append(synthetic_trace)

        self.info(f"Generated {len(synthetic_traces)} synthetic traces")
        return synthetic_traces

    def _apply_multipliers(self, hash_ids: list[int]) -> list[int]:
        """Apply multiplier transformations to hash IDs.

        New hash IDs are generated as consecutive integers continuing from
        the last hash ID in the specific list. Two hash_ids sharing the same
        integers represents prefix overlap.

        Args:
            hash_ids: Original hash IDs from trace.

        Returns:
            Transformed hash IDs with new consecutive integers for extensions.
        """
        if not hash_ids:
            return []

        # Start with original hash_ids (preserves prefix overlap)
        scaled_ids = hash_ids[:]
        # Next ID continues from the last one in this specific list (not the max)
        next_id = hash_ids[-1] + 1

        if self.params.prefix_len_multiplier != 1.0:
            scale = self.params.prefix_len_multiplier
            if scale > 1.0:
                # Extend prefix with NEW consecutive hash IDs
                num_to_add = int(len(hash_ids) * (scale - 1))
                for _ in range(num_to_add):
                    scaled_ids.append(next_id)
                    next_id += 1
            elif scale < 1.0:
                # Shorten prefix
                num_to_keep = max(1, int(len(scaled_ids) * scale))
                scaled_ids = scaled_ids[:num_to_keep]

        # Apply prompt length multiplier (affects unique prompt portion)
        if self.params.prompt_len_multiplier != 1.0 and scaled_ids:
            prompt_scale = self.params.prompt_len_multiplier
            if prompt_scale > 1.0:
                # Add new hash IDs for extended prompt portion
                num_to_add = int(prompt_scale - 1)
                for _ in range(num_to_add):
                    scaled_ids.append(next_id)
                    next_id += 1

        # Apply root replication (creates independent trees with new IDs)
        if self.params.prefix_root_multiplier > 1:
            original_len = len(scaled_ids)
            for _ in range(self.params.prefix_root_multiplier - 1):
                # Generate new IDs for each replication
                for _ in range(original_len):
                    scaled_ids.append(next_id)
                    next_id += 1

        return scaled_ids

    def _sample_isl(self) -> int:
        """Sample input sequence length from learned distribution.

        Returns:
            Sampled ISL.
        """
        if self._isl_sampler:
            return self._isl_sampler.sample(self._rng)
        return 512

    def _sample_osl(self) -> int:
        """Sample output sequence length from learned distribution.

        Returns:
            Sampled OSL.
        """
        if self._osl_sampler:
            return self._osl_sampler.sample(self._rng)
        return 64

    def get_stats(self) -> dict[str, Any]:
        """Get synthesizer statistics.

        Returns:
            Dictionary with synthesis stats.
        """
        return {
            "tree_nodes": len(self._tree.get_all_nodes()),
            "tree_depth": self._tree.get_stats().get("max_depth", 0),
            "params": self.params.model_dump(),
        }
