# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesizer for generating synthetic traces with prefix patterns."""

import math
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from aiperf.common import random_generator as rng
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.radix_tree import RadixTree
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher


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
        self._rng = rng.derive("dataset.synthesis.synthesizer")

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
                    traces.append(orjson.loads(line))

        return self.synthesize_traces(traces)

    def synthesize_traces(self, traces: list[dict]) -> list[dict]:
        """Synthesize traces from a list of trace dictionaries.

        Args:
            traces: List of input trace dictionaries.

        Returns:
            List of synthetic trace dictionaries.
        """
        self.info(f"Synthesizing {len(traces)} traces")

        # Build radix tree
        for trace in traces:
            hash_ids = trace.get("hash_ids", [])
            if hash_ids:
                self._tree.add_path(hash_ids)

        # Apply multipliers to all hash_ids at once
        all_hash_ids = [trace.get("hash_ids", []) for trace in traces]
        all_input_lens = [trace.get("input_length", 0) for trace in traces]
        all_new_hash_ids, all_new_isls = self._apply_multipliers(
            all_hash_ids, all_input_lens
        )

        # Generate synthetic traces (pass-through all fields, update only what we modify)
        synthetic_traces = []
        for trace, new_hash_ids, isl in zip(
            traces, all_new_hash_ids, all_new_isls, strict=True
        ):
            # Start with a copy of the original trace to preserve arbitrary fields
            synthetic_trace = dict(trace)

            if not new_hash_ids:
                isl = trace.get("input_length", self.params.block_size)
                # Remove hash_ids if we couldn't compute new ones
                synthetic_trace.pop("hash_ids", None)
            else:
                synthetic_trace["hash_ids"] = new_hash_ids

            # Apply max_isl filter
            if self.params.max_isl and isl > self.params.max_isl:
                isl = self.params.max_isl

            # Only set input_length if the original trace didn't use text_input
            # to avoid validation error (MooncakeTrace doesn't allow both)
            if "text_input" not in trace:
                synthetic_trace["input_length"] = isl

            # Apply timestamp scaling if present
            timestamp = trace.get("timestamp")
            if timestamp is not None and self.params.speedup_ratio > 0:
                synthetic_trace["timestamp"] = int(
                    timestamp / self.params.speedup_ratio
                )

            synthetic_traces.append(synthetic_trace)

        self.info(f"Generated {len(synthetic_traces)} synthetic traces")
        return synthetic_traces

    def synthesize_grouped_traces(
        self, data: dict[str, list[dict]]
    ) -> dict[str, list[dict]]:
        """Synthesize traces while preserving session grouping.

        Args:
            data: Dictionary mapping session_id to list of trace dicts.

        Returns:
            Dictionary mapping session_id to list of synthesized trace dicts.
        """
        # Flatten with session_id embedded
        traces = [
            {**trace, "session_id": session_id}
            for session_id, session_traces in data.items()
            for trace in session_traces
        ]

        synthesized = self.synthesize_traces(traces)

        # Re-group by session_id (preserve empty sessions from input)
        result: dict[str, list[dict]] = {sid: [] for sid in data}
        for trace in synthesized:
            session_id = trace.pop("session_id", "default")
            result.setdefault(session_id, []).append(trace)

        return result

    def _apply_multipliers(
        self, all_hash_ids: list[list[int]], all_input_lens: list[int]
    ) -> tuple[list[list[int]], list[int]]:
        """Apply all multiplier transformations to hash IDs.

        Order of operations:
        1. Find shared prefixes (appearing 2+ times) and max_hash_id from shared set
        2. Filter each trace to keep only shared prefixes, compute prefix_len/prompt_len
        3. Apply prefix_len_mult: stretch hash_ids and prefix_len
        4. Compute new prompt blocks from scaled prompt_len
        5. Compute new_input_len
        6. Apply width multiplier (prefix_root_mult)
        7. Apply rolling hasher to renormalize hash_ids

        Args:
            all_hash_ids: List of hash ID lists from all traces.
            all_input_lens: List of input lengths (token counts) from all traces.

        Returns:
            Tuple of (transformed hash_ids lists, new input lengths).
        """
        prefix_mult = self.params.prefix_len_multiplier
        prompt_mult = self.params.prompt_len_multiplier
        root_mult = self.params.prefix_root_multiplier
        block_size = self.params.block_size

        # Step 1: Find shared prefixes (hash_ids appearing 2+ times)
        hash_counter: Counter[int] = Counter(h for ids in all_hash_ids for h in ids)
        shared_hashes = {h for h, count in hash_counter.items() if count > 1}

        # Get max_hash_id from shared set, pre-scale for interleaved stretch
        # With interleave: hash h -> [h*mult, h*mult+1, ..., h*mult+(mult-1)]
        # So max stretched = (max_shared + 1) * mult - 1
        max_hash_id = max(shared_hashes) if shared_hashes else 0
        mult_int = int(prefix_mult) if prefix_mult > 1.0 else 1
        max_hash_id = (max_hash_id + 1) * mult_int

        # Step 2-5: Process each trace
        results: list[list[int]] = []
        new_input_lens: list[int] = []

        for ids, input_len in zip(all_hash_ids, all_input_lens, strict=True):
            if not ids:
                results.append([])
                new_input_lens.append(input_len)
                continue

            # Filter to keep only shared prefixes (stop at first miss)
            prefix_ids: list[int] = []
            for h in ids:
                if h in shared_hashes:
                    prefix_ids.append(h)
                else:
                    break

            # Compute prefix_len and prompt_len
            prefix_len = len(prefix_ids) * block_size
            prompt_len = input_len - prefix_len
            if prompt_len < 0:
                raise ValueError(
                    f"input_len ({input_len}) < prefix_len ({prefix_len}): "
                    f"trace has fewer tokens than its shared prefix blocks"
                )

            # Step 3: Apply prefix_len_mult - stretch or squeeze prefix
            if prefix_mult > 1.0:
                # Stretch: interleave new blocks [0, 1, 2] with mult=2 -> [0, 1, 2, 3, 4, 5]
                mult_int = int(prefix_mult)
                stretched_ids = [
                    h * mult_int + j for h in prefix_ids for j in range(mult_int)
                ]
                # Add extra blocks for fractional part of the multiplier
                target_prefix_blocks = math.ceil(len(prefix_ids) * prefix_mult)
                extra_needed = target_prefix_blocks - len(stretched_ids)
                if extra_needed > 0:
                    stretched_ids.extend(
                        range(max_hash_id + 1, max_hash_id + 1 + extra_needed)
                    )
                    max_hash_id += extra_needed
                new_prefix_len = len(stretched_ids) * block_size
            elif prefix_mult < 1.0:
                # Squeeze: slice to fewer blocks
                new_prefix_blocks = max(1, int(len(prefix_ids) * prefix_mult))
                stretched_ids = prefix_ids[:new_prefix_blocks]
                new_prefix_len = new_prefix_blocks * block_size
            else:
                stretched_ids = prefix_ids[:]
                new_prefix_len = prefix_len

            # Step 4: Compute new prompt blocks from scaled prompt_len
            new_prompt_len = int(prompt_len * prompt_mult)
            num_prompt_blocks = (new_prompt_len + block_size - 1) // block_size

            # Add prompt blocks with unique hash_ids
            stretched_ids.extend(
                range(max_hash_id + 1, max_hash_id + 1 + num_prompt_blocks)
            )
            max_hash_id += num_prompt_blocks

            # Step 5: Compute new_input_len
            new_input_len = new_prefix_len + new_prompt_len
            results.append(stretched_ids)
            new_input_lens.append(new_input_len)

        # Step 6: Apply width multiplier (prefix_root_mult)
        if root_mult > 1:
            offset_base = max_hash_id + 1
            for i, ids in enumerate(results):
                if not ids:
                    continue
                tree_index = self._rng.randint(0, root_mult - 1)
                if tree_index > 0:
                    offset = tree_index * offset_base
                    results[i] = [h + offset for h in ids]

        # Step 7: Optionally apply rolling hasher to renormalize hash_ids
        if self.params.renormalize_hash_ids:
            hasher = RollingHasher(block_size=block_size)
            for i, ids in enumerate(results):
                if ids:
                    # Treat each hash_id as a unique "block" for rolling hash
                    results[i] = hasher.hash_token_blocks([(h,) for h in ids])

        return results, new_input_lens

    def get_stats(self) -> dict[str, Any]:
        """Get synthesizer statistics.

        Returns:
            Dictionary with synthesis stats.
        """
        return {
            "tree_nodes": len(self._tree.get_all_nodes()),
            "tree_depth": self._tree.get_stats().max_depth,
            "params": self.params.model_dump(),
        }
