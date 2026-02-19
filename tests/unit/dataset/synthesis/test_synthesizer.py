# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Synthesizer."""

from collections import defaultdict

import pytest

from aiperf.dataset.synthesis import Synthesizer
from aiperf.dataset.synthesis.models import SynthesisParams


class TestSynthesizer:
    """Tests for Synthesizer class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_default(self) -> None:
        """Test Synthesizer initialization with defaults."""
        synthesizer = Synthesizer()
        assert synthesizer.params is not None
        assert synthesizer.params.speedup_ratio == 1.0

    def test_initialization_with_params(self) -> None:
        """Test Synthesizer initialization with custom params."""
        params = SynthesisParams(speedup_ratio=2.0, prefix_len_multiplier=1.5)
        synthesizer = Synthesizer(params=params)
        assert synthesizer.params.speedup_ratio == 2.0
        assert synthesizer.params.prefix_len_multiplier == 1.5

    # ============================================================================
    # Synthesis Tests
    # ============================================================================

    def test_synthesize_single_trace(self, sample_trace_data) -> None:
        """Test synthesizing a single trace."""
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(sample_trace_data[:1])

        assert len(synthetic) == 1
        assert "input_length" in synthetic[0]
        assert "output_length" in synthetic[0]

    def test_synthesize_multiple_traces(self, sample_trace_data) -> None:
        """Test synthesizing multiple traces."""
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(sample_trace_data)

        assert len(synthetic) == len(sample_trace_data)

    def test_synthesize_preserves_session_id(self) -> None:
        """Test that synthesis preserves session_id."""
        traces = [
            {
                "input_length": 1024,
                "output_length": 20,
                "hash_ids": [1, 2],
                "session_id": "test-session",
            },
            {
                "input_length": 1024,
                "output_length": 20,
                "hash_ids": [1, 3],
                "session_id": "test-session",
            },
        ]
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(traces)

        assert synthetic[0].get("session_id") == "test-session"

    def test_synthesize_preserves_delay(self) -> None:
        """Test that synthesis preserves delay."""
        traces = [
            {
                "input_length": 100,
                "output_length": 20,
                "hash_ids": [1, 2],
                "delay": 1000,
            }
        ]
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(traces)

        assert synthetic[0].get("delay") == 1000

    # ============================================================================
    # Timestamp Scaling Tests
    # ============================================================================

    def test_speedup_ratio_1(self) -> None:
        """Test speedup_ratio of 1 (no change)."""
        traces = [{"input_length": 100, "output_length": 20, "timestamp": 1000}]
        params = SynthesisParams(speedup_ratio=1.0)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        assert synthetic[0].get("timestamp") == 1000

    def test_speedup_ratio_2(self) -> None:
        """Test speedup_ratio of 2 (2x faster)."""
        traces = [{"input_length": 100, "output_length": 20, "timestamp": 1000}]
        params = SynthesisParams(speedup_ratio=2.0)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Timestamp should be divided by speedup_ratio
        assert synthetic[0].get("timestamp") == 500

    @pytest.mark.parametrize(
        "speedup,input_ts,expected_ts",
        [
            (1.0, 1000, 1000),
            (2.0, 1000, 500),
            (0.5, 1000, 2000),
        ],
    )
    def test_speedup_ratio_variations(
        self, speedup: float, input_ts: int, expected_ts: int
    ) -> None:
        """Test various speedup ratios."""
        traces = [{"input_length": 100, "output_length": 20, "timestamp": input_ts}]
        params = SynthesisParams(speedup_ratio=speedup)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        assert synthetic[0].get("timestamp") == expected_ts

    # ============================================================================
    # Prefix Multiplier Tests
    # ============================================================================

    def test_prefix_len_multiplier_1(self) -> None:
        """Test prefix_len_multiplier of 1 (no change)."""
        # Need multiple traces with shared prefixes for the algorithm to work
        # block_size=512, input_length=1536 = 3 blocks
        # Shared prefix [1, 2], prompt = 1536 - 1024 = 512 tokens = 1 block
        traces = [
            {"input_length": 1536, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 1536, "output_length": 20, "hash_ids": [1, 2, 4]},
        ]
        params = SynthesisParams(prefix_len_multiplier=1.0)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Shared prefix [1, 2] (2 blocks) + 1 prompt block = 3 blocks
        # new_prefix_len = 1024, new_prompt_len = 512, new_input_len = 1536
        assert len(synthetic[0].get("hash_ids", [])) == 3
        assert len(synthetic[1].get("hash_ids", [])) == 3
        assert synthetic[0]["input_length"] == 1536
        assert synthetic[1]["input_length"] == 1536

    def test_prefix_len_multiplier_2(self) -> None:
        """Test prefix_len_multiplier of 2 (double prefix length)."""
        # Need multiple traces with shared prefixes
        # block_size=512, input_length=1024 = 2 blocks
        # Shared prefix [1], prompt = 1024 - 512 = 512 tokens = 1 block
        traces = [
            {"input_length": 1024, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 1024, "output_length": 20, "hash_ids": [1, 3]},
        ]
        params = SynthesisParams(prefix_len_multiplier=2.0)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Shared prefix [1] stretched to 2 blocks (interleaved) + 1 prompt block = 3 blocks
        # new_prefix_len = 512 * 2 = 1024, new_prompt_len = 512, new_input_len = 1536
        assert len(synthetic[0].get("hash_ids", [])) == 3
        assert len(synthetic[1].get("hash_ids", [])) == 3
        assert synthetic[0]["input_length"] == 1536
        assert synthetic[1]["input_length"] == 1536

    @pytest.mark.parametrize(
        "prefix_mult, num_shared_prefix_blocks, expected_prefix_blocks",
        [
            (1.1, 10, 11),  # ceil(10 * 1.1) = 11
            (1.5, 10, 15),  # ceil(10 * 1.5) = 15
            (2.3, 4, 10),  # ceil(4 * 2.3) = 10: interleave gives 8, +2 extra
            (1.01, 3, 4),  # ceil(3 * 1.01) = 4: interleave gives 3, +1 extra
        ],
    )
    def test_prefix_len_multiplier_fractional(
        self,
        prefix_mult: float,
        num_shared_prefix_blocks: int,
        expected_prefix_blocks: int,
    ) -> None:
        """Test fractional prefix_len_multiplier adds extra blocks."""
        # Each trace has num_shared_prefix_blocks shared + 1 unique prompt block
        shared = list(range(num_shared_prefix_blocks))
        traces = [
            {
                "input_length": (num_shared_prefix_blocks + 1) * 512,
                "output_length": 20,
                "hash_ids": shared + [100],
            },
            {
                "input_length": (num_shared_prefix_blocks + 1) * 512,
                "output_length": 20,
                "hash_ids": shared + [200],
            },
        ]
        params = SynthesisParams(prefix_len_multiplier=prefix_mult, block_size=512)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        for trace in synthetic:
            hash_ids = trace["hash_ids"]
            input_length = trace["input_length"]
            # Prefix blocks + 1 prompt block
            assert len(hash_ids) == expected_prefix_blocks + 1
            # Input length must be consistent with block count
            assert input_length == len(hash_ids) * 512

    # ============================================================================
    # Max ISL Filter Tests
    # ============================================================================

    def test_max_isl_filter_applied(self) -> None:
        """Test that max_isl filter caps input length."""
        # 10 blocks * 512 = 5120 tokens, capped to 4096
        traces = [
            {"input_length": 5120, "output_length": 20, "hash_ids": list(range(10))},
            {"input_length": 5120, "output_length": 20, "hash_ids": list(range(10))},
        ]
        params = SynthesisParams(max_isl=4096, block_size=512)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        assert synthetic[0]["input_length"] == 4096

    def test_max_isl_filter_not_applied(self) -> None:
        """Test that max_isl filter doesn't apply when None."""
        # 4 blocks * 512 = 2048 tokens
        traces = [
            {"input_length": 2048, "output_length": 20, "hash_ids": [1, 2, 3, 4]},
            {"input_length": 2048, "output_length": 20, "hash_ids": [1, 2, 3, 5]},
        ]
        params = SynthesisParams(max_isl=None, block_size=512)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Should not be filtered
        assert synthetic[0]["input_length"] == 2048

    # ============================================================================
    # Statistics Tests
    # ============================================================================

    def test_get_stats(self, sample_trace_data) -> None:
        """Test getting synthesizer statistics."""
        synthesizer = Synthesizer()
        synthesizer.synthesize_traces(sample_trace_data)
        stats = synthesizer.get_stats()

        assert "tree_nodes" in stats
        assert "tree_depth" in stats
        assert "params" in stats

    def test_stats_after_synthesis(self, sample_trace_data) -> None:
        """Test that stats are populated after synthesis."""
        synthesizer = Synthesizer()
        synthesizer.synthesize_traces(sample_trace_data)
        stats = synthesizer.get_stats()

        assert stats["tree_nodes"] >= 1

    # ============================================================================
    # Distribution Sampling Tests
    # ============================================================================

    def test_isl_osl_sampling(self, sample_trace_data) -> None:
        """Test that ISL/OSL are sampled from learned distributions."""
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(sample_trace_data)

        # Check that sampled values are in reasonable ranges
        for trace in synthetic:
            assert isinstance(trace["input_length"], int)
            assert isinstance(trace["output_length"], int)
            assert trace["input_length"] > 0
            assert trace["output_length"] > 0

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_synthesize_trace_without_hashes(self, sample_trace_without_hashes) -> None:
        """Test synthesizing traces without hash IDs."""
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(sample_trace_without_hashes)

        assert len(synthetic) == len(sample_trace_without_hashes)
        for trace in synthetic:
            # Should still have ISL/OSL
            assert "input_length" in trace
            assert "output_length" in trace

    def test_synthesize_empty_traces(self) -> None:
        """Test synthesizing empty trace list."""
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces([])

        assert len(synthetic) == 0

    def test_root_replication_multiplier(self) -> None:
        """Test prefix_root_multiplier distributes traces across independent trees."""
        # Use many traces with shared prefixes to test distribution
        traces = [
            {"input_length": 1024, "output_length": 20, "hash_ids": [1, 2]}
            for _ in range(100)
        ]
        params = SynthesisParams(prefix_root_multiplier=3)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Collect all unique hash_id[0] values (representing different tree roots)
        first_ids = {trace["hash_ids"][0] for trace in synthetic}

        # With multiplier=3, traces should be distributed across 3 independent trees
        # Each tree has its own offset, so first_ids should have up to 3 different values
        # With 100 traces and 3 trees, statistically we should see multiple trees
        assert len(first_ids) > 1

        # All traces should have hash_ids
        for trace in synthetic:
            assert len(trace["hash_ids"]) >= 1

    def test_root_multiplier_with_prefix_multiplier_no_collisions(self) -> None:
        """Test that prefix_root_multiplier doesn't collide when prefix_len_multiplier extends hash_ids.

        When prefix_len_multiplier > 1, new hash IDs are generated beyond _max_hash_id.
        The root multiplier offsets must account for these extended IDs to prevent collisions.
        """
        traces = [
            {"input_length": 1024, "output_length": 20, "hash_ids": [1, 2]}
            for _ in range(100)
        ]
        params = SynthesisParams(prefix_len_multiplier=2.0, prefix_root_multiplier=3)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Each trace now has 4 hash_ids: [1, 2] extended to [1, 2, 3, 4]
        # With prefix_root_multiplier=3, we have 3 independent trees
        # Tree 0: offset 0 -> IDs 1-4
        # Tree 1: offset 5 (max_hash_id=4, so offset=5) -> IDs 6-9
        # Tree 2: offset 10 -> IDs 11-14
        # Key: offsets must be > 4 to avoid collision with extended IDs

        # Collect all hash_ids across all synthetic traces
        all_hash_id_sets: list[set[int]] = []
        for trace in synthetic:
            all_hash_id_sets.append(set(trace["hash_ids"]))

        # Verify each trace has 4 hash_ids (2 original + 2 from multiplier)
        for trace in synthetic:
            assert len(trace["hash_ids"]) == 4

        # Check for collisions: no two traces from different trees should share hash_ids
        # Group traces by their first hash_id (tree root indicator)
        trees: dict[int, list[set[int]]] = defaultdict(list)
        for trace in synthetic:
            root = trace["hash_ids"][0]
            trees[root].append(set(trace["hash_ids"]))

        # Verify trees don't overlap
        tree_roots = sorted(trees.keys())
        for i, root_i in enumerate(tree_roots):
            all_ids_in_tree_i = set().union(*trees[root_i])
            for root_j in tree_roots[i + 1 :]:
                all_ids_in_tree_j = set().union(*trees[root_j])
                # No hash_id should appear in both trees
                overlap = all_ids_in_tree_i & all_ids_in_tree_j
                assert not overlap, (
                    f"Trees {root_i} and {root_j} share hash_ids: {overlap}"
                )

    def test_text_input_not_modified(self) -> None:
        """Test that traces with text_input don't get input_length added."""
        traces = [
            {"text_input": "What is AI?", "output_length": 50},
            {"text_input": "Explain quantum computing", "output_length": 100},
        ]
        synthesizer = Synthesizer()
        synthetic = synthesizer.synthesize_traces(traces)

        # text_input should be preserved, input_length should NOT be added
        assert synthetic[0]["text_input"] == "What is AI?"
        assert synthetic[1]["text_input"] == "Explain quantum computing"
        assert "input_length" not in synthetic[0]
        assert "input_length" not in synthetic[1]

    # ============================================================================
    # Input Length Preservation Tests
    # ============================================================================

    def test_input_length_preserved_no_multiplier(self) -> None:
        """Test that input_length is preserved when no multipliers applied."""
        traces = [
            {"input_length": 700, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 700, "output_length": 20, "hash_ids": [1, 3]},
        ]
        params = SynthesisParams(block_size=512)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Input length should be preserved
        assert synthetic[0]["input_length"] == 700
        assert synthetic[1]["input_length"] == 700

    def test_input_length_scaled_with_multiplier(self) -> None:
        """Test that input_length scales with prefix and prompt multipliers."""
        # 2 traces sharing prefix [1], with prompt portions
        # block_size=512, input_length=1024 = 2 blocks
        # Shared prefix [1] = 512 tokens, prompt = 512 tokens
        traces = [
            {"input_length": 1024, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 1024, "output_length": 20, "hash_ids": [1, 3]},
        ]
        params = SynthesisParams(
            block_size=512, prefix_len_multiplier=2.0, prompt_len_multiplier=2.0
        )
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # new_prefix_len = 512 * 2 = 1024
        # new_prompt_len = 512 * 2 = 1024
        # new_input_len = 1024 + 1024 = 2048
        # num_prompt_blocks = (1024 + 511) // 512 = 2
        # Stretched prefix = 2 blocks + 2 prompt blocks = 4 blocks
        for trace in synthetic:
            assert trace["input_length"] == 2048
            assert len(trace["hash_ids"]) == 4


class TestSynthesizeGroupedTraces:
    """Tests for synthesize_grouped_traces method."""

    # ============================================================================
    # Basic Functionality
    # ============================================================================

    def test_preserves_session_grouping(self) -> None:
        """Test that session grouping is preserved through synthesis."""
        data = {
            "session-a": [
                {"input_length": 1024, "output_length": 20, "hash_ids": [1, 2]},
                {"input_length": 1536, "output_length": 30, "hash_ids": [1, 2, 3]},
            ],
            "session-b": [
                {"input_length": 1024, "output_length": 40, "hash_ids": [4, 5]},
            ],
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        assert set(result.keys()) == {"session-a", "session-b"}
        assert len(result["session-a"]) == 2
        assert len(result["session-b"]) == 1

    def test_empty_input(self) -> None:
        """Test synthesizing empty grouped data."""
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces({})

        assert result == {}

    def test_single_session_single_trace(self) -> None:
        """Test with minimal input: one session, one trace."""
        data = {
            "only-session": [
                {"input_length": 100, "output_length": 20, "hash_ids": [1]},
            ],
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        assert "only-session" in result
        assert len(result["only-session"]) == 1
        assert "input_length" in result["only-session"][0]
        assert "output_length" in result["only-session"][0]

    # ============================================================================
    # Synthesis Parameters Applied
    # ============================================================================

    def test_speedup_ratio_applied(self) -> None:
        """Test that speedup_ratio is applied to grouped traces."""
        data = {
            "session-1": [
                {"input_length": 100, "output_length": 20, "timestamp": 1000},
                {"input_length": 150, "output_length": 30, "timestamp": 2000},
            ],
        }
        params = SynthesisParams(speedup_ratio=2.0)
        synthesizer = Synthesizer(params=params)
        result = synthesizer.synthesize_grouped_traces(data)

        assert result["session-1"][0]["timestamp"] == 500
        assert result["session-1"][1]["timestamp"] == 1000

    def test_prefix_multiplier_applied(self) -> None:
        """Test that prefix_len_multiplier is applied to grouped traces."""
        # Need multiple traces with shared prefixes
        # Shared prefix [1] = 512 tokens, prompt = 512 tokens
        data = {
            "session-1": [
                {"input_length": 1024, "output_length": 20, "hash_ids": [1, 2]},
                {"input_length": 1024, "output_length": 20, "hash_ids": [1, 3]},
            ],
        }
        params = SynthesisParams(prefix_len_multiplier=2.0)
        synthesizer = Synthesizer(params=params)
        result = synthesizer.synthesize_grouped_traces(data)

        # new_prefix_len = 512 * 2 = 1024, new_prompt_len = 512
        # new_input_len = 1024 + 512 = 1536
        assert result["session-1"][0]["input_length"] == 1536
        assert result["session-1"][1]["input_length"] == 1536

    def test_max_isl_filter_applied(self) -> None:
        """Test that max_isl filter is applied to grouped traces."""
        # 10 blocks * 512 = 5120 tokens, capped to 4096
        data = {
            "session-1": [
                {
                    "input_length": 5120,
                    "output_length": 20,
                    "hash_ids": list(range(10)),
                },
                {
                    "input_length": 5120,
                    "output_length": 20,
                    "hash_ids": list(range(10)),
                },
            ],
        }
        params = SynthesisParams(max_isl=4096, block_size=512)
        synthesizer = Synthesizer(params=params)
        result = synthesizer.synthesize_grouped_traces(data)

        assert result["session-1"][0]["input_length"] == 4096

    # ============================================================================
    # Field Preservation
    # ============================================================================

    def test_delay_preserved(self) -> None:
        """Test that delay field is preserved through grouped synthesis."""
        data = {
            "session-1": [
                {"input_length": 100, "output_length": 20, "delay": 500},
                {"input_length": 150, "output_length": 30, "delay": 1000},
            ],
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        assert result["session-1"][0]["delay"] == 500
        assert result["session-1"][1]["delay"] == 1000

    def test_session_id_not_in_output_traces(self) -> None:
        """Test that session_id is removed from individual trace dicts."""
        data = {
            "my-session": [
                {"input_length": 100, "output_length": 20},
            ],
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        # session_id should be the key, not in the trace dict
        assert "session_id" not in result["my-session"][0]

    # ============================================================================
    # Multiple Sessions
    # ============================================================================

    def test_many_sessions(self) -> None:
        """Test with many sessions."""
        data = {
            f"session-{i}": [{"input_length": 100 + i, "output_length": 20}]
            for i in range(10)
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        assert len(result) == 10
        for i in range(10):
            assert f"session-{i}" in result

    def test_traces_without_hash_ids(self) -> None:
        """Test grouped synthesis with traces lacking hash_ids."""
        data = {
            "session-1": [
                {"input_length": 100, "output_length": 20},
                {"input_length": 150, "output_length": 30},
            ],
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        assert len(result["session-1"]) == 2
        # Should still have ISL/OSL
        for trace in result["session-1"]:
            assert "input_length" in trace
            assert "output_length" in trace

    def test_empty_sessions_preserved(self) -> None:
        """Test that empty sessions are preserved, not dropped."""
        data = {
            "empty-session": [],
            "non-empty": [{"input_length": 100, "output_length": 20}],
        }
        synthesizer = Synthesizer()
        result = synthesizer.synthesize_grouped_traces(data)

        assert set(result.keys()) == {"empty-session", "non-empty"}
        assert result["empty-session"] == []
        assert len(result["non-empty"]) == 1
