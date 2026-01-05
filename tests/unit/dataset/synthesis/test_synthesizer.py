# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Synthesizer."""

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
                "input_length": 100,
                "output_length": 20,
                "hash_ids": [1, 2],
                "session_id": "test-session",
            }
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
        traces = [
            {
                "input_length": 100,
                "output_length": 20,
                "hash_ids": [1, 2, 3],
            }
        ]
        params = SynthesisParams(prefix_len_multiplier=1.0)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        hash_ids = synthetic[0].get("hash_ids", [])
        assert len(hash_ids) == 3

    def test_prefix_len_multiplier_2(self) -> None:
        """Test prefix_len_multiplier of 2 (double length)."""
        traces = [
            {
                "input_length": 100,
                "output_length": 20,
                "hash_ids": [1, 2],
            }
        ]
        params = SynthesisParams(prefix_len_multiplier=2.0)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        hash_ids = synthetic[0].get("hash_ids", [])
        # Should have roughly doubled
        assert len(hash_ids) > 2

    # ============================================================================
    # Max ISL Filter Tests
    # ============================================================================

    def test_max_isl_filter_applied(self) -> None:
        """Test that max_isl filter caps input length."""
        traces = [
            {"input_length": 5000, "output_length": 20},
        ]
        params = SynthesisParams(max_isl=4096)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        assert synthetic[0]["input_length"] <= 4096

    def test_max_isl_filter_not_applied(self) -> None:
        """Test that max_isl filter doesn't apply when None."""
        traces = [
            {"input_length": 2048, "output_length": 20},
        ]
        params = SynthesisParams(max_isl=None)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        # Should not be filtered
        assert synthetic[0]["input_length"] <= 2048

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
        """Test prefix_root_multiplier effect."""
        traces = [
            {
                "input_length": 100,
                "output_length": 20,
                "hash_ids": [1, 2],
            }
        ]
        params = SynthesisParams(prefix_root_multiplier=3)
        synthesizer = Synthesizer(params=params)
        synthetic = synthesizer.synthesize_traces(traces)

        hash_ids = synthetic[0].get("hash_ids", [])
        # With root multiplier of 3, should replicate the tree
        assert len(hash_ids) > 2
