# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PrefixAnalyzer."""

import pytest

from aiperf.dataset.synthesis import PrefixAnalyzer


class TestPrefixAnalyzer:
    """Tests for PrefixAnalyzer class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_default(self) -> None:
        """Test PrefixAnalyzer initialization with defaults."""
        analyzer = PrefixAnalyzer()
        assert analyzer.block_size == 512

    def test_initialization_custom_block_size(self) -> None:
        """Test PrefixAnalyzer initialization with custom block size."""
        analyzer = PrefixAnalyzer(block_size=256)
        assert analyzer.block_size == 256

    # ============================================================================
    # Analysis Tests
    # ============================================================================

    def test_analyze_single_trace(self, sample_trace_data) -> None:
        """Test analyzing a single trace."""
        analyzer = PrefixAnalyzer()
        traces = sample_trace_data[:1]
        stats = analyzer.analyze_traces(traces)

        assert stats.total_requests == 1
        assert stats.min_isl == 100
        assert stats.max_isl == 100

    def test_analyze_multiple_traces(self, sample_trace_data) -> None:
        """Test analyzing multiple traces."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(sample_trace_data)

        assert stats.total_requests == 5
        assert stats.min_isl == 100
        assert stats.max_isl == 200

    def test_analyze_file(self, trace_file_simple) -> None:
        """Test analyzing traces from file."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_file(trace_file_simple)

        assert stats.total_requests > 0

    # ============================================================================
    # Statistics Computation Tests
    # ============================================================================

    def test_isl_osl_extraction(self, sample_trace_data) -> None:
        """Test ISL/OSL extraction."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(sample_trace_data)

        assert stats.min_isl == 100
        assert stats.max_isl == 200
        assert 100 <= stats.avg_isl <= 200

        assert stats.min_osl == 20
        assert stats.max_osl == 40

    @pytest.mark.parametrize(
        "isl_values,expected_avg",
        [
            ([100, 100, 100], 100),
            ([100, 200, 300], 200),
            ([50, 100, 150], 100),
        ],
    )
    def test_average_isl_computation(self, isl_values, expected_avg) -> None:
        """Test average ISL computation."""
        traces = [{"input_length": isl, "output_length": 20} for isl in isl_values]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        assert stats.avg_isl == expected_avg

    # ============================================================================
    # Cache Hit Rate Tests
    # ============================================================================

    def test_cache_hit_rate_no_hashes(self, sample_trace_without_hashes) -> None:
        """Test cache hit rate with no hash IDs."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(sample_trace_without_hashes)

        assert stats.cache_hit_rate == 0.0  # No hashes means no cache reuse

    def test_cache_hit_rate_all_unique(self) -> None:
        """Test cache hit rate with all unique hash IDs."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1]},
            {"input_length": 100, "output_length": 20, "hash_ids": [2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [3]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        assert stats.cache_hit_rate == 0.0  # No shared prefixes

    def test_cache_hit_rate_with_reuse(self) -> None:
        """Test cache hit rate with prefix reuse."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # hash_id 1 appears 3 times, so at least some reuse
        assert stats.cache_hit_rate > 0.0

    def test_cache_hit_rate_100_percent(self) -> None:
        """Test cache hit rate when all requests use same prefix."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # First trace: 2 misses (cold cache)
        # Second trace: 2 hits (already seen)
        # Third trace: 2 hits (already seen)
        # Total: 6 blocks, 4 hits = 4/6 = 0.6667
        assert abs(stats.cache_hit_rate - (4 / 6)) < 0.01

    # ============================================================================
    # Prefix Reuse Tests
    # ============================================================================

    def test_prefix_reuse_ratio_no_reuse(self) -> None:
        """Test prefix reuse ratio with no reuse."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1]},
            {"input_length": 100, "output_length": 20, "hash_ids": [2]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        assert stats.prefix_reuse_ratio == 0.0

    def test_prefix_reuse_ratio_with_reuse(self) -> None:
        """Test prefix reuse ratio with reuse."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        assert stats.prefix_reuse_ratio > 0.0

    # ============================================================================
    # Unique Prefixes Tests
    # ============================================================================

    def test_unique_prefixes_count(self) -> None:
        """Test unique prefix counting.

        Counts all unique prefix subsequences: (1), (1,2), (1,2,3), etc.
        """
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 4]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 5]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # Prefixes: (1), (1,2), (1,2,3), (1,2,4), (1,5) = 5 unique
        assert stats.unique_prefixes == 5

    def test_unique_prefixes_no_hashes(self, sample_trace_without_hashes) -> None:
        """Test unique prefix counting with no hashes."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(sample_trace_without_hashes)

        assert stats.unique_prefixes == 0

    # ============================================================================
    # Prefix Groups Tests
    # ============================================================================

    def test_num_prefix_groups_single_shared(self) -> None:
        """Test prefix groups with one shared first block.

        Counts distinct first blocks that appear in 2+ sequences.
        """
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 4]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 5]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # First block 1 appears in all 3 sequences
        assert stats.num_prefix_groups == 1

    def test_num_prefix_groups_multiple_shared(self) -> None:
        """Test prefix groups with multiple shared first blocks."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 3]},
            {"input_length": 100, "output_length": 20, "hash_ids": [2, 4]},
            {"input_length": 100, "output_length": 20, "hash_ids": [2, 5]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # First block 1 shared by 2, first block 2 shared by 2
        assert stats.num_prefix_groups == 2

    def test_num_prefix_groups_no_shared(self) -> None:
        """Test prefix groups when no first blocks are shared."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 100, "output_length": 20, "hash_ids": [3, 4]},
            {"input_length": 100, "output_length": 20, "hash_ids": [5, 6]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # All first blocks are unique (1, 3, 5)
        assert stats.num_prefix_groups == 0

    def test_num_prefix_groups_identical_sequences(self) -> None:
        """Test prefix groups when all sequences are identical."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # First block 1 is shared across all 3
        assert stats.num_prefix_groups == 1

    def test_num_prefix_groups_no_hashes(self, sample_trace_without_hashes) -> None:
        """Test prefix groups with no hashes."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(sample_trace_without_hashes)

        assert stats.num_prefix_groups == 0

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_analyze_empty_traces(self) -> None:
        """Test analyzing empty trace list."""
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces([])

        assert stats.total_requests == 0

    def test_analyze_trace_with_missing_fields(self) -> None:
        """Test analyzing trace with missing fields."""
        traces = [
            {"input_length": 100},  # Missing output_length
            {"output_length": 20},  # Missing input_length
        ]
        analyzer = PrefixAnalyzer()
        stats = analyzer.analyze_traces(traces)

        # Should handle gracefully
        assert stats.total_requests == 2

    def test_multiple_analyses_separate_state(self) -> None:
        """Test that multiple analyses don't interfere."""
        analyzer1 = PrefixAnalyzer()
        analyzer2 = PrefixAnalyzer()

        traces1 = [{"input_length": 100, "output_length": 20}]
        traces2 = [{"input_length": 200, "output_length": 30}]

        stats1 = analyzer1.analyze_traces(traces1)
        stats2 = analyzer2.analyze_traces(traces2)

        assert stats1.min_isl == 100
        assert stats2.min_isl == 200
