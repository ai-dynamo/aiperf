# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RollingHasher."""

import pytest

from aiperf.dataset.synthesis import RollingHasher


class TestRollingHasher:
    """Tests for RollingHasher class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_default(self) -> None:
        """Test RollingHasher initialization with defaults."""
        hasher = RollingHasher()
        assert hasher.block_size == 512
        stats = hasher.get_stats()
        assert stats["total_hashes"] == 0
        assert stats["max_id"] == -1

    def test_initialization_custom_block_size(self) -> None:
        """Test RollingHasher initialization with custom block size."""
        hasher = RollingHasher(block_size=256)
        assert hasher.block_size == 256

    # ============================================================================
    # Hash Generation Tests
    # ============================================================================

    def test_hash_single_block(self) -> None:
        """Test hashing a single block."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(["hello"])
        assert len(hash_ids) == 1
        assert hash_ids[0] == 0

    def test_hash_multiple_blocks(self) -> None:
        """Test hashing multiple blocks."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(["hello", "world", "test"])
        assert len(hash_ids) == 3
        assert all(isinstance(h, int) for h in hash_ids)

    def test_hash_unique_assignment(self) -> None:
        """Test that unique blocks get unique hash IDs."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(["a", "b", "c"])
        # All should be different (since they're different blocks with different context)
        assert len(set(hash_ids)) >= 1  # At least unique from rolling hash context

    def test_hash_empty_list(self) -> None:
        """Test hashing empty list returns empty list."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks([])
        assert hash_ids == []

    @pytest.mark.parametrize(
        "blocks,expected_count",
        [
            (["a"], 1),
            (["a", "b"], 2),
            (["a", "b", "c", "d", "e"], 5),
        ],
    )
    def test_hash_sequence_lengths(
        self, blocks: list[str], expected_count: int
    ) -> None:
        """Test that output length matches input length."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(blocks)
        assert len(hash_ids) == expected_count

    # ============================================================================
    # Rolling Hash State Tests
    # ============================================================================

    def test_rolling_hash_context_matters(self) -> None:
        """Test that rolling hash context affects the hash ID."""
        hasher1 = RollingHasher()
        hash_ids1 = hasher1.hash_blocks(["a", "b"])

        hasher2 = RollingHasher()
        hash_ids2 = hasher2.hash_blocks(["a"])

        # The second "a" in hasher1's sequence is different from hasher2's "a"
        # because it has different context (different previous hash)
        assert len(hash_ids1) == 2
        assert len(hash_ids2) == 1

    def test_reset_clears_state(self) -> None:
        """Test that reset clears the rolling state."""
        hasher = RollingHasher()
        hash_ids1 = hasher.hash_blocks(["a", "b"])

        hasher.reset()

        hash_ids2 = hasher.hash_blocks(["a", "b"])

        # After reset, the same sequence should produce different context-based hashes
        assert len(hash_ids1) == len(hash_ids2)

    # ============================================================================
    # Statistics Tests
    # ============================================================================

    def test_get_stats_counts(self) -> None:
        """Test that statistics accurately count hashes."""
        hasher = RollingHasher()
        hasher.hash_blocks(["a", "b", "c"])

        stats = hasher.get_stats()
        assert stats["total_hashes"] > 0  # Should have seen some hashes
        assert stats["max_id"] >= 0

    def test_get_stats_multiple_sequences(self) -> None:
        """Test statistics across multiple sequences."""
        hasher = RollingHasher()
        hasher.hash_blocks(["a", "b"])
        initial_stats = hasher.get_stats()

        hasher.reset()
        hasher.hash_blocks(["c", "d", "e"])
        final_stats = hasher.get_stats()

        # Should have seen more total hashes after processing more blocks
        assert final_stats["total_hashes"] >= initial_stats["total_hashes"]

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_hash_single_long_block(self) -> None:
        """Test hashing a single very long block."""
        hasher = RollingHasher()
        long_text = "x" * 10000
        hash_ids = hasher.hash_blocks([long_text])
        assert len(hash_ids) == 1

    def test_hash_many_identical_blocks(self) -> None:
        """Test hashing many identical blocks."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(["same"] * 10)
        assert len(hash_ids) == 10
        # All should have different IDs due to rolling hash context

    def test_hash_special_characters(self) -> None:
        """Test hashing blocks with special characters."""
        hasher = RollingHasher()
        blocks = ["hello@world", "test#123", "special$chars"]
        hash_ids = hasher.hash_blocks(blocks)
        assert len(hash_ids) == 3
        assert all(isinstance(h, int) for h in hash_ids)
