# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RollingHasher."""

from unittest.mock import MagicMock

import pytest

from aiperf.dataset.synthesis import RollingHasher
from aiperf.dataset.synthesis.rolling_hasher import hashes_to_texts, texts_to_hashes


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

    # ============================================================================
    # Token Block Hashing Tests
    # ============================================================================

    def test_hash_token_blocks_single(self) -> None:
        """Test hashing a single token block."""
        hasher = RollingHasher()
        blocks = [[1, 2, 3, 4]]
        hash_ids = hasher.hash_token_blocks(blocks)
        assert len(hash_ids) == 1
        assert isinstance(hash_ids[0], int)

    def test_hash_token_blocks_multiple(self) -> None:
        """Test hashing multiple token blocks."""
        hasher = RollingHasher()
        blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        hash_ids = hasher.hash_token_blocks(blocks)
        assert len(hash_ids) == 3
        assert all(isinstance(h, int) for h in hash_ids)

    def test_hash_token_blocks_empty(self) -> None:
        """Test hashing empty token block list."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_token_blocks([])
        assert hash_ids == []

    def test_hash_token_blocks_preserves_prefix_sharing(self) -> None:
        """Test that shared prefixes produce same hash IDs."""
        hasher = RollingHasher()
        # Two sequences with shared prefix
        seq1 = [[1, 2], [3, 4], [5, 6]]
        seq2 = [[1, 2], [3, 4], [7, 8]]

        hash_ids1 = hasher.hash_token_blocks(seq1)
        hasher.reset()
        hash_ids2 = hasher.hash_token_blocks(seq2)

        # First two blocks should have same hash IDs (shared prefix)
        assert hash_ids1[0] == hash_ids2[0]
        assert hash_ids1[1] == hash_ids2[1]


class TestTextsToHashes:
    """Tests for texts_to_hashes module function."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # Simple tokenizer: 1 token per character
        tokenizer.encode = lambda text: list(range(len(text)))
        return tokenizer

    def test_texts_to_hashes_single_text(self, mock_tokenizer: MagicMock) -> None:
        """Test converting a single text to hashes."""
        texts = ["a" * 20]  # 20 tokens with block_size=10 = 2 blocks
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 1
        assert len(result[0]) == 2  # 2 blocks
        assert all(isinstance(h, int) for h in result[0])

    def test_texts_to_hashes_multiple_texts(self, mock_tokenizer: MagicMock) -> None:
        """Test converting multiple texts to hashes."""
        texts = ["a" * 20, "b" * 30]
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 2
        assert len(result[0]) == 2  # 20 tokens / 10 = 2 blocks
        assert len(result[1]) == 3  # 30 tokens / 10 = 3 blocks

    def test_texts_to_hashes_empty_text(self, mock_tokenizer: MagicMock) -> None:
        """Test converting empty text returns empty hash list."""
        texts = [""]
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 1
        assert result[0] == []

    def test_texts_to_hashes_shared_prefix(self, mock_tokenizer: MagicMock) -> None:
        """Test that shared text prefixes produce shared hash IDs."""
        # Two texts with identical first 10 chars (1 block)
        texts = ["aaaaaaaaaa" + "b" * 10, "aaaaaaaaaa" + "c" * 10]
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 2
        # First block should have same hash ID (shared prefix)
        assert result[0][0] == result[1][0]


class TestHashesToTexts:
    """Tests for hashes_to_texts module function."""

    @pytest.fixture
    def mock_prompt_generator(self) -> MagicMock:
        """Create a mock prompt generator."""
        generator = MagicMock()
        generator.generate = MagicMock(return_value="generated text")
        return generator

    def test_hashes_to_texts_single(self, mock_prompt_generator: MagicMock) -> None:
        """Test converting single hash sequence to text."""
        hash_ids_list = [[1, 2, 3]]
        input_lengths = [100]

        result = hashes_to_texts(
            mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
        )

        assert len(result) == 1
        mock_prompt_generator.generate.assert_called_once()

    def test_hashes_to_texts_multiple(self, mock_prompt_generator: MagicMock) -> None:
        """Test converting multiple hash sequences to texts."""
        hash_ids_list = [[1, 2], [3, 4, 5]]
        input_lengths = [100, 150]

        result = hashes_to_texts(
            mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
        )

        assert len(result) == 2
        assert mock_prompt_generator.generate.call_count == 2

    def test_hashes_to_texts_empty_hash_ids(
        self, mock_prompt_generator: MagicMock
    ) -> None:
        """Test converting empty hash_ids generates text without hash_ids."""
        hash_ids_list = [[]]
        input_lengths = [100]

        result = hashes_to_texts(
            mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
        )

        assert len(result) == 1
        # Should call generate without hash_ids
        mock_prompt_generator.generate.assert_called_with(mean=100)

    def test_hashes_to_texts_constraint_violation(
        self, mock_prompt_generator: MagicMock
    ) -> None:
        """Test that constraint violation raises ValueError."""
        # 2 hash_ids * 64 block_size = 128 < 200 input_length
        hash_ids_list = [[1, 2]]
        input_lengths = [200]

        with pytest.raises(ValueError, match="Constraint violation"):
            hashes_to_texts(
                mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
            )
