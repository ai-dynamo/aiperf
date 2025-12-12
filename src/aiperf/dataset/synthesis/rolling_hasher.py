# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling hasher for converting text blocks to unique hash IDs."""

from collections.abc import Sequence


class RollingHasher:
    """Converts sequences of text blocks into globally unique hash IDs.

    Uses a rolling hash approach where each block's hash depends on:
    1. The block content itself
    2. The previous block's hash (for sequential chaining)

    This creates a stateful hash function where the same text block
    may produce different hash IDs depending on context.
    """

    def __init__(self, block_size: int = 512) -> None:
        """Initialize the rolling hasher.

        Args:
            block_size: Number of tokens per block for hashing (default: 512).
        """
        self.block_size = block_size
        self._hash_to_id: dict[int, int] = {}  # Maps hash values to unique IDs
        self._id_counter = 0  # Counter for assigning unique IDs
        self._prev_hash = 0  # State: previous hash for rolling computation

    def hash_blocks(self, blocks: Sequence[str]) -> list[int]:
        """Convert a sequence of text blocks to hash IDs.

        Args:
            blocks: Sequence of text strings representing blocks.

        Returns:
            List of unique hash IDs corresponding to each block.
        """
        hash_ids = []
        self._prev_hash = 0

        for block in blocks:
            hash_id = self._hash_block(block)
            hash_ids.append(hash_id)

        return hash_ids

    def _hash_block(self, block: str) -> int:
        """Hash a single block using rolling hash.

        Args:
            block: Text block to hash.

        Returns:
            Unique hash ID for this block in its context.
        """
        # Compute hash of current block
        block_hash = hash(block)

        # Rolling hash: combine with previous hash for sequential context
        combined_hash = hash((self._prev_hash, block_hash))

        # Map to unique ID, creating new ID if not seen before
        if combined_hash not in self._hash_to_id:
            self._hash_to_id[combined_hash] = self._id_counter
            self._id_counter += 1

        hash_id = self._hash_to_id[combined_hash]
        self._prev_hash = combined_hash

        return hash_id

    def reset(self) -> None:
        """Reset the hasher state for hashing new sequences."""
        self._prev_hash = 0
        # Note: We keep _hash_to_id and _id_counter to maintain global uniqueness

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the hasher.

        Returns:
            Dictionary with 'total_hashes' (unique hashes seen) and
            'max_id' (highest hash ID assigned).
        """
        return {
            "total_hashes": len(self._hash_to_id),
            "max_id": self._id_counter - 1 if self._id_counter > 0 else 0,
        }
