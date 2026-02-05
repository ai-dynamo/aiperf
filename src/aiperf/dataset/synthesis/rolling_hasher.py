# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling hasher for converting text blocks to unique hash IDs.

Provides functions for converting between texts and hash IDs:
- texts_to_hashes: Convert texts to hash ID sequences
- hashes_to_texts: Convert hash IDs back to reproducible texts

The hash IDs are consecutive integers where shared values between
sequences represent prefix overlap (cache hits).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from aiperf.common.config.config_defaults import InputTokensDefaults
from aiperf.common.tokenizer import Tokenizer


def _stable_hash(data: str | tuple) -> int:
    """Compute a stable hash consistent across Python sessions.

    Uses SHA-256 truncated to 64 bits for deterministic hashing.
    Unlike Python's built-in hash(), this produces identical results
    across different Python processes and runs.

    Args:
        data: String or tuple to hash.

    Returns:
        64-bit integer hash value.
    """
    # For tuples, serialize to string first; strings encode directly
    encoded = data.encode() if isinstance(data, str) else str(data).encode()
    # Use first 8 bytes of SHA-256 as 64-bit int
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


if TYPE_CHECKING:
    from aiperf.dataset.generator.prompt import PromptGenerator


class RollingHasher:
    """Converts sequences of text blocks into globally unique hash IDs.

    Uses a rolling hash approach where each block's hash depends on:
    1. The block content itself
    2. The previous block's hash (for sequential chaining)

    This creates a stateful hash function where the same text block
    may produce different hash IDs depending on context.
    """

    def __init__(self, block_size: int = InputTokensDefaults.BLOCK_SIZE) -> None:
        """Initialize the rolling hasher.

        Args:
            block_size: Number of tokens per block for hashing.
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
        # Compute stable hash of current block (consistent across Python sessions)
        block_hash = _stable_hash(block)

        # Rolling hash: combine with previous hash for sequential context
        combined_hash = _stable_hash((self._prev_hash, block_hash))

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
            "max_id": self._id_counter - 1 if self._id_counter > 0 else -1,
        }

    def hash_token_blocks(self, blocks: Sequence[Sequence[int]]) -> list[int]:
        """Convert a sequence of token blocks to hash IDs.

        Args:
            blocks: Sequence of token blocks (each block is a sequence of token IDs).

        Returns:
            List of unique hash IDs corresponding to each block.
        """
        hash_ids: list[int] = []
        parent_hash = 0

        for block in blocks:
            block_tuple = tuple(block) if not isinstance(block, tuple) else block
            combined = (parent_hash, _stable_hash(block_tuple))
            global_hash = _stable_hash(combined)

            if global_hash not in self._hash_to_id:
                self._hash_to_id[global_hash] = self._id_counter
                self._id_counter += 1

            hash_ids.append(self._hash_to_id[global_hash])
            parent_hash = global_hash

        return hash_ids


def texts_to_hashes(
    tokenizer: Tokenizer,
    texts: list[str],
    block_size: int = InputTokensDefaults.BLOCK_SIZE,
) -> list[list[int]]:
    """Convert a list of texts to hash ID sequences.

    Tokenizes texts, splits into blocks, and generates consecutive hash IDs.
    Shared hash IDs between texts represent prefix overlap (cache hits).

    Args:
        tokenizer: Tokenizer for encoding texts.
        texts: List of input text strings.
        block_size: Number of tokens per block.

    Returns:
        List of hash ID sequences, one per input text.
        len(hash_ids) == input_length // block_size for each text.
    """
    hasher = RollingHasher(block_size=block_size)
    results: list[list[int]] = []

    for text in texts:
        tokens = tokenizer.encode(text)
        blocks: list[list[int]] = [
            tokens[i : i + block_size] for i in range(0, len(tokens), block_size)
        ]
        if blocks:
            hashes = hasher.hash_token_blocks(blocks)
            results.append(hashes)
        else:
            results.append([])

    return results


def hashes_to_texts(
    prompt_generator: PromptGenerator,
    hash_ids_list: list[list[int]],
    input_lengths: list[int],
    block_size: int = InputTokensDefaults.BLOCK_SIZE,
) -> list[str]:
    """Convert hash ID sequences back to text strings.

    Uses the PromptGenerator's cache to ensure the same hash ID always produces
    the same token block, enabling prefix sharing. Text is generated from the
    Shakespeare corpus used by PromptGenerator.

    Args:
        prompt_generator: PromptGenerator instance for generating text from hash_ids.
        hash_ids_list: List of hash ID sequences.
        input_lengths: Target input lengths (in tokens) for each sequence.
        block_size: Number of tokens per block.

    Returns:
        List of text strings, one per hash ID sequence.

    Raises:
        ValueError: If len(hash_ids) * block_size < input_length for any sequence.
    """
    results: list[str] = []

    for hash_ids, input_len in zip(hash_ids_list, input_lengths, strict=True):
        # Verify constraint: len(hash_ids) * block_size >= input_len
        if hash_ids and len(hash_ids) * block_size < input_len:
            raise ValueError(
                f"Constraint violation: len(hash_ids) * block_size "
                f"({len(hash_ids) * block_size}) < input_len ({input_len})"
            )

        if hash_ids:
            # Use PromptGenerator to generate text from hash_ids
            # This uses the Shakespeare corpus and caches blocks by hash_id
            text = prompt_generator.generate(mean=input_len, hash_ids=hash_ids)
        else:
            # No hash_ids - generate plain text of target length
            text = prompt_generator.generate(mean=input_len)

        results.append(text)

    return results
