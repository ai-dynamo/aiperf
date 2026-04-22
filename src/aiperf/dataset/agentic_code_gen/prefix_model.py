# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV cache prefix model for hash ID generation.

L1: Tools + system prompt. Identical across ALL sessions (globally cached).
L1.5: Group-shared prefix (repo instructions and context). Shared within a group.
L2: Session-specific prefix (initial files). Unique per session at turn 0.
L3: Conversation history added from turn 1 onward. Grows each turn, unique per session.

Hash ID layout:
    L1:     [0 .. L1_blocks-1]                                      (global)
    L1.5:   [L1_blocks + group_id * MAX_GROUP_BLOCKS .. +N]          (per group)
    L2+L3:  [session_base .. session_base+M]                         (per session)

On session reset: new session gets fresh L2+L3, L1 and L1.5 preserved.
"""

from __future__ import annotations

import math

from aiperf.dataset.agentic_code_gen.models import CacheLayerConfig

MAX_GROUP_BLOCKS = 200
MAX_GROUPS = 1_000
MAX_SESSION_BLOCKS = 4_000


class PrefixAllocator:
    """Allocates deterministic hash IDs for the layered prefix model."""

    def __init__(self, config: CacheLayerConfig, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self._block_size = block_size
        self._l1_blocks = math.ceil(config.layer1_tokens / block_size)
        self._l15_blocks = math.ceil(config.layer1_5_tokens / block_size)
        num_groups = config.layer1_5_groups.num_groups
        if num_groups > MAX_GROUPS:
            raise ValueError(f"num_groups={num_groups} exceeds MAX_GROUPS={MAX_GROUPS}")
        if self._l15_blocks > MAX_GROUP_BLOCKS:
            raise ValueError(
                f"layer1_5_tokens={config.layer1_5_tokens} needs "
                f"{self._l15_blocks} blocks, exceeding "
                f"MAX_GROUP_BLOCKS={MAX_GROUP_BLOCKS}"
            )
        self._l1_ids = list(range(self._l1_blocks))
        self._prefix_blocks = self._l1_blocks + self._l15_blocks
        # Session region starts after all group regions
        self._session_region_base = self._l1_blocks + MAX_GROUPS * MAX_GROUP_BLOCKS

    @property
    def l1_blocks(self) -> int:
        return self._l1_blocks

    @property
    def l15_blocks(self) -> int:
        return self._l15_blocks

    @property
    def prefix_blocks(self) -> int:
        """L1 + L1.5 blocks (shared prefix before session-specific content)."""
        return self._prefix_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    def group_base(self, group_id: int) -> int:
        """Compute the base hash ID offset for a group's L1.5 blocks."""
        return self._l1_blocks + group_id * MAX_GROUP_BLOCKS

    def session_base(self, session_index: int) -> int:
        """Compute the base hash ID offset for a session's L2+L3 blocks."""
        return self._session_region_base + session_index * MAX_SESSION_BLOCKS

    def _ensure_session_block_budget(
        self, session_needed: int, input_length: int
    ) -> None:
        if session_needed > MAX_SESSION_BLOCKS:
            raise ValueError(
                f"input_length={input_length} needs {session_needed} session blocks, "
                f"exceeding MAX_SESSION_BLOCKS={MAX_SESSION_BLOCKS}"
            )

    def _l15_ids(self, group_id: int) -> list[int]:
        """Get the L1.5 block IDs for a group."""
        base = self.group_base(group_id)
        return list(range(base, base + self._l15_blocks))

    def turn_hash_ids(
        self,
        session_index: int,
        group_id: int,
        input_length: int,
        prev_session_ids: list[int] | None = None,
    ) -> list[int]:
        """Generate the full hash_ids array for a turn.

        Returns L1 ++ L1.5 ++ session_ids where len(result) == ceil(input_length / block_size).
        On turn 0: everything beyond L1+L1.5 is session prefix (L2).
        On turn 1+: carried session blocks + new L3 growth.
        """
        total_blocks = (
            math.ceil(input_length / self._block_size) if input_length > 0 else 0
        )

        l1_used = min(self._l1_blocks, total_blocks)
        l1 = self._l1_ids[:l1_used]

        remaining = total_blocks - l1_used
        if remaining <= 0:
            return l1

        l15_used = min(self._l15_blocks, remaining)
        l15 = self._l15_ids(group_id)[:l15_used]

        session_needed = max(0, remaining - l15_used)
        if session_needed == 0:
            return l1 + l15

        self._ensure_session_block_budget(session_needed, input_length)
        base = self.session_base(session_index)

        if prev_session_ids is None:
            session = list(range(base, base + session_needed))
        else:
            new_blocks = session_needed - len(prev_session_ids)
            if new_blocks > 0:
                next_id = prev_session_ids[-1] + 1 if prev_session_ids else base
                session = prev_session_ids + list(range(next_id, next_id + new_blocks))
            else:
                session = prev_session_ids[:session_needed]

        return l1 + l15 + session

    def session_blocks_needed(self, input_length: int) -> int:
        """Compute how many session-level blocks (L2+L3) are needed for a given input length."""
        total_blocks = (
            math.ceil(input_length / self._block_size) if input_length > 0 else 0
        )
        l1_used = min(self._l1_blocks, total_blocks)
        remaining = total_blocks - l1_used
        l15_used = min(self._l15_blocks, remaining)
        return max(0, remaining - l15_used)

    def build_hash_ids(
        self,
        group_id: int,
        input_length: int,
        session_ids: list[int],
    ) -> list[int]:
        """Construct full hash_ids from L1 + L1.5 + provided session_ids."""
        total_blocks = (
            math.ceil(input_length / self._block_size) if input_length > 0 else 0
        )
        l1_used = min(self._l1_blocks, total_blocks)
        l1 = self._l1_ids[:l1_used]

        remaining = total_blocks - l1_used
        if remaining <= 0:
            return l1

        l15_used = min(self._l15_blocks, remaining)
        l15 = self._l15_ids(group_id)[:l15_used]

        session_needed = max(0, remaining - l15_used)
        if session_needed == 0:
            return l1 + l15

        self._ensure_session_block_budget(session_needed, input_length)
        if len(session_ids) < session_needed:
            raise ValueError(
                f"input_length={input_length} needs {session_needed} session IDs, "
                f"got {len(session_ids)}"
            )

        return l1 + l15 + session_ids[:session_needed]

    def extract_session_ids(self, hash_ids: list[int]) -> list[int]:
        """Extract everything after L1 + L1.5 from a full hash_ids array."""
        if len(hash_ids) <= self._prefix_blocks:
            return []
        return hash_ids[self._prefix_blocks :]
