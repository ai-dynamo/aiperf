# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LCP-driven conversation reconstructor for byte-exact weka trace replay.

The module name ``synth_buf`` is short for "synthesis buffer" — the
multi-segment in-progress chat-message tile this module maintains across
turns. The canonical reconstructor lives in :class:`ConversationReconstructor`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal


def compose_weka_prompt_tokens(
    *,
    hash_ids: list[int],
    input_length: int,
    decode_block_tokens: Callable[[list[int]], list[int]],
    sample_partial_tail_tokens: Callable[[int, str], list[int]],
    seed: str,
) -> list[int]:
    """Build the prompt token sequence for a weka turn.

    Replaces the ``synthesize_prompts_from_hash_ids`` parent-side phase: the
    same hash-id-seeded RNG used by :class:`ConversationReconstructor` for
    LCP segments is reused here for the prompt itself, so workers can
    produce both byte-deterministically without a separate ``parallel_decode``
    pool.

    Three layouts:

    - ``hash_ids`` empty: prompt is entirely a sha256-keyed sample of length
      ``input_length``.
    - ``input_length <= len(hash_ids) * block_size``: exact-tile or
      last-block-partial; truncate the hashed prefix to ``input_length``.
      Byte-identical to ``_build_token_sequence``'s last-block-partial path
      because ``sample_tokens_from_corpus`` calls ``randrange`` exactly once
      regardless of block size, so prefix truncation matches.
    - ``input_length > len(hash_ids) * block_size``: prefix-only — append a
      sha256-keyed partial tail. Byte content of the tail differs from
      ``_build_token_sequence``'s order-dependent ``_corpus_rng`` path; the
      sha256-keyed seed makes the tail position-deterministic and
      reproducible across processes.
    """
    if not hash_ids:
        return sample_partial_tail_tokens(input_length, seed)
    block_tokens = decode_block_tokens(hash_ids)
    if input_length <= len(block_tokens):
        return block_tokens[:input_length]
    tail = input_length - len(block_tokens)
    return block_tokens + sample_partial_tail_tokens(tail, seed)


@dataclass
class RoleSegment:
    """One role-tagged segment of the reconstructed conversation.

    Block ranges of adjacent segments form a contiguous tile of [0, M_curr).
    Only the final segment may carry a partial-tail beyond its block range
    (encoded into ``tokens`` but not ``block_count``).

    ``tokens`` is the canonical size source — it holds the exact Qwen token IDs
    for this segment. ``content`` is the decoded text and is always equal to
    ``decode_tokens_to_text(tokens)`` at the time the segment was emitted.

    Post-P17 invariant: every segment except the trailing user holds exactly
    ``block_count * block_size`` tokens; the trailing user segment may hold
    ``block_count * block_size + partial_tail`` tokens. The tokens for any
    given hash_id are byte-identical across every segment they appear in.
    """

    role: Literal["system", "user", "assistant"]
    block_start: int
    block_count: int
    tokens: list[int]
    content: str

    @property
    def content_token_count(self) -> int:
        """Token count == ``len(tokens)``. Kept as a property for back-compat."""
        return len(self.tokens)


@dataclass
class ConversationReconstructor:
    """Walks a conversation's turns, maintaining synth_buf segments.

    Caller invariants:
    - ``decode_block_tokens(hash_ids)`` returns the deterministic Qwen token
      sequence for the given blocks (exactly ``len(hash_ids) * block_size``
      tokens).
    - ``sample_partial_tail_tokens(n, seed)`` returns deterministic Qwen
      token IDs that total exactly ``n``. ``seed`` must be position-keyed
      (e.g. sha256((conv_id, turn_index, "partial_tail"))).
    - ``decode_tokens_to_text(tokens)`` decodes a token list to text using
      the same tokenizer, with no special-token insertion.

    ``bpe_stable_terminator_tokens`` is plumbed through to PromptGenerator
    but the reconstructor algorithm intentionally does not consume it:
    rewriting trailing tokens of every segment would violate the
    hash-content invariant (a given ``hash_id`` must decode to the
    identical token sequence in every segment of every turn).
    """

    block_size: int
    decode_block_tokens: Callable[[list[int]], list[int]]
    sample_partial_tail_tokens: Callable[[int, str], list[int]]
    decode_tokens_to_text: Callable[[list[int]], str]
    bpe_stable_terminator_tokens: list[int] = field(default_factory=list)
    _segments: list[RoleSegment] = field(default_factory=list)

    def init_turn_0(
        self,
        hash_ids: list[int],
        in_tokens: int,
        tool_tokens: int,
        system_tokens: int,
        seed: str,
    ) -> None:
        """Initialize segments for turn 0 from a tool+system / user prefix split.

        See spec §4.3. hash_ids tile the first ``floor(in_tokens / bs)`` blocks;
        any partial tail of ``in_tokens % bs`` tokens is appended to the user
        segment via ``sample_partial_tail_tokens``.

        Post-P19: tool_tokens and system_tokens are merged into a SINGLE
        ``role="system"`` segment of ``ceil((tool+system)/bs) * bs`` tokens.
        Some serving stacks (Anthropic API, certain Qwen deployments) reject
        chat requests containing multiple adjacent system messages; trace
        audit confirmed tool/system token counts are constant per scope, so
        merging once at turn 0 is safe. The merged segment consumes the same
        hash blocks ``[0..ceil((tool+system)/bs))`` the two-segment form did,
        preserving the KV-cache prefix byte-for-byte. The user segment
        receives the remainder of the block tile plus any partial tail. This
        guarantees ``sum(len(seg.tokens)) == in_tokens`` exactly and that
        every segment decodes the cached hash content byte-identically.
        """
        bs = self.block_size
        m_full = in_tokens // bs
        partial_tail_tokens_n = in_tokens - m_full * bs
        if len(hash_ids) < m_full:
            raise ValueError(
                f"weka trace turn-0 has {len(hash_ids)} hash_ids but in_tokens="
                f"{in_tokens} with block_size={bs} requires at least {m_full} "
                f"(floor(in_tokens / block_size)). Either the recorded hash_ids "
                f"list is truncated or in_tokens is wrong."
            )

        cursor = 0
        segs: list[RoleSegment] = []

        if tool_tokens > 0 or system_tokens > 0:
            prefix_tokens = tool_tokens + system_tokens
            prefix_blocks = math.ceil(prefix_tokens / bs)
            if prefix_blocks > 0:
                seg_tokens = self.decode_block_tokens(
                    hash_ids[cursor : cursor + prefix_blocks]
                )
                segs.append(
                    RoleSegment(
                        role="system",
                        block_start=cursor,
                        block_count=prefix_blocks,
                        tokens=seg_tokens,
                        content=self.decode_tokens_to_text(seg_tokens),
                    )
                )
                cursor += prefix_blocks

        user_blocks = m_full - cursor
        user_tokens = self.decode_block_tokens(hash_ids[cursor : cursor + user_blocks])
        if partial_tail_tokens_n > 0:
            user_tokens.extend(
                self.sample_partial_tail_tokens(partial_tail_tokens_n, seed)
            )
        segs.append(
            RoleSegment(
                role="user",
                block_start=cursor,
                block_count=user_blocks,
                tokens=user_tokens,
                content=self.decode_tokens_to_text(user_tokens),
            )
        )

        self._segments = segs

    def advance_turn(
        self,
        prev_hash_ids: list[int],
        prev_in_tokens: int,
        prev_out_tokens: int,
        curr_hash_ids: list[int],
        curr_in_tokens: int,
        seed: str,
    ) -> None:
        """Advance synth_buf to turn k via LCP-driven symmetric attribution.

        Implements spec §4.4: truncate at LCP, synthesize the post-LCP region,
        attribute ``ceil(prev_out / bs)`` blocks to an assistant segment and
        the remainder (blocks + partial tail) to a user segment. The same
        rule applies across all three structural patterns (append-only,
        mid-seq replace, pull-back); see §4.4.1.

        Post-P17: assistant size is block-aligned UP via
        ``ceil(prev_out_tokens / bs) * bs``, clamped to fit the new region.
        This makes the asst content slightly larger than the recorded
        ``prev_out_tokens`` (by up to ``bs - 1`` tokens) but preserves the
        hash-content invariant — every cached block emits its full content,
        unmodified by any terminator stamp.
        """
        bs = self.block_size
        m_curr = len(curr_hash_ids)
        lcp = longest_common_prefix(prev_hash_ids, curr_hash_ids)
        prev_partial_tail = prev_in_tokens % bs

        truncate_synth_buf_at_block(
            self._segments,
            lcp,
            bs,
            decode_tokens_to_text=self.decode_tokens_to_text,
            prev_partial_tail=prev_partial_tail,
        )

        new_blocks = curr_hash_ids[lcp:m_curr]
        new_partial_tail_n = curr_in_tokens % bs
        new_region_tokens = self.decode_block_tokens(new_blocks)
        if new_partial_tail_n > 0:
            new_region_tokens.extend(
                self.sample_partial_tail_tokens(new_partial_tail_n, seed)
            )
        new_blocks_count = m_curr - lcp

        asst_blocks_target = (
            math.ceil(prev_out_tokens / bs) if prev_out_tokens > 0 else 0
        )
        asst_blocks = min(asst_blocks_target, new_blocks_count)
        asst_emit_size = asst_blocks * bs

        cursor = lcp
        if asst_blocks > 0:
            asst_tokens = new_region_tokens[:asst_emit_size]
            self._segments.append(
                RoleSegment(
                    role="assistant",
                    block_start=cursor,
                    block_count=asst_blocks,
                    tokens=asst_tokens,
                    content=self.decode_tokens_to_text(asst_tokens),
                )
            )
            cursor += asst_blocks

        user_blocks = new_blocks_count - asst_blocks
        user_tokens = new_region_tokens[asst_emit_size:]
        if len(user_tokens) > 0:
            self._segments.append(
                RoleSegment(
                    role="user",
                    block_start=cursor,
                    block_count=user_blocks,
                    tokens=user_tokens,
                    content=self.decode_tokens_to_text(user_tokens),
                )
            )

    def snapshot_messages(self) -> list[dict[str, str]]:
        """Return the current synth_buf as a list of OpenAI-style chat messages.

        Each segment becomes one ``{"role": ..., "content": ...}`` dict, in
        order. ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``.
        The returned list is a fresh list of fresh dicts — callers may mutate
        without affecting the reconstructor's internal state.

        Used by ``WekaTraceLoader`` (and the parallel-convert worker path) to
        fill ``Turn.raw_messages`` so AIPerf can replay the byte-exact prompt
        seen by the original recording. The list represents the FULL chat
        prefix at this turn (NOT just the latest appended user message); the
        orchestrator concatenates them with the serving stack's chat template
        at request time.
        """
        return [{"role": s.role, "content": s.content} for s in self._segments]


def longest_common_prefix(prev_hash_ids: list[int], curr_hash_ids: list[int]) -> int:
    """Return the index of the first differing element of the two sequences.

    Returns 0 when the first elements differ; returns
    ``min(len(prev_hash_ids), len(curr_hash_ids))`` when one sequence is a
    complete prefix of the other.
    """
    n = min(len(prev_hash_ids), len(curr_hash_ids))
    for i in range(n):
        if prev_hash_ids[i] != curr_hash_ids[i]:
            return i
    return n


def truncate_synth_buf_at_block(
    segments: list[RoleSegment],
    target_blocks: int,
    block_size: int,
    decode_tokens_to_text: Callable[[list[int]], str] | None = None,
    prev_partial_tail: int = 0,
) -> None:
    """Truncate ``segments`` in place so cumulative block_count == target_blocks.

    Post-P17: every segment except the trailing user holds exactly
    ``block_count * block_size`` tokens, and the trailing user holds
    ``block_count * block_size + prev_partial_tail`` tokens. So:

    Boundary case (``cursor + seg.block_count == target_blocks``):
    Trailing tokens past ``block_count * block_size`` are exactly the
    ``prev_partial_tail`` tokens (no asst-block-rounding overhead to
    disambiguate from). Strip them; the next turn's tiling will
    re-introduce the right partial tail for ``curr_in_tokens % bs``.

    Mid-segment case (truncation lands inside a segment): token-level slice
    to ``kept_blocks * block_size``. Now this slice is guaranteed to end on
    a hash-block boundary because every segment is block-aligned.

    When ``decode_tokens_to_text`` is provided, ``content`` is re-derived
    from the surviving tokens to keep the (tokens, content) invariant.
    """
    if target_blocks <= 0:
        segments.clear()
        return

    cursor = 0
    for i, seg in enumerate(segments):
        if cursor + seg.block_count < target_blocks:
            cursor += seg.block_count
            continue
        if cursor + seg.block_count == target_blocks:
            # Boundary cut: strip the trailing partial_tail tokens (post-P17
            # the only tokens past block_count*bs are the partial tail).
            if prev_partial_tail > 0 and len(seg.tokens) > 0:
                stripped_n = min(prev_partial_tail, len(seg.tokens))
                seg.tokens = seg.tokens[:-stripped_n]
                if decode_tokens_to_text is not None:
                    seg.content = decode_tokens_to_text(seg.tokens)
            del segments[i + 1 :]
            return
        if cursor == target_blocks:
            del segments[i:]
            return
        # Mid-segment cut: token-level slice on a guaranteed block boundary.
        kept_blocks = target_blocks - cursor
        kept_tokens_n = min(len(seg.tokens), kept_blocks * block_size)
        seg.block_count = kept_blocks
        seg.tokens = seg.tokens[:kept_tokens_n]
        if decode_tokens_to_text is not None:
            seg.content = decode_tokens_to_text(seg.tokens)
        del segments[i + 1 :]
        return
