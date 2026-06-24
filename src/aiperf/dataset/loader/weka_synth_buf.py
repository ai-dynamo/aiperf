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

# Re-exported for backwards compatibility; the composer lives in its own
# module since it is independent of the synthesis-buffer state machine.
from aiperf.dataset.loader.weka_prompt_compose import (  # noqa: E402
    compose_weka_prompt_tokens as compose_weka_prompt_tokens,
)
from aiperf.dataset.loader.weka_tool_shape import (
    demote_unpaired_tool_marks,
    tool_shape_segment_messages,
)


@dataclass
class TurnDelta:
    """Per-turn emission for delta-encoded conversation reconstruction.

    Returned by :meth:`ConversationReconstructor.turn_delta` after each
    ``init_turn_0`` / ``advance_turn`` call.
    """

    delta_messages: list[dict[str, str]]
    reset_context: bool


@dataclass
class RoleSegment:
    """One role-tagged segment of the reconstructed conversation.

    Block ranges of adjacent segments form a contiguous tile of [0, M_curr).
    Only the final segment may carry a partial-tail beyond its block range
    (encoded into ``tokens`` but not ``block_count``).

    ``tokens`` is the canonical size source — it holds the exact Qwen token IDs
    for this segment. ``content`` is the decoded text and is always equal to
    ``decode_tokens_to_text(tokens)`` at the time the segment was emitted.

    Block-alignment invariant: every segment except the trailing user holds exactly
    ``block_count * block_size`` tokens; the trailing user segment may hold
    ``block_count * block_size + partial_tail`` tokens. The tokens for any
    given hash_id are byte-identical across every segment they appear in.
    """

    role: Literal["system", "user", "assistant"]
    block_start: int
    block_count: int
    tokens: list[int]
    content: str
    tool_result_turn: int | None = None
    """Set on a user segment whose content is the tool output answering the
    immediately-preceding assistant segment's tool call. Holds the turn index
    the segment was created on, keying the deterministic synthetic call id so
    re-emissions reproduce the id the turn was first sent with."""

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
    emit_assistant_segments: bool = True
    """When False, ``turn_delta`` filters role=='assistant' segments out of
    the emitted ``delta_messages``. The segments remain in ``_segments`` for
    LCP/truncation accounting on subsequent turns. Used to switch the weka
    loader from pre-canned trace assistant text (preserves recorded hash_id
    chain, but invalidates server KV every turn) to live server-generated
    assistant turns threaded back via ``DELTAS_WITHOUT_RESPONSES`` (preserves
    cache-hit reuse across turns at the cost of hash-id fidelity past
    turn 0).
    """
    tool_shaped_messages: bool = False
    """When True, ``turn_delta`` emits marked tool-result user segments in
    the OpenAI tool-call wire shape (see weka_tool_shape). Requires emitted
    assistant segments for pairing, so live-assistant deltas stay plain."""
    _segments: list[RoleSegment] = field(default_factory=list)
    _emitted_segment_count: int = 0
    _last_disturbance_at: int | None = None
    _turn_index: int = 0

    def init_turn_0(
        self,
        hash_ids: list[int],
        in_tokens: int,
        tool_tokens: int,
        system_tokens: int,
        seed: str,
        is_tool_result: bool = False,
    ) -> None:
        """Initialize segments for turn 0 from a tool+system / user prefix split.

        See spec §4.3. hash_ids tile the first ``floor(in_tokens / bs)`` blocks
        when fully recorded; any partial tail of ``in_tokens % bs`` tokens is
        appended to the user segment via ``sample_partial_tail_tokens``.

        When ``hash_ids`` is **truncated** relative to
        ``floor(in_tokens / bs)`` (a common shape for real captures where the
        recorder only stored a prefix of the hash blocks), the missing block
        region is synthesized as additional partial-tail tokens on the
        trailing user segment. The resulting prompt has exactly ``in_tokens``
        tokens but a smaller hash-derived prefix than the recording had —
        KV-cache fidelity for the covered prefix is preserved; the uncovered
        suffix carries sha256-keyed synth tokens whose hashes don't match
        any recorded block. This matches the relaxed model already used by
        :func:`compose_weka_prompt_tokens`.

        If even the system/tool prefix can't be filled from hash_ids, the
        function still raises: synthesizing the system segment from random
        tokens would silently fake the KV-cache prefix the whole trace
        exists to measure.

        tool_tokens and system_tokens are merged into a SINGLE
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
        covered_blocks = min(m_full, len(hash_ids))
        missing_block_tokens = (m_full - covered_blocks) * bs

        cursor = 0
        segs: list[RoleSegment] = []

        if tool_tokens > 0 or system_tokens > 0:
            prefix_tokens = tool_tokens + system_tokens
            prefix_blocks = math.ceil(prefix_tokens / bs)
            if prefix_blocks > 0:
                # The system/tool prefix MUST come from hash_ids — synthesizing
                # it from random tokens would silently corrupt the KV-cache
                # prefix measurement (the whole point of the trace).
                if prefix_blocks > len(hash_ids):
                    raise ValueError(
                        f"weka trace turn-0 system prefix requires "
                        f"{prefix_blocks} hash blocks but only "
                        f"{len(hash_ids)} were recorded "
                        f"(tool_tokens={tool_tokens}, "
                        f"system_tokens={system_tokens}, block_size={bs}). "
                        f"The hash_ids list is too truncated to even "
                        f"reconstruct the prefix; aborting to avoid faking "
                        f"the cache structure."
                    )
                # Clamp to the prompt's own covered-block count. When the
                # declared prefix block count exceeds it -- tool+system tokens
                # exceed in_tokens, or ceil() rounding overshoots
                # floor(in_tokens/bs) with over-recorded hash blocks -- the
                # un-clamped system segment would claim more blocks than the
                # prompt contains: user_blocks (covered_blocks - cursor) goes
                # negative, the recorded user/tail slice silently empties, and
                # sum(seg.tokens) overshoots in_tokens, breaking the byte-exact
                # ISL invariant and poisoning the cumulative cursor in
                # truncate_synth_buf_at_block on every later turn. The
                # over-budget prefix tokens fold into the synth tail below.
                prefix_blocks = min(prefix_blocks, covered_blocks)
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

        # User segment: consume whatever hash_ids remain, then synthesize the
        # missing-blocks region + the recorded partial tail as one synth-tail
        # call. When the system prefix covers the entire prompt, appending a
        # zero-token user segment would be deleted by the next turn's
        # boundary cut and flag a spurious disturbance (reset_context) on a
        # pure-growth turn — skip it unless it is the only segment.
        user_blocks = covered_blocks - cursor
        user_tokens = self.decode_block_tokens(hash_ids[cursor : cursor + user_blocks])
        synth_tail_n = missing_block_tokens + partial_tail_tokens_n
        if synth_tail_n > 0:
            user_tokens.extend(self.sample_partial_tail_tokens(synth_tail_n, seed))
        if user_tokens or not segs:
            segs.append(
                RoleSegment(
                    role="user",
                    block_start=cursor,
                    block_count=user_blocks,
                    tokens=user_tokens,
                    content=self.decode_tokens_to_text(user_tokens),
                )
            )

        self._turn_index = 0
        del is_tool_result  # turn 0 has no preceding assistant to pair with
        self._segments = segs
        self._emitted_segment_count = 0
        self._last_disturbance_at = None

    def advance_turn(
        self,
        prev_hash_ids: list[int],
        prev_in_tokens: int,
        prev_out_tokens: int,
        curr_hash_ids: list[int],
        curr_in_tokens: int,
        seed: str,
        is_tool_result: bool = False,
    ) -> None:
        """Advance synth_buf to turn k via LCP-driven symmetric attribution.

        Implements spec §4.4: truncate at LCP, synthesize the post-LCP region,
        attribute ``ceil(prev_out / bs)`` blocks to an assistant segment and
        the remainder (blocks + partial tail) to a user segment. The same
        rule applies across all three structural patterns (append-only,
        mid-seq replace, pull-back); see §4.4.1.

        When ``curr_hash_ids`` is **truncated** relative to
        ``curr_in_tokens // bs`` (common in real captures where the recorder
        stored only a prefix of the hash blocks), the missing block region
        is synthesized as additional partial-tail tokens on the trailing
        user segment. Total tokens still equal ``curr_in_tokens`` exactly;
        only the uncovered suffix carries synth tokens whose hashes don't
        match any recorded block. Mirrors the relaxed shape in
        :meth:`init_turn_0`.

        Assistant size is block-aligned UP via
        ``ceil(prev_out_tokens / bs) * bs``, clamped to fit the new region.
        This makes the asst content slightly larger than the recorded
        ``prev_out_tokens`` (by up to ``bs - 1`` tokens) but preserves the
        hash-content invariant — every cached block emits its full content,
        unmodified by any terminator stamp.

        ``prev_in_tokens`` is retained for the recorded-request contract
        (mirroring ``prev_hash_ids`` / ``prev_out_tokens``) but no longer
        feeds the truncation: the buffer self-describes its trailing
        overhang, which is also correct when the boundary segment is not the
        trailing one.
        """
        bs = self.block_size
        m_curr = len(curr_hash_ids)
        m_curr_full = curr_in_tokens // bs
        # Mirror init_turn_0's ``covered_blocks = min(m_full, len(hash_ids))``:
        # a hashed-but-partial last block (len(curr_hash_ids) > curr_in_tokens
        # // bs, e.g. in=250 with hash_ids=[..,4] at bs=64) covers only its
        # ``curr_in_tokens % bs`` partial tail, not a full ``bs``-token block.
        # Decoding it as a full block AND appending the partial tail below
        # double-counts ~bs tokens and breaks the sum(seg.tokens) ==
        # curr_in_tokens invariant the byte-exact contract depends on.
        m_curr_covered = min(m_curr, m_curr_full)
        missing_block_tokens = max(0, (m_curr_full - m_curr) * bs)
        lcp = longest_common_prefix(prev_hash_ids, curr_hash_ids)

        truncate_disturbance = truncate_synth_buf_at_block(
            self._segments,
            lcp,
            bs,
            decode_tokens_to_text=self.decode_tokens_to_text,
        )
        self._last_disturbance_at = truncate_disturbance

        new_blocks = curr_hash_ids[lcp:m_curr_covered]
        new_partial_tail_n = curr_in_tokens % bs
        new_region_tokens = self.decode_block_tokens(new_blocks)
        synth_tail_n = missing_block_tokens + new_partial_tail_n
        if synth_tail_n > 0:
            new_region_tokens.extend(
                self.sample_partial_tail_tokens(synth_tail_n, seed)
            )
        new_blocks_count = max(0, m_curr_covered - lcp)

        self._turn_index += 1
        asst_blocks_target = (
            math.ceil(prev_out_tokens / bs) if prev_out_tokens > 0 else 0
        )
        if not any(seg.role == "user" for seg in self._segments):
            # Context-loss rule: the truncation removed every user segment
            # (or turn 0 was system-only), so the conversation resumes at a
            # USER turn — the wire cannot present assistant output before
            # any user input. The whole new region becomes user content.
            asst_blocks_target = 0
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
                    # The mark is unconditional here; turn_delta demotes it at
                    # first emission if no assistant directly precedes it in
                    # the emitted window (post-context-loss heads, missing
                    # assistant, assistant emitted in an earlier delta), so
                    # the shape a turn is first sent with is the shape every
                    # reset re-emission reproduces.
                    tool_result_turn=self._turn_index if is_tool_result else None,
                )
            )

    def turn_delta(self) -> TurnDelta:
        """Compute the raw_messages to emit for the just-completed turn.

        Three cases:
          1. First call after ``init_turn_0`` (``_emitted_segment_count == 0``):
             emit ALL current segments, ``reset_context=False``. This is
             turn 0's baseline state.
          2. Strict append (no disturbance, or disturbance only touched
             segments at index ``>= _emitted_segment_count``): emit segments
             at index ``>= _emitted_segment_count``, ``reset_context=False``.
          3. Disturbance touched a previously-emitted segment (index
             ``< _emitted_segment_count``): emit ALL current segments,
             ``reset_context=True``.

        Updates ``_emitted_segment_count`` to ``len(self._segments)`` on
        return. Clears ``_last_disturbance_at`` to ``None``.
        """
        disturbed_emitted = (
            self._last_disturbance_at is not None
            and self._last_disturbance_at < self._emitted_segment_count
        )
        if self._emitted_segment_count == 0 or disturbed_emitted:
            source = self._segments
            reset = self._emitted_segment_count != 0 and disturbed_emitted
        else:
            source = self._segments[self._emitted_segment_count :]
            reset = False

        messages = [{"role": s.role, "content": s.content} for s in source]
        if self.tool_shaped_messages and self.emit_assistant_segments:
            # A mark that cannot pair in THIS window ships plain and must
            # stay plain on every later re-emission — demote it now so reset
            # full re-emits cannot retroactively shape already-sent context.
            demote_unpaired_tool_marks(source)
            messages = tool_shape_segment_messages(messages, source)
        if not self.emit_assistant_segments:
            messages = [m for m in messages if m["role"] != "assistant"]

        self._emitted_segment_count = len(self._segments)
        self._last_disturbance_at = None
        return TurnDelta(delta_messages=messages, reset_context=reset)

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
) -> int | None:
    """Truncate ``segments`` in place so cumulative block_count == target_blocks.

    Block-aligned shape: every segment except the trailing one holds exactly
    ``block_count * block_size`` tokens; only the trailing segment may hold
    an overhang past its block range (the synthesized missing-block region
    plus the partial tail). So:

    Boundary case (``cursor + seg.block_count == target_blocks``): strip the
    segment's own overhang past ``block_count * block_size`` — zero for any
    non-trailing segment, the full missing-blocks + partial-tail synth region
    for the trailing one; the next turn's tiling re-introduces the right tail
    for ``curr_in_tokens``. Sizing the strip from the segment itself (not the
    previous turn's ``in_tokens % bs``) matters when the boundary segment is
    NOT the trailing one (e.g. a tail-only tool-result segment sits past it):
    stripping recorded hash-block tokens there would corrupt the hash-content
    invariant and destabilize every reset re-emission.

    Mid-segment case (truncation lands inside a segment): token-level slice
    to ``kept_blocks * block_size`` — guaranteed to end on a hash-block
    boundary because every segment is block-aligned.

    When ``decode_tokens_to_text`` is provided, ``content`` is re-derived
    from the surviving tokens to keep the (tokens, content) invariant.

    Returns the earliest segment index whose content was disturbed — modified
    in place (overhang strip, mid-segment slice) or removed entirely (cleared
    buffer, cut at a segment start, deletion past a boundary cut) — or
    ``None`` when no segment was touched (target beyond every segment, or
    empty buffer). :meth:`ConversationReconstructor.turn_delta` compares the
    index against the emitted count: disturbing a previously-emitted segment
    forces a context reset on the next emission.
    """
    if target_blocks <= 0:
        had_segments = len(segments) > 0
        segments.clear()
        return 0 if had_segments else None

    cursor = 0
    for i, seg in enumerate(segments):
        if cursor + seg.block_count < target_blocks:
            cursor += seg.block_count
            continue
        if cursor + seg.block_count == target_blocks:
            # Boundary cut: strip this segment's overhang past its block
            # range (only the trailing segment has one — missing-block synth
            # tokens plus the partial tail), then drop any segments past the
            # boundary.
            disturbed: int | None = None
            overhang = len(seg.tokens) - seg.block_count * block_size
            if overhang > 0:
                seg.tokens = seg.tokens[:-overhang]
                if decode_tokens_to_text is not None:
                    seg.content = decode_tokens_to_text(seg.tokens)
                disturbed = i
            deleted_past_boundary = i + 1 < len(segments)
            del segments[i + 1 :]
            if disturbed is None and deleted_past_boundary:
                disturbed = i + 1
            return disturbed
        if cursor == target_blocks:
            # Cut lands exactly at the start of segment i: segments[i:] are
            # all deleted. Earliest disturbed index is i.
            del segments[i:]
            return i
        # Mid-segment cut: token-level slice on a guaranteed block boundary.
        kept_blocks = target_blocks - cursor
        kept_tokens_n = min(len(seg.tokens), kept_blocks * block_size)
        seg.block_count = kept_blocks
        seg.tokens = seg.tokens[:kept_tokens_n]
        if decode_tokens_to_text is not None:
            seg.content = decode_tokens_to_text(seg.tokens)
        del segments[i + 1 :]
        return i
    return None
