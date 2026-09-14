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

from aiperf.common.aiperf_logger import AIPerfLogger

# Re-exported for backwards compatibility; the composer lives in its own
# module since it is independent of the synthesis-buffer state machine.
from aiperf.dataset.loader.weka_prompt_compose import (  # noqa: E402
    compose_weka_prompt_tokens as compose_weka_prompt_tokens,
)
from aiperf.dataset.loader.weka_tool_shape import (
    demote_unpaired_tool_marks,
    tool_shape_segment_messages,
)

_logger = AIPerfLogger(__name__)


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
    tool_shaped_messages: bool = False
    """When True, ``turn_delta`` emits marked tool-result user segments in
    the OpenAI tool-call wire shape (see weka_tool_shape)."""
    _segments: list[RoleSegment] = field(default_factory=list)
    _emitted_segment_count: int = 0
    _last_disturbance_at: int | None = None
    _turn_index: int = 0
    _trailing_non_user_turns: list[int] = field(default_factory=list)
    """Turn indices whose segment list could NOT be made to end with a user
    block. The wire invariant (every turn ends with a user message) is
    enforced for every turn that contributes any new content, but a turn that
    appends zero new tokens (a fully block-aligned pull-back/compaction:
    ``new_blocks_count == 0`` with no partial/missing tail) has nothing to
    relabel as user, and a turn-0 prompt entirely consumed by the tool/system
    prefix has only the cached system segment. Those degenerate shapes leave a
    trailing non-user segment; they are recorded here (and warned once) rather
    than faked, since synthesizing tokens would break the byte-exact
    ``sum(seg.tokens) == in_tokens`` invariant."""

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
        self._assert_trailing_user()

    def advance_turn(
        self,
        prev_hash_ids: list[int],
        prev_in_tokens: int,
        prev_out_tokens: int,
        curr_hash_ids: list[int],
        curr_in_tokens: int,
        seed: str,
        is_tool_result: bool = False,
        *,
        max_asst_blocks: int | None = None,
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

        Wire invariant: the segment list always ends with a user segment. A
        chat request cannot ask the model to continue from a trailing
        assistant message, so when the assistant target would consume the
        entire new region and no partial/missing tail remains to seed a
        trailing user segment, the final new block is handed back to the user
        (relabeled, not resized — byte-exactness holds). The only turns that
        cannot satisfy this are recorded on ``_trailing_non_user_turns`` (see
        :meth:`_assert_trailing_user`): a turn that appends zero new tokens has
        nothing to relabel.

        ``prev_in_tokens`` is retained for the recorded-request contract
        (mirroring ``prev_hash_ids`` / ``prev_out_tokens``) but no longer
        feeds the truncation: the buffer self-describes its trailing
        overhang, which is also correct when the boundary segment is not the
        trailing one.

        ``max_asst_blocks`` is the optional two-pass planning cap from
        :func:`compute_asst_block_caps`: an upper bound on this turn's assistant
        block count chosen so a future block-aligned pull-back truncates onto a
        user block instead of an assistant one. ``None`` means no cap. It only
        ever lowers ``asst_blocks`` (an extra ``min``), so it composes with the
        context-loss rule and the trailing-user reservation and never breaks the
        byte-exact ``sum(seg.tokens) == curr_in_tokens`` invariant.
        """
        bs = self.block_size
        # Single source of truth for this turn's block accounting (covered
        # blocks, LCP, new-region size, synth tail), shared with the Pass-1
        # planner ``compute_asst_block_caps`` so the two never drift.
        geo = compute_turn_block_geometry(
            prev_hash_ids, curr_hash_ids, curr_in_tokens, bs
        )
        lcp = geo.lcp
        m_curr_covered = geo.m_curr_covered
        synth_tail_n = geo.synth_tail_n
        new_blocks_count = geo.new_blocks_count

        truncate_disturbance = truncate_synth_buf_at_block(
            self._segments,
            lcp,
            bs,
            decode_tokens_to_text=self.decode_tokens_to_text,
        )
        self._last_disturbance_at = truncate_disturbance

        new_blocks = curr_hash_ids[lcp:m_curr_covered]
        new_region_tokens = self.decode_block_tokens(new_blocks)
        if synth_tail_n > 0:
            new_region_tokens.extend(
                self.sample_partial_tail_tokens(synth_tail_n, seed)
            )

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
        if max_asst_blocks is not None:
            # Two-pass planning cap (see compute_asst_block_caps): a future
            # block-aligned pull-back will truncate exactly to a boundary inside
            # this turn's region. Shrinking the assistant here so that boundary
            # block lands in THIS turn's user segment makes that future turn end
            # with a user block — decided up front and stable across every turn,
            # so the rendered prefix never diverges and KV-cache reuse holds.
            asst_blocks = min(asst_blocks, max_asst_blocks)
        if synth_tail_n == 0 and asst_blocks == new_blocks_count and asst_blocks > 0:
            # Wire invariant: every turn must end with a user message — a chat
            # request cannot ask the model to continue from a trailing
            # assistant message. When the assistant would consume the entire
            # new region and no partial/missing tail remains to seed a trailing
            # user segment, hand the final new block back to the user. This
            # relabels block content (role only), so the byte-exact
            # sum(seg.tokens) == curr_in_tokens invariant is preserved. When
            # the region is a single block the assistant segment vanishes and
            # the turn becomes user-only.
            asst_blocks -= 1
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

        self._assert_trailing_user()

    def _assert_trailing_user(self) -> None:
        """Record (and warn once) when the segment list does not end with a user.

        The wire invariant — every turn ends with a user message — is enforced
        for any turn that contributes new content. The only shapes that cannot
        satisfy it without faking tokens are recorded on
        ``_trailing_non_user_turns``: a turn appending zero new tokens (a fully
        block-aligned pull-back with ``new_blocks_count == 0`` and no tail) has
        nothing to relabel as user, and a turn-0 prompt entirely consumed by
        the cached tool/system prefix has only a system segment. Synthesizing a
        user block in either case would violate the byte-exact
        ``sum(seg.tokens) == in_tokens`` invariant, so the caveat is surfaced
        rather than hidden.
        """
        if not self._segments or self._segments[-1].role == "user":
            return
        self._trailing_non_user_turns.append(self._turn_index)
        trailing_role = self._segments[-1].role
        _logger.warning(
            f"weka reconstructor: turn {self._turn_index} ends with a "
            f"'{trailing_role}' segment, not 'user'. The turn added no new "
            f"content to relabel as a trailing user block (block-aligned "
            f"pull-back or system-only turn-0); synthesizing one would break "
            f"the byte-exact ISL invariant, so it is left as-is."
        )

    def turn_delta(self) -> TurnDelta:
        """Compute the raw_messages to emit for the just-completed turn.

        Four cases:
          1. First call after ``init_turn_0`` (``_emitted_segment_count == 0``):
             emit ALL current segments, ``reset_context=False``. This is
             turn 0's baseline state.
          2. Strict append (no disturbance, or disturbance only touched
             segments at index ``>= _emitted_segment_count``): emit segments
             at index ``>= _emitted_segment_count``, ``reset_context=False``.
          3. Disturbance touched a previously-emitted segment (index
             ``< _emitted_segment_count``): emit ALL current segments,
             ``reset_context=True``.
          4. An equal-context retry appended no segments: re-emit ALL current
             segments with ``reset_context=True``. An empty delta would
             otherwise render as an invalid empty user message instead of the
             recorded repeated request.

        Updates ``_emitted_segment_count`` to ``len(self._segments)`` on
        return. Clears ``_last_disturbance_at`` to ``None``.
        """
        disturbed_emitted = (
            self._last_disturbance_at is not None
            and self._last_disturbance_at < self._emitted_segment_count
        )
        unchanged_retry = (
            self._emitted_segment_count > 0
            and self._emitted_segment_count == len(self._segments)
            and not disturbed_emitted
        )
        if self._emitted_segment_count == 0 or disturbed_emitted or unchanged_retry:
            source = self._segments
            reset = self._emitted_segment_count != 0 and (
                disturbed_emitted or unchanged_retry
            )
        else:
            source = self._segments[self._emitted_segment_count :]
            reset = False

        messages = [{"role": s.role, "content": s.content} for s in source]
        if self.tool_shaped_messages:
            # A mark that cannot pair in THIS window ships plain and must
            # stay plain on every later re-emission — demote it now so reset
            # full re-emits cannot retroactively shape already-sent context.
            demote_unpaired_tool_marks(source)
            messages = tool_shape_segment_messages(messages, source)

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


@dataclass(frozen=True)
class TurnBlockGeometry:
    """Role-independent block accounting for one turn transition.

    Single source of truth shared by
    :meth:`ConversationReconstructor.advance_turn` (which truncates to ``lcp``
    then appends the new region) and :func:`compute_asst_block_caps` (which
    replays the same tile to plan assistant caps). Depends only on hash ids and
    token counts — never on role labeling — so the two callers cannot drift.
    """

    lcp: int
    """Longest common prefix (in blocks) of prev/curr hash ids — the block the
    buffer truncates back to before appending this turn's new region."""
    m_curr_covered: int
    """Covered block count ``min(len(curr_hash_ids), curr_in_tokens // bs)``. A
    partial last hashed block (``len > in // bs``) contributes only its tail,
    not a full block."""
    new_blocks_count: int
    """Full blocks appended after truncating to ``lcp``: ``max(0, m_curr_covered - lcp)``."""
    synth_tail_n: int
    """Synthesized trailing tokens: the partial tail (``in % bs``) plus any
    missing-block region when the recording stored fewer hash blocks than
    ``in // bs``."""


def compute_turn_block_geometry(
    prev_hash_ids: list[int],
    curr_hash_ids: list[int],
    curr_in_tokens: int,
    block_size: int,
) -> TurnBlockGeometry:
    """Compute the role-independent block geometry for one turn transition.

    A hashed-but-partial last block (``len(curr_hash_ids) > curr_in_tokens //
    bs``, e.g. ``in=250`` with ``hash_ids=[..,4]`` at ``bs=64``) covers only its
    ``curr_in_tokens % bs`` partial tail, not a full ``bs``-token block —
    counting it as a full block AND appending the tail would double-count ~``bs``
    tokens and break the ``sum(seg.tokens) == curr_in_tokens`` invariant.
    """
    bs = block_size
    m_curr = len(curr_hash_ids)
    m_curr_full = curr_in_tokens // bs
    m_curr_covered = min(m_curr, m_curr_full)
    missing_block_tokens = max(0, (m_curr_full - m_curr) * bs)
    lcp = longest_common_prefix(prev_hash_ids, curr_hash_ids)
    return TurnBlockGeometry(
        lcp=lcp,
        m_curr_covered=m_curr_covered,
        new_blocks_count=max(0, m_curr_covered - lcp),
        synth_tail_n=curr_in_tokens % bs + missing_block_tokens,
    )


def compute_asst_block_caps(
    turns: list[tuple[list[int], int]],
    block_size: int,
) -> list[int | None]:
    """Pass-1 planner: per-turn upper bounds on assistant block count.

    ``turns`` is one ``(hash_ids, in_tokens)`` pair per turn, in order.

    Returns ``caps`` where ``caps[k]`` bounds the assistant blocks
    :meth:`ConversationReconstructor.advance_turn` may attribute at turn ``k``
    (``None`` = no constraint, including turn 0). The bounds make every future
    block-aligned pull-back truncate onto a user block rather than an assistant
    one, so each such turn ends with a user message.

    Pure and role-independent: LCP, the per-turn covered-block region, and the
    block tile depend only on ``hash_ids`` / ``in_tokens``, never on how blocks
    are later labeled. So the caps can be computed once up front and applied at
    the turn that *creates* each block — the boundary is fixed on first emission
    and never relabeled, preserving cross-turn KV-cache reuse.

    Mirrors ``advance_turn`` + ``truncate_synth_buf_at_block``: replays the
    block tile (owning turn per block) via ``compute_turn_block_geometry``,
    clamping ``eff_lcp = min(lcp, len(tile))`` since truncate can't grow the
    buffer. A turn that appends nothing (``new_blocks_count == 0`` and
    ``synth_tail_n == 0``) truncates onto block ``T-1`` (``T = eff_lcp``); its
    owning turn ``j = tile[T-1]`` must keep that block in its user segment, so
    ``cap_j = min((T-1) - eff_lcp_j)``. Turn-0 owners are skipped (no assistant).
    """
    n_turns = len(turns)
    caps: list[int | None] = [None] * n_turns
    if n_turns == 0:
        return caps

    bs = block_size
    tile: list[int] = []
    eff_lcp_per_turn: list[int] = [0] * n_turns

    for k in range(n_turns):
        hash_ids, in_tokens = turns[k]
        prev_hash_ids = turns[k - 1][0] if k else []
        # Same geometry function advance_turn uses, so the two never drift.
        geo = compute_turn_block_geometry(prev_hash_ids, hash_ids, in_tokens, bs)
        if k == 0:
            eff_lcp_per_turn[0] = 0
            tile = [0] * geo.m_curr_covered
            continue

        # eff_lcp clamps the raw LCP to the prior buffer's real block count:
        # truncate is a no-op past the buffer end and cannot grow the tile.
        eff_lcp = min(geo.lcp, len(tile))
        eff_lcp_per_turn[k] = eff_lcp
        new_blocks_count = max(0, geo.m_curr_covered - eff_lcp)
        synth_tail_n = geo.synth_tail_n

        if new_blocks_count == 0 and synth_tail_n == 0:
            # Degenerate pull-back: truncate to T=eff_lcp exposes block T-1.
            # target == eff_lcp <= len(tile) by construction, so the only guard
            # needed is target >= 1 (a turn-0-boundary pull-back has nothing to
            # cap).
            target = eff_lcp
            if target >= 1:
                owner = tile[target - 1]
                if owner != 0:
                    bound = (target - 1) - eff_lcp_per_turn[owner]
                    if bound >= 0:
                        prev = caps[owner]
                        caps[owner] = bound if prev is None else min(prev, bound)

        # Mutate the tile suffix in place (O(new)) instead of rebuilding it.
        del tile[eff_lcp:]
        tile.extend([k] * new_blocks_count)

    return caps


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
