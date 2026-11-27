# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LCP-driven conversation reconstructor for byte-exact weka trace replay.

The module name ``synth_buf`` is short for "synthesis buffer" — the
multi-segment in-progress chat-message tile this module maintains across
turns. The canonical reconstructor lives in :class:`ConversationReconstructor`.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Collection
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

_Role = Literal["system", "user", "assistant"]


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

    Segments tile the request's token positions ``[0, in_tokens)`` in order.
    Global block ``b`` holds positions ``[b * bs, (b + 1) * bs)``, and a
    segment may start or end inside a block.

    ``tokens`` is the canonical size source — it holds the exact Qwen token IDs
    for this segment. ``content`` is the decoded text and is always equal to
    ``decode_tokens_to_text(tokens)`` at the time the segment was emitted.

    Hash-content invariant: a position covered by a recorded hash id holds
    that hash's decoded token at the same offset, in every segment of every
    turn. Only positions with no recorded hash hold sampled filler.
    """

    role: _Role
    start: int
    """Token position of the first token."""
    tokens: list[int]
    content: str
    tool_result_turn: int | None = None
    """Set on a user segment whose content is the tool output answering the
    immediately-preceding assistant segment's tool call. Holds the turn index
    the segment was created on, keying the deterministic synthetic call id so
    re-emissions reproduce the id the turn was first sent with."""

    @property
    def end(self) -> int:
        """Token position past the last token."""
        return self.start + len(self.tokens)

    @property
    def content_token_count(self) -> int:
        """Token count == ``len(tokens)``. Kept as a property for back-compat."""
        return len(self.tokens)


@dataclass(frozen=True, slots=True)
class LayoutSegment:
    role: _Role
    start: int
    """Token position of the message's first token."""
    tool_result_turn: int | None = None


@dataclass(frozen=True, slots=True)
class ParentPrefix:
    """One parent turn's message layout, for a child's turn 0.

    A segment ends where the next one starts, which may be inside a block.
    """

    hash_ids: tuple[int, ...]
    in_tokens: int
    """The parent request's token count."""
    segments: tuple[LayoutSegment, ...]
    next_tool_turn: int
    """First tool-call id number past the parent's, so child ids never collide."""

    def spans(self) -> list[tuple[LayoutSegment, int]]:
        """Each segment with the token position where it ends (the next
        segment's ``start``, or ``in_tokens``)."""
        if not self.segments:
            return []
        ends = [seg.start for seg in self.segments[1:]] + [self.in_tokens]
        return list(zip(self.segments, ends, strict=True))


@dataclass(slots=True)
class ParentPrefixes:
    """Parent turn layouts that child turn 0s need, keyed by ``(session_id, turn)``.

    Shared by the serial and parallel reconstruction paths so both capture the
    same layouts at the same point.
    """

    wanted: set[tuple[str, int] | None]
    """Turns some child forks from; only these are captured."""
    layouts: dict[tuple[str, int], ParentPrefix] = field(default_factory=dict)

    def capture(
        self,
        session_id: str,
        turn: int,
        recon: ConversationReconstructor,
        hash_ids: list[int],
    ) -> None:
        if (session_id, turn) in self.wanted:
            self.layouts[(session_id, turn)] = recon.parent_prefix(hash_ids)


@dataclass
class ConversationReconstructor:
    """Walks a conversation's turns, maintaining synth_buf segments.

    Caller invariants:
    - ``decode_block_tokens(hash_ids)`` returns the deterministic Qwen token
      sequence for the given blocks (exactly ``len(hash_ids) * block_size``
      tokens).
    - ``sample_partial_tail_tokens(n, seed)`` returns deterministic Qwen
      token IDs that total exactly ``n``. ``seed`` must be position-keyed
      (e.g. sha256((conv_id, turn_index, "partial_tail"))). It fills only
      positions with no recorded hash.
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
    _tool_turn_offset: int = 0
    """Added to the turn index in synthetic tool-call ids (non-zero only for a
    child that inherited its parent's ids)."""
    _trailing_non_user_turns: list[int] = field(default_factory=list)
    """Turn indices whose segment list could NOT be made to end with a user
    block. The wire invariant (every turn ends with a user message) is
    enforced for every turn that contributes any new content, but a turn that
    appends zero new tokens (a pull-back/compaction that cuts to exactly
    ``in_tokens``) has nothing to relabel as user, and a turn-0 prompt
    entirely consumed by the tool/system prefix has only the cached system
    segment. Those degenerate shapes leave a trailing non-user segment; they
    are recorded here (and warned once) rather than faked, since
    synthesizing tokens would break the byte-exact
    ``sum(seg.tokens) == in_tokens`` invariant."""

    def init_turn_0(
        self,
        hash_ids: list[int],
        in_tokens: int,
        tool_tokens: int,
        system_tokens: int,
        seed: str,
        is_tool_result: bool = False,
        parent_prefix: ParentPrefix | None = None,
        turn_0_pullbacks: Collection[int] = (),
    ) -> None:
        """Initialize segments for turn 0 from a tool+system / user prefix split.

        With ``parent_prefix``, see :meth:`_init_turn_0_from_parent`;
        ``turn_0_pullbacks`` comes from :func:`compute_asst_block_caps`.

        See spec §4.3. hash_ids tile the request's blocks when fully recorded;
        a hashed partial last block holds the first ``in_tokens % bs`` tokens
        of its own decoding (see :meth:`_request_tokens`).

        When ``hash_ids`` is **truncated** relative to
        ``floor(in_tokens / bs)`` (a common shape for real captures where the
        recorder only stored a prefix of the hash blocks), the positions with
        no hash get sampled filler on the trailing user segment. The resulting
        prompt has exactly ``in_tokens`` tokens but a smaller hash-derived
        prefix than the recording had — KV-cache fidelity for the covered
        prefix is preserved; the uncovered suffix carries sha256-keyed synth
        tokens whose hashes don't match any recorded block. This matches the
        relaxed model already used by :func:`compose_weka_prompt_tokens`.

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
        receives the rest of the request. This guarantees
        ``sum(len(seg.tokens)) == in_tokens`` exactly and that every segment
        decodes the cached hash content byte-identically.
        """
        tokens = self._request_tokens(hash_ids, in_tokens, start=0, seed=seed)
        if parent_prefix is not None and self._init_turn_0_from_parent(
            parent_prefix,
            hash_ids=hash_ids,
            in_tokens=in_tokens,
            tokens=tokens,
            is_tool_result=is_tool_result,
            turn_0_pullbacks=turn_0_pullbacks,
        ):
            return
        bs = self.block_size

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
                # Clamp to the prompt's own full hashed blocks. When the
                # declared prefix exceeds them -- tool+system tokens exceed
                # in_tokens, or ceil() rounding overshoots floor(in_tokens/bs)
                # -- the over-budget prefix tokens go to the user segment, so
                # sum(seg.tokens) stays exactly in_tokens.
                prefix_blocks = min(
                    prefix_blocks, self._full_blocks(hash_ids, in_tokens)
                )
            if prefix_blocks > 0:
                cursor = prefix_blocks * bs
                segs.append(self._segment("system", 0, tokens[:cursor]))

        # When the system prefix covers the entire prompt, appending a
        # zero-token user segment would be deleted by the next turn's cut and
        # flag a spurious disturbance (reset_context) on a pure-growth turn —
        # skip it unless it is the only segment.
        if len(tokens) > cursor or not segs:
            segs.append(self._segment("user", cursor, tokens[cursor:]))

        self._turn_index = 0
        self._tool_turn_offset = 0
        del is_tool_result  # turn 0 has no preceding assistant to pair with
        self._segments = segs
        self._emitted_segment_count = 0
        self._last_disturbance_at = None
        self._assert_trailing_user()

    def _init_turn_0_from_parent(
        self,
        parent: ParentPrefix,
        *,
        hash_ids: list[int],
        in_tokens: int,
        tokens: list[int],
        is_tool_result: bool,
        turn_0_pullbacks: Collection[int],
    ) -> bool:
        """Split turn 0 into the parent's messages over the shared prefix.

        Only roles and message starts come from the parent; the rest is one
        user message, or the tool result for an inherited assistant's call.
        ``tokens`` are this request's own, as in the flat split.

        Returns False and changes nothing when:
        - the shared prefix ends inside the parent's first non-system message
          (the flat split already matches the parent there);
        - the child adds no tokens after a non-user parent message; or
        - a later turn would cut back to the end of an inherited non-user
          message and add nothing (``turn_0_pullbacks``).
        A request cannot end with an assistant message.
        """
        # The shared prefix is the cut a turn from the parent request to this
        # one would make, so it may end inside a block.
        shared = turn_cut(
            list(parent.hash_ids),
            parent.in_tokens,
            hash_ids,
            in_tokens,
            block_size=self.block_size,
        )
        tool_turn = None
        if self.tool_shaped_messages:
            # Keep the assistant only if the child adds a tail and no later
            # turn cuts back to its end.
            answers_call = (
                is_tool_result and in_tokens > shared and shared not in turn_0_pullbacks
            )
            shared, tool_turn = _shared_without_split_tool_pair(
                parent, shared, answers_call
            )
        if shared <= _first_message_end(parent):
            return False
        if any(
            0 < target <= shared and _role_at(parent, target - 1) != "user"
            for target in turn_0_pullbacks
        ):
            return False

        segs = self._inherited_segments(parent, tokens, shared)
        if in_tokens == shared and segs[-1].role != "user":
            return False

        if in_tokens > shared:
            segs.append(self._segment("user", shared, tokens[shared:], tool_turn))
        self._segments = segs
        self._turn_index = 0
        self._tool_turn_offset = parent.next_tool_turn
        self._emitted_segment_count = 0
        self._last_disturbance_at = None
        return True

    def _inherited_segments(
        self, parent: ParentPrefix, tokens: list[int], shared: int
    ) -> list[RoleSegment]:
        """Segments with the parent's roles that tile token positions
        ``[0, shared)``, holding this request's ``tokens``."""
        segs: list[RoleSegment] = []
        cursor = 0
        for seg, end in parent.spans():
            if cursor >= shared:
                break
            stop = min(end, shared)
            if stop == cursor:
                continue
            segs.append(
                self._segment(
                    seg.role, cursor, tokens[cursor:stop], seg.tool_result_turn
                )
            )
            cursor = stop
        return segs

    def _request_tokens(
        self, hash_ids: list[int], in_tokens: int, *, start: int, seed: str
    ) -> list[int]:
        """The request's tokens from position ``start`` to ``in_tokens``.

        Each hashed block holds its own decoding; a hashed partial last block
        holds the first ``in_tokens % bs`` tokens of it. Positions with no
        recorded hash get sampled filler. Same layout as
        :func:`compose_weka_prompt_tokens`.
        """
        bs = self.block_size
        first = start // bs
        hashed_end = min(in_tokens, len(hash_ids) * bs)
        tokens: list[int] = []
        if start < hashed_end:
            last = math.ceil(hashed_end / bs)
            tokens = self.decode_block_tokens(hash_ids[first:last])
            tokens = tokens[start - first * bs : hashed_end - first * bs]
        filler = in_tokens - max(start, hashed_end)
        if filler > 0:
            tokens.extend(self.sample_partial_tail_tokens(filler, seed))
        return tokens

    def _full_blocks(self, hash_ids: list[int], in_tokens: int) -> int:
        """Blocks of the request that are full and have a recorded hash."""
        return min(len(hash_ids), in_tokens // self.block_size)

    def _segment(
        self,
        role: _Role,
        start: int,
        tokens: list[int],
        tool_result_turn: int | None = None,
    ) -> RoleSegment:
        return RoleSegment(
            role=role,
            start=start,
            tokens=tokens,
            content=self.decode_tokens_to_text(tokens),
            tool_result_turn=tool_result_turn,
        )

    def parent_prefix(self, hash_ids: list[int]) -> ParentPrefix:
        """Capture the current turn's layout for a child's turn 0.

        ``hash_ids`` are the current turn's hash ids. Call this after
        :meth:`turn_delta`, so tool-result marks are in the state they were
        sent with.
        """
        return ParentPrefix(
            hash_ids=tuple(hash_ids),
            in_tokens=self._segments[-1].end if self._segments else 0,
            segments=tuple(
                LayoutSegment(seg.role, seg.start, seg.tool_result_turn)
                for seg in self._segments
            ),
            next_tool_turn=self._turn_index + self._tool_turn_offset + 1,
        )

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

        Implements spec §4.4: cut the buffer at the shared prefix, build the
        new region, attribute about ``ceil(prev_out / bs)`` blocks to an
        assistant segment and the rest to a user segment. The same rule
        applies across all three structural patterns (append-only, mid-seq
        replace, pull-back); see §4.4.1.

        The cut is at token position
        ``min(lcp * bs, prev_in_tokens, curr_in_tokens)`` (see :func:`turn_cut`),
        so it may fall inside a block. When turn k-1 ended in a hashed partial
        block that turn k shares, nothing is cut: turn k-1's messages stay and
        the new region continues that block from its next token. The request
        is turn k-1's tokens plus the new ones, ``curr_in_tokens`` in total.

        When ``curr_hash_ids`` is **truncated** relative to the request
        (common in real captures where the recorder stored only a prefix of
        the hash blocks), the positions with no hash get sampled filler on the
        trailing user segment.

        The assistant starts at the cut and ends on the block boundary at or
        after ``cut + prev_out_tokens``, clamped to the full hashed blocks of
        the request. ``max_asst_blocks`` and the cap from
        :func:`compute_asst_block_caps` count the blocks the assistant touches,
        from block ``cut // bs``.

        Wire invariant: the segment list always ends with a user segment. A
        chat request cannot ask the model to continue from a trailing
        assistant message, so when the assistant would consume the entire new
        region, its last block is handed back to the user (relabeled, not
        resized — byte-exactness holds). The only turns that cannot satisfy
        this are recorded on ``_trailing_non_user_turns`` (see
        :meth:`_assert_trailing_user`): a turn that appends zero new tokens has
        nothing to relabel.

        ``max_asst_blocks`` is the optional two-pass planning cap from
        :func:`compute_asst_block_caps`: an upper bound on this turn's assistant
        block count chosen so a future pull-back cuts onto a user token instead
        of an assistant one. ``None`` means no cap. It only ever lowers the
        assistant block count, so it composes with the context-loss rule and
        the trailing-user reservation and never breaks the byte-exact
        ``sum(seg.tokens) == curr_in_tokens`` invariant.
        """
        bs = self.block_size
        # Same cut the Pass-1 planner compute_asst_block_caps uses, so the two
        # never drift.
        cut = turn_cut(
            prev_hash_ids, prev_in_tokens, curr_hash_ids, curr_in_tokens, block_size=bs
        )
        self._last_disturbance_at = truncate_segments_at(
            self._segments, cut, decode_tokens_to_text=self.decode_tokens_to_text
        )
        region = self._request_tokens(
            curr_hash_ids, curr_in_tokens, start=cut, seed=seed
        )

        self._turn_index += 1
        first = cut // bs
        asst_blocks = (
            math.ceil((cut + prev_out_tokens) / bs) - first
            if prev_out_tokens > 0
            else 0
        )
        if not any(seg.role == "user" for seg in self._segments):
            # Context-loss rule: the truncation removed every user segment
            # (or turn 0 was system-only), so the conversation resumes at a
            # USER turn — the wire cannot present assistant output before
            # any user input. The whole new region becomes user content.
            asst_blocks = 0
        # Only full hashed blocks; filler and the partial block go to the user.
        full_blocks = self._full_blocks(curr_hash_ids, curr_in_tokens)
        asst_blocks = min(asst_blocks, max(0, full_blocks - first))
        if max_asst_blocks is not None:
            # Two-pass planning cap (see compute_asst_block_caps): a future
            # pull-back will cut exactly to a position inside this turn's
            # region. Shrinking the assistant here so that position lands in
            # THIS turn's user segment makes that future turn end with a user
            # message — decided up front and stable across every turn, so the
            # rendered prefix never diverges and KV-cache reuse holds.
            asst_blocks = min(asst_blocks, max_asst_blocks)
        if asst_blocks > 0 and (first + asst_blocks) * bs >= curr_in_tokens:
            # Wire invariant: every turn must end with a user message. When
            # the assistant would consume the entire new region, hand its last
            # block back to the user (role only, so sum(seg.tokens) ==
            # curr_in_tokens still holds). When that block is the whole
            # region the turn becomes user-only.
            asst_blocks -= 1
        asst_n = max(0, (first + asst_blocks) * bs - cut)

        if asst_n > 0:
            self._segments.append(self._segment("assistant", cut, region[:asst_n]))
        if len(region) > asst_n:
            # The mark is unconditional here; turn_delta demotes it at first
            # emission if no assistant directly precedes it in the emitted
            # window (post-context-loss heads, missing assistant, assistant
            # emitted in an earlier delta), so the shape a turn is first sent
            # with is the shape every reset re-emission reproduces.
            tool_turn = (
                self._turn_index + self._tool_turn_offset if is_tool_result else None
            )
            self._segments.append(
                self._segment("user", cut + asst_n, region[asst_n:], tool_turn)
            )

        self._assert_trailing_user()

    def _assert_trailing_user(self) -> None:
        """Record (and warn once) when the segment list does not end with a user.

        The wire invariant — every turn ends with a user message — is enforced
        for any turn that contributes new content. The only shapes that cannot
        satisfy it without faking tokens are recorded on
        ``_trailing_non_user_turns``: a turn appending zero new tokens (a
        pull-back that cuts to exactly ``curr_in_tokens``) has nothing to
        relabel as user, and a turn-0 prompt entirely consumed by
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
            f"content to relabel as a trailing user block (zero-token "
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


def _first_message_end(parent: ParentPrefix) -> int:
    for seg, end in parent.spans():
        if seg.role != "system":
            return end
    return parent.in_tokens


def _shared_without_split_tool_pair(
    parent: ParentPrefix, shared: int, answers_call: bool
) -> tuple[int, int | None]:
    """Move ``shared`` back to the start of an assistant segment it cuts or
    ends at, when the next segment is that assistant's tool result.

    The parent sent that assistant with ``tool_calls`` because the tool result
    followed it. Without the tool result, the child would send the assistant
    without ``tool_calls``, a message the parent never sent. Its tokens go to
    the child's new user message instead.

    When ``answers_call`` and ``shared`` ends exactly at the assistant, the
    child's tail is the tool result instead: ``shared`` stays and the parent's
    call turn is returned, so the assistant keeps the parent's call id.
    """
    spans = parent.spans()
    for (seg, end), (nxt, _) in zip(spans, spans[1:], strict=False):
        if seg.start < shared <= end:
            if seg.role == "assistant" and nxt.tool_result_turn is not None:
                if answers_call and shared == end:
                    return shared, nxt.tool_result_turn
                return seg.start, None
            break
    return shared, None


def _role_at(parent: ParentPrefix, position: int) -> _Role:
    """Defaults to user when no parent segment covers token ``position``."""
    for seg, end in parent.spans():
        if seg.start <= position < end:
            return seg.role
    return "user"


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


def turn_cut(
    prev_hash_ids: list[int],
    prev_in_tokens: int,
    curr_hash_ids: list[int],
    curr_in_tokens: int,
    *,
    block_size: int,
) -> int:
    """Token position that one turn transition keeps the buffer up to.

    Single source of truth shared by
    :meth:`ConversationReconstructor.advance_turn` (which cuts here, then
    appends the new region) and :func:`compute_asst_block_caps` (which replays
    the same cuts to plan assistant caps). It depends only on hash ids and
    token counts, never on role labels, so the two callers cannot drift.

    The cut is ``min(lcp * bs, prev_in, curr_in)``. Every kept position holds
    the same hash content in both requests. A shared partial last block
    (``in % bs`` tokens of a hash that the next request also uses) is kept,
    and the new region continues it.
    """
    lcp = longest_common_prefix(prev_hash_ids, curr_hash_ids)
    return min(lcp * block_size, prev_in_tokens, curr_in_tokens)


def compute_asst_block_caps(
    turns: list[tuple[list[int], int]],
    block_size: int,
    turn_0_pullbacks: set[int] | None = None,
) -> list[int | None]:
    """Pass-1 planner: per-turn upper bounds on assistant block count.

    ``turns`` is one ``(hash_ids, in_tokens)`` pair per turn, in order.

    Returns ``caps`` where ``caps[k]`` bounds the assistant blocks
    :meth:`ConversationReconstructor.advance_turn` may attribute at turn ``k``
    (``None`` = no constraint, including turn 0). The bounds make every future
    pull-back that adds nothing cut onto a user token rather than an assistant
    one, so each such turn ends with a user message.

    Pure and role-independent: the cuts depend only on ``hash_ids`` /
    ``in_tokens``, never on how tokens are later labeled. So the caps can be
    computed once up front and applied at the turn that *creates* each
    token — the boundary is fixed on first emission and never relabeled,
    preserving cross-turn KV-cache reuse.

    Mirrors ``advance_turn`` + ``truncate_segments_at``: replays which turn
    owns each token range, using :func:`turn_cut`. A turn that adds
    nothing (``cut == in_tokens``) ends at token ``C - 1`` (``C = cut``); its
    owning turn ``j`` must keep that token in its user segment, so its
    assistant, which starts at block ``cut_j // bs``, must end by block
    ``(C - 1) // bs``. Turn-0 owners are skipped (no assistant); their cuts
    ``C`` are added to ``turn_0_pullbacks`` when given (a forked turn 0 must
    not leave an inherited assistant there).
    """
    n_turns = len(turns)
    caps: list[int | None] = [None] * n_turns
    bs = block_size
    # Owner turn of each token range: range i is [starts[i], starts[i + 1]).
    starts: list[int] = []
    owners: list[int] = []
    cuts: list[int] = [0] * n_turns

    for k in range(n_turns):
        hash_ids, in_tokens = turns[k]
        # Same cut function advance_turn uses, so the two never drift.
        cut = turn_cut(*turns[k - 1], hash_ids, in_tokens, block_size=bs) if k else 0
        cuts[k] = cut
        if cut == in_tokens and cut >= 1:
            owner = owners[bisect.bisect_right(starts, cut - 1) - 1]
            if owner != 0:
                bound = (cut - 1) // bs - cuts[owner] // bs
                if bound >= 0:
                    prev = caps[owner]
                    caps[owner] = bound if prev is None else min(prev, bound)
            elif turn_0_pullbacks is not None:
                turn_0_pullbacks.add(cut)

        del_at = bisect.bisect_left(starts, cut)
        del starts[del_at:]
        del owners[del_at:]
        if in_tokens > cut:
            starts.append(cut)
            owners.append(k)

    return caps


def truncate_segments_at(
    segments: list[RoleSegment],
    cut: int,
    decode_tokens_to_text: Callable[[list[int]], str] | None = None,
) -> int | None:
    """Truncate ``segments`` in place so they hold token positions ``[0, cut)``.

    ``segments`` must tile positions from 0 with no gaps, as
    :class:`RoleSegment` requires.

    A cut inside a segment slices its tokens. When ``decode_tokens_to_text``
    is provided, ``content`` is re-derived from the surviving tokens to keep
    the (tokens, content) invariant.

    Returns the earliest segment index whose content was disturbed — sliced
    or removed — or ``None`` when no segment was touched (cut at or past the
    end, or empty buffer). :meth:`ConversationReconstructor.turn_delta`
    compares the index against the emitted count: disturbing a
    previously-emitted segment forces a context reset on the next emission.
    """
    if cut <= 0:
        had_segments = bool(segments)
        segments.clear()
        return 0 if had_segments else None
    for i, seg in enumerate(segments):
        if seg.end < cut:
            continue
        if seg.end > cut:
            seg.tokens = seg.tokens[: cut - seg.start]
            if decode_tokens_to_text is not None:
                seg.content = decode_tokens_to_text(seg.tokens)
            del segments[i + 1 :]
            return i
        deleted = i + 1 < len(segments)
        del segments[i + 1 :]
        return i + 1 if deleted else None
    return None
