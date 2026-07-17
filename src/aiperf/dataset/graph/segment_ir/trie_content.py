# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format-agnostic LCP-trie content lowering for the segment-trie IR.

The shared core behind :func:`~aiperf.dataset.graph.adapters.weka.trie_build.build_trie_graph`,
moved verbatim from ``adapters/weka/trie_build.py`` so any adapter that can
normalize its recorded requests into :class:`TrieRequest`/:class:`TrieNode`
reuses ONE pipeline (weka and dynamo both lower through it). Any behavior
change here breaks weka byte-exactness -- treat edits as frozen.

The pipeline (:func:`build_trie_ir`), in exactly the weka build order:

1. **Content parents** (:func:`resolve_content_parents`): a node's
   content-parent is the earlier node whose ``hash_ids`` is the longest full
   prefix (tie-break most recent), else the longest partial-LCP branch point
   (tie-break earliest), else a fresh root.
2. **Idle warp + ranks + interval edges**: the active-interval idle-gap warp
   (:func:`apply_idle_gap_warp`),
   the time-consistent global rank, and the finished-before interval-order
   edges (:mod:`~aiperf.dataset.graph.segment_ir.interval_order`).
3. **Frozen block tags** (:func:`compute_asst_caps` + :func:`assign_block_tags`):
   every covered block gets one ``(role, starts_new_message)`` tag, assigned by
   the node that first materializes it and then FROZEN, so a shared block
   prefix yields identical tags -> identical messages -> a real cache prefix.
4. **Per-node emission** (:func:`_assemble_messages_from` + covered-count ISL
   gate :func:`assert_covered_isl` + :func:`emit_response_segment`): group the
   tags into messages, emit one content-addressed pool entry per message chained
   root->tip, gate the reconstructed token count, and append the recorded
   assistant output as one trailing pool segment. Because the content-parent's
   tags are copied VERBATIM into the inherited prefix, every WHOLE message
   strictly inside a node's ``inherited`` block count re-derives the exact sid
   the parent already emitted; the driver therefore SPLICES the parent's sid
   chain for those messages (from its :class:`_EmissionRecord`) and emits fresh
   only the straddling fragment plus the new-region messages -- byte-identical
   output (a spliced ``pool.add`` would have been a dedup no-op) without the
   quadratic re-decode+re-hash of the shared prefix. :func:`assemble_messages`
   stays the whole-prefix (``start_block=0``) wrapper for callers/tests.
"""

from __future__ import annotations

import math
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import msgspec

from aiperf.dataset.graph.models import (
    START_NODE_ID,
    ChannelRequirement,
    ChannelSpec,
    GraphRecord,
    LlmNode,
    ProvenanceSpec,
    StaticEdge,
)
from aiperf.dataset.graph.segment_ir.interval_order import (
    apply_start_anchors,
    build_interval_edges,
    compute_ranks,
)
from aiperf.dataset.graph.segment_ir.pool import SegmentPool

# --- block geometry (merged verbatim from segment_ir/block_geometry.py) -----
# Format-agnostic block-aligned LCP geometry: operates on plain
# ``(hash_ids, token_count, block_size)`` values, originally moved verbatim from
# ``adapters/shared/content.py`` so weka and dynamo share ONE geometry source
# (any behavior change here breaks weka byte-exactness -- treat edits as frozen).


def longest_common_prefix(prev_hash_ids: list[int], curr_hash_ids: list[int]) -> int:
    """Return the index of the first differing element of the two sequences."""
    n = min(len(prev_hash_ids), len(curr_hash_ids))
    for i in range(n):
        if prev_hash_ids[i] != curr_hash_ids[i]:
            return i
    return n


@dataclass(slots=True)
class TurnBlockGeometry:
    """Block-aligned geometry of one turn relative to its content parent.

    All four fields are WHOLE-block counts/token-counts on the block-aligned
    grid; the ``curr_in_tokens % bs`` partial tail is deliberately excluded from
    every field (it is synthesized separately at message assembly, never frozen
    into a block role). ``synth_tail_n`` counts ONLY the tokens of missing WHOLE
    blocks (hash_ids truncated below ``curr_in_tokens // bs``), not the partial
    remainder.
    """

    lcp: int
    m_curr_covered: int
    new_blocks_count: int
    synth_tail_n: int


def compute_turn_block_geometry(
    prev_hash_ids: list[int],
    curr_hash_ids: list[int],
    curr_in_tokens: int,
    block_size: int,
) -> TurnBlockGeometry:
    """Compute the block-aligned geometry of ``curr`` relative to ``prev``.

    The SINGLE geometry source shared by :func:`block_role_split` and the
    trailing-user cap planner (``_compute_asst_caps``). ``m_curr_covered`` is the
    number of covered WHOLE blocks (``min(len(curr_hash_ids), in // bs)``);
    ``lcp`` is the longest common prefix of the two hash-id lists;
    ``new_blocks_count`` is the covered blocks past the LCP; ``synth_tail_n`` is
    the token count of missing WHOLE blocks only (never the ``in % bs`` partial
    tail).
    """
    bs = block_size
    m_full = curr_in_tokens // bs
    m_covered = min(len(curr_hash_ids), m_full)
    lcp = longest_common_prefix(prev_hash_ids, curr_hash_ids)
    return TurnBlockGeometry(
        lcp=lcp,
        m_curr_covered=m_covered,
        new_blocks_count=max(0, m_covered - lcp),
        synth_tail_n=max(0, (m_full - len(curr_hash_ids)) * bs),
    )


def block_role_split(
    *,
    prev_hash_ids: list[int],
    curr_hash_ids: list[int],
    curr_in_tokens: int,
    prev_out_tokens: int,
    block_size: int,
    max_asst_blocks: int | None,
    parent_has_user: bool,
    parent_covered_blocks: int | None = None,
) -> tuple[int, list[str]]:
    """Split a turn's new blocks into per-block ``assistant``/``user`` roles.

    Returns ``(inherited, roles)`` where ``inherited`` is the block count carried
    over from the content parent and ``roles`` is the creation-time role tag for
    each new block. ``inherited`` is ``min(lcp, len(prev_hash_ids),
    m_curr_covered)`` -- clamped to this node's OWN covered-block count so a child
    that shares a full block prefix but declares a smaller ``in`` (``in // bs <
    lcp``) never inherits more tags than it emits (which would trip the ISL gate
    on an otherwise-legitimate trace). When ``parent_covered_blocks`` is provided
    it is additionally clamped to the parent's actual covered-block count, so a
    parent that UNDER-COVERS (fewer covered blocks than the child's LCP) yields
    the covered-but-uninherited blocks as NEW blocks the child materializes and
    tags itself. Passing ``None`` preserves the original behavior for every
    existing caller.
    ``ceil(prev_out / bs)`` leading new blocks are attributed to the assistant
    (the previous turn's response), clamped to the available new-block width and
    to ``max_asst_blocks`` when set; when the parent has no user context (a
    context-loss branch) every new block becomes ``user``.

    Trailing-user is guaranteed at BLOCK CREATION (frozen): a node whose new
    region is all-assistant flips its OWN last new block to user. Because the
    tag is inherited verbatim by every inheritor, the flip is consistent across
    the whole subtree and cache-safe (turn boundaries align framing).
    """
    geo = compute_turn_block_geometry(
        prev_hash_ids, curr_hash_ids, curr_in_tokens, block_size
    )
    inherited = min(geo.lcp, len(prev_hash_ids), geo.m_curr_covered)
    if parent_covered_blocks is not None:
        inherited = min(inherited, parent_covered_blocks)
    new_n = max(0, geo.m_curr_covered - inherited)
    asst = math.ceil(prev_out_tokens / block_size) if prev_out_tokens > 0 else 0
    if not parent_has_user:
        asst = 0
    asst = min(asst, new_n)
    if max_asst_blocks is not None:
        asst = min(asst, max_asst_blocks)
    if asst == new_n and asst > 0:
        asst -= 1  # trailing-user (frozen at creation): a node whose new region is all-assistant
        # flips its OWN last new block to user; inherited verbatim, so consistent
        # across all inheritors and cache-safe (turn boundaries align framing).
    roles = ["assistant"] * asst + ["user"] * (new_n - asst)
    return inherited, roles


# --- idle-gap warp (merged verbatim from segment_ir/idle_warp.py) -----------
# Active-interval idle-gap warp: operates on plain ``(raw_start, raw_end)``
# intervals and the trie-node protocol, originally moved verbatim from
# ``adapters/weka/trie_build.py`` so weka and dynamo share ONE idle-warp source
# (any behavior change here breaks weka byte-exactness -- treat edits as frozen).


class ActiveIdleWarp:
    """Idle-gap warp over the UNION of request ACTIVE INTERVALS, not their starts.

    Plot every request in the trace as its active interval ``[raw_start,
    raw_end]`` (``raw_end = t + api_time``) on one line. A true IDLE gap is a
    stretch where NOTHING is running: between the latest end of everything
    started so far (the running max end) and the next request's start. Any idle
    gap longer than ``cap`` is collapsed to ``cap``; every later timestamp shifts
    left by the excess.

    This is the crucial difference from capping START-to-START gaps: an active
    stretch -- a single long request's processing, OR overlapping subagents --
    is NEVER cut, so every request keeps its EXACT temporal shape (durations and
    overlaps preserved) and only the dead air between requests is removed.
    Capping start-to-start gaps instead eats into a long request's own
    ``api_time`` (warping its end PAST the next request's start), which distorts
    the shape and manufactures false overlaps. Because no cut ever falls inside
    an active interval, ``warped_end == warped_start + api_time`` always holds and
    a request that genuinely finished before another (raw) still does so warped.
    """

    def __init__(self, intervals: list[tuple[float, float]], cap: float) -> None:
        # ``_cuts`` is an ascending list of ``(next_start, cumulative_excess)``:
        # every timestamp at/after ``next_start`` shifts left by that cumulative
        # excess. Built by a sweep over intervals sorted by start.
        self._cuts: list[tuple[float, float]] = []
        if not intervals:
            return
        ordered = sorted(intervals)
        running_end = ordered[0][1]
        cumulative = 0.0
        for start, end in ordered[1:]:
            if start > running_end:  # nothing active in (running_end, start)
                idle = start - running_end
                if idle > cap:
                    cumulative += idle - cap
                    self._cuts.append((start, cumulative))
            if end > running_end:
                running_end = end

    def map(self, t: float) -> float:
        shift = 0.0
        for next_start, cumulative in self._cuts:
            if t < next_start:
                break
            shift = cumulative
        return t - shift


def apply_idle_gap_warp(nodes: list, idle_gap_cap_seconds: float | None) -> None:
    """Stamp each node's ``warped_start`` from the active-interval idle warp.

    ``nodes`` must expose ``raw_start`` / ``raw_end`` / ``request.t`` and a
    writable ``warped_start`` (the trie-node protocol). ``None`` cap disables
    the warp (raw ``request.t`` passthrough).
    """
    if idle_gap_cap_seconds is None:
        for node in nodes:
            node.warped_start = node.request.t
        return
    warp = ActiveIdleWarp(
        [(n.raw_start, n.raw_end) for n in nodes], idle_gap_cap_seconds
    )
    for node in nodes:
        node.warped_start = warp.map(node.request.t)


# --- content-addressed message chaining (merged verbatim from segment_ir/message_addressing.py) ---
# The shared root->tip discipline both adapters use to turn a per-turn message
# sequence into a ``SegmentPool`` path: add one segment per message, threading
# each segment's ``parent_id`` onto the previous segment's content-addressed id
# (the first is a root, ``parent_id=None``). Adapters differ only in how they
# DERIVE each message's ``(role, content, tokens)``; the chaining loop -- and
# therefore the pool insertion order and resulting segment ids -- is identical.


# One message to content-address: (role, content, tokens).
MessageUnit = tuple[str, str, list[int]]


def add_message_chain(pool: SegmentPool, messages: Iterable[MessageUnit]) -> list[str]:
    """Add each message as one pool segment, threading ``parent_id`` root->tip.

    Returns the ordered per-message segment ids (the ``prompt_segment_ids`` path).
    The first message is a root (``parent_id=None``); each subsequent segment is
    parented on the prior segment's id.
    """
    ids: list[str] = []
    prev_id: str | None = None
    for role, content, tokens in messages:
        prev_id = pool.add(role=role, content=content, tokens=tokens, parent_id=prev_id)
        ids.append(prev_id)
    return ids


@dataclass(frozen=True)
class ReconCallbacks:
    """The three deterministic content callbacks the message-unit builder needs.

    Injectable so unit tests can drive the builder with collision-free stub
    decoders (no tokenizer / corpus build), while the production default wires a
    :class:`~aiperf.dataset.graph.adapters.shared.content.CorpusContentSynthesizer`'s
    byte-faithful callbacks (see
    :func:`~aiperf.dataset.graph.adapters.weka.trie_build._default_callbacks`).

    DETERMINISM CONTRACT (relied on by prefix-path reuse in :func:`build_trie_ir`):
    ``decode_block_tokens`` MUST be a pure per-build function of its hash-id
    arguments -- the SAME hash id decodes to the SAME tokens for every call
    within one :func:`build_trie_ir` invocation. Every production callback
    satisfies this by construction (each caches per ``(root seed, trace_id,
    hash_id, block_size)``; see ``_default_callbacks``). The reuse path decodes
    each shared block ONCE, at first materialization, and splices the resulting
    sid into every inheritor; a callback that drifted between the parent's and a
    child's emission (a contract violation) would no longer be re-caught at the
    child by the ISL gate. Successful-build store bytes are unaffected either
    way (a re-decoded duplicate segment was already discarded by pool dedup).
    """

    decode_block_tokens: Callable[[list[int]], list[int]]
    sample_partial_tail_tokens: Callable[[int, str], list[int]]
    decode_tokens_to_text: Callable[[list[int]], str]
    block_exact: bool = True
    """``decode_block_tokens`` returns EXACTLY ``block_size`` tokens per block.

    Test-only escape hatch; all production callbacks leave this True (the
    assembled-token ISL gate hard-aborts the build on drift). Set False ONLY by
    unit-test stub callbacks whose tiny block runs would otherwise trip the gate
    on every node."""


@dataclass(slots=True)
class TrieRequest:
    """Normalized per-request view of one recorded LLM call."""

    hash_ids: list[int]
    """Full-prompt block-hash list (recorded or virtual; opaque position keys)."""
    input_length: int
    """Recorded prompt token count."""
    output_length: int
    """Recorded completion token count."""
    t: float
    """Trace-relative request start, seconds."""
    api_time: float
    """Recorded request duration, seconds."""
    model: str | None = None
    """Recorded model name when present."""
    streaming: bool = False
    """Whether the recorded request streamed."""
    ttft: float | None = None
    """Recorded time-to-first-token, seconds; None for non-streaming requests."""


@dataclass
class TrieNode:
    """One recorded leaf request plus its derived structural context."""

    node_id: str
    request: TrieRequest
    # Index into the flattened recorded-order list (also the recorded ``t``
    # order tiebreak key).
    order: int
    # Enclosing async_launched subtree-root ids (transitive). Timing async-exclusion only.
    async_ancestors: frozenset[str] = field(default_factory=frozenset)
    # Global time-consistent rank (stamped by ``interval_order.compute_ranks``).
    rank: int = 0
    # Resolved content-parent (longest hash-id prefix / branch point), else
    # ``None`` for a fresh root. Filled in a second pass. CONTENT/PROMPT ONLY:
    # selects which segment-pool prefix this turn materializes; it is NOT a
    # timing/dependency cause (a branch point can be arbitrarily far back, which
    # would make the firing delay the cumulative warped distance -- the
    # aggregate-timestamp bug). Timing anchors on the interval-order finished-
    # before frontier (:func:`~aiperf.dataset.graph.segment_ir.interval_order.build_interval_edges`) instead.
    content_parent: TrieNode | None = field(default=None)
    # Recorded raw start ``request.t`` mapped onto the idle-gap-warped clock
    # (see :func:`apply_idle_gap_warp`).
    # Equals ``request.t`` when no cap is active. Stamped in a pass after
    # content-parent resolution, before edge building.
    warped_start: float = field(default=0.0)
    # Node id of this request's CAUSAL predecessor: the spawning request for a
    # subagent's first inner request, else the previous request in its own
    # chain; None for chain roots. Consumed by apply_start_anchors -- when the
    # causal parent is still IN FLIGHT at this node's recorded start, the
    # node's incoming edges are replaced with one start-anchored edge.
    causal_parent_id: str | None = field(default=None)
    # Dynamo-only recorded metadata attached during trie lowering; None for
    # other adapters.
    dynamo_meta: dict[str, Any] | None = None

    @property
    def start(self) -> float:
        """Node start on the idle-gap-warped clock (raw ``request.t`` when uncapped)."""
        return self.warped_start

    @property
    def raw_start(self) -> float:
        """Recorded start on the RAW clock (``request.t``); interval-order input."""
        return self.request.t

    @property
    def raw_end(self) -> float:
        """Recorded completion on the RAW clock (``request.t + api_time``).

        The who-finished-before-whom ground truth for the interval-order edge rule
        and the active-interval idle-gap warp -- both read raw timestamps.
        """
        return self.request.t + (self.request.api_time or 0.0)

    @property
    def end(self) -> float:
        """Completion time on the warped clock = warped start + raw ``api_time``.

        ``api_time`` is the request's own server-processing duration, not an
        inter-request idle gap, so it is NOT warped -- it is added raw to the
        warped start. End-to-start delays therefore subtract a RAW ``api_time``
        from a warped start-to-start gap.
        """
        return self.warped_start + (self.request.api_time or 0.0)


@dataclass(slots=True)
class TrieNodeBuild:
    """Per-node output of the shared trie build."""

    prompt_path: list[str]
    """Ordered per-message SegmentPool ids for the node's prompt."""
    response_id: str
    """Pool id of the node's synthesized assistant response segment."""
    small_prompt: bool = False
    """True when the covered-count was 0 and the tiny-prompt fallback fired."""


@dataclass(slots=True)
class TrieBuild:
    """Whole-graph output of the shared trie build."""

    builds: dict[str, TrieNodeBuild]
    """node_id -> per-node build artifacts."""
    edges_by_node: dict[str, list[StaticEdge]]
    """node_id -> incoming interval-order StaticEdges."""


@dataclass(slots=True)
class _EmissionRecord:
    """Per-node prompt-message bookkeeping for prefix-path reuse.

    Local to ONE :func:`build_trie_ir` call (one trace per worker task); dies
    with the build. ``array("q")`` int arrays keep the per-message state compact
    on deep, wide traces (~16 MB worst single trace vs ~70 MB with int lists).
    """

    msg_end_blocks: array
    """Exclusive end block index of each emitted prompt message (absolute)."""
    cum_token_counts: array
    """Cumulative ACTUAL decoded token count through each message (inclusive)."""
    prompt_path: list[str]
    """This node's ordered per-message pool sids (the same list object as its
    :attr:`TrieNodeBuild.prompt_path` -- referenced, never duplicated)."""


# --- content-parent resolution --------------------------------------------


def resolve_content_parents(nodes: list[TrieNode]) -> None:
    """Fill each node's content-parent from the hash-id prefix tree.

    For node R, the content-parent is the earlier node (lower ``order``) whose
    ``hash_ids`` is the longest FULL prefix of R's, tie-broken toward the most
    recent (highest ``order``). When no earlier node is a full prefix, it is the
    earlier node with the longest partial ``hash_ids`` LCP (the branch point).
    With no overlap at all (LCP 0 and no full prefix), R stays a fresh root.

    This is an O(sum of ``hash_ids`` lengths) incremental prefix-automaton pass
    that yields byte-for-byte the SAME selection as scanning all earlier nodes
    pairwise with a full-prefix / longest-common-prefix comparison (see the
    brute-force oracle in test_weka_trie_build_resolution.py), but without the
    O(n^2 * m) double loop. Each node is resolved against the automaton built from all
    strictly-earlier nodes, then inserted. Empty-``hash_ids`` nodes are never
    inserted (they can never be a full prefix and contribute LCP 0, so skipping
    them cannot change the selection).

    Representation: a flat int-state automaton, NOT a tree of node objects (that
    tree cost 312 B/position; this costs 175 B). State 0 is the root;
    ``transitions[(state, h)]`` is the state reached by consuming hash ``h`` from
    ``state``, created EXACTLY where the node-object trie created a child
    (``children.get(h) is None`` <-> ``transitions.get((state, h)) is None``), so
    the reachable state graph IS that trie. Parallel ``terminal``/``passer`` lists
    hold, per state, the MOST-RECENT full-prefix owner (overwrite-always: the
    full-prefix tie-break favors the most recent) and the EARLIEST pass-through
    owner (set-once-if-None: the partial-LCP tie-break favors the earliest). The
    walk visits the same depths with the same ``matched > best_full_len`` /
    ``matched > best_partial_lcp`` comparisons, so the ``content_parent``
    assignment is identical as a theorem about dict semantics. Tuple keys
    ``(int, int)`` are hashed/compared by value, so arbitrary ids -- negative
    virtual (dynamo) or >64-bit weka JSON -- key identically to the old nested
    ``dict[int, ...]`` (the combined single-int-key variant was REJECTED because
    weka ids need not fit 64 bits). Nothing iterates a dict anywhere (point
    get/set only), so no ordering sensitivity exists to disturb.
    """
    transitions: dict[tuple[int, int], int] = {}
    terminal: list[TrieNode | None] = [None]
    passer: list[TrieNode | None] = [None]
    # Hoist the transition probe to a local: the dict object is mutated in place
    # by ``_insert_flat`` (never rebound), so the bound method stays valid across
    # nodes and this drops one attribute lookup per walk step (pure speedup, no
    # semantic change).
    tget = transitions.get
    for r in nodes:
        r_hashes = r.request.hash_ids
        best_full: TrieNode | None = None
        best_full_len = -1
        best_partial: TrieNode | None = None
        best_partial_lcp = 0

        state = 0
        matched = 0  # hashes consumed so far == the trie depth reached
        for h in r_hashes:
            nxt = tget((state, h))
            if nxt is None:
                break
            state = nxt
            matched += 1
            # A node terminating here is a full prefix of R of length ``matched``.
            term = terminal[state]
            if term is not None and matched > best_full_len:
                best_full_len = matched
                best_full = term
            # Any node walking through here shares an LCP of ``matched`` with R.
            pas = passer[state]
            if pas is not None and matched > best_partial_lcp:
                best_partial_lcp = matched
                best_partial = pas

        if best_full is not None:
            r.content_parent = best_full
        elif best_partial is not None:
            r.content_parent = best_partial

        if r_hashes:
            # Hand off the state the walk already reached: ``r_hashes[:matched]``
            # follows existing transitions whose ``passer`` an earlier inserter
            # already stamped, so re-walking that prefix is a no-op -- resuming
            # the insert at ``(state, matched)`` is byte-identical and skips the
            # redundant probes on deep shared-prefix chains.
            _insert_flat(
                transitions,
                terminal,
                passer,
                r_hashes,
                r,
                start_state=state,
                start_index=matched,
            )


def _insert_flat(
    transitions: dict[tuple[int, int], int],
    terminal: list[TrieNode | None],
    passer: list[TrieNode | None],
    hashes: list[int],
    node: TrieNode,
    *,
    start_state: int = 0,
    start_index: int = 0,
) -> None:
    """Insert ``node``'s ``hash_ids`` into the flat automaton, stamping passer/terminal.

    Insertion is in ``order`` (ascending). A missing transition ``(state, h)``
    creates a fresh state id ``len(terminal)`` -- appending ``None`` to BOTH the
    ``terminal`` and ``passer`` lists (kept in lockstep, so ``len`` is the next
    id) -- exactly where the node-object trie created a child. ``passer`` is set
    only on its FIRST inserter at each state (the earliest/lowest-order node) --
    the partial-LCP tie-break favors the earliest node. ``terminal`` is
    overwritten on every inserter (the latest wins) -- the full-prefix tie-break
    favors the most recent.

    ``start_state`` / ``start_index`` resume the insert from a prefix
    :func:`resolve_content_parents`' walk already traversed. The contract (relied
    on ONLY by that caller): ``hashes[:start_index]`` leads from the root to
    ``start_state`` via EXISTING transitions, and every existing state was
    created by an earlier inserter that stamped its ``passer`` -- so the
    set-once-if-None checks on that prefix are all no-ops and skipping them is
    byte-identical. Called with the defaults it is a plain full insert from the
    root (its standalone/measurement use).
    """
    tget = transitions.get
    t_append = terminal.append
    p_append = passer.append
    state = start_state
    suffix = hashes if start_index == 0 else hashes[start_index:]
    for h in suffix:
        key = (state, h)
        nxt = tget(key)
        if nxt is None:
            nxt = len(terminal)
            transitions[key] = nxt
            t_append(None)
            p_append(None)
        state = nxt
        if passer[state] is None:
            passer[state] = node
    terminal[state] = node


# --- frozen block tags ------------------------------------------------------


def compute_asst_caps(nodes: list[TrieNode], block_size: int) -> dict[str, int | None]:
    """Pass-1 trailing-user planner over the GLOBAL ``content_parent`` tree.

    A degenerate pull-back (``new_blocks_count == 0`` and ``synth_tail_n == 0``)
    at ``eff_lcp >= 1`` re-exposes block ``eff_lcp - 1`` of the parent lineage; to
    keep the frozen boundary landing on a user block, the owner ancestor of that
    block is capped. Owners are tracked as a per-node "tile" list (block index ->
    owning ``node_id``), rebuilt off :func:`compute_turn_block_geometry` so the
    role split and this planner share the SINGLE geometry source. The inherited
    count uses the SAME three-way clamp as :func:`assign_block_tags`
    (``min(lcp, len(parent_tiles), m_curr_covered)``) so on an over-share row
    (``in // bs < lcp``) the re-exposed boundary block -- and therefore the
    capped owner -- matches the frozen tags.

    Skips capping a root owner (a node whose ``content_parent is None``),
    matching agentx's ``owner != 0`` guard. Requires :func:`resolve_content_parents`
    to have run so each node's ``content_parent`` is set.
    """
    is_root = {n.node_id: (n.content_parent is None) for n in nodes}
    caps: dict[str, int | None] = {}
    tiles: dict[str, list[str]] = {}
    eff: dict[str, int] = {}
    for node in nodes:
        parent = node.content_parent
        curr = node.request.hash_ids
        if parent is None:
            g = compute_turn_block_geometry(
                [], curr, node.request.input_length, block_size
            )
            tiles[node.node_id] = [node.node_id] * g.m_curr_covered
            eff[node.node_id] = 0
            continue
        pt = tiles.get(parent.node_id, [])
        g = compute_turn_block_geometry(
            parent.request.hash_ids, curr, node.request.input_length, block_size
        )
        e = min(g.lcp, len(pt), g.m_curr_covered)
        eff[node.node_id] = e
        new_n = max(0, g.m_curr_covered - e)
        if new_n == 0 and g.synth_tail_n == 0 and e >= 1:
            owner = pt[e - 1]
            if not is_root.get(owner, True):
                bound = (e - 1) - eff.get(owner, 0)
                if bound >= 0:
                    caps[owner] = (
                        bound if caps.get(owner) is None else min(caps[owner], bound)
                    )
        tiles[node.node_id] = pt[:e] + [node.node_id] * new_n
    return caps


def _assign_block_tags_and_inheritance(
    nodes: list[TrieNode],
    block_size: int,
    caps: dict[str, int | None],
) -> tuple[dict[str, list[tuple[str, bool]]], dict[str, int]]:
    """Freeze block tags AND return each node's inherited-block count.

    The single source of the frozen ``(role, starts_new_message)`` tags (see
    :func:`assign_block_tags`) and, alongside, the geometric ``inherited`` count
    each node's tags were built from. :func:`build_trie_ir`'s prefix-path reuse
    consumes this EXACT ``inherited`` value (never a recomputed one) so the reuse
    boundary can never drift from the frozen tags -- the two are computed once,
    together.

    Returns ``(tags, inherited_by_node)`` where ``tags`` is
    ``node_id -> [(role, starts_new_message), ...]`` (one entry per covered
    block) and ``inherited_by_node`` is ``node_id -> inherited`` (the count of
    leading blocks whose tags were copied verbatim from the content-parent).
    Requires :func:`resolve_content_parents` to have run.
    """
    tags: dict[str, list[tuple[str, bool]]] = {}
    inherited_by_node: dict[str, int] = {}
    for node in nodes:
        parent = node.content_parent
        parent_tags = tags.get(parent.node_id, []) if parent is not None else []
        prev_hash = list(parent.request.hash_ids) if parent is not None else []
        prev_out = parent.request.output_length if parent is not None else 0
        geo = compute_turn_block_geometry(
            prev_hash,
            list(node.request.hash_ids),
            node.request.input_length,
            block_size,
        )
        # Clamp inherited to BOTH ends: a content-parent that UNDER-COVERS holds
        # fewer frozen tags than its LCP with this child (inherit only what the
        # parent tagged, covered-but-uninherited blocks become NEW blocks here);
        # and this child may OVER-share -- a full block prefix (lcp) but a smaller
        # declared in (in // bs < lcp), so clamp to its own covered count too or
        # it would carry more tags than it emits and trip the ISL gate.
        inherited = min(geo.lcp, len(parent_tags), geo.m_curr_covered)
        parent_has_user = any(role == "user" for role, _ in parent_tags[:inherited])
        inh2, new_roles = block_role_split(
            prev_hash_ids=prev_hash,
            curr_hash_ids=list(node.request.hash_ids),
            curr_in_tokens=node.request.input_length,
            prev_out_tokens=prev_out,
            block_size=block_size,
            max_asst_blocks=caps.get(node.node_id),
            parent_has_user=parent_has_user,
            parent_covered_blocks=len(parent_tags),
        )
        # Geometry is the single source of the inherited-block count; the split
        # helper must agree or the frozen prefix would drift from the tags.
        if inh2 != inherited:
            raise ValueError(
                f"node {node.node_id}: block-tag/geometry disagreement "
                f"{inh2} != {inherited}"
            )
        node_tags: list[tuple[str, bool]] = list(parent_tags[:inherited])
        for j, role in enumerate(new_roles):
            starts = (j == 0) or (role != new_roles[j - 1])
            node_tags.append((role, starts))
        tags[node.node_id] = node_tags
        inherited_by_node[node.node_id] = inherited
    return tags, inherited_by_node


def assign_block_tags(
    nodes: list[TrieNode],
    block_size: int,
    caps: dict[str, int | None],
) -> dict[str, list[tuple[str, bool]]]:
    """Freeze a ``(role, starts_new_message)`` tag per COVERED block at creation.

    The CONTENT invariant's enforcement core: every covered block position of
    every node gets exactly one ``(role, starts_new_message)`` tag, assigned by
    the node that FIRST materializes it and then FROZEN. Two requests sharing a
    block prefix therefore read the SAME tags on that prefix -> identical
    messages -> cache-safe. Re-tagging a shared block on a later turn would
    silently break that prefix identity, which is why tags freeze at creation.

    Nodes are processed in recorded (``order``) order so a content-parent is
    tagged before any child reads its frozen tags. For each node:

    * Inherited blocks ``[0, inherited)`` reuse the content-parent's frozen tags
      VERBATIM (``inherited = min(lcp, len(parent.hash_ids))`` from the shared
      geometry). A fresh root (``content_parent is None``) inherits nothing.
    * ``parent_has_user`` for the context-loss rule is computed over the
      INHERITED PREFIX ONLY -- the blocks this node actually carries forward.
    * New blocks ``[inherited, m_covered)`` get roles from :func:`block_role_split`
      (assistant run then user run, capped, context-loss aware).
    * ``starts_new_message`` is True for the node's FIRST new block (a new
      recorded turn always opens a message, even when it continues the parent's
      tail role -- this preserves contiguous same-role turns) OR at a role
      transition within the new region; otherwise False.

    Returns ``node_id -> [(role, starts_new_message), ...]`` (one entry per
    covered block, ``min(len(hash_ids), in // bs)`` of them). Requires
    :func:`resolve_content_parents` to have run. Thin wrapper over
    :func:`_assign_block_tags_and_inheritance` (drops the inheritance map) so the
    public tag output and every existing caller/oracle stay unchanged.
    """
    tags, _inherited_by_node = _assign_block_tags_and_inheritance(
        nodes, block_size, caps
    )
    return tags


# --- message + segment emission --------------------------------------------


def _assemble_messages_from(
    hash_ids: list[int],
    block_tags: list[tuple[str, bool]],
    start_block: int,
    seed_parent_id: str | None,
    pool: SegmentPool,
    decode_block_tokens: Callable[[list[int]], list[int]],
    decode_tokens_to_text: Callable[[list[int]], str],
) -> tuple[list[str], int, list[int], list[int]]:
    """Emit the messages of ``block_tags[start_block:]``, chained onto a seed.

    The reuse-aware emission core: identical grouping and content-addressing to
    the whole-prefix emission, but starting at ``start_block`` and threading the
    first fresh message's ``parent_id`` onto ``seed_parent_id`` (the spliced
    content-parent tip; ``None`` at a chain root). A new message begins at
    ``start_block`` and at every block whose frozen ``starts_new_message`` is
    True; consecutive False blocks join it. Each message's tokens are the
    block-aligned concatenation of its blocks (NO partial tail).

    Returns ``(fresh_segment_ids, fresh_token_count, group_end_blocks,
    group_token_counts)``: the ordered per-message pool ids emitted here, their
    ACTUAL decoded token total, and per-message ABSOLUTE exclusive end block
    indices + actual token counts (the driver folds these into the node's
    :class:`_EmissionRecord`). Absolute end blocks are valid because a child's
    inherited-prefix tags ARE the parent's, copied verbatim.
    """
    groups: list[tuple[str, list[int]]] = []
    for k in range(start_block, len(block_tags)):
        role, starts = block_tags[k]
        if starts or not groups:
            groups.append((role, [k]))
        else:
            groups[-1][1].append(k)
    fresh_ids: list[str] = []
    fresh_token_count = 0
    group_end_blocks: list[int] = []
    group_token_counts: list[int] = []
    prev_id = seed_parent_id
    for role, idxs in groups:
        toks: list[int] = []
        for k in idxs:
            toks.extend(decode_block_tokens([hash_ids[k]]))
        prev_id = pool.add(
            role=role,
            content=decode_tokens_to_text(toks),
            tokens=toks,
            parent_id=prev_id,
        )
        fresh_ids.append(prev_id)
        fresh_token_count += len(toks)
        group_end_blocks.append(idxs[-1] + 1)
        group_token_counts.append(len(toks))
    return fresh_ids, fresh_token_count, group_end_blocks, group_token_counts


def assemble_messages(
    hash_ids: list[int],
    block_tags: list[tuple[str, bool]],
    pool: SegmentPool,
    decode_block_tokens: Callable[[list[int]], list[int]],
    decode_tokens_to_text: Callable[[list[int]], str],
) -> tuple[list[str], int]:
    """Group frozen per-block tags into messages and emit one content-addressed
    pool entry per message, chained root->tip.

    A new message begins at the first covered block and at every block whose
    frozen ``starts_new_message`` is True; consecutive ``starts_new_message=False``
    blocks join the current message. Each message's tokens are the block-aligned
    concatenation of its blocks (NO partial tail), and it is emitted as one
    ``pool.add`` parented at the previous message's id (first message's
    ``parent_id`` is ``None``).

    Because the tags are frozen per trie position, a shared block prefix yields
    identical message grouping -> identical ``(parent_id, role, tokens)`` chain
    -> identical content-addressed ids -> a real cache prefix. There is NO role
    coercion and NO relabeling here: trailing-user is already frozen in the tags.

    Returns ``(prompt_segment_ids, assembled_token_count)`` -- the ordered
    per-message pool ids plus the ACTUAL decoded token total across all
    messages, so the ISL gate can compare what was assembled (not a value
    re-derived from the tag count, which would be tautological). The whole-prefix
    (``start_block=0``, no seed) wrapper over :func:`_assemble_messages_from`;
    behavior-identical to the pre-reuse emission, retained for callers and the
    differential oracle.
    """
    fresh_ids, token_count, _ends, _counts = _assemble_messages_from(
        hash_ids, block_tags, 0, None, pool, decode_block_tokens, decode_tokens_to_text
    )
    return fresh_ids, token_count


def emit_response_segment(
    node: TrieNode,
    *,
    pool: SegmentPool,
    parent_id: str | None,
    callbacks: ReconCallbacks,
) -> str:
    """Append the node's assistant output as one pool segment; return its id.

    The recorded ``out`` token count is synthesized into deterministic tokens
    keyed on the node id so the response is content-addressed and stable. An
    ``out == 0`` response still emits an (empty) segment so ``response_id``
    always names a real pool entry.
    """
    n_out = max(0, node.request.output_length)
    tokens = callbacks.sample_partial_tail_tokens(n_out, f"{node.node_id}:response")
    content = callbacks.decode_tokens_to_text(tokens)
    return pool.add(
        role="assistant", content=content, tokens=tokens, parent_id=parent_id
    )


# --- ISL gate + fan-in ------------------------------------------------------


class TrieISLMismatchError(ValueError):
    """Reconstructed prompt token count did not equal the block-aligned covered-count target."""


def assert_covered_isl(
    node: TrieNode, prompt_token_count: int, block_size: int
) -> None:
    """Hard build-abort when a reconstructed prompt misses its covered-count target.

    The target is the COVERED-count: only the blocks actually emitted (message-unit
    emission covers ``min(recorded hash blocks, in // block_size)`` blocks, block-
    aligned, no partial tail, no synthesis of missing whole blocks). It is NOT
    ``(in // block_size) * block_size`` -- a recorded request may store fewer hash
    blocks than ``in // block_size``, and demanding that would hard-abort the build
    on a legitimate trace. :func:`build_trie_ir` wires the call site.
    """
    expected = (
        min(len(node.request.hash_ids), node.request.input_length // block_size)
        * block_size
    )
    if prompt_token_count != expected:
        raise TrieISLMismatchError(
            f"node {node.node_id}: {prompt_token_count} tokens != covered-count "
            f"{expected} (in={node.request.input_length}, bs={block_size}, "
            f"{len(node.request.hash_ids)} blocks)"
        )


def with_fan_in_inputs(llm: LlmNode, node_edges: list[StaticEdge]) -> LlmNode:
    """Attach one AND-fan-in input requirement per non-START predecessor edge.

    A multi-predecessor trie node declares NO input channels otherwise, so the
    executor's ``await_inputs`` gate is a no-op and the node fires on its FIRST
    completing predecessor (then a later predecessor re-schedules it past the
    cycle guard). Reading each non-START predecessor's already-write-declared
    ``{source}_out`` channel turns ``await_inputs`` into the recorded AND-join.
    ``count=1``: every predecessor writes its ``_out`` exactly once. This adds
    only the WAIT; the LlmNode prompt still comes from the segment pool, not the
    channel value. START-rooted nodes (no non-START edge) keep empty ``inputs``.
    """
    inputs = [
        ChannelRequirement(channel=f"{e.source}_out", count=1)
        for e in node_edges
        if e.source != START_NODE_ID and e.delay_after_predecessor_start_us is None
    ]
    if not inputs:
        return llm
    return msgspec.structs.replace(llm, inputs=inputs)


# --- driver -----------------------------------------------------------------


def build_trie_ir(
    nodes: list[TrieNode],
    *,
    block_size: int,
    callbacks: ReconCallbacks,
    pool: SegmentPool,
    idle_gap_cap_seconds: float | None,
    small_prompt_fallback: bool = False,
) -> TrieBuild:
    """Run the shared trie-content pipeline over normalized nodes.

    Exactly the weka build order: content parents -> idle warp -> ranks ->
    interval edges -> assistant caps -> frozen block tags -> per-node message
    assembly + covered-count ISL gate + response segment. The gate compares
    the ACTUAL assembled token count (skipped when
    ``callbacks.block_exact`` is False -- deliberate placeholder content).

    Recorded inter-request delays always replay on the edges, warped through
    :func:`apply_idle_gap_warp` (``idle_gap_cap_seconds=None`` disables the
    warp, never the delays).
    ``small_prompt_fallback`` emits a single user message sized to
    ``input_length`` for covered-count-0 nodes instead of an empty prompt (the
    ISL gate is skipped for those nodes); both recorded adapters pass True (a
    recorded sub-block prompt was a real prompt -- an empty messages array is
    unreplayable). False remains for callers that must preserve empty prompts.
    """
    resolve_content_parents(nodes)
    apply_idle_gap_warp(nodes, idle_gap_cap_seconds)
    compute_ranks(nodes)
    edges_by_node = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges_by_node)
    caps = compute_asst_caps(nodes, block_size)
    tags, inherited_by_node = _assign_block_tags_and_inheritance(
        nodes, block_size, caps
    )

    builds: dict[str, TrieNodeBuild] = {}
    # node_id -> emission record; local to this build (one trace per worker task).
    records: dict[str, _EmissionRecord] = {}
    for node in nodes:
        node_tags = tags[node.node_id]
        covered = len(node_tags)
        small = False
        if covered == 0 and small_prompt_fallback and node.request.input_length > 0:
            toks = callbacks.sample_partial_tail_tokens(
                node.request.input_length, f"{node.node_id}:tiny"
            )
            prompt_path = [
                pool.add(
                    role="user",
                    content=callbacks.decode_tokens_to_text(toks),
                    tokens=toks,
                    parent_id=None,
                )
            ]
            small = True
            # Small-prompt nodes publish NO record: their single message is not
            # tag-derived, and any child structurally inherits 0 (empty tags), so
            # nothing can reuse from them.
        else:
            parent = node.content_parent
            # The reuse boundary MUST come from the geometric ``inherited``
            # (single-sourced from the tag pass above), never a tag-prefix
            # comparison: tags can coincide beyond the lcp while hash ids differ,
            # and splicing there would ship a wrong content-addressed sid.
            inherited = inherited_by_node[node.node_id]
            # Missing-parent-record guard: a resolved content_parent may publish
            # no record (small-prompt parent), so gate on BOTH the record
            # existing AND inherited > 0 before attempting any splice.
            rec = records.get(parent.node_id) if parent is not None else None
            j = 0
            splice: list[str] = []
            reused_tokens = 0
            start_block = 0
            seed_parent_id: str | None = None
            if rec is not None and inherited > 0:
                # Count whole parent messages ending at or before ``inherited``;
                # bisect_right includes a message ending exactly at inherited.
                j = bisect_right(rec.msg_end_blocks, inherited)
                if j > 0:
                    splice = rec.prompt_path[:j]
                    reused_tokens = rec.cum_token_counts[j - 1]
                    start_block = rec.msg_end_blocks[j - 1]
                    seed_parent_id = splice[-1]
                    # The resume block must OPEN a message: it is a parent message
                    # boundary copied verbatim into this child's frozen tags, so a
                    # False here means the grouping drifted in this frozen file.
                    if start_block < covered:
                        assert node_tags[start_block][1], (
                            f"node {node.node_id}: reuse resume block "
                            f"{start_block} does not start a message (grouping drift)"
                        )
            fresh_ids, fresh_tokens, group_ends, group_counts = _assemble_messages_from(
                node.request.hash_ids,
                node_tags,
                start_block,
                seed_parent_id,
                pool,
                callbacks.decode_block_tokens,
                callbacks.decode_tokens_to_text,
            )
            prompt_path = splice + fresh_ids
            # Gate on the ACTUAL assembled token count (reused + fresh) so decode
            # drift (blocks not sized block_size) aborts the build; placeholder
            # callbacks (block_exact=False, test-only stubs) skip it. Reused
            # blocks contribute the parent's actual decoded counts: they are
            # ISL-verified ONCE, at first materialization, and re-verifying here
            # would be identical only under the ReconCallbacks determinism
            # contract (all production callbacks satisfy it).
            if callbacks.block_exact:
                assert_covered_isl(node, reused_tokens + fresh_tokens, block_size)
            # Publish this node's record: reused ends/counts (cumulative) copied
            # from the parent, then fresh groups appended with the running total
            # continued from ``reused_tokens``.
            end_blocks = rec.msg_end_blocks[:j] if j else array("q")
            cum_counts = rec.cum_token_counts[:j] if j else array("q")
            running = reused_tokens
            for end_block, count in zip(group_ends, group_counts, strict=True):
                end_blocks.append(end_block)
                running += count
                cum_counts.append(running)
            records[node.node_id] = _EmissionRecord(
                msg_end_blocks=end_blocks,
                cum_token_counts=cum_counts,
                prompt_path=prompt_path,
            )
        response_id = emit_response_segment(
            node,
            pool=pool,
            parent_id=prompt_path[-1] if prompt_path else None,
            callbacks=callbacks,
        )
        builds[node.node_id] = TrieNodeBuild(
            prompt_path=prompt_path, response_id=response_id, small_prompt=small
        )
    return TrieBuild(builds=builds, edges_by_node=edges_by_node)


# --- graph assembly ---------------------------------------------------------


def assemble_trie_graph(
    nodes: list[TrieNode],
    result: TrieBuild,
    build_node: Callable[[TrieNode], LlmNode],
    provenance: ProvenanceSpec,
) -> GraphRecord:
    """Shared assembly epilogue for trie adapters: build LlmNodes, attach
    fan-in edges, stamp the theoretical prefix cache, declare output channels,
    and wrap in a ``version="2.0"`` :class:`GraphRecord`.

    Each node is turned into a pre-fan-in :class:`LlmNode` by ``build_node`` --
    the per-adapter builder closed over ``result.builds`` and the adapter's own
    dispatch knobs (e.g. weka's per-node ``max_osl``
    cap). Its interval-order edges are appended to the
    graph edge list and turned into AND-fan-in input requirements
    (:func:`with_fan_in_inputs`). After every node is assembled the shared
    per-trace theoretical prefix cache is stamped across the node map, then
    each node's ``{node_id}_out`` scratch channel is declared and the whole
    topology is wrapped under ``provenance``. The assembly order and every
    constructed value are byte-identical across adapters -- only ``build_node``
    and ``provenance`` differ.

    Every ``LlmNode`` writes its response into a per-node ``{node_id}_out``
    channel; the executor's channel store rejects writes to any channel not
    declared in ``graph.state`` (``UnknownChannelError``), so each output
    channel is declared with the default TEXT / overwrite spec. Successors read
    these channels ONLY to enforce the AND-fan-in WAIT (one ``count=1``
    requirement per predecessor) -- the prompt itself comes from the segment
    pool, never the channel value -- so the default single-producer overwrite
    spec is contractually sufficient.
    """
    # call-scoped: prefix_cache imports TrieNode from this module
    from aiperf.dataset.graph.segment_ir.prefix_cache import (
        stamp_theoretical_prefix_cache,
    )

    llm_nodes: dict[str, LlmNode] = {}
    edges: list[StaticEdge] = []
    for node in nodes:
        llm = build_node(node)
        node_edges = result.edges_by_node[node.node_id]
        edges.extend(node_edges)
        llm_nodes[node.node_id] = with_fan_in_inputs(llm, node_edges)
    stamp_theoretical_prefix_cache(llm_nodes, nodes)

    state = {f"{nid}_out": ChannelSpec() for nid in llm_nodes}
    return GraphRecord(
        version="2.0",
        provenance=provenance,
        state=state,
        nodes=dict(llm_nodes),
        edges=edges,
    )


__all__ = [
    "ActiveIdleWarp",
    "MessageUnit",
    "ReconCallbacks",
    "TrieBuild",
    "TrieISLMismatchError",
    "TrieNode",
    "TrieNodeBuild",
    "TrieRequest",
    "TurnBlockGeometry",
    "add_message_chain",
    "apply_idle_gap_warp",
    "assemble_messages",
    "assemble_trie_graph",
    "assert_covered_isl",
    "assign_block_tags",
    "block_role_split",
    "build_trie_ir",
    "compute_asst_caps",
    "compute_turn_block_geometry",
    "emit_response_segment",
    "longest_common_prefix",
    "resolve_content_parents",
    "with_fan_in_inputs",
]
