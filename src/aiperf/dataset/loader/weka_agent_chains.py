# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flattened-agent chain detection for Weka traces via hash_id LCP evidence.

Untagged agent fan-outs (Workflow agents, headerless subagents) are recorded
as interleaved flat top-level requests. This module partitions a trace's
top-level requests back into per-agent chains and classifies every hash-list
divergence as either a join seam (the same agent continuing after a context
edit such as compaction) or a spawn (a new agent forked from a shared
prefix).

Seam-vs-spawn rule: on a shrink — a request sharing only a proper prefix of
its best-matching chain tail — the request is the same agent's continuation
(join seam) iff no future request ever pulls back to the longer, pre-shrink
state; otherwise it is a separate spawned agent. Implemented offline:
phase 1 builds chains greedily, forking on every shrink; phase 2 splices the
elected continuation back onto tails whose longer state turned out to be
dead. A deeper sibling fork from the same tail IS the "future pullback"
that demotes shallower forks to spawns, so the election encodes the rule
directly.

Same-model rule: a chain is only ever continued — by extension or by seam
splice — by requests of the SAME model. Cross-model attachment is always a
spawn, even on a full-prefix match (a Haiku worker reading Opus context is
a different agent; the rare mid-session /model switch is accepted as a
split).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
)

_EPSILON_SECONDS = 1e-6

_NormalRequestT = WekaNormalRequest | WekaStreamingRequest
IndexedRequest = tuple[int, _NormalRequestT]
"""One retained top-level request: (outer index in trace.requests, request)."""


def _req_end(req: _NormalRequestT) -> float:
    """Interval end in seconds; missing/negative/non-finite api_time counts as zero.

    The finite check matters for temporal feasibility: a NaN or +inf
    ``api_time`` would otherwise poison a chain tail's end and either block
    every same-context extension forever (+inf) or make comparisons
    unreliable (NaN).
    """
    duration = (
        req.api_time
        if req.api_time is not None and math.isfinite(req.api_time)
        else 0.0
    )
    return req.t + max(duration, 0.0)


def _np_lcp(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two int64 hash arrays."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    neq = a[:n] != b[:n]
    i = int(neq.argmax())
    return i if neq[i] else n


@dataclass(slots=True)
class ChainFork:
    """Where a chain split off another chain."""

    parent_chain: int | None
    """Phase-1 index of the chain forked from; None = no shared context.
    Rewritten to the live (post-splice) chain index by phase 2."""
    fork_outer_idx: int | None
    """Outer index of the tail request T this chain forked from."""
    depth: int
    """Blocks shared with T at fork time (LCP)."""
    fork_time: float
    """t of this chain's first request."""


@dataclass(slots=True)
class AgentChain:
    """One detected agent: a time-ordered run of requests."""

    requests: list[IndexedRequest] = field(default_factory=list)
    """The chain's requests in (t, outer_idx) order."""
    fork: ChainFork | None = None
    """How this chain came to exist; None only for the first chain."""
    spliced_into: int | None = None
    """Set by phase 2 when this chain was a join-seam continuation."""
    tail_outer_idx: int = -1
    """Outer index of the last hash-bearing request (phase-1 state)."""
    tail_hash: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    """Hash array of the last hash-bearing request (phase-1 state)."""
    tail_end: float = 0.0
    """Interval end of the last hash-bearing request (phase-1 state)."""
    tail_model: str = ""
    """Model of the last hash-bearing request (phase-1 state). A chain is
    only ever continued by same-model requests; cross-model = spawn."""


@dataclass(slots=True)
class ChainDetectionResult:
    """Output of detect_agent_chains. Spliced chains stay in ``chains`` for
    fork-history logging but are excluded from ``worker_indices``."""

    chains: list[AgentChain]
    """All phase-1 chains, including spliced (dead) ones."""
    main_index: int
    """Index of the chain owning the trace's first retained request."""
    worker_indices: list[int]
    """Live non-main chains, ordered by first request (t, outer_idx)."""
    seams_merged: int
    """Number of join-seam splices performed in phase 2."""
    unclassified_empty_hash: int
    """Requests with empty hash_ids kept on the main chain as-is."""


def is_aux_chain(
    requests: list[_NormalRequestT],
    main_peak_isl: int,
    *,
    max_requests: int,
    isl_ratio: float,
    isl_floor: int,
    main_model: str | None = None,
    cross_model: bool = True,
) -> bool:
    """Classify a detected worker chain as an auxiliary one-shot call.

    A detected worker chain is reclassified as *auxiliary* -- a sidecar rather
    than a genuine agent -- when it is short and either (a) starts from a small,
    fresh context, or (b) runs on a different model than the enclosing agent.
    Both are signatures of a tool-issued one-shot (web fetch / search summary,
    title generation, a classifier) that rides alongside the agent for a single
    call, not a sustained sub-agent. Applied at both layers -- top-level flat
    chains (genuine ``::fa:`` vs sidecar ``::aux:``) and a subagent's nested-LCP
    overflow chains (genuine ``:fa:`` vs sidecar ``:aux:``).

    ``requests`` is the chain's request list (time-ordered). Returns ``True``
    iff it holds at most ``max_requests`` requests AND either:

    - **cross-model** (``cross_model`` set and ``main_model`` given): the first
      request runs on a model other than the enclosing main chain's. An agent
      does not switch models for its own reasoning, so a one-shot on a different
      model is a tool-internal call (e.g. a Haiku ``WebFetch`` summary under an
      Opus agent) regardless of payload size; or
    - **small fresh context**: the first request's input length is below
      ``max(isl_floor, isl_ratio * main_peak_isl)`` -- small both absolutely and
      relative to the surrounding context. ``main_peak_isl`` is the largest
      input length on the enclosing main chain (the trace's for flat chains, the
      subagent's for overflow).

    Set ``max_requests=0`` to disable reclassification entirely (every worker
    chain keeps its agent tag); ``cross_model=False`` disables only arm (a).
    """
    if not requests or len(requests) > max_requests:
        return False
    first = requests[0]
    if cross_model and main_model is not None and first.model != main_model:
        return True
    threshold = max(isl_floor, int(isl_ratio * main_peak_isl))
    return first.input_length < threshold


def is_reduction_chain(
    requests: list[_NormalRequestT],
    *,
    osl_max: int,
    ratio: float,
    isl_floor: int,
) -> bool:
    """Classify a single-request worker chain as an auxiliary *reduction* call.

    A reduction is a large-input / short-output one-shot -- a context
    compaction, a subagent-result summary, or a tool-output digest -- that
    consumes a large body and emits a short summary. It runs on the SAME model
    as the enclosing agent (so :func:`is_aux_chain`'s cross-model arm misses it)
    and starts from a large context (so the small-fresh-context size arm misses
    it too), yet it is auxiliary, not generative: a real agent emits long
    completions, while a reduction's output stays bounded and its input/output
    ratio is high. The signature is stable across every weka capture (the one
    flat-chain class present from the earliest snapshot onward).

    Returns ``True`` iff ``requests`` is exactly one request whose output length
    is in ``(0, osl_max)``, whose input length is at least ``isl_floor``, and
    whose input/output ratio exceeds ``ratio``. Set ``osl_max=0`` to disable.
    Intended to be checked only AFTER :func:`is_aux_chain` returns False, so a
    cross-model or small-fresh-context one-shot is never relabeled a reduction.
    """
    if osl_max <= 0 or len(requests) != 1:
        return False
    first = requests[0]
    osl = first.output_length or 0
    if osl <= 0 or osl >= osl_max:
        return False
    if first.input_length < isl_floor:
        return False
    return first.input_length > ratio * osl


_IntervalCand = tuple[float, float, int, int]  # (t0, t1, first_outer, chain_index)


def _overlap_components(cands: list[_IntervalCand]) -> list[list[_IntervalCand]]:
    """Connected components of ``[t0, t1)`` interval overlap (sweep + union-find).

    Processing in start order, every interval still "active" (end > this start)
    overlaps the current one; the active set is always one component, so unioning
    with any active member joins the whole component. Touching intervals
    (``end == start``) do not overlap."""
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: (cands[i][0], cands[i][2]))
    parent = list(range(len(cands)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active: list[tuple[float, int]] = []  # min-heap of (t1, idx)
    for i in order:
        t0 = cands[i][0]
        while active and active[0][0] <= t0:
            heapq.heappop(active)
        if active:
            parent[_find(i)] = _find(active[0][1])
        heapq.heappush(active, (cands[i][1], i))

    comps: dict[int, list[_IntervalCand]] = {}
    for i in range(len(cands)):
        comps.setdefault(_find(i), []).append(cands[i])
    return list(comps.values())


def worker_group_assignment(
    result: ChainDetectionResult,
    *,
    group_min: int,
) -> dict[int, tuple[int, int]]:
    """Assign each worker-group member a ``(group, member)`` coordinate.

    A parallel sub-agent fan-out is a set of persistent workers that share a deep
    spawned context AND run CONCURRENTLY. We require BOTH, the way the corpus
    analysis prescribes (overlapping intervals AND a shared prefix):

    1. **Shared-context scope**: bucket workers that forked from shared context
       (``fork.depth > 0``) by their fork point
       (``fork.parent_chain`` + ``fork.fork_outer_idx``) -- workers off the same
       parent request share that parent's deep prefix. This scope keeps unrelated
       fan-outs apart; without it, pure interval overlap bridges every worker in
       a busy trace into one blob (a long-running chain of overlaps spans the
       whole session).
    2. **Temporal overlap within the scope**: within each fork-point bucket,
       split into connected components of overlapping ``[t0, t1)`` intervals
       (``t0`` = first request time, ``t1`` = latest request-interval end). This
       drops members that share the fork point but never actually run
       concurrently (e.g. a chain seam-re-keyed onto a parent tail it never ran
       beside) -- the group is a concurrent swarm, not a session-wide lineage
       (the failure of keying on ``hash_ids[0]``).

    - **group**: a (fork-point, overlap-component) with at least ``group_min``
      members, numbered by earliest start.
    - **member**: index within the group, in ``(t0, outer_idx)`` order.

    Returns ``{chain_index: (group, member)}`` for members only (a subset of
    ``result.worker_indices``); empty when ``group_min <= 0``."""
    if group_min <= 0:
        return {}

    buckets: dict[tuple[int, int], list[_IntervalCand]] = {}
    for ci in result.worker_indices:
        chain = result.chains[ci]
        fork = chain.fork
        if (
            fork is None
            or fork.depth <= 0
            or fork.parent_chain is None
            or fork.fork_outer_idx is None
            or not chain.requests
        ):
            continue
        t0 = chain.requests[0][1].t
        t1 = max(_req_end(req) for _, req in chain.requests)
        key = (fork.parent_chain, fork.fork_outer_idx)
        buckets.setdefault(key, []).append((t0, t1, chain.requests[0][0], ci))

    components: list[list[_IntervalCand]] = []
    for cands in buckets.values():
        components.extend(_overlap_components(cands))

    groups = [comp for comp in components if len(comp) >= group_min]
    groups.sort(key=lambda comp: min((c[0], c[2]) for c in comp))

    out: dict[int, tuple[int, int]] = {}
    for group, comp in enumerate(groups):
        comp.sort(key=lambda c: (c[0], c[2]))
        for member, (_, _, _, ci) in enumerate(comp):
            out[ci] = (group, member)
    return out


def worker_group_members(result: ChainDetectionResult, *, group_min: int) -> set[int]:
    """Worker chain indices belonging to a parallel fan-out group.

    Thin wrapper over :func:`worker_group_assignment` returning just the member
    set. See that function for the fork-point ``(group, member)`` decomposition."""
    return set(worker_group_assignment(result, group_min=group_min))


@dataclass(slots=True)
class _Phase1State:
    """Working state for the greedy forward pass."""

    chains: list[AgentChain] = field(default_factory=list)
    chain_of_request: dict[int, int] = field(default_factory=dict)
    forks_by_tail: dict[int, list[int]] = field(default_factory=dict)
    req_by_outer: dict[int, _NormalRequestT] = field(default_factory=dict)
    unclassified: int = 0

    def _append(self, chain_idx: int, outer_idx: int, req: _NormalRequestT) -> None:
        c = self.chains[chain_idx]
        c.requests.append((outer_idx, req))
        self.chain_of_request[outer_idx] = chain_idx
        if req.hash_ids:
            c.tail_outer_idx = outer_idx
            c.tail_hash = np.asarray(req.hash_ids, dtype=np.int64)
            c.tail_end = _req_end(req)
            c.tail_model = req.model

    def classify(self, outer_idx: int, req: _NormalRequestT) -> None:
        self.req_by_outer[outer_idx] = req
        if not req.hash_ids:
            # No LCP evidence: keep on the main chain, invisible to tails
            # and forks (it must not found a chain or serve as a witness).
            self.unclassified += 1
            if not self.chains:
                self.chains.append(AgentChain())
            self.chains[0].requests.append((outer_idx, req))
            self.chain_of_request[outer_idx] = 0
            return

        h = np.asarray(req.hash_ids, dtype=np.int64)
        if not self.chains:
            # The trace's first request founds the main chain. It forked
            # from nothing, so fork stays None (the AgentChain contract).
            self.chains.append(AgentChain())
            self._append(0, outer_idx, req)
            return
        target = _find_extension_target(self.chains, h, req.t, req.model)
        if target is not None:
            self._append(target, outer_idx, req)
            return

        parent, depth = _max_lcp_chain(self.chains, h)
        if parent is None and all(c.tail_hash.shape[0] == 0 for c in self.chains):
            # No chain carries hash evidence yet (only leading empty-hash
            # rows): this is the trace's first hash-bearing request and IS
            # the main agent — joining chain 0 keeps the empty rows "on the
            # main chain" instead of exiling all real load to a spawn.
            self._append(0, outer_idx, req)
            return
        fork = ChainFork(
            parent_chain=parent,
            fork_outer_idx=(
                self.chains[parent].tail_outer_idx if parent is not None else None
            ),
            depth=depth,
            fork_time=req.t,
        )
        new_idx = len(self.chains)
        self.chains.append(AgentChain(fork=fork))
        self._append(new_idx, outer_idx, req)
        if fork.fork_outer_idx is not None and depth > 0:
            self.forks_by_tail.setdefault(fork.fork_outer_idx, []).append(new_idx)


def detect_agent_chains(
    normals: list[IndexedRequest],
    *,
    seam_max_gap_seconds: float = 3600.0,
    seam_min_overlap_ratio: float = 0.5,
) -> ChainDetectionResult:
    """Partition retained top-level requests into per-agent chains.

    ``seam_max_gap_seconds`` / ``seam_min_overlap_ratio`` gate join-seam
    splicing (see :func:`_elect_continuation`): a low-overlap continuation far
    past the chain tail is spawned as its own conversation rather than stitched
    on, so a distinct session that merely shares a base prefix does not get
    folded into one conversation with a multi-hour internal gap. Defaults match
    ``Environment.DATASET.WEKA_SEAM_*``; the WekaTraceLoader call sites pass the
    configured values, while direct callers (tests) use these literals."""
    if not normals:
        return ChainDetectionResult(
            chains=[],
            main_index=0,
            worker_indices=[],
            seams_merged=0,
            unclassified_empty_hash=0,
        )

    # Spec §4: process in (t, outer_idx) order. Trace files are normally
    # time-ordered already; sorting makes the algorithm independent of file
    # order and keeps every chain's request list t-ordered for delay math.
    ordered = sorted(normals, key=lambda item: (item[1].t, item[0]))

    state = _Phase1State()
    for outer_idx, req in ordered:
        state.classify(outer_idx, req)
    chains = state.chains

    seams = _resolve_seams(
        chains,
        state.forks_by_tail,
        state.chain_of_request,
        state.req_by_outer,
        max_gap_seconds=seam_max_gap_seconds,
        min_overlap_ratio=seam_min_overlap_ratio,
    )

    alias = {
        i: c.spliced_into for i, c in enumerate(chains) if c.spliced_into is not None
    }

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    for c in chains:
        if (
            c.spliced_into is None
            and c.fork is not None
            and c.fork.parent_chain is not None
        ):
            c.fork.parent_chain = _resolve(c.fork.parent_chain)

    main_index = _resolve(state.chain_of_request[ordered[0][0]])
    workers = [
        i for i, c in enumerate(chains) if c.spliced_into is None and i != main_index
    ]
    workers.sort(key=lambda i: (chains[i].requests[0][1].t, chains[i].requests[0][0]))
    return ChainDetectionResult(
        chains=chains,
        main_index=main_index,
        worker_indices=workers,
        seams_merged=seams,
        unclassified_empty_hash=state.unclassified,
    )


def _find_extension_target(
    chains: list[AgentChain], h: np.ndarray, t: float, model: str
) -> int | None:
    """Chain whose tail is a complete prefix of ``h``, has ended by ``t``,
    and ran on the same ``model`` (cross-model attachment is always a spawn).

    Deepest tail wins; ties go to the lowest chain index (ascending scan
    with strict ``>`` keeps the first)."""
    best: int | None = None
    best_len = -1
    hn = h.shape[0]
    for idx, c in enumerate(chains):
        tl = c.tail_hash.shape[0]
        if tl == 0 or tl > hn or tl <= best_len:
            continue
        if c.tail_model != model:
            continue
        if c.tail_end > t + _EPSILON_SECONDS:
            continue
        if c.tail_hash[tl - 1] != h[tl - 1]:
            continue
        if bool((h[:tl] == c.tail_hash).all()):
            best, best_len = idx, tl
    return best


def _max_lcp_chain(chains: list[AgentChain], h: np.ndarray) -> tuple[int | None, int]:
    """Chain tail with the deepest LCP against ``h`` (ties: deeper tail,
    then lower index). Returns (None, 0) when nothing shares a prefix."""
    best_idx: int | None = None
    best_key = (0, 0)
    for idx, c in enumerate(chains):
        if c.tail_hash.shape[0] == 0:
            continue
        d = _np_lcp(c.tail_hash, h)
        if d == 0:
            continue
        key = (d, c.tail_hash.shape[0])
        if key > best_key:
            best_idx, best_key = idx, key
    return best_idx, best_key[0]


def _observed_group_prefix(result: ChainDetectionResult, members: list[int]) -> int:
    """LCP over the group members' first-request hash lists (0 if < 2)."""
    firsts = [
        np.asarray(result.chains[ci].requests[0][1].hash_ids, dtype=np.int64)
        for ci in members
    ]
    firsts = [f for f in firsts if f.shape[0] > 0]
    if len(firsts) < 2:
        return 0
    observed = firsts[0].shape[0]
    for other in firsts[1:]:
        observed = min(observed, _np_lcp(firsts[0][:observed], other))
    return observed


def compute_chain_prefix_blocks(
    result: ChainDetectionResult, *, declared_prefix_blocks: int
) -> dict[int, int]:
    """Effective setup-prefix block count per live chain (spec §5.4).

    Chains are grouped by fork ancestry into namespace groups (each
    zero-depth fork roots a new group). Per group, the observed prefix is
    the LCP over members' first-request hash lists (0 for singletons).
    The main chain keeps the LONGER of declared vs observed; other chains
    use observed — they only prove the shared region, anything past it is
    not common to the group.
    """
    live = [i for i, c in enumerate(result.chains) if c.spliced_into is None]
    if not live:
        return {}

    def _group_root(ci: int) -> int:
        while True:
            c = result.chains[ci]
            if c.fork is None or c.fork.parent_chain is None or c.fork.depth == 0:
                return ci
            ci = c.fork.parent_chain

    groups: dict[int, list[int]] = {}
    for ci in live:
        groups.setdefault(_group_root(ci), []).append(ci)

    prefixes: dict[int, int] = {}
    for members in groups.values():
        observed = _observed_group_prefix(result, members)
        for ci in members:
            if ci == result.main_index:
                prefixes[ci] = max(declared_prefix_blocks, observed)
            else:
                prefixes[ci] = observed
    return prefixes


def _last_hash_outer_idx(chain: AgentChain) -> int | None:
    """Outer index of the chain's last HASH-BEARING request.

    Empty-hash rows carry no LCP evidence and must stay invisible to
    detection (spec §8), so they cannot make a dead tail look "extended"."""
    for oi, req in reversed(chain.requests):
        if req.hash_ids:
            return oi
    return None


def _elect_continuation(
    chains: list[AgentChain],
    registered: list[int],
    t_req: _NormalRequestT,
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int | None:
    """Pick the seam continuation among forks registered on a dead tail.

    Eligibility: positive depth, no temporal overlap with the tail, same
    model. Election: deepest LCP, tie-break earliest fork_time, then lowest
    chain index.

    Seam guard (``max_gap_seconds`` / ``min_overlap_ratio``): a genuine context
    compaction continues promptly, so a candidate that starts more than
    ``max_gap_seconds`` after the tail ends AND shares less than
    ``min_overlap_ratio`` of the tail's blocks is rejected -- it is a distinct
    session that merely reuses the base prefix, and stitching it on would
    fabricate a multi-hour intra-conversation idle gap. Both conditions must
    hold, so prompt compactions (any overlap) and verbatim long-gap resumes
    (high overlap) still elect."""
    t_end = _req_end(t_req)
    tail_blocks = len(t_req.hash_ids)

    def _seam_blocked(ci: int) -> bool:
        if tail_blocks == 0:
            return False
        gap = chains[ci].requests[0][1].t - t_end
        overlap = chains[ci].fork.depth / tail_blocks
        return gap > max_gap_seconds and overlap < min_overlap_ratio

    candidates = [
        ci
        for ci in registered
        if chains[ci].fork is not None
        and chains[ci].fork.depth > 0
        and t_end <= chains[ci].requests[0][1].t + _EPSILON_SECONDS
        and chains[ci].requests[0][1].model == t_req.model
        and not _seam_blocked(ci)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda ci: (
            chains[ci].fork.depth,
            -chains[ci].fork.fork_time,
            -ci,
        ),
    )


def _rekey_leftover_forks(
    *,
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    req_by_outer: dict[int, _NormalRequestT],
    registered: list[int],
    elected: int,
    owner: int,
    new_tail_outer: int,
) -> bool:
    """Re-evaluate non-elected forks against the merged chain's new tail.

    A fork's LCP relationship is to the evolving conversation state, not to
    one frozen snapshot — without re-keying, phase-1's deepest-tail
    tie-break can register a later compaction continuation under an
    already-consumed tail and strand it as a spurious spawn. Returns True
    when any fork was re-keyed (the caller re-arms that tail's key)."""
    new_tail_hash = np.asarray(req_by_outer[new_tail_outer].hash_ids, np.int64)
    rekeyed = False
    for ci in registered:
        c = chains[ci]
        if ci == elected or c.spliced_into is not None or c.fork is None:
            continue
        d = _np_lcp(
            new_tail_hash, np.asarray(c.requests[0][1].hash_ids, dtype=np.int64)
        )
        if d <= 0:
            continue
        c.fork.fork_outer_idx = new_tail_outer
        c.fork.depth = d
        c.fork.parent_chain = owner
        forks_by_tail.setdefault(new_tail_outer, []).append(ci)
        rekeyed = True
    return rekeyed


def _resolve_seams(
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    chain_of_request: dict[int, int],
    req_by_outer: dict[int, _NormalRequestT],
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int:
    """Phase 2: splice join-seam continuations onto dead tails.

    For each fork-source request T in time order: if T is still the final
    hash-bearing request of its (post-splice) chain, elect among its
    temporally-feasible same-model forks the deepest one (tie: earliest
    fork_time, then lowest index) as the same agent's continuation and
    splice it on. Everything else stays a spawn.

    Cascades: a splice moves the chain's tail to a later request, whose own
    forks are visited later in the worklist, and the non-elected forks of
    the consumed tail are re-keyed against the merged chain's new tail (see
    :func:`_rekey_leftover_forks`)."""
    alias: dict[int, int] = {}

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    seams = 0
    keys = sorted(forks_by_tail)
    heapq.heapify(keys)
    processed: set[int] = set()
    while keys:
        fork_outer_idx = heapq.heappop(keys)
        if fork_outer_idx in processed:
            continue
        processed.add(fork_outer_idx)
        owner = _resolve(chain_of_request[fork_outer_idx])
        owner_chain = chains[owner]
        if _last_hash_outer_idx(owner_chain) != fork_outer_idx:
            continue  # longer state was extended -> all forks are spawns
        registered = [
            ci
            for ci in forks_by_tail[fork_outer_idx]
            if chains[ci].spliced_into is None
        ]
        elected = _elect_continuation(
            chains,
            registered,
            req_by_outer[fork_outer_idx],
            max_gap_seconds=max_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
        )
        if elected is None:
            continue
        target = chains[elected]
        owner_chain.requests.extend(target.requests)
        for oi, _ in target.requests:
            chain_of_request[oi] = owner
        target.spliced_into = owner
        alias[elected] = owner
        seams += 1

        new_tail_outer = _last_hash_outer_idx(owner_chain)
        if new_tail_outer is None:
            continue
        if _rekey_leftover_forks(
            chains=chains,
            forks_by_tail=forks_by_tail,
            req_by_outer=req_by_outer,
            registered=registered,
            elected=elected,
            owner=owner,
            new_tail_outer=new_tail_outer,
        ):
            # The new tail's key may sort below the current one or may have
            # been visited already as a no-op; allow it to (re)process so
            # the re-keyed candidates get their election.
            processed.discard(new_tail_outer)
            heapq.heappush(keys, new_tail_outer)
    return seams
