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
    """Interval end in seconds; missing/negative api_time counts as zero."""
    return req.t + max(req.api_time or 0.0, 0.0)


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
        target = _find_extension_target(self.chains, h, req.t, req.model)
        if target is not None:
            self._append(target, outer_idx, req)
            return

        parent, depth = _max_lcp_chain(self.chains, h)
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


def detect_agent_chains(normals: list[IndexedRequest]) -> ChainDetectionResult:
    """Partition retained top-level requests into per-agent chains."""
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
        chains, state.forks_by_tail, state.chain_of_request, state.req_by_outer
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


@dataclass(slots=True, frozen=True)
class MetricRecord:
    """One request's contribution to the per-trace shared seen-set."""

    sort_key: tuple[float, int, int, int]
    """(absolute_t, outer_idx, stream_idx, k) — deterministic global order."""
    session_id: str
    """Conversation the value is looked up under at emission time."""
    k: int
    """Turn index within that conversation."""
    hash_ids: list[int]
    """The request's input hash blocks."""


def compute_shared_prefix_cache_metrics(
    records: list[MetricRecord],
) -> dict[tuple[str, int], tuple[int, int]]:
    """{(session_id, k): (hit_blocks, total_blocks)} over ONE shared
    per-trace seen-set, consumed in global time order (spec §5.5)."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    seen: set[int] = set()
    for rec in sorted(records, key=lambda r: r.sort_key):
        hits = 0
        for hid in rec.hash_ids:
            if hid not in seen:
                break
            hits += 1
        out[(rec.session_id, rec.k)] = (hits, len(rec.hash_ids))
        seen.update(rec.hash_ids)
    return out


def looks_hash_poisoned(
    result: ChainDetectionResult, *, min_chains: int = 8, ratio: float = 0.5
) -> bool:
    """True when detection is dominated by zero-depth chain founders —
    the signature of per-request-nonce-poisoned chained block hashes
    (spec §8). Legitimate disjoint-namespace batches produce only a few
    zero-depth founders and stay far below the threshold."""
    total = len(result.chains)
    if total < min_chains:
        return False
    zero = sum(1 for c in result.chains if c.fork is not None and c.fork.depth == 0)
    return zero / total > ratio


def _resolve_seams(
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    chain_of_request: dict[int, int],
    req_by_outer: dict[int, _NormalRequestT],
) -> int:
    """Phase 2: splice join-seam continuations onto dead tails.

    For each fork-source request T in time order: if T is still the final
    request of its (post-splice) chain, elect among its temporally-feasible
    forks the deepest one (tie: earliest fork_time, then lowest index) as
    the same agent's continuation and splice it on. Everything else stays a
    spawn. Cascades fall out of the forward pass: a splice moves the chain's
    tail to a later request, whose own forks are visited later."""
    alias: dict[int, int] = {}

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    seams = 0
    for fork_outer_idx in sorted(forks_by_tail):
        owner = _resolve(chain_of_request[fork_outer_idx])
        owner_chain = chains[owner]
        if owner_chain.requests[-1][0] != fork_outer_idx:
            continue  # longer state was extended -> all forks are spawns
        t_req = req_by_outer[fork_outer_idx]
        t_end = _req_end(t_req)
        candidates = [
            ci
            for ci in forks_by_tail[fork_outer_idx]
            if chains[ci].fork is not None
            and chains[ci].fork.depth > 0
            and t_end <= chains[ci].requests[0][1].t + _EPSILON_SECONDS
            and chains[ci].requests[0][1].model == t_req.model
        ]
        if not candidates:
            continue
        elected = max(
            candidates,
            key=lambda ci: (
                chains[ci].fork.depth,
                -chains[ci].fork.fork_time,
                -ci,
            ),
        )
        target = chains[elected]
        owner_chain.requests.extend(target.requests)
        for oi, _ in target.requests:
            chain_of_request[oi] = owner
        target.spliced_into = owner
        alias[elected] = owner
        seams += 1
    return seams
