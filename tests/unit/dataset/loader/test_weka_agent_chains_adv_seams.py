# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for phase-2 seam resolution (``_resolve_seams``).

Attack surface: the join-seam vs spawn election (spec section 4 "Phase 2").
Every test encodes the spec's expected behavior and throws a hostile input
at the detector. Helpers mirror the reference suite
(``test_weka_agent_chains.py``); the reference file is never modified.
"""

from aiperf.dataset.loader.weka_agent_chains import (
    _EPSILON_SECONDS,
    detect_agent_chains,
)
from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest


def _req(
    t: float, hash_ids: list[int], api_time: float = 1.0, model: str = "m"
) -> WekaNormalRequest:
    return WekaNormalRequest(
        type="n",
        t=t,
        model=model,
        input_length=len(hash_ids) * 64,
        output_length=10,
        hash_ids=hash_ids,
        api_time=api_time,
    )


def _normals(*reqs: WekaNormalRequest) -> list[tuple[int, WekaNormalRequest]]:
    return list(enumerate(reqs))


def _chain_outer_indices(result, chain_index: int) -> list[int]:
    return [oi for oi, _ in result.chains[chain_index].requests]


def _all_emitted_outer_indices(result) -> list[int]:
    """Every retained request as emitted: main chain + every worker chain.

    Spliced (dead) chains are excluded — their requests are owned by the
    chain they were spliced into, so each request appears exactly once.
    """
    out = [oi for oi, _ in result.chains[result.main_index].requests]
    for ci in result.worker_indices:
        out.extend(oi for oi, _ in result.chains[ci].requests)
    return out


# --------------------------------------------------------------------------
# Election tie-breaks: equal depth -> earliest fork_time; then lowest index.
# --------------------------------------------------------------------------


def test_election_equal_depth_earliest_fork_time_wins():
    # Two forks off the same dead tail, EQUAL depth (2). Spec phase 2:
    # tie-break is earliest fork_time, so fork B (t=2.0) is elected FIRST
    # and splices before fork A (t=3.0). The seam cascade then re-evaluates
    # A against the merged chain's new tail [1,2,80,81]: that longer state
    # is also dead (no future pullback) and A keeps its [1,2] prefix, so by
    # the governing rule A is a SECOND compaction continuation, not a spawn.
    # The election order is observable in the splice order: B lands before A.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M1, ends t=0.5
            _req(3.0, [1, 2, 90, 91], api_time=0.1),  # fork A, fork_time 3.0
            _req(2.0, [1, 2, 80, 81], api_time=0.1),  # fork B, fork_time 2.0
        )
    )
    assert r.seams_merged == 2
    # Fork B (outer_idx 2, earlier fork_time) spliced first, then A (outer 1).
    assert _chain_outer_indices(r, r.main_index) == [0, 2, 1]
    assert r.worker_indices == []


def test_election_equal_depth_equal_fork_time_lowest_index_wins():
    # Two forks off one dead tail with EQUAL depth (2) AND EQUAL fork_time
    # (t=3.0). Spec final tie-break: lowest chain index. Phase 1 processes in
    # (t, outer_idx) order so the earlier-outer fork gets the lower chain
    # index and must be elected.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(3.0, [1, 2, 90, 91], api_time=0.5),  # outer 1 -> chain idx 1
            _req(3.0, [1, 2, 80, 81], api_time=0.5),  # outer 2 -> chain idx 2
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]  # lowest index won
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [2]


# --------------------------------------------------------------------------
# The "T must be final" rule and depth-recorded-at-fork-time.
# --------------------------------------------------------------------------


def test_t_extended_after_fork_makes_fork_a_spawn():
    # Fork recorded from M1, then M1's chain extended AFTER (M2 pulls back to
    # the longer state). Spec phase 2: T not final -> every fork is a spawn.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M1
            _req(2.0, [1, 2, 90, 91], api_time=0.5),  # fork off M1, depth 2
            _req(4.0, [1, 2, 3, 4, 5, 6, 7], api_time=0.5),  # M2 extends M1
        )
    )
    assert r.seams_merged == 0
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_fork_depth_is_recorded_against_tail_at_fork_time():
    # Spec section 4 note: forking records the tail request T at fork time; a
    # spawn whose true ancestor is an older mid-chain state still yields the
    # correct depth because the tail extends that older state.
    # Main: M1[1,2,3] -> M2[1,2,3,4,5]. Fork diverges at block 3 (depth 3) and
    # the tail at fork time is M2 (outer_idx 1), not M1. A later M3 keeps the
    # fork a spawn so we can read the recorded depth.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),  # M1
            _req(2.0, [1, 2, 3, 4, 5], api_time=0.5),  # M2 (tail at fork time)
            _req(4.0, [1, 2, 3, 80], api_time=0.5),  # fork: depth 3 vs M2
            _req(6.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M3 keeps fork a spawn
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.depth == 3
    assert worker.fork.fork_outer_idx == 1  # M2, the tail at fork time


# --------------------------------------------------------------------------
# Cascading splices and alias resolution.
# --------------------------------------------------------------------------


def test_cascaded_three_compactions_collapse_to_one_chain():
    # Three consecutive compactions: M1 -> C2 -> C3 -> C4. Each shrink's
    # longer state is dead, so all three seams splice into one chain (spec
    # section 11 test 7). Hostile twist: C2 keeps a longer tail (len 6) than
    # C3 (len 5), so phase-1's deeper-tail tie-break registers C4's fork
    # against C2's tail — a slot consumed before C3 splices in. Phase 2's
    # re-keying (non-elected forks re-evaluated against the merged chain's
    # new tail) is what rescues C4 from being stranded as a spurious spawn.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M1
            _req(2.0, [1, 2, 70, 71, 72, 73], api_time=0.5),  # C2 (depth 2)
            _req(4.0, [1, 2, 70, 80, 81], api_time=0.5),  # C3 (depth 3 off C2)
            _req(6.0, [1, 2, 70, 90], api_time=0.5),  # C4 (depth 3 off C3)
        )
    )
    assert r.seams_merged == 3
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2, 3]


def test_cascaded_three_compactions_collapse_when_tails_grow():
    # Control for the re-keying case above: same three-compaction shape, but each
    # compaction's tail is LONGER than the prior, so C4's fork registers
    # against C3's tail (the later-processed slot). The cascade then works
    # exactly as the spec promises -> one chain, three seams. This isolates
    # the bug to the phase-1 fork-registration tie-break, not the cascade
    # mechanism itself.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4], api_time=0.5),  # M1 (tail len 4)
            _req(2.0, [1, 2, 70, 71], api_time=0.5),  # C2 (depth 2, tail len 4)
            _req(4.0, [1, 2, 70, 80, 81, 82, 83], api_time=0.5),  # C3 deeper tail
            _req(6.0, [1, 2, 70, 90], api_time=0.5),  # C4 forks off C3's tail
        )
    )
    assert r.seams_merged == 3
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2, 3]


def test_live_worker_fork_parent_rewritten_to_live_chain_after_splice():
    # A LIVE worker forks off C2 (chain idx 1). C2 is itself a compaction-seam
    # of M1 and gets spliced into main (chain idx 0). Spec section 4: phase 2
    # rewrites fork.parent_chain to live (post-splice) chains, so the live
    # worker's recorded parent must resolve from the dead chain 1 to chain 0.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M1, ends 0.5
            _req(2.0, [1, 2, 80, 81, 82, 83], api_time=5.0),  # C2 seam, runs 2..7
            _req(3.0, [1, 2, 80, 81, 70], api_time=1.0),  # worker off C2, overlaps
            _req(9.0, [1, 2, 80, 81, 82, 83, 99], api_time=1.0),  # extends C2
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 3]
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    # chain 1 was spliced into chain 0; the worker's parent must follow.
    assert worker.spliced_into is None
    assert worker.fork is not None
    assert worker.fork.parent_chain == r.main_index


def test_cascade_independent_of_file_order_when_outer_precedes_time():
    # File order (outer_idx) deliberately diverges from time order so the
    # phase-2 iteration (sorted by fork_outer_idx) visits a dependent fork's
    # source BEFORE its prerequisite splice. Spec section 4 promises cascades
    # collapse regardless; the alias map must make the result order-invariant.
    # Time order A(t0) -> B(t1 seam of A) -> D(t2 seam of B); outer A=2,B=0,D=1.
    reqs = [
        _req(1.0, [1, 2, 80, 81, 82, 83], api_time=0.5),  # idx0 = B
        _req(2.0, [1, 2, 80, 70], api_time=0.5),  # idx1 = D
        _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # idx2 = A (root)
    ]
    r = detect_agent_chains(list(enumerate(reqs)))
    assert r.seams_merged == 2
    assert r.worker_indices == []
    # Emitted in time order within the single chain: A, B, D.
    assert _chain_outer_indices(r, r.main_index) == [2, 0, 1]


# --------------------------------------------------------------------------
# Temporal veto at the epsilon boundary.
# --------------------------------------------------------------------------


def test_temporal_veto_seam_allowed_exactly_at_epsilon_boundary():
    # Spec phase 2 (b): continuation feasible iff end(T) <= start(C) + eps.
    # Construct end(T) == start(C) + eps exactly -> the boundary is inclusive,
    # so the seam IS elected.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=10.0),  # ends t=10.0
            _req(10.0 - _EPSILON_SECONDS, [1, 2, 90, 91], api_time=0.1),
        )
    )
    assert r.seams_merged == 1
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_temporal_veto_spawn_just_past_epsilon_boundary():
    # One epsilon past the boundary: end(T) > start(C) + eps -> the longer
    # state overlaps the continuation candidate -> spawn even though dead.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=10.0),  # ends t=10.0
            _req(10.0 - 2 * _EPSILON_SECONDS, [1, 2, 90, 91], api_time=0.1),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_zero_api_time_tail_allows_seam_at_equal_timestamp():
    # api_time=0 -> end(T)==start(T). A continuation at the same timestamp
    # satisfies end(T) <= start(C) + eps and seams. Boundary of the
    # zero-api_time interval.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.0),  # ends t=0
            _req(0.0, [1, 2, 90, 91], api_time=0.0),  # starts t=0
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


# --------------------------------------------------------------------------
# Same-model rule in phase 2.
# --------------------------------------------------------------------------


def test_deepest_fork_cross_model_shallower_same_model_elected():
    # Deepest fork (depth 4) is cross-model -> excluded by phase 2 (c).
    # The shallower (depth 2) SAME-model fork is elected as the seam.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus", api_time=0.5),
            _req(2.0, [1, 2, 3, 4, 80, 81], model="haiku", api_time=0.5),  # deep
            _req(3.0, [1, 2, 90], model="opus", api_time=0.5),  # shallow same-model
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_all_candidates_cross_model_no_splice_at_all():
    # Every feasible fork off the dead tail is a different model -> phase 2
    # elects nothing; all forks remain spawns.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus", api_time=0.5),
            _req(2.0, [1, 2, 90, 91], model="haiku", api_time=0.5),
            _req(3.0, [1, 2, 80, 81], model="sonnet", api_time=0.5),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 2
    assert _chain_outer_indices(r, r.main_index) == [0]


# --------------------------------------------------------------------------
# In-flight sibling at depth == len(tail): never electable.
# --------------------------------------------------------------------------


def test_in_flight_full_prefix_sibling_never_seams():
    # M1 runs t=[10,30]. r reads M1's FULL hash list while M1 is still in
    # flight (overlap). Spec section 4: full-prefix LCP but temporal overlap
    # -> fork at depth == len(tail) -> spawn. Phase 2 must never splice it
    # back: even though M1 is final, the temporal veto (end(M1)=30 > start(r))
    # blocks the seam.
    r = detect_agent_chains(
        _normals(
            _req(10.0, [1, 2, 3], api_time=20.0),  # M1 runs 10..30
            _req(15.0, [1, 2, 3, 7], api_time=0.5),  # full-prefix sibling
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.depth == 3  # depth == len(tail) by construction


def test_full_prefix_sibling_near_miss_extension_at_exact_end():
    # Near-miss to the in-flight sibling: the same full-prefix request now
    # starts EXACTLY when the tail ends (end(M1)=2.0 == start(r)). The
    # extension check end(tail) <= start + eps now passes -> EXTENSION, one
    # chain, never a fork. Confirms the in-flight veto is purely temporal.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=2.0),  # ends t=2.0
            _req(2.0, [1, 2, 3, 4], api_time=0.5),  # starts t=2.0
        )
    )
    assert r.seams_merged == 0
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


# --------------------------------------------------------------------------
# Spawn whose own tail dies and absorbs its own seam continuation.
# --------------------------------------------------------------------------


def test_worker_compacts_midlife_absorbs_own_seam_main_unaffected():
    # Main stays alive. A worker forks (depth 2), grows, then compacts itself.
    # The worker's own continuation must splice onto the WORKER chain (it is
    # the post-splice tail owner), leaving main untouched. Spec section 4:
    # seams elect against the fork-source's own chain.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),  # M1
            _req(2.0, [1, 2, 50, 51, 52, 53], api_time=0.5),  # W1 fork depth 2
            _req(3.0, [1, 2, 50, 51, 52, 53, 54], api_time=0.5),  # W2 extends W1
            _req(5.0, [1, 2, 3, 4, 5], api_time=0.5),  # M2 keeps main alive
            _req(6.0, [1, 2, 70, 71], api_time=0.5),  # W3 worker self-compaction
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 3]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1, 2, 4]


# --------------------------------------------------------------------------
# worker_indices ordering / seams_merged counting / spliced bookkeeping.
# --------------------------------------------------------------------------


def test_worker_indices_ordered_by_first_request_time_not_outer():
    # Two live workers whose first-request times invert their outer order.
    # Spec phase 3 / result invariant: worker_indices is ordered by
    # first-request (t, outer_idx), so the earlier-starting worker comes first.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),  # main
            _req(5.0, [1, 2, 50, 51], api_time=10.0),  # worker A first t=5
            _req(2.0, [1, 2, 60, 61], api_time=10.0),  # worker B first t=2
            _req(9.0, [1, 2, 3, 4], api_time=0.5),  # main grows
        )
    )
    firsts = [r.chains[i].requests[0][0] for i in r.worker_indices]
    assert firsts == [2, 1]  # worker B (t=2) before worker A (t=5)


def test_spliced_chains_excluded_from_workers_but_present_in_chains():
    # Invariant from ChainDetectionResult docstring + spec section 4: spliced
    # (dead) chains stay in `chains` for fork-history logging, carry
    # `spliced_into`, and are excluded from `worker_indices`.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M1
            _req(2.0, [1, 2, 90, 91], api_time=0.5),  # compaction seam of M1
        )
    )
    assert r.seams_merged == 1
    assert r.worker_indices == []
    spliced = [i for i, c in enumerate(r.chains) if c.spliced_into is not None]
    assert spliced == [1]  # chain 1 still present in chains
    assert r.chains[1].spliced_into == r.main_index
    assert 1 not in r.worker_indices


def test_every_retained_request_appears_in_exactly_one_chain_once():
    # Cross-cutting invariant: across main + worker chains, every retained
    # request appears exactly once (spliced chains contribute nothing extra).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # M1
            _req(2.0, [1, 2, 70, 71, 72, 73], api_time=0.5),  # seam of M1
            _req(3.0, [1, 2, 70, 80], api_time=0.5),  # cascade seam
            _req(4.0, [1, 2, 50, 51], api_time=5.0),  # worker (overlaps)
            _req(5.0, [1, 2, 50, 51, 52], api_time=0.5),  # worker turn 2
            _req(9.0, [1, 2, 70, 80, 90], api_time=0.5),  # main grows past seams
        )
    )
    emitted = sorted(_all_emitted_outer_indices(r))
    assert emitted == [0, 1, 2, 3, 4, 5]
    assert len(emitted) == len(set(emitted))  # no duplicates


def test_seams_merged_counts_one_per_splice_across_independent_tails():
    # Two independent dead tails (disjoint namespaces), each with one elected
    # seam. seams_merged must total exactly 2 (one per splice), not collapse.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),  # main M1
            _req(2.0, [1, 2, 90, 91], api_time=0.5),  # seam of main
            _req(1.0, [100, 101, 102, 103], api_time=0.5),  # disjoint batch founder
            _req(3.0, [100, 101, 80, 81], api_time=0.5),  # seam of disjoint batch
        )
    )
    assert r.seams_merged == 2
    # Main absorbed its seam; the disjoint batch is one live worker that
    # absorbed its own seam.
    assert _chain_outer_indices(r, r.main_index) == [0, 1]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [2, 3]


# --------------------------------------------------------------------------
# Seam guard: a far-future low-overlap continuation is a distinct session
# (spawn), not the same agent resuming. Prompt compactions and high-overlap
# long-gap resumes are preserved.
# --------------------------------------------------------------------------


def test_seam_guard_splits_far_low_overlap_continuation():
    # Tail [1,2,3,4] dies; a request 10000s later shares only [1] (overlap
    # 0.25 < 0.5) — a distinct session reusing the base prefix, not a
    # compaction. With the default guard (gap 3600s / overlap 0.5) it stays
    # its own worker chain instead of being spliced onto main.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4], api_time=0.5),
            _req(10000.0, [1, 9], api_time=0.5),
        )
    )
    assert r.seams_merged == 0
    assert _chain_outer_indices(r, r.main_index) == [0]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_seam_guard_keeps_near_low_overlap_compaction():
    # Same low overlap (0.25), but the continuation fires 2s later — a real
    # mid-session compaction. The guard's gap condition is not met, so it
    # splices onto main (one chain, no worker).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4], api_time=0.5),
            _req(2.0, [1, 9], api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]
    assert len(r.worker_indices) == 0


def test_seam_guard_keeps_far_high_overlap_resume():
    # 10000s gap but the resume keeps [1,2,3] of the [1,2,3,4] tail (overlap
    # 0.75 >= 0.5) — a verbatim resume of the same conversation. Not blocked.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4], api_time=0.5),
            _req(10000.0, [1, 2, 3, 9], api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]
    assert len(r.worker_indices) == 0


def test_seam_guard_disabled_by_zero_overlap_threshold_merges():
    # Escape hatch: min_overlap_ratio=0.0 disables the overlap half, so even a
    # far low-overlap join splices on (pre-guard behavior).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4], api_time=0.5),
            _req(10000.0, [1, 9], api_time=0.5),
        ),
        seam_min_overlap_ratio=0.0,
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]
    assert len(r.worker_indices) == 0
