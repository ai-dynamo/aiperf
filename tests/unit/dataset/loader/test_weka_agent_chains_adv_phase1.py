# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for phase-1 greedy chain building in detect_agent_chains.

Targets the sharp edges of spec section 4 "Phase 1" (design spec
2026-06-10-weka-flattened-agent-lcp-detection-design.md): the 1e-6 temporal
epsilon boundary, extension and fork-witness tie-breaks (deepest tail vs
lowest index), input ordering by (t, outer_idx), degenerate api_time values,
the same-model rule among multiple extension candidates, and the spec
section 8 empty-hash invisibility rules.
"""

import pytest

from aiperf.dataset.loader.weka_agent_chains import detect_agent_chains
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
)


def _req(
    t: float,
    hash_ids: list[int],
    api_time: float | None = 1.0,
    model: str = "m",
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


def _sreq(
    t: float,
    hash_ids: list[int],
    api_time: float | None = 1.0,
    model: str = "m",
) -> WekaStreamingRequest:
    return WekaStreamingRequest(
        type="s",
        t=t,
        model=model,
        input_length=len(hash_ids) * 64,
        output_length=10,
        hash_ids=hash_ids,
        api_time=api_time,
    )


def _normals(*reqs) -> list[tuple[int, WekaNormalRequest | WekaStreamingRequest]]:
    return list(enumerate(reqs))


def _chain_outer_indices(result, chain_index: int) -> list[int]:
    return [oi for oi, _ in result.chains[chain_index].requests]


def _worker_by_first_outer(result) -> dict[int, int]:
    """Map each worker chain's first request outer_idx -> chain index."""
    return {result.chains[i].requests[0][0]: i for i in result.worker_indices}


# ---------------------------------------------------------------------------
# Temporal epsilon boundary (spec 4 phase 1: end(tail) <= start(r) + eps)
# ---------------------------------------------------------------------------


def test_detect_agent_chains_overlap_within_epsilon_extends():
    # Tail ends at t=2.0; next request starts 5e-7 earlier — overlap smaller
    # than eps must still count as a sequential extension.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=2.0),
            _req(2.0 - 5e-7, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_detect_agent_chains_overlap_beyond_epsilon_forks_at_full_tail_depth():
    # Overlap of 2e-6 > eps: full-prefix match is an in-flight sibling.
    # Spec 4 step 2: falls through to the fork case at depth == len(tail),
    # and phase 2 must not splice it back (same temporal veto).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=2.0),
            _req(2.0 - 2e-6, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.depth == 3
    assert worker.fork.parent_chain == r.main_index


def test_detect_agent_chains_tail_end_exactly_equal_to_start_extends():
    # end(tail) == start(r) exactly: zero idle gap is a legal continuation.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=2.0),
            _req(2.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


@pytest.mark.parametrize("api_time", [0.0, None, -5.0])
def test_detect_agent_chains_degenerate_api_time_zero_duration_tail_extends(
    api_time: float | None,
):
    # Spec 3: end(r) = t + max(api_time or 0, 0). Zero, missing, and negative
    # api_time all yield a zero-duration tail that never overlaps a request
    # arriving at the same instant.
    r = detect_agent_chains(
        _normals(
            _req(2.0, [1, 2, 3], api_time=api_time),
            _req(2.0, [1, 2, 3, 4], api_time=0.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


# ---------------------------------------------------------------------------
# Extension target tie-breaks (deepest tail, then lowest chain index)
# ---------------------------------------------------------------------------


def test_detect_agent_chains_equal_length_tails_extension_lowest_index_wins():
    # Identical duplicate requests (same t-window) force two chains with the
    # exact same tail [1,2,3]. A later extension must land on the
    # lowest-index chain (spec 4 step 1 tie-break).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=50.0),
            _req(1.0, [1, 2, 3], api_time=50.0),  # in-flight retry -> fork
            _req(60.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_detect_agent_chains_deeper_tail_at_higher_index_wins_extension():
    # Chain 0 tail [1,2] (shallow, low index) vs chain 1 tail [1,2,3]
    # (deep, high index): deepest-tail-wins must beat lowest-index.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=10.0),
            _req(1.0, [1, 2, 3], api_time=1.0),  # overlaps main -> fork
            _req(20.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1, 2]


def test_detect_agent_chains_deeper_cross_model_tail_skipped_for_extension():
    # The deepest full-prefix tail is the wrong model; the same-model rule
    # must exclude it BEFORE the deepest-tail preference applies.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.1, model="opus"),
            _req(1.0, [1, 2, 3, 4], api_time=0.1, model="haiku"),
            _req(2.0, [1, 2, 3, 4, 5], api_time=0.1, model="opus"),
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    haiku = r.chains[r.worker_indices[0]]
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]
    assert haiku.fork is not None
    assert haiku.fork.depth == 3  # cross-model full-prefix forks at len(tail)


def test_detect_agent_chains_in_flight_deeper_tail_skipped_shallower_extends():
    # The deeper full-prefix tail is still executing; the temporal veto must
    # exclude it, leaving the shallower (ended) tail as the extension target.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=2.0),
            _req(1.0, [1, 2, 3], api_time=100.0),  # overlaps main -> fork
            _req(5.0, [1, 2, 3, 4], api_time=1.0),  # chain 1 still running
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_detect_agent_chains_last_element_match_full_mismatch_not_extension():
    # h shares its element at index len(tail)-1 with the tail but differs
    # earlier — a hostile input for any last-element fast-path. Must fork as
    # a disjoint founder (LCP 0), never extend.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(2.0, [7, 8, 3, 4], api_time=0.5),
        )
    )
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.parent_chain is None
    assert worker.fork.depth == 0


# ---------------------------------------------------------------------------
# Fork-witness tie-breaks (spec 4 step 2: deepest LCP, deeper tail, low idx)
# ---------------------------------------------------------------------------


def test_detect_agent_chains_equal_lcp_deeper_tail_wins_fork_witness():
    # Two chains share LCP=2 with the new request; the deeper tail (higher
    # index) must be recorded as the fork parent.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=100.0),  # chain 0: shallow, in-flight
            _req(1.0, [1, 2, 3, 4, 5], api_time=0.1),  # chain 1: deep tail
            _req(1.05, [1, 2, 99], api_time=0.1),  # LCP 2 with both
        )
    )
    assert len(r.worker_indices) == 2
    by_first = _worker_by_first_outer(r)
    deep = by_first[1]
    forked = by_first[2]
    assert r.chains[forked].fork is not None
    assert r.chains[forked].fork.parent_chain == deep
    assert r.chains[forked].fork.depth == 2


def test_detect_agent_chains_equal_lcp_equal_tails_fork_witness_lowest_index():
    # Both chains have the identical tail [1,2,3] (equal LCP, equal tail
    # length): the lowest-index chain must be the fork witness.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=50.0),
            _req(1.0, [1, 2, 3], api_time=50.0),  # in-flight retry -> fork
            _req(2.0, [1, 2, 99], api_time=0.1),
        )
    )
    assert len(r.worker_indices) == 2
    by_first = _worker_by_first_outer(r)
    forked = by_first[2]
    assert r.chains[forked].fork is not None
    assert r.chains[forked].fork.parent_chain == r.main_index
    assert r.chains[forked].fork.depth == 2


def test_detect_agent_chains_fork_depth_matches_older_midchain_ancestor():
    # A spawn diverging from an OLD mid-chain state still gets the right
    # depth off the current tail (spec 4: LCP(r, tail) == LCP(r, old state)
    # when the divergence point precedes both).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=0.1),
            _req(1.0, [1, 2, 3, 4], api_time=0.1),  # main grows
            _req(1.5, [1, 2, 99], api_time=0.1),  # diverges at the OLD state
            _req(2.0, [1, 2, 3, 4, 5], api_time=0.1),  # pullback -> spawn
        )
    )
    assert r.seams_merged == 0
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 3]
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.parent_chain == r.main_index
    assert worker.fork.depth == 2


# ---------------------------------------------------------------------------
# Input ordering (spec 4: process in (t, outer_idx) order)
# ---------------------------------------------------------------------------


def test_detect_agent_chains_unsorted_input_processed_in_time_order():
    # File order is hostile (t descending-ish); the function must sort by
    # (t, outer_idx) and recover one clean chain in t order.
    normals = [
        (0, _req(4.0, [1, 2, 3, 4, 5], api_time=0.5)),
        (1, _req(0.0, [1, 2, 3], api_time=0.5)),
        (2, _req(2.0, [1, 2, 3, 4], api_time=0.5)),
    ]
    r = detect_agent_chains(normals)
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [1, 2, 0]


def test_detect_agent_chains_equal_t_ties_broken_by_outer_idx():
    # Two requests at the same t, passed in reversed list order: outer_idx
    # must decide processing order, making [1,2,3] the founder and
    # [1,2,3,4] its extension — a single chain.
    normals = [
        (1, _req(1.0, [1, 2, 3, 4], api_time=0.0)),
        (0, _req(1.0, [1, 2, 3], api_time=0.0)),
    ]
    r = detect_agent_chains(normals)
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


# ---------------------------------------------------------------------------
# Empty-hash requests (spec 8: invisible, kept on main, counted)
# ---------------------------------------------------------------------------


def test_detect_agent_chains_first_request_empty_hash_keeps_single_chain():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [], api_time=0.5),
            _req(1.0, [1, 2, 3], api_time=0.5),
            _req(2.0, [1, 2, 3, 4], api_time=0.5),
        )
    )
    assert r.unclassified_empty_hash == 1
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_detect_agent_chains_mid_trace_empty_hash_rows_counted_and_invisible():
    # Consecutive empty-hash rows stay on the main chain in t order and the
    # next hash-bearing request extends the last REAL tail across them.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=0.1),
            _req(1.0, [], api_time=0.1),
            _req(2.0, [], api_time=0.1),
            _req(3.0, [1, 2, 3], api_time=0.1),
        )
    )
    assert r.worker_indices == []
    assert r.unclassified_empty_hash == 2
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2, 3]


def test_detect_agent_chains_empty_hash_between_fanout_turns_breaks_neither_chain():
    # An empty-hash row interleaved between a main turn and a worker turn
    # must not break either chain's extension continuity.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=1.0),  # main turn 0, ends t=1
            _req(2.0, [1, 2, 50, 51], api_time=10.0),  # worker, ends t=12
            _req(3.0, [], api_time=0.1),  # empty row, lands on main
            _req(4.0, [1, 2, 3, 4], api_time=1.0),  # main turn 1
            _req(13.0, [1, 2, 50, 51, 52], api_time=1.0),  # worker turn 2
        )
    )
    assert r.seams_merged == 0
    assert r.unclassified_empty_hash == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2, 3]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1, 4]


def test_detect_agent_chains_empty_hash_after_dead_tail_does_not_demote_seam():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=1.0),
            _req(2.0, [], api_time=0.1),  # no hash evidence: not a pullback
            _req(4.0, [1, 2, 90, 91], api_time=1.0),  # compaction shrink
        )
    )
    assert r.unclassified_empty_hash == 1
    assert r.seams_merged == 1
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Degenerate traces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hash_ids", "expected_unclassified"),
    [([1, 2, 3], 0), ([], 1)],
)
def test_detect_agent_chains_single_request_trace_yields_single_main_chain(
    hash_ids: list[int], expected_unclassified: int
):
    r = detect_agent_chains(_normals(_req(0.0, hash_ids, api_time=0.5)))
    assert len(r.chains) == 1
    assert r.worker_indices == []
    # Spec 4 step 2: "no chains yet" founds with fork record
    # (None, None, 0, t); the empty-hash path founds with fork=None.
    # Either way the main chain must have no parent and zero depth.
    main = r.chains[r.main_index]
    assert main.fork is None or (
        main.fork.parent_chain is None and main.fork.depth == 0
    )
    assert _chain_outer_indices(r, r.main_index) == [0]
    assert r.unclassified_empty_hash == expected_unclassified


def test_detect_agent_chains_length_one_hash_retry_then_growth_single_chain():
    # Hash lists of length 1, including an equal-hash retry (zero-growth
    # extension at equal length).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1], api_time=0.1),
            _req(1.0, [1], api_time=0.1),
            _req(2.0, [1, 2], api_time=0.1),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_detect_agent_chains_identical_duplicates_zero_duration_one_chain():
    # Identical (t, hash_ids) duplicates with zero-duration intervals: the
    # second is an equal-hash retry extension, never a fork.
    r = detect_agent_chains(
        _normals(
            _req(1.0, [1, 2, 3], api_time=0.0),
            _req(1.0, [1, 2, 3], api_time=0.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


# ---------------------------------------------------------------------------
# Global invariants on a hostile composite trace
# ---------------------------------------------------------------------------


def test_detect_agent_chains_partition_and_order_invariants_composite_trace():
    # Fan-out (two workers, one founded off the other's deeper tail), an
    # empty-hash row, an equal-hash retry, a streaming-type turn, and a
    # final main compaction seam — all at once.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),  # o0 main turn 0
            _req(1.0, [1, 2, 50, 51], api_time=8.0),  # o1 worker A founder
            _req(1.2, [1, 2, 60, 61], api_time=2.0),  # o2 worker B founder
            _req(2.0, [], api_time=0.1),  # o3 empty row -> main
            _req(4.0, [1, 2, 3, 4], api_time=0.5),  # o4 main turn 1
            _req(5.0, [1, 2, 60, 61], api_time=0.5),  # o5 worker B retry
            _sreq(10.0, [1, 2, 50, 51, 52], api_time=0.5),  # o6 worker A turn 2
            _req(12.0, [1, 2, 3, 99], api_time=0.5),  # o7 main compaction seam
        )
    )
    assert r.seams_merged == 1
    assert r.unclassified_empty_hash == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 3, 4, 7]
    assert len(r.worker_indices) == 2
    # Workers ordered by first-request (t, outer_idx).
    assert [_chain_outer_indices(r, i) for i in r.worker_indices] == [[1, 6], [2, 5]]

    # Invariant: every input request appears in exactly one LIVE chain
    # exactly once (spliced chains are dead duplicates by design).
    live = [i for i, c in enumerate(r.chains) if c.spliced_into is None]
    seen = [oi for i in live for oi in _chain_outer_indices(r, i)]
    assert sorted(seen) == list(range(8))

    # Invariant: each live chain's request list is strictly
    # (t, outer_idx)-ordered.
    for i in live:
        keys = [(req.t, oi) for oi, req in r.chains[i].requests]
        assert keys == sorted(set(keys))
