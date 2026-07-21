# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the namespace-group setup prefix (spec section 5.4)
and the shared prefix-cache metric pre-pass (spec section 5.5) in
``weka_agent_chains``."""

import random

import pytest

from aiperf.dataset.loader.weka_agent_chains import (
    AgentChain,
    ChainDetectionResult,
    ChainFork,
    compute_chain_prefix_blocks,
    detect_agent_chains,
)
from aiperf.dataset.loader.weka_metric_prepass import (
    MetricRecord,
    compute_shared_prefix_cache_metrics,
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


def _live_outer_indices(result) -> list[int]:
    out: list[int] = []
    for c in result.chains:
        if c.spliced_into is None:
            out.extend(oi for oi, _ in c.requests)
    return out


# ---------------------------------------------------------------------------
# compute_chain_prefix_blocks — namespace groups (spec section 5.4)
# ---------------------------------------------------------------------------


def test_compute_chain_prefix_blocks_group_survives_cascaded_seam_splices():
    # Two cascading compactions splice chains 1 and 2 into the main chain.
    # A worker forked in flight off the SECOND spliced chain's tail must have
    # its fork ancestry rewritten to the live owner so the namespace group
    # does not fracture (spec section 5.4: groups follow fork ancestry).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 80, 81, 82, 83], api_time=0.5),  # seam 1
            _req(4.0, [1, 2, 80, 91], api_time=10.0),  # seam 2, runs to t=14
            _req(6.0, [1, 2, 80, 91, 99], api_time=0.5),  # in-flight sibling
            _req(20.0, [1, 2, 80, 91, 92], api_time=0.5),  # extends seam 2
        )
    )
    assert r.seams_merged == 2
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    # The phase-1 fork parent was a spliced (dead) chain; phase 2 must have
    # rewritten it to the live owner — the main chain.
    assert worker.fork.parent_chain == r.main_index
    assert worker.fork.depth == 4
    # Invariant: every retained request lives in exactly one live chain.
    assert sorted(_live_outer_indices(r)) == [0, 1, 2, 3, 4]

    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    # Prefixes are reported for live chains only, and the worker shares the
    # main group's observed prefix (no fracture into a singleton group).
    assert set(prefixes) == {r.main_index, r.worker_indices[0]}
    assert prefixes[r.main_index] == 2
    assert prefixes[r.worker_indices[0]] == 2


def test_compute_chain_prefix_blocks_three_level_fork_ancestry_single_group():
    # Worker forked from a worker forked from main: all three chains form
    # one namespace group via transitive depth>0 ancestry.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2, 10, 11], api_time=0.5),  # W1 forks from main
            _req(2.0, [1, 2, 3, 4], api_time=0.5),  # main pullback
            _req(3.0, [1, 2, 10, 20], api_time=0.5),  # W2 forks from W1
            _req(5.0, [1, 2, 10, 11, 12], api_time=0.5),  # W1 pullback
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 2
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    by_first = {
        r.chains[i].requests[0][0]: _chain_outer_indices(r, i) for i in r.worker_indices
    }
    assert by_first == {1: [1, 4], 3: [3]}
    w2 = next(i for i in r.worker_indices if r.chains[i].requests[0][0] == 3)
    w1 = next(i for i in r.worker_indices if r.chains[i].requests[0][0] == 1)
    assert r.chains[w2].fork.parent_chain == w1  # true grandchild edge

    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    assert prefixes[r.main_index] == 2
    assert [prefixes[i] for i in r.worker_indices] == [2, 2]


def test_compute_chain_prefix_blocks_zero_depth_root_group_with_internal_forks():
    # A zero-depth fork roots a second namespace group; later members fork
    # from each other (transitively) inside that group. The singleton main
    # group keeps its declared header prefix and must not leak it into the
    # disjoint group (and vice versa).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),  # main, singleton group
            _req(1.0, [100, 101, 110], api_time=10.0),  # zero-depth root W1
            _req(2.0, [100, 101, 120], api_time=10.0),  # W2 forks from W1
            _req(3.0, [100, 101, 120, 130], api_time=0.5),  # W3 forks from W2
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 3
    w1, w2, w3 = r.worker_indices
    assert r.chains[w1].fork.parent_chain is None
    assert r.chains[w1].fork.depth == 0
    assert r.chains[w2].fork.parent_chain == w1
    assert r.chains[w3].fork.parent_chain == w2

    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=7)
    assert prefixes[r.main_index] == 7  # singleton main -> declared passes
    assert [prefixes[i] for i in r.worker_indices] == [2, 2, 2]


@pytest.mark.parametrize("declared", [0, 1, 2])
def test_compute_chain_prefix_blocks_member_first_request_is_exact_common_prefix(
    declared: int,
):
    # Worker A's entire first hash list IS the group's common prefix
    # (observed == its full length); members have three different first
    # lengths. declared in {0, 1} loses to observed=2; declared == 2 is the
    # equality boundary where max() must pass the same value through.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2], api_time=0.5),  # worker A: exactly the prefix
            _req(1.2, [1, 2, 9, 10, 11], api_time=0.5),  # worker B: longer
            _req(3.0, [1, 2, 3, 4], api_time=0.5),  # main pullback
        )
    )
    assert len(r.worker_indices) == 2
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=declared)
    assert prefixes[r.main_index] == 2
    assert [prefixes[i] for i in r.worker_indices] == [2, 2]


def test_compute_chain_prefix_blocks_declared_win_applies_to_main_only():
    """Resolved spec ambiguity (5.4): when P_declared beats P_observed, the
    longer declared boundary applies to the MAIN chain only. Workers only
    provably share the observed region — their first request may not even
    contain the declared prefix blocks, so a group-wide declared boundary
    is physically unrenderable for them. The boundaries diverge only on
    old declared-token vintages where declared > observed, where byte
    sharing past the observed region is impossible regardless."""
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2, 50], api_time=0.5),  # worker in the main group
            _req(3.0, [1, 2, 3, 4], api_time=0.5),  # main pullback
        )
    )
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=3)
    assert prefixes[r.main_index] == 3  # keep the longer one (declared)
    assert prefixes[r.worker_indices[0]] == 2  # workers prove only observed


def test_compute_chain_prefix_blocks_excludes_empty_first_hash_from_fold():
    # A group member whose first request carries no hash evidence must not
    # zero the group's observed prefix (spec section 8: empty hash_ids are
    # no witness of anything). Hand-constructed result: detection itself
    # never founds a chain on an empty-hash request mid-trace.
    c0 = AgentChain(requests=[(0, _req(0.0, [1, 2, 3, 4]))])
    c1 = AgentChain(
        requests=[(1, _req(1.0, [])), (2, _req(2.0, [1, 2, 9]))],
        fork=ChainFork(parent_chain=0, fork_outer_idx=0, depth=2, fork_time=1.0),
    )
    c2 = AgentChain(
        requests=[(3, _req(3.0, [1, 2, 7]))],
        fork=ChainFork(parent_chain=0, fork_outer_idx=0, depth=2, fork_time=3.0),
    )
    result = ChainDetectionResult(
        chains=[c0, c1, c2],
        main_index=0,
        worker_indices=[1, 2],
        seams_merged=0,
        unclassified_empty_hash=1,
    )
    prefixes = compute_chain_prefix_blocks(result, declared_prefix_blocks=0)
    assert prefixes == {0: 2, 1: 2, 2: 2}


def test_detect_agent_chains_leading_empty_hash_request_keeps_single_chain():
    r = detect_agent_chains(
        _normals(
            _req(0.0, []),
            _req(1.0, [1, 2, 3], api_time=0.5),
            _req(2.0, [1, 2, 3, 4], api_time=0.5),
        )
    )
    assert r.unclassified_empty_hash == 1
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_compute_chain_prefix_blocks_empty_detection_returns_empty():
    prefixes = compute_chain_prefix_blocks(
        detect_agent_chains([]), declared_prefix_blocks=9
    )
    assert prefixes == {}


# ---------------------------------------------------------------------------
# compute_shared_prefix_cache_metrics — shared seen-set pre-pass (spec 5.5)
# ---------------------------------------------------------------------------


def test_compute_shared_prefix_cache_metrics_hits_stop_at_first_unseen_block():
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(
                sort_key=(0.0, 0, 0, 0), session_id="r", k=0, hash_ids=[1, 2, 3]
            ),
            # Leading block 9 unseen while blocks 2 and 3 ARE seen: a prefix
            # cache cannot hit past the first miss -> hits == 0.
            MetricRecord(
                sort_key=(1.0, 1, 0, 0), session_id="r", k=1, hash_ids=[9, 2, 3]
            ),
            # All of [9, 2, 3] entered the seen-set above (the full input is
            # cached once the request ran), so the next request hits 3 deep.
            MetricRecord(
                sort_key=(2.0, 2, 0, 0), session_id="w", k=0, hash_ids=[9, 2, 3, 4]
            ),
            MetricRecord(
                sort_key=(3.0, 3, 0, 0), session_id="w", k=1, hash_ids=[9, 2, 3, 4]
            ),
        ]
    )
    assert out[("r", 0)] == (0, 3)
    assert out[("r", 1)] == (0, 3)
    assert out[("w", 0)] == (3, 4)
    assert out[("w", 1)] == (4, 4)


def test_compute_shared_prefix_cache_metrics_empty_inputs():
    assert compute_shared_prefix_cache_metrics([]) == {}
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(sort_key=(0.0, 0, 0, 0), session_id="r", k=0, hash_ids=[]),
            MetricRecord(sort_key=(1.0, 1, 0, 0), session_id="r", k=1, hash_ids=[5]),
        ]
    )
    assert out[("r", 0)] == (0, 0)
    # The empty record must not seed the seen-set.
    assert out[("r", 1)] == (0, 1)


def test_compute_shared_prefix_cache_metrics_tie_on_t_orders_outer_stream_then_k():
    # Equal absolute_t across all three records: the sort key must rank
    # outer_idx first, then stream_idx, then k (spec 5.5 stable_position).
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(sort_key=(5.0, 7, 1, 0), session_id="a", k=0, hash_ids=[1, 2]),
            MetricRecord(sort_key=(5.0, 7, 0, 5), session_id="b", k=5, hash_ids=[1, 3]),
            MetricRecord(sort_key=(5.0, 6, 9, 9), session_id="c", k=9, hash_ids=[1, 9]),
        ]
    )
    assert out[("c", 9)] == (0, 2)  # lowest outer_idx goes first
    assert out[("b", 5)] == (1, 2)  # stream_idx outranks k on the outer tie
    assert out[("a", 0)] == (1, 2)


def test_compute_shared_prefix_cache_metrics_input_order_independent():
    # The function must sort by sort_key itself: feeding the records in
    # reversed or shuffled order yields identical output.
    records = [
        MetricRecord(sort_key=(3.0, 5, 0, 0), session_id="w1", k=0, hash_ids=[1, 2, 7]),
        MetricRecord(sort_key=(0.0, 0, 0, 0), session_id="root", k=0, hash_ids=[1, 2]),
        MetricRecord(
            sort_key=(1.0, 2, 0, 0), session_id="root", k=1, hash_ids=[1, 2, 3]
        ),
        MetricRecord(sort_key=(2.0, 1, 1, 4), session_id="sa", k=4, hash_ids=[1, 9]),
        MetricRecord(sort_key=(4.0, 9, 0, 0), session_id="w1", k=1, hash_ids=[]),
    ]
    expected = {
        ("root", 0): (0, 2),
        ("root", 1): (2, 3),
        ("sa", 4): (1, 2),
        ("w1", 0): (2, 3),
        ("w1", 1): (0, 0),
    }
    assert compute_shared_prefix_cache_metrics(records) == expected
    assert compute_shared_prefix_cache_metrics(list(reversed(records))) == expected
    shuffled = records.copy()
    random.Random(7).shuffle(shuffled)
    assert compute_shared_prefix_cache_metrics(shuffled) == expected


def test_compute_shared_prefix_cache_metrics_shorter_prefix_request_full_hit():
    # A request that re-sends a strict prefix of an earlier request's
    # blocks hits on its entire (shorter) length: hits == total.
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(
                sort_key=(0.0, 0, 0, 0), session_id="r", k=0, hash_ids=[1, 2, 3, 4]
            ),
            MetricRecord(sort_key=(1.0, 1, 0, 0), session_id="w", k=0, hash_ids=[1, 2]),
        ]
    )
    assert out[("w", 0)] == (2, 2)
