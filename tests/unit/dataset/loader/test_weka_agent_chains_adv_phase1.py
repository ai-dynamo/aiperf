# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tests.unit.dataset.loader._shared_helpers import _chain_outer_indices

"""Adversarial tests for phase-1 greedy chain building in detect_agent_chains."""

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


def _worker_by_first_outer(result) -> dict[int, int]:
    """Map each worker chain's first request outer_idx -> chain index."""
    return {result.chains[i].requests[0][0]: i for i in result.worker_indices}


def test_detect_agent_chains_overlap_within_epsilon_extends():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=2.0),
            _req(2.0 - 5e-7, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_detect_agent_chains_overlap_beyond_epsilon_forks_at_full_tail_depth():
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
    r = detect_agent_chains(
        _normals(
            _req(2.0, [1, 2, 3], api_time=api_time),
            _req(2.0, [1, 2, 3, 4], api_time=0.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_detect_agent_chains_equal_length_tails_extension_lowest_index_wins():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=50.0),
            _req(1.0, [1, 2, 3], api_time=50.0),
            _req(60.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_detect_agent_chains_deeper_tail_at_higher_index_wins_extension():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=10.0),
            _req(1.0, [1, 2, 3], api_time=1.0),
            _req(20.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1, 2]


def test_detect_agent_chains_deeper_cross_model_tail_skipped_for_extension():
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
    assert haiku.fork.depth == 3


def test_detect_agent_chains_in_flight_deeper_tail_skipped_shallower_extends():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=2.0),
            _req(1.0, [1, 2, 3], api_time=100.0),
            _req(5.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_detect_agent_chains_last_element_match_full_mismatch_not_extension():
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


def test_detect_agent_chains_equal_lcp_deeper_tail_wins_fork_witness():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=100.0),
            _req(1.0, [1, 2, 3, 4, 5], api_time=0.1),
            _req(1.05, [1, 2, 99], api_time=0.1),
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
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=50.0),
            _req(1.0, [1, 2, 3], api_time=50.0),
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
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2], api_time=0.1),
            _req(1.0, [1, 2, 3, 4], api_time=0.1),
            _req(1.5, [1, 2, 99], api_time=0.1),
            _req(2.0, [1, 2, 3, 4, 5], api_time=0.1),
        )
    )
    assert r.seams_merged == 0
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 3]
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.parent_chain == r.main_index
    assert worker.fork.depth == 2


def test_detect_agent_chains_unsorted_input_processed_in_time_order():
    normals = [
        (0, _req(4.0, [1, 2, 3, 4, 5], api_time=0.5)),
        (1, _req(0.0, [1, 2, 3], api_time=0.5)),
        (2, _req(2.0, [1, 2, 3, 4], api_time=0.5)),
    ]
    r = detect_agent_chains(normals)
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [1, 2, 0]


def test_detect_agent_chains_equal_t_ties_broken_by_outer_idx():
    normals = [
        (1, _req(1.0, [1, 2, 3, 4], api_time=0.0)),
        (0, _req(1.0, [1, 2, 3], api_time=0.0)),
    ]
    r = detect_agent_chains(normals)
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


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
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=1.0),
            _req(2.0, [1, 2, 50, 51], api_time=10.0),
            _req(3.0, [], api_time=0.1),
            _req(4.0, [1, 2, 3, 4], api_time=1.0),
            _req(13.0, [1, 2, 50, 51, 52], api_time=1.0),
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
            _req(2.0, [], api_time=0.1),
            _req(4.0, [1, 2, 90, 91], api_time=1.0),
        )
    )
    assert r.unclassified_empty_hash == 1
    assert r.seams_merged == 1
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


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
    main = r.chains[r.main_index]
    assert main.fork is None or (
        main.fork.parent_chain is None and main.fork.depth == 0
    )
    assert _chain_outer_indices(r, r.main_index) == [0]
    assert r.unclassified_empty_hash == expected_unclassified


def test_detect_agent_chains_length_one_hash_retry_then_growth_single_chain():
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
    r = detect_agent_chains(
        _normals(
            _req(1.0, [1, 2, 3], api_time=0.0),
            _req(1.0, [1, 2, 3], api_time=0.0),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_detect_agent_chains_partition_and_order_invariants_composite_trace():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2, 50, 51], api_time=8.0),
            _req(1.2, [1, 2, 60, 61], api_time=2.0),
            _req(2.0, [], api_time=0.1),
            _req(4.0, [1, 2, 3, 4], api_time=0.5),
            _req(5.0, [1, 2, 60, 61], api_time=0.5),
            _sreq(10.0, [1, 2, 50, 51, 52], api_time=0.5),
            _req(12.0, [1, 2, 3, 99], api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert r.unclassified_empty_hash == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 3, 4, 7]
    assert len(r.worker_indices) == 2
    assert [_chain_outer_indices(r, i) for i in r.worker_indices] == [[1, 6], [2, 5]]

    live = [i for i, c in enumerate(r.chains) if c.spliced_into is None]
    seen = [oi for i in live for oi in _chain_outer_indices(r, i)]
    assert sorted(seen) == list(range(8))

    for i in live:
        keys = [(req.t, oi) for oi, req in r.chains[i].requests]
        assert keys == sorted(set(keys))
