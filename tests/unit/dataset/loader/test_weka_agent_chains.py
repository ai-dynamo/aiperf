# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.dataset.loader.weka_agent_chains import detect_agent_chains
from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest


def _req(t: float, hash_ids: list[int], api_time: float = 1.0) -> WekaNormalRequest:
    return WekaNormalRequest(
        type="n",
        t=t,
        model="m",
        input_length=len(hash_ids) * 64,
        output_length=10,
        hash_ids=hash_ids,
        api_time=api_time,
    )


def _normals(*reqs: WekaNormalRequest) -> list[tuple[int, WekaNormalRequest]]:
    return list(enumerate(reqs))


def _chain_outer_indices(result, chain_index: int) -> list[int]:
    return [oi for oi, _ in result.chains[chain_index].requests]


def test_pure_growth_single_chain():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3]),
            _req(2.0, [1, 2, 3, 4, 5]),
            _req(4.0, [1, 2, 3, 4, 5, 6]),
        )
    )
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_equal_hash_retry_is_zero_growth_extension():
    r = detect_agent_chains(_normals(_req(0.0, [1, 2, 3]), _req(2.0, [1, 2, 3])))
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_in_flight_full_prefix_sibling_forks():
    # M1 runs t=[10, 30]; r starts t=15 with M1's full hash list as prefix.
    # A single agent cannot overlap itself -> r must be a separate chain.
    r = detect_agent_chains(
        _normals(
            _req(10.0, [1, 2, 3], api_time=20.0),
            _req(15.0, [1, 2, 3, 7], api_time=1.0),
        )
    )
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.depth == 3


def test_zero_lcp_request_founds_disjoint_chain():
    r = detect_agent_chains(_normals(_req(0.0, [1, 2, 3]), _req(2.0, [9, 10])))
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.parent_chain is None
    assert worker.fork.depth == 0


def test_empty_hash_ids_stays_on_main_and_is_invisible():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3]),
            _req(2.0, []),
            _req(4.0, [1, 2, 3, 4]),  # extends turn 0, not the empty req
        )
    )
    assert r.worker_indices == []
    assert r.unclassified_empty_hash == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_deepest_tail_wins_extension_tiebreak():
    # Two chains: main grows to [1,2,3,4]; sibling forked at [1,2]+[8].
    # A new request [1,2,3,4,5] fully extends main only.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4]),
            _req(0.5, [1, 2, 8], api_time=0.1),  # overlaps main -> fork
            _req(3.0, [1, 2, 3, 4, 5]),
        )
    )
    assert _chain_outer_indices(r, r.main_index)[-1] == 2


def test_empty_input_returns_empty_result():
    r = detect_agent_chains([])
    assert r.chains == []
    assert r.worker_indices == []
