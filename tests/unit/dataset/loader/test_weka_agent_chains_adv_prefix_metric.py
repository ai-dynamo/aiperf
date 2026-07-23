# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tests.unit.dataset.loader._shared_helpers import (
    _chain_outer_indices,
    _normals,
    _req,
)

"""Adversarial tests for the namespace-group setup prefix and shared prefix-cache metric pre-pass in ``weka_agent_chains``."""

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


def _live_outer_indices(result) -> list[int]:
    out: list[int] = []
    for c in result.chains:
        if c.spliced_into is None:
            out.extend(oi for oi, _ in c.requests)
    return out


def test_compute_chain_prefix_blocks_group_survives_cascaded_seam_splices():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 80, 81, 82, 83], api_time=0.5),
            _req(4.0, [1, 2, 80, 91], api_time=10.0),
            _req(6.0, [1, 2, 80, 91, 99], api_time=0.5),
            _req(20.0, [1, 2, 80, 91, 92], api_time=0.5),
        )
    )
    assert r.seams_merged == 2
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.parent_chain == r.main_index
    assert worker.fork.depth == 4
    assert sorted(_live_outer_indices(r)) == [0, 1, 2, 3, 4]

    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    assert set(prefixes) == {r.main_index, r.worker_indices[0]}
    assert prefixes[r.main_index] == 2
    assert prefixes[r.worker_indices[0]] == 2


def test_compute_chain_prefix_blocks_three_level_fork_ancestry_single_group():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2, 10, 11], api_time=0.5),
            _req(2.0, [1, 2, 3, 4], api_time=0.5),
            _req(3.0, [1, 2, 10, 20], api_time=0.5),
            _req(5.0, [1, 2, 10, 11, 12], api_time=0.5),
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
    assert r.chains[w2].fork.parent_chain == w1

    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    assert prefixes[r.main_index] == 2
    assert [prefixes[i] for i in r.worker_indices] == [2, 2]


def test_compute_chain_prefix_blocks_zero_depth_root_group_with_internal_forks():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [100, 101, 110], api_time=10.0),
            _req(2.0, [100, 101, 120], api_time=10.0),
            _req(3.0, [100, 101, 120, 130], api_time=0.5),
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
    assert prefixes[r.main_index] == 7
    assert [prefixes[i] for i in r.worker_indices] == [2, 2, 2]


@pytest.mark.parametrize("declared", [0, 1, 2])
def test_compute_chain_prefix_blocks_member_first_request_is_exact_common_prefix(
    declared: int,
):
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2], api_time=0.5),
            _req(1.2, [1, 2, 9, 10, 11], api_time=0.5),
            _req(3.0, [1, 2, 3, 4], api_time=0.5),
        )
    )
    assert len(r.worker_indices) == 2
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=declared)
    assert prefixes[r.main_index] == 2
    assert [prefixes[i] for i in r.worker_indices] == [2, 2]


def test_compute_chain_prefix_blocks_declared_win_applies_to_main_only():
    """Resolved spec ambiguity (5.4): when P_declared beats P_observed, the"""
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [1, 2, 50], api_time=0.5),
            _req(3.0, [1, 2, 3, 4], api_time=0.5),
        )
    )
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=3)
    assert prefixes[r.main_index] == 3
    assert prefixes[r.worker_indices[0]] == 2


def test_compute_chain_prefix_blocks_excludes_empty_first_hash_from_fold():
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


def test_compute_shared_prefix_cache_metrics_hits_stop_at_first_unseen_block():
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(
                sort_key=(0.0, 0, 0, 0), session_id="r", k=0, hash_ids=[1, 2, 3]
            ),
            MetricRecord(
                sort_key=(1.0, 1, 0, 0), session_id="r", k=1, hash_ids=[9, 2, 3]
            ),
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
    assert out[("r", 1)] == (0, 1)


def test_compute_shared_prefix_cache_metrics_tie_on_t_orders_outer_stream_then_k():
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(sort_key=(5.0, 7, 1, 0), session_id="a", k=0, hash_ids=[1, 2]),
            MetricRecord(sort_key=(5.0, 7, 0, 5), session_id="b", k=5, hash_ids=[1, 3]),
            MetricRecord(sort_key=(5.0, 6, 9, 9), session_id="c", k=9, hash_ids=[1, 9]),
        ]
    )
    assert out[("c", 9)] == (0, 2)
    assert out[("b", 5)] == (1, 2)
    assert out[("a", 0)] == (1, 2)


def test_compute_shared_prefix_cache_metrics_input_order_independent():
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
    out = compute_shared_prefix_cache_metrics(
        [
            MetricRecord(
                sort_key=(0.0, 0, 0, 0), session_id="r", k=0, hash_ids=[1, 2, 3, 4]
            ),
            MetricRecord(sort_key=(1.0, 1, 0, 0), session_id="w", k=0, hash_ids=[1, 2]),
        ]
    )
    assert out[("w", 0)] == (2, 2)
