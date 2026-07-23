# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.dataset.loader.weka_agent_chains import (
    compute_chain_prefix_blocks,
    detect_agent_chains,
)
from aiperf.dataset.loader.weka_metric_prepass import (
    MetricRecord,
    compute_shared_prefix_cache_metrics,
)
from tests.unit.dataset.loader._shared_helpers import (
    _chain_outer_indices,
    _normals,
    _req,
)


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
            _req(4.0, [1, 2, 3, 4]),
        )
    )
    assert r.worker_indices == []
    assert r.unclassified_empty_hash == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_deepest_tail_wins_extension_tiebreak():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4]),
            _req(0.5, [1, 2, 8], api_time=0.1),
            _req(3.0, [1, 2, 3, 4, 5]),
        )
    )
    assert _chain_outer_indices(r, r.main_index)[-1] == 2


def test_empty_input_returns_empty_result():
    r = detect_agent_chains([])
    assert r.chains == []
    assert r.worker_indices == []


def test_compaction_shrink_with_dead_longer_state_is_join_seam():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6]),
            _req(2.0, [1, 2, 90, 91]),
        )
    )
    assert r.worker_indices == []
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_shrink_with_live_longer_state_is_spawn():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6]),
            _req(2.0, [1, 2, 90, 91]),
            _req(4.0, [1, 2, 3, 4, 5, 6, 7]),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_election_deepest_fork_wins_seam_shallower_stays_spawn():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6]),
            _req(2.0, [1, 2, 90]),
            _req(3.0, [1, 2, 3, 4, 80, 81]),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_temporal_overlap_vetoes_seam():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=10.0),
            _req(2.0, [1, 2, 90, 91]),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1


def test_cascaded_compactions_stay_one_chain():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6]),
            _req(2.0, [1, 2, 80, 81, 82, 83]),
            _req(4.0, [1, 2, 80, 91]),
        )
    )
    assert r.seams_merged == 2
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2]


def test_fanout_with_continuing_main_yields_worker_chains():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=1.0),
            _req(2.0, [1, 2, 50, 51], api_time=5.0),
            _req(2.5, [1, 2, 60, 61], api_time=5.0),
            _req(9.0, [1, 2, 3, 4, 5], api_time=1.0),
            _req(8.0, [1, 2, 50, 51, 52], api_time=1.0),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 2
    assert _chain_outer_indices(r, r.main_index) == [0, 3]
    by_first = {
        r.chains[i].requests[0][0]: _chain_outer_indices(r, i) for i in r.worker_indices
    }
    assert by_first == {1: [1, 4], 2: [2]}


def test_observed_prefix_recovers_zero_declared_boundary():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=1.0),
            _req(0.5, [1, 2, 50], api_time=1.0),
            _req(3.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    assert prefixes[r.main_index] == 2
    assert prefixes[r.worker_indices[0]] == 2


def test_declared_wins_when_longer_for_main_chain_only():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=1.0),
            _req(0.5, [1, 2, 50], api_time=1.0),
            _req(3.0, [1, 2, 3, 4], api_time=1.0),
        )
    )
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=3)
    assert prefixes[r.main_index] == 3
    assert prefixes[r.worker_indices[0]] == 2


def test_disjoint_group_gets_own_observed_prefix():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [100, 101, 110], api_time=5.0),
            _req(1.5, [100, 101, 120], api_time=5.0),
        )
    )
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    assert prefixes[r.main_index] == 0
    disjoint = [prefixes[i] for i in r.worker_indices]
    assert disjoint == [2, 2]


def test_singleton_trace_keeps_declared_prefix():
    r = detect_agent_chains(_normals(_req(0.0, [1, 2, 3])))
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=5)
    assert prefixes == {r.main_index: 5}


def test_shared_seen_set_counts_cross_conversation_hits_in_time_order():
    records = [
        MetricRecord(
            sort_key=(0.0, 0, 0, 0), session_id="root", k=0, hash_ids=[1, 2, 3]
        ),
        MetricRecord(sort_key=(1.0, 2, 0, 0), session_id="w0", k=0, hash_ids=[1, 2, 9]),
        MetricRecord(
            sort_key=(2.0, 3, 0, 0), session_id="root", k=1, hash_ids=[1, 2, 3, 4]
        ),
        MetricRecord(sort_key=(1.0, 1, 0, 0), session_id="sa", k=0, hash_ids=[1, 5]),
    ]
    out = compute_shared_prefix_cache_metrics(records)
    assert out[("root", 0)] == (0, 3)
    assert out[("sa", 0)] == (1, 2)
    assert out[("w0", 0)] == (2, 3)
    assert out[("root", 1)] == (3, 4)


def test_cross_model_full_prefix_extension_is_spawn():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], model="opus"),
            _req(2.0, [1, 2, 3, 7], model="haiku"),
        )
    )
    assert len(r.worker_indices) == 1
    assert r.seams_merged == 0


def test_cross_model_shrink_never_seams():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus"),
            _req(2.0, [1, 2, 90, 91], model="haiku"),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1


def test_same_model_fork_elected_over_deeper_cross_model_fork():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus"),
            _req(2.0, [1, 2, 3, 4, 80, 81], model="haiku"),
            _req(3.0, [1, 2, 90], model="opus"),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]
