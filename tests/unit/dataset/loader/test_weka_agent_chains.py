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


def test_compaction_shrink_with_dead_longer_state_is_join_seam():
    # M1 grows to 6 blocks and ends; M2 keeps only the 2-block prefix.
    # Nothing ever returns to the longer state -> same agent (seam).
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
    # Same shrink shape, but a later request extends the longer state.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6]),
            _req(2.0, [1, 2, 90, 91]),
            _req(4.0, [1, 2, 3, 4, 5, 6, 7]),  # pullback: M1's state lives
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_election_deepest_fork_wins_seam_shallower_stays_spawn():
    # Two forks off the same dead tail: depth 2 and depth 4.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6]),
            _req(2.0, [1, 2, 90]),  # shallow fork (depth 2)
            _req(3.0, [1, 2, 3, 4, 80, 81]),  # deep fork (depth 4)
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_temporal_overlap_vetoes_seam():
    # Shrink that starts before the tail's interval ends: cannot be the
    # same agent even though the longer state is dead.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=10.0),  # ends t=10
            _req(2.0, [1, 2, 90, 91]),  # starts t=2
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1


def test_cascaded_compactions_stay_one_chain():
    # Compact twice; both seams splice into one chain.
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
    # Main keeps growing; two overlapping workers fork at the shared prefix.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=1.0),
            _req(2.0, [1, 2, 50, 51], api_time=5.0),  # worker A
            _req(2.5, [1, 2, 60, 61], api_time=5.0),  # worker B (overlaps A)
            _req(9.0, [1, 2, 3, 4, 5], api_time=1.0),  # main turn 2
            _req(8.0, [1, 2, 50, 51, 52], api_time=1.0),  # worker A turn 2
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
    # 0/0-declared trace: main + one worker sharing blocks [1, 2].
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
    assert prefixes[r.main_index] == 3  # keep the longer one
    assert prefixes[r.worker_indices[0]] == 2  # workers only prove observed


def test_disjoint_group_gets_own_observed_prefix():
    # Main namespace plus a disjoint 2-worker batch sharing [100, 101].
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(1.0, [100, 101, 110], api_time=5.0),
            _req(1.5, [100, 101, 120], api_time=5.0),
        )
    )
    prefixes = compute_chain_prefix_blocks(r, declared_prefix_blocks=0)
    assert prefixes[r.main_index] == 0  # singleton group -> declared (0)
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
        # Same t as the w0 row above: stable tiebreak by position.
        MetricRecord(sort_key=(1.0, 1, 0, 0), session_id="sa", k=0, hash_ids=[1, 5]),
    ]
    out = compute_shared_prefix_cache_metrics(records)
    assert out[("root", 0)] == (0, 3)
    assert out[("sa", 0)] == (1, 2)  # processed before w0 (position 1 < 2)
    assert out[("w0", 0)] == (2, 3)
    assert out[("root", 1)] == (3, 4)  # block 4 is globally novel


def test_cross_model_full_prefix_extension_is_spawn():
    # haiku request fully extends the opus tail after it ended — the
    # same-model rule still forces a fork (cross-model = different agent).
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], model="opus"),
            _req(2.0, [1, 2, 3, 7], model="haiku"),
        )
    )
    assert len(r.worker_indices) == 1
    assert r.seams_merged == 0  # and phase 2 must not splice it back


def test_cross_model_shrink_never_seams():
    # Dead longer state, temporally feasible — but the continuation
    # candidate is a different model, so it stays a spawn.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus"),
            _req(2.0, [1, 2, 90, 91], model="haiku"),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1


def test_same_model_fork_elected_over_deeper_cross_model_fork():
    # Deeper fork is cross-model (excluded); shallower same-model fork
    # is elected as the seam continuation.
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus"),
            _req(2.0, [1, 2, 3, 4, 80, 81], model="haiku"),  # deep, wrong model
            _req(3.0, [1, 2, 90], model="opus"),  # shallow, same model
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]
