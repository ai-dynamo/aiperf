# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tests.unit.dataset.loader._shared_helpers import (
    _chain_outer_indices,
    _normals,
    _req,
)

"""Adversarial tests for phase-2 seam resolution (``_resolve_seams``)."""

from aiperf.dataset.loader.weka_agent_chains import (
    _EPSILON_SECONDS,
    detect_agent_chains,
)


def _all_emitted_outer_indices(result) -> list[int]:
    """Every retained request as emitted: main chain + every worker chain."""
    out = [oi for oi, _ in result.chains[result.main_index].requests]
    for ci in result.worker_indices:
        out.extend(oi for oi, _ in result.chains[ci].requests)
    return out


def test_election_equal_depth_earliest_fork_time_wins():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(3.0, [1, 2, 90, 91], api_time=0.1),
            _req(2.0, [1, 2, 80, 81], api_time=0.1),
        )
    )
    assert r.seams_merged == 2
    assert _chain_outer_indices(r, r.main_index) == [0, 2, 1]
    assert r.worker_indices == []


def test_election_equal_depth_equal_fork_time_lowest_index_wins():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(3.0, [1, 2, 90, 91], api_time=0.5),
            _req(3.0, [1, 2, 80, 81], api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [2]


def test_t_extended_after_fork_makes_fork_a_spawn():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 90, 91], api_time=0.5),
            _req(4.0, [1, 2, 3, 4, 5, 6, 7], api_time=0.5),
        )
    )
    assert r.seams_merged == 0
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_fork_depth_is_recorded_against_tail_at_fork_time():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(2.0, [1, 2, 3, 4, 5], api_time=0.5),
            _req(4.0, [1, 2, 3, 80], api_time=0.5),
            _req(6.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.depth == 3
    assert worker.fork.fork_outer_idx == 1


def test_cascaded_three_compactions_collapse_to_one_chain():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 70, 71, 72, 73], api_time=0.5),
            _req(4.0, [1, 2, 70, 80, 81], api_time=0.5),
            _req(6.0, [1, 2, 70, 90], api_time=0.5),
        )
    )
    assert r.seams_merged == 3
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2, 3]


def test_cascaded_three_compactions_collapse_when_tails_grow():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4], api_time=0.5),
            _req(2.0, [1, 2, 70, 71], api_time=0.5),
            _req(4.0, [1, 2, 70, 80, 81, 82, 83], api_time=0.5),
            _req(6.0, [1, 2, 70, 90], api_time=0.5),
        )
    )
    assert r.seams_merged == 3
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 2, 3]


def test_live_worker_fork_parent_rewritten_to_live_chain_after_splice():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 80, 81, 82, 83], api_time=5.0),
            _req(3.0, [1, 2, 80, 81, 70], api_time=1.0),
            _req(9.0, [1, 2, 80, 81, 82, 83, 99], api_time=1.0),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1, 3]
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.spliced_into is None
    assert worker.fork is not None
    assert worker.fork.parent_chain == r.main_index


def test_cascade_independent_of_file_order_when_outer_precedes_time():
    reqs = [
        _req(1.0, [1, 2, 80, 81, 82, 83], api_time=0.5),
        _req(2.0, [1, 2, 80, 70], api_time=0.5),
        _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
    ]
    r = detect_agent_chains(list(enumerate(reqs)))
    assert r.seams_merged == 2
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [2, 0, 1]


def test_temporal_veto_seam_allowed_exactly_at_epsilon_boundary():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=10.0),
            _req(10.0 - _EPSILON_SECONDS, [1, 2, 90, 91], api_time=0.1),
        )
    )
    assert r.seams_merged == 1
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_temporal_veto_spawn_just_past_epsilon_boundary():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=10.0),
            _req(10.0 - 2 * _EPSILON_SECONDS, [1, 2, 90, 91], api_time=0.1),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_zero_api_time_tail_allows_seam_at_equal_timestamp():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.0),
            _req(0.0, [1, 2, 90, 91], api_time=0.0),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_deepest_fork_cross_model_shallower_same_model_elected():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], model="opus", api_time=0.5),
            _req(2.0, [1, 2, 3, 4, 80, 81], model="haiku", api_time=0.5),
            _req(3.0, [1, 2, 90], model="opus", api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 2]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1]


def test_all_candidates_cross_model_no_splice_at_all():
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


def test_in_flight_full_prefix_sibling_never_seams():
    r = detect_agent_chains(
        _normals(
            _req(10.0, [1, 2, 3], api_time=20.0),
            _req(15.0, [1, 2, 3, 7], api_time=0.5),
        )
    )
    assert r.seams_merged == 0
    assert len(r.worker_indices) == 1
    worker = r.chains[r.worker_indices[0]]
    assert worker.fork is not None
    assert worker.fork.depth == 3


def test_full_prefix_sibling_near_miss_extension_at_exact_end():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=2.0),
            _req(2.0, [1, 2, 3, 4], api_time=0.5),
        )
    )
    assert r.seams_merged == 0
    assert r.worker_indices == []
    assert _chain_outer_indices(r, r.main_index) == [0, 1]


def test_worker_compacts_midlife_absorbs_own_seam_main_unaffected():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(2.0, [1, 2, 50, 51, 52, 53], api_time=0.5),
            _req(3.0, [1, 2, 50, 51, 52, 53, 54], api_time=0.5),
            _req(5.0, [1, 2, 3, 4, 5], api_time=0.5),
            _req(6.0, [1, 2, 70, 71], api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert _chain_outer_indices(r, r.main_index) == [0, 3]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [1, 2, 4]


def test_worker_indices_ordered_by_first_request_time_not_outer():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3], api_time=0.5),
            _req(5.0, [1, 2, 50, 51], api_time=10.0),
            _req(2.0, [1, 2, 60, 61], api_time=10.0),
            _req(9.0, [1, 2, 3, 4], api_time=0.5),
        )
    )
    firsts = [r.chains[i].requests[0][0] for i in r.worker_indices]
    assert firsts == [2, 1]


def test_spliced_chains_excluded_from_workers_but_present_in_chains():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 90, 91], api_time=0.5),
        )
    )
    assert r.seams_merged == 1
    assert r.worker_indices == []
    spliced = [i for i, c in enumerate(r.chains) if c.spliced_into is not None]
    assert spliced == [1]
    assert r.chains[1].spliced_into == r.main_index
    assert 1 not in r.worker_indices


def test_every_retained_request_appears_in_exactly_one_chain_once():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 70, 71, 72, 73], api_time=0.5),
            _req(3.0, [1, 2, 70, 80], api_time=0.5),
            _req(4.0, [1, 2, 50, 51], api_time=5.0),
            _req(5.0, [1, 2, 50, 51, 52], api_time=0.5),
            _req(9.0, [1, 2, 70, 80, 90], api_time=0.5),
        )
    )
    emitted = sorted(_all_emitted_outer_indices(r))
    assert emitted == [0, 1, 2, 3, 4, 5]
    assert len(emitted) == len(set(emitted))


def test_seams_merged_counts_one_per_splice_across_independent_tails():
    r = detect_agent_chains(
        _normals(
            _req(0.0, [1, 2, 3, 4, 5, 6], api_time=0.5),
            _req(2.0, [1, 2, 90, 91], api_time=0.5),
            _req(1.0, [100, 101, 102, 103], api_time=0.5),
            _req(3.0, [100, 101, 80, 81], api_time=0.5),
        )
    )
    assert r.seams_merged == 2
    assert _chain_outer_indices(r, r.main_index) == [0, 1]
    assert len(r.worker_indices) == 1
    assert _chain_outer_indices(r, r.worker_indices[0]) == [2, 3]


def test_seam_guard_splits_far_low_overlap_continuation():
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
