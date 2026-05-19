# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models.branch_stats import BranchStats


class TestBranchStatsDefaults:
    def test_all_counters_zero(self):
        s = BranchStats()
        assert s.children_spawned == 0
        assert s.children_completed == 0
        assert s.children_errored == 0
        assert s.parents_suspended == 0
        assert s.parents_resumed == 0
        assert s.parents_failed_due_to_child_error == 0
        assert s.joins_suppressed == 0


class TestBranchStatsIncrement:
    def test_set_and_serialize(self):
        s = BranchStats(
            children_spawned=5,
            children_completed=4,
            children_errored=1,
            parents_suspended=2,
            parents_resumed=2,
            joins_suppressed=3,
        )
        d = s.stats_dict()
        assert d["children_spawned"] == 5
        assert d["children_completed"] == 4
        assert d["joins_suppressed"] == 3
        assert d["parents_failed_due_to_child_error"] == 0

    def test_round_trip(self):
        s = BranchStats(joins_suppressed=7, children_spawned=10)
        restored = BranchStats.model_validate(s.model_dump())
        assert restored == s
