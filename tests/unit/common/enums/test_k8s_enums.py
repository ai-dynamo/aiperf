# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import SweepType, WorkerStartupState


def test_sweep_type_members_present():
    assert SweepType.GRID == "grid"
    assert SweepType.ADAPTIVE_SEARCH == "adaptive_search"
    assert {s.value for s in SweepType} == {
        "grid",
        "zip",
        "scenarios",
        "adaptive_search",
        "sobol",
    }


def test_sweep_type_is_case_insensitive():
    assert SweepType("GRID") is SweepType.GRID


def test_worker_startup_state_members_present():
    assert WorkerStartupState.READY == "ready"
    assert {s.value for s in WorkerStartupState} == {
        "starting",
        "waiting_for_dataset",
        "router_probing",
        "ready",
        "shutting_down",
    }
