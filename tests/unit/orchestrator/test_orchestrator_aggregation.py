# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the current aggregation/export ownership boundaries."""

from __future__ import annotations

from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy


def test_aggregation_delegation_api_is_owned_by_cli_runner() -> None:
    """Aggregation/export belongs to cli_runner, not orchestrator strategies."""
    assert not hasattr(MultiRunOrchestrator, "execute_and_export"), (
        "MultiRunOrchestrator must not own aggregation/export"
    )
    assert not hasattr(FixedTrialsStrategy, "export_aggregates"), (
        "strategies must not own aggregate export layout"
    )

    # The current aggregation/export home must exist (catches an accidental
    # delete of the re-homed entry point).
    from aiperf.cli_runner import _aggregate

    assert hasattr(_aggregate, "aggregate_and_export")
    assert hasattr(_aggregate, "_stamp_scenario_submission_metadata")
