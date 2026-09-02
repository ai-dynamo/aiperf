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


def test_scenario_sweep_stamps_submission_invalid(monkeypatch) -> None:
    """A scenario + sweep (only reachable under --unsafe-override) must stamp
    submission_valid=False even when per-variation apply_scenario is clean.

    apply_scenario re-resolves ONE expanded variation and cannot see the sweep,
    so it returns a clean outcome; the envelope-level override violation must be
    carried forward off plan.is_sweep.
    """
    from types import SimpleNamespace

    from aiperf.cli_runner import _aggregate

    base_config = SimpleNamespace(scenario="inferencex-agentx-mvp")
    plan = SimpleNamespace(configs=[base_config], is_sweep=True)
    aggregate = SimpleNamespace(metadata={})

    # Clean per-variation outcome (submission_valid=True) — the sweep is invisible here.
    clean_run = SimpleNamespace(
        resolved=SimpleNamespace(
            scenario_outcome=SimpleNamespace(
                submission_valid=True, submission_invalid_reasons=[]
            )
        )
    )
    monkeypatch.setattr(
        "aiperf.dataset.provenance.public_dataset_provenance", lambda _c: None
    )
    monkeypatch.setattr("aiperf.cli_runner._make_benchmark_run", lambda _c: clean_run)
    monkeypatch.setattr("aiperf.common.scenario.apply_scenario", lambda _r: None)
    monkeypatch.setattr(_aggregate, "_sum_runtime_response_counts", lambda _r: (0, 0))

    _aggregate._stamp_scenario_submission_metadata(aggregate, [], plan)

    assert aggregate.metadata["_validator_submission_valid"] is False
    assert (
        "scenario_with_sweep"
        in aggregate.metadata["_validator_submission_invalid_reasons"]
    )
