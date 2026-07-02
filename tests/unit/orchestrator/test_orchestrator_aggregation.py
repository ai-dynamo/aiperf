# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supersession marker for the v1 orchestrator aggregation/export tests.

The v1 suite verified that ``MultiRunOrchestrator.execute_and_export`` delegated
aggregation/export to the active strategy: it asserted ``strategy.aggregate()``
was called with ``(results, config)``, that ``strategy.export_aggregates()``
fired only when the aggregate was non-None, and exercised
``FixedTrialsStrategy`` / ``ParameterSweepStrategy`` / ``SweepConfidenceStrategy``
``export_aggregates()`` path layouts (``aggregate/``, ``sweep_aggregate/``,
per-value confidence dirs under repeated vs independent modes).

main's #1035 removed ``execute_and_export`` from the orchestrator entirely.
On v2 the orchestrator only iterates ``(variation, trial)`` cells via a
``RunExecutor``; aggregation + export is a separate cli_runner concern:

- per-cell confidence aggregation + JSON/CSV/detailed export lives in
  ``aiperf.cli_runner._aggregate.aggregate_and_export`` (calls
  ``ConfidenceAggregation.aggregate`` then the
  ``AggregateConfidence{Json,Csv}Exporter`` / ``AggregateDetailedJsonExporter``);
- sweep-wide aggregation across variations lives in
  ``aiperf.cli_runner._sweep_aggregate`` and the per-cell Pareto rollup in
  ``aiperf.cli_runner._pareto._aggregate_one_cell``.

There is no v2 ``strategy.aggregate`` / ``strategy.export_aggregates`` contract
to port against; the v1 export-path-layout assertions describe directory shapes
that no longer exist. The v2 mechanics are covered by:

| v1 test concern                                       | v2 coverage |
| ----------------------------------------------------- | ----------- |
| execute_and_export -> strategy.aggregate(results,cfg) | tests/unit/orchestrator/test_aggregation.py + tests/unit/test_aggregate_one_cell.py |
| export fires only when aggregate is non-None          | tests/unit/cli_runner / aiperf.cli_runner._aggregate.aggregate_and_export |
| Confidence / Sweep / SweepConfidence export layouts   | tests/unit/orchestrator/test_strategies.py + exporters/aggregate tests |

The ONE genuinely-ported behavior touching aggregation -- the scenario-submission
carrier keys stamped onto ``AggregateResult.metadata`` -- moved (P8) to
``aiperf.cli_runner._aggregate._stamp_scenario_submission_metadata`` and is
verified for real against a REAL ``BenchmarkPlan`` in
``tests/unit/orchestrator/test_orchestrator.py``
(``test_stamp_scenario_submission_metadata_*``).

This module keeps a single guard so the supersession is self-verifying: if
``execute_and_export`` (or the v1 ``strategy.export_aggregates`` contract)
re-appears, the guard fails and forces a real re-port of the suite mapped above.
"""

from __future__ import annotations

from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy


def test_v1_aggregation_delegation_api_is_gone() -> None:
    """The v1 ``execute_and_export`` / ``export_aggregates`` contract is gone.

    Guards the supersession documented in this module's docstring. The v1
    orchestrator delegated aggregation/export to the strategy; on v2 that lives
    in ``aiperf.cli_runner._aggregate.aggregate_and_export``. If
    ``execute_and_export`` re-appears on the orchestrator or
    ``export_aggregates`` re-appears on the strategies, the v1 delegation model
    was re-introduced and its tests must be properly re-ported (see the table in
    this module's docstring) rather than left as this stub.
    """
    assert not hasattr(MultiRunOrchestrator, "execute_and_export"), (
        "v1 MultiRunOrchestrator.execute_and_export re-appeared -- re-port the v1 "
        "aggregation-delegation suite (see module docstring)."
    )
    assert not hasattr(FixedTrialsStrategy, "export_aggregates"), (
        "v1 strategy.export_aggregates re-appeared -- re-port the v1 export-layout "
        "suite (see module docstring)."
    )

    # The current aggregation/export home must exist (catches an accidental
    # delete of the re-homed entry point).
    from aiperf.cli_runner import _aggregate

    assert hasattr(_aggregate, "aggregate_and_export")
    assert hasattr(_aggregate, "_stamp_scenario_submission_metadata")
