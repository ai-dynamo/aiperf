# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PhaseRunner rejects end_conversation_on_truncation where a branch orchestrator
owns session lifecycle (DAG datasets, agentic replay)."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.plugin.enums import TimingMode
from aiperf.timing.phase.runner import PhaseRunner


def _runner(*, enabled: bool, is_dag: bool, timing_mode: TimingMode) -> MagicMock:
    runner = MagicMock()
    runner._branch_orchestrator = None
    runner._config.timing_mode = timing_mode
    runner._is_dag_dataset.return_value = is_dag
    runner._run.cfg.get_default_dataset.return_value = MagicMock(
        end_conversation_on_truncation=enabled
    )
    return runner


@pytest.mark.parametrize(
    "is_dag,timing_mode",
    [
        param(True, TimingMode.REQUEST_RATE, id="dag-dataset"),
        param(False, TimingMode.AGENTIC_REPLAY, id="agentic-replay"),
    ],
)  # fmt: skip
def test_construct_branch_orchestrator_option_on_raises(
    is_dag: bool, timing_mode: TimingMode
) -> None:
    runner = _runner(enabled=True, is_dag=is_dag, timing_mode=timing_mode)
    with pytest.raises(ValueError, match="not supported for DAG datasets"):
        PhaseRunner._maybe_construct_branch_orchestrator(runner, MagicMock())


def test_construct_branch_orchestrator_option_off_builds() -> None:
    runner = _runner(enabled=False, is_dag=True, timing_mode=TimingMode.REQUEST_RATE)
    with patch("aiperf.timing.phase.runner.BranchOrchestrator") as orchestrator:
        PhaseRunner._maybe_construct_branch_orchestrator(runner, MagicMock())
    orchestrator.assert_called_once()


def test_construct_branch_orchestrator_non_dag_option_on_is_noop() -> None:
    runner = _runner(enabled=True, is_dag=False, timing_mode=TimingMode.REQUEST_RATE)
    PhaseRunner._maybe_construct_branch_orchestrator(runner, MagicMock())
    assert runner._branch_orchestrator is None


@pytest.mark.parametrize(
    "ended,skipped,expected_fragment",
    [
        param(3, 5, "ended_on_truncation=3 (skipped_turns=5)", id="ended"),
        param(0, 0, None, id="none-ended"),
    ],
)  # fmt: skip
def test_format_phase_complete_reports_truncation_counts(
    ended: int, skipped: int, expected_fragment: str | None
) -> None:
    stats = CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        start_ns=1,
        requests_end_ns=2,
        final_requests_completed=4,
        final_requests_cancelled=0,
        final_request_errors=0,
        sessions_ended_on_truncation=ended,
        turns_skipped_on_truncation=skipped,
    )
    message = PhaseRunner._format_phase_complete(MagicMock(), stats)
    if expected_fragment is None:
        assert "ended_on_truncation" not in message
    else:
        assert expected_fragment in message
