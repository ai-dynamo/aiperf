# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``errored_traces`` must be observable: partial failure logs, total failure raises.

The counter was previously incremented and never read anywhere in ``src/`` --
no exporter, no exit-code gate, no phase-failure check -- so a graph run in
which every trace failed still completed, exported, and exited 0.
"""

from __future__ import annotations

import logging

import pytest

from aiperf.common.exceptions import InvalidStateError
from tests.unit.graph.test_runtime_hardening_round3 import _cfg, _minimal_strategy


def _strategy_with_failures(errored: int, finished: int, samples: int = 0):
    strategy = _minimal_strategy(_cfg())
    strategy._errored_traces = errored
    # The failure ratio's denominator is finished attempts (admitted or not);
    # every trace in these fixtures reached the executor.
    strategy._finished_traces = finished
    strategy._errored_trace_samples = [
        f"t-{i}: RuntimeError('boom')" for i in range(samples)
    ]
    return strategy


def test_clean_phase_reports_nothing(caplog) -> None:
    """No errored traces -> no log, no raise."""
    strategy = _strategy_with_failures(errored=0, finished=10)
    with caplog.at_level(logging.ERROR):
        strategy.report_trace_failures()
    assert caplog.records == []


def test_partial_failure_logs_but_does_not_raise(caplog) -> None:
    """Some traces failing is normal on a large corpus and must not abort."""
    strategy = _strategy_with_failures(errored=3, finished=100, samples=3)
    with caplog.at_level(logging.ERROR):
        strategy.report_trace_failures()

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "3/100" in message
    assert "t-0" in message


def test_total_failure_raises(caplog) -> None:
    """Every completed trace failing is not a successful run."""
    strategy = _strategy_with_failures(errored=5, finished=5, samples=2)
    with pytest.raises(InvalidStateError) as exc:
        strategy.report_trace_failures()

    message = str(exc.value)
    assert "every graph trace failed" in message
    assert "5/5" in message
    assert "t-0" in message


def test_sample_overflow_reports_true_total() -> None:
    """Past the sample cap the message still carries the real count."""
    strategy = _strategy_with_failures(errored=42, finished=42, samples=5)
    with pytest.raises(InvalidStateError) as exc:
        strategy.report_trace_failures()

    message = str(exc.value)
    assert "and 37 more (total 42)" in message
    assert "42/42" in message


class TestRunnerWiring:
    """The reporter is only useful if PhaseRunner actually calls it.

    The original defect was not a missing implementation but a missing caller,
    so assert the wiring explicitly rather than only the reporter's behavior.
    """

    def test_runner_has_trace_failure_hook(self) -> None:
        from aiperf.timing.phase.runner import PhaseRunner

        assert hasattr(PhaseRunner, "_report_trace_failures")

    def test_hook_is_duck_typed_and_tolerates_other_strategies(self) -> None:
        """Non-graph strategies do not implement it; the hook must no-op."""
        from unittest.mock import MagicMock

        from aiperf.timing.phase.runner import PhaseRunner

        runner = PhaseRunner.__new__(PhaseRunner)
        strategy = MagicMock(spec=[])  # exposes no attributes at all
        PhaseRunner._report_trace_failures(runner, strategy)

    def test_hook_invokes_strategy_reporter(self) -> None:
        from unittest.mock import MagicMock

        from aiperf.timing.phase.runner import PhaseRunner

        runner = PhaseRunner.__new__(PhaseRunner)
        strategy = MagicMock()
        strategy.report_trace_failures = MagicMock()

        PhaseRunner._report_trace_failures(runner, strategy)

        strategy.report_trace_failures.assert_called_once_with()

    def test_hook_propagates_total_failure_raise(self) -> None:
        """A raise from the strategy must reach run()'s failure handler."""
        from aiperf.timing.phase.runner import PhaseRunner

        runner = PhaseRunner.__new__(PhaseRunner)
        strategy = _strategy_with_failures(errored=2, finished=2, samples=1)

        with pytest.raises(InvalidStateError):
            PhaseRunner._report_trace_failures(runner, strategy)
