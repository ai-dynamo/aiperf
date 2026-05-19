# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAG-aware completion gating on PhaseRunner.

The phase must NOT declare itself complete while ``BranchOrchestrator``
reports pending DAG work, even when ``requests_sent`` has reached the
configured ``--request-count`` cap. Without this gate, the runner would
freeze sent counts mid-DAG and drop in-flight children.
"""

from unittest.mock import MagicMock

from aiperf.timing.phase.runner import PhaseRunner


class TestPhaseRunnerDagCompletion:
    def test_completion_blocks_while_orchestrator_has_pending_joins(self) -> None:
        runner = PhaseRunner.__new__(PhaseRunner)
        orch = MagicMock()
        orch.has_pending_branch_work.return_value = True
        runner._branch_orchestrator = orch

        counter = MagicMock()
        counter.requests_sent = 100
        progress = MagicMock()
        progress.counter = counter
        runner._progress = progress

        config = MagicMock()
        config.total_expected_requests = 100
        runner._config = config

        # Even at cap, completion should not fire while joins are pending.
        assert runner._is_phase_complete() is False

    def test_completion_fires_when_orchestrator_drained(self) -> None:
        runner = PhaseRunner.__new__(PhaseRunner)
        orch = MagicMock()
        orch.has_pending_branch_work.return_value = False
        runner._branch_orchestrator = orch

        counter = MagicMock()
        counter.requests_sent = 100
        progress = MagicMock()
        progress.counter = counter
        runner._progress = progress

        config = MagicMock()
        config.total_expected_requests = 100
        runner._config = config

        assert runner._is_phase_complete() is True

    def test_completion_blocks_below_cap_regardless_of_orchestrator(self) -> None:
        runner = PhaseRunner.__new__(PhaseRunner)
        orch = MagicMock()
        orch.has_pending_branch_work.return_value = False
        runner._branch_orchestrator = orch

        counter = MagicMock()
        counter.requests_sent = 50
        progress = MagicMock()
        progress.counter = counter
        runner._progress = progress

        config = MagicMock()
        config.total_expected_requests = 100
        runner._config = config

        assert runner._is_phase_complete() is False

    def test_completion_without_orchestrator_uses_only_counter(self) -> None:
        runner = PhaseRunner.__new__(PhaseRunner)
        runner._branch_orchestrator = None

        counter = MagicMock()
        counter.requests_sent = 100
        progress = MagicMock()
        progress.counter = counter
        runner._progress = progress

        config = MagicMock()
        config.total_expected_requests = 100
        runner._config = config

        # Non-DAG runs: no orchestrator means no DAG gate; cap reached => complete.
        assert runner._is_phase_complete() is True

    def test_completion_without_orchestrator_below_cap(self) -> None:
        runner = PhaseRunner.__new__(PhaseRunner)
        runner._branch_orchestrator = None

        counter = MagicMock()
        counter.requests_sent = 50
        progress = MagicMock()
        progress.counter = counter
        runner._progress = progress

        config = MagicMock()
        config.total_expected_requests = 100
        runner._config = config

        assert runner._is_phase_complete() is False

    def test_completion_with_no_request_cap_uncapped_is_incomplete(self) -> None:
        # When no request cap is configured, _is_phase_complete is never True
        # via this helper alone — completion is driven by other stop conditions.
        runner = PhaseRunner.__new__(PhaseRunner)
        runner._branch_orchestrator = None

        counter = MagicMock()
        counter.requests_sent = 1_000_000
        progress = MagicMock()
        progress.counter = counter
        runner._progress = progress

        config = MagicMock()
        config.total_expected_requests = None
        runner._config = config

        assert runner._is_phase_complete() is False
