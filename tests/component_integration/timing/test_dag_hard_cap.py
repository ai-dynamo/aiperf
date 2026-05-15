# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ``--request-count`` hard-cap pin for DAG runs (in-process).

Pins the literal-cap contract through the full callback-handler +
orchestrator + issuer pipeline using ``FakeTransport`` (no real HTTP):
when ``--request-count N`` is set on a forking dataset, the run must
terminate at exactly N wire requests with no leaked credits and no
hang at-cap.

These cases caught (and now regression-pin) the ``CallbackHandler``
short-circuit bug fixed in 485b1441b: when intercept drains the DAG
synchronously inside the same call (every spawned child gets refused
at the cap gate — most acutely cap=1, but also cap=4/5/7 in the
truncation regime), the early ``return intercepted`` skipped the
``all_credits_returned_event`` check. No future credit return was
coming, so the runner blocked forever. Pre-fix this file would hang
on every truncation case; post-fix it completes.

Topology: ``tests/fixtures/dag/full.dag.jsonl`` — 1 root + 2 forks ×
2 turns each = 5 wire requests per session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "dag" / "full.dag.jsonl"


def _build_dag_command(
    input_file: Path, request_count: int, concurrency: int = 1
) -> str:
    """Build a dag_jsonl profile command with the given hard cap and concurrency.

    Concurrency stays ≤ smallest cap we test so AIPerf's CLI validator
    (which rejects ``concurrency > request_count``) doesn't reject us.
    """
    return f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --custom-dataset-type dag_jsonl \
            --input-file {input_file} \
            --concurrency {concurrency} \
            --request-count {request_count} \
            --record-processor-service-count 1 \
            --workers-max 2 \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """


@pytest.mark.component_integration
class TestDagHardCap:
    """End-to-end cap-compliance pins. Each parametrized case forces the
    cap to land at a different point in the DAG dispatch sequence,
    exercising every cap-boundary code path that participates in the
    literal-cap invariant.

    Pre-485b1441b, the truncation cases (cap=1/3/4/5) all hung because
    the callback handler skipped the all-credits-returned check when
    intercept short-circuited — even when intercept drained the DAG
    inline. The cap=5+session_size and cap=10 cases (clean session
    boundaries) worked. This is now the regression test for that bug.
    """

    @pytest.mark.parametrize(
        "request_count",
        [1, 3, 4, 5, 7, 10],
        ids=[
            "cap-1-root-only-fork-spawn-refused",
            "cap-3-mid-fork-spawn",
            "cap-4-one-child-fits-mid-arc",
            "cap-5-exact-session-boundary",
            "cap-7-mid-second-session",
            "cap-10-two-clean-sessions",
        ],
    )
    def test_request_count_is_a_hard_cap_on_wire_credits(
        self, cli: AIPerfCLI, request_count: int
    ) -> None:
        """``--request-count N`` is a literal cap on wire requests:
        exactly N credits issued, all returned, no leaks, no hang."""
        result = cli.run_sync(
            _build_dag_command(FIXTURE, request_count),
            timeout=60.0,
            assert_success=True,
        )

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.total_credits == request_count, (
            f"Hard-cap violated: expected exactly {request_count} wire credits, "
            f"got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced(), (
            f"Credit leak: {analyzer.total_credits} issued, "
            f"{analyzer.total_returns} returned"
        )


@pytest.mark.component_integration
class TestDagHardCapConcurrencyStress:
    """Concurrency-stress pins. With ``concurrency >= 2``, multiple
    root sessions are in flight simultaneously, and multiple intercepts
    can run concurrently when several roots return at once. Pins that
    the gate-check + ``increment_sent`` stay atomic under contention
    (no overshoot, no undershoot, no race-induced hang).
    """

    @pytest.mark.parametrize(
        "concurrency,request_count",
        [
            (2, 5),  # cap = single session size; both concurrent roots race
            (2, 10),  # clean two-session boundary at higher concurrency
            (2, 7),  # truncating cap with 2 in flight
            (4, 10),  # 4 roots concurrent, cap=10 → some must be refused
            (4, 20),  # cap = 4 × session size; should run cleanly
        ],
        ids=[
            "conc=2-cap=5",
            "conc=2-cap=10",
            "conc=2-cap=7-truncating",
            "conc=4-cap=10",
            "conc=4-cap=20-clean",
        ],
    )
    def test_cap_holds_under_concurrency(
        self, cli: AIPerfCLI, concurrency: int, request_count: int
    ) -> None:
        """``concurrency=N`` runs N root sessions in flight simultaneously.
        Multiple intercepts can fire concurrently, multiple gate checks
        race the increment. Pins that this still produces exactly the
        configured number of wire requests, balanced, no hang."""
        result = cli.run_sync(
            _build_dag_command(FIXTURE, request_count, concurrency=concurrency),
            timeout=60.0,
            assert_success=True,
        )

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.total_credits == request_count, (
            f"Concurrency={concurrency} cap={request_count} violated hard cap: "
            f"expected exactly {request_count} wire credits, got "
            f"{analyzer.total_credits}. Likely a gate-check/increment race "
            f"under contention."
        )
        assert analyzer.credits_balanced(), (
            f"Concurrency={concurrency} cap={request_count} leaked credits: "
            f"{analyzer.total_credits} issued, {analyzer.total_returns} returned"
        )


@pytest.mark.component_integration
class TestDagBranchStatsExport:
    """``BranchStats`` (including ``children_truncated``) must flow
    through the exporter pipeline into ``profile_export_aiperf.json``.
    Without this, a user has no post-run signal that their DAG was cut
    short — only the in-flight log line (which they may have missed).
    """

    def test_truncated_count_lands_in_json_export(self, cli: AIPerfCLI) -> None:
        """A truncating cap (cap=4 on a 5-wire/session fixture) must
        surface ``children_truncated > 0`` in the exported JSON."""
        result = cli.run_sync(
            _build_dag_command(FIXTURE, request_count=4),
            timeout=60.0,
            assert_success=True,
        )
        assert result.json is not None, (
            "profile_export_aiperf.json must exist after a successful run"
        )
        bs = result.json.branch_stats
        assert bs is not None, (
            "branch_stats must be present in JSON export when DAG fanout occurred"
        )
        assert bs.children_truncated > 0, (
            f"cap=4 must produce children_truncated > 0; got {bs.children_truncated}"
        )

    def test_clean_run_reports_zero_truncated(self, cli: AIPerfCLI) -> None:
        """A clean ``--num-conversations 1`` run (1 root, full DAG, no
        cap) must report ``children_truncated == 0`` — pin that the
        counter isn't accidentally incremented on success paths.
        Using ``--num-conversations`` rather than ``--request-count``
        because the latter cannot guarantee "exactly one full session"
        with concurrency=1 (the root's slot releases after its 1-turn
        return, letting the strategy sample a second root and truncate
        its tree)."""
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --custom-dataset-type dag_jsonl \
                --input-file {FIXTURE} \
                --concurrency 1 --num-conversations 1 \
                --record-processor-service-count 1 \
                --workers-max 2 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=60.0, assert_success=True)
        assert result.json is not None
        bs = result.json.branch_stats
        assert bs is not None
        assert bs.children_truncated == 0, (
            f"clean num-conversations=1 must report 0 truncated; got "
            f"{bs.children_truncated}"
        )
        assert bs.children_completed > 0, (
            f"clean run must report some completed children; got "
            f"{bs.children_completed}"
        )


@pytest.mark.component_integration
class TestDagDrainObserverRace:
    """Regression pin for the DAG completion-signal race fixed in commit
    7cd4180b7. Under ``--concurrency >= 2`` against a truncating DAG cap,
    the orchestrator's final drain step (last ``_handle_child_done``
    decrement, or ``dispatch_join_turn`` returning False under cap and
    falling through to ``joins_suppressed``) can land BETWEEN the
    concurrent ``on_credit_return`` callbacks. If the deferred
    ``_maybe_signal_dag_completion`` only ran inside callbacks (no
    drain-observer hook), ``all_credits_returned_event`` would never
    fire and the phase runner would block on ``asyncio.run`` forever.

    Pre-fix this case hung ~40% of the time. Post-fix the orchestrator
    publishes ``set_drain_observer`` and the callback handler re-runs
    the deferred check after every drain.
    """

    @pytest.mark.parametrize(
        "concurrency,request_count",
        [
            (4, 10),  # the original reproducer
            (4, 13),  # different truncation point
            (8, 10),  # higher concurrency, more interleaving
            (8, 17),  # higher concurrency, mid-session truncation
        ],
        ids=[
            "conc=4-cap=10-original-repro",
            "conc=4-cap=13",
            "conc=8-cap=10",
            "conc=8-cap=17",
        ],
    )
    def test_no_hang_on_truncated_dag_under_concurrency(
        self, cli: AIPerfCLI, concurrency: int, request_count: int
    ) -> None:
        """Test simply completes — that's the assertion. Pre-fix this
        timed out at the cli.run_sync 60s deadline. Post-fix it returns
        in ~2s.
        """
        result = cli.run_sync(
            _build_dag_command(FIXTURE, request_count, concurrency=concurrency),
            timeout=30.0,
            assert_success=True,
        )
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.total_credits == request_count
        assert analyzer.credits_balanced()
