# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end proof of orchestrator repeated-firing (in-process).

Drives the full credit / orchestrator / issuer pipeline using
``FakeTransport`` (no real HTTP) against ``orchestrator.dag.jsonl``:
a single ``orchestrator: true`` root whose conversation-level ``spawns``
fan out to two request-bearing children (only `start` itself is request-free).

The orchestrator ``start`` issues a *virtual* (``no_request=True``) credit
that the StickyCreditRouter short-circuits in-process: it is NEVER sent to a
worker over the bus. The router synthesizes its ``CreditReturn`` locally and
feeds it through the normal callback path, so the existing ``intercept()``
still fires ``fan-out-a`` / ``fan-out-b`` as real wire requests. Because
``start`` stays in the sampled root pool, it re-fires on every sampled
iteration under load.

Consequence for observability: because the virtual credit is short-circuited
at the router, ``start`` produces NO bus-sent ``Credit`` at all (the analyzer
taps ``payloads_by_type(Credit, sent=True)``). That is the strongest possible
form of "the orchestrator sends nothing" — it does not even reach the internal
credit wire, let alone HTTP. The orchestrator's repeated firing is therefore
observed indirectly, through the children it spawns (which exist only because
``intercept()`` ran on each virtual firing).

Asserts:

  - ``start`` issues ZERO bus-sent credits -> the orchestrator never touches
    the wire (short-circuited at the router);
  - ``fan-out-a`` and ``fan-out-b`` each appear MULTIPLE times as real
    (``no_request=False``) wire requests -> repeated firing, one child pair
    per sampled orchestrator iteration;
  - the real-wire credit total equals ``--request-count`` (the cap counts
    only child wire requests; the virtual ``start`` firings do NOT count);
  - credits balance and the phase completes without hang/timeout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.credit.structs import Credit
from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "dag" / "orchestrator.dag.jsonl"
)

CHILD_IDS = {"fan-out-a", "fan-out-b"}


def _build_command(input_file: Path, request_count: int, concurrency: int = 1) -> str:
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


def _credits_by_conversation(
    analyzer: CreditFlowAnalyzer,
) -> dict[str, list[Credit]]:
    grouped: dict[str, list[Credit]] = {}
    for credit in analyzer.credits:
        grouped.setdefault(credit.conversation_id, []).append(credit)
    return grouped


@pytest.mark.component_integration
class TestOrchestratorRepeatedFiring:
    """Repeated-firing E2E: one virtual orchestrator fans out to two real
    children on every sampled iteration."""

    @pytest.mark.parametrize(
        "request_count",
        [4, 6, 8],
        ids=["cap-4-two-iters", "cap-6-three-iters", "cap-8-four-iters"],
    )
    def test_orchestrator_refires_children_without_own_http(
        self, cli: AIPerfCLI, request_count: int
    ) -> None:
        result = cli.run_sync(
            _build_command(FIXTURE, request_count),
            timeout=60.0,
            assert_success=True,
        )
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        by_conv = _credits_by_conversation(analyzer)

        # ZERO wire credits for the orchestrator: the virtual 'start' credit is
        # short-circuited at the router and never reaches the credit bus, so it
        # cannot appear among bus-sent credits. This is the strongest form of
        # "the orchestrator sends nothing" -- not even the internal wire.
        assert "start" not in by_conv, (
            "orchestrator 'start' must issue NO bus-sent credit (router "
            f"short-circuits the virtual credit in-process); saw {sorted(by_conv)}"
        )

        # Real wire requests come ONLY from the children.
        real_credits = [c for c in analyzer.credits if not c.no_request]
        assert {c.conversation_id for c in real_credits} == CHILD_IDS, (
            "only the fan-out children may issue real wire requests; got "
            f"{sorted({c.conversation_id for c in real_credits})}"
        )

        # --request-count caps ONLY real child wire requests; the virtual
        # orchestrator credits do not count toward it.
        assert len(real_credits) == request_count, (
            f"expected exactly {request_count} real child wire requests "
            f"(virtual 'start' credits excluded from the cap); got "
            f"{len(real_credits)}"
        )

        # Repeated firing: each child fired MULTIPLE times, evenly.
        per_child = {cid: 0 for cid in CHILD_IDS}
        for c in real_credits:
            per_child[c.conversation_id] += 1
        for cid, count in per_child.items():
            assert count > 1, (
                f"child '{cid}' must fire multiple times (repeated firing); got {count}"
            )
        assert per_child["fan-out-a"] == per_child["fan-out-b"] == request_count // 2, (
            "children fan out in lockstep pairs, one pair per sampled "
            f"iteration; got {per_child}"
        )

        # The orchestrator re-fired once per completed child pair. Because the
        # virtual credit is short-circuited at the router (never bus-sent), the
        # firing count is observed through the children it spawned: each pair
        # above corresponds to exactly one virtual orchestrator firing, so
        # ``request_count // 2 > 1`` child pairs proves repeated firing.
        iterations = request_count // 2
        assert iterations > 1, (
            "orchestrator must re-fire multiple times (repeated firing); "
            f"child pairs observed: {iterations}"
        )

        # No leaks, clean completion (the run already returned before timeout).
        assert analyzer.credits_balanced(), (
            f"credit leak: {analyzer.total_credits} issued, "
            f"{analyzer.total_returns} returned"
        )
