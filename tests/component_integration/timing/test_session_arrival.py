# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session arrival pacing through the service and credit lifecycle."""

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_SECOND
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import ConcurrencyAnalyzer, CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
@pytest.mark.parametrize("pattern", ["constant", "poisson"])
@pytest.mark.parametrize("dispatch", ["rate_slot", "direct"])
def test_session_arrivals_complete_think_times(
    cli: AIPerfCLI, tmp_path: Path, pattern: str, dispatch: str
) -> None:
    """Both policies complete the session budget without shortening think time."""
    result = cli.run_sync(
        f"aiperf profile --model {defaults.model} --streaming "
        f"--num-sessions 6 --session-turns-mean 3 --session-turns-stddev 0 "
        f"--session-turn-delay-mean 500 --session-turn-delay-stddev 0 "
        f"--request-rate 4 --request-rate-mode {pattern} "
        f"--continuation-dispatch {dispatch} --concurrency 16 "
        f"--isl 16 --osl 10 --extra-inputs ignore_eos:true "
        f"--random-seed 42 --ui {defaults.ui}",
        timeout=30.0,
    )
    flow = CreditFlowAnalyzer(result.runner_result)
    assert result.request_count == 18
    assert flow.credits_balanced()
    assert flow.session_credits_match(expected_turns=3)
    assert flow.turn_indices_sequential()

    returns = {
        (p.payload.credit.x_correlation_id, p.payload.credit.turn_index): p.timestamp_ns
        for p in flow.return_payloads
    }
    events = [
        {
            "session": p.payload.x_correlation_id,
            "turn": p.payload.turn_index,
            "sent_ns": p.timestamp_ns,
            "returned_ns": returns[(p.payload.x_correlation_id, p.payload.turn_index)],
        }
        for p in flow.credit_payloads
    ]
    gaps = [
        (
            p.timestamp_ns
            - returns[(p.payload.x_correlation_id, p.payload.turn_index - 1)]
        )
        / NANOS_PER_SECOND
        for p in flow.credit_payloads
        if p.payload.turn_index > 0
    ]
    (tmp_path / "arrival-events.json").write_bytes(
        orjson.dumps(
            {
                "pattern": pattern,
                "dispatch": dispatch,
                "events": events,
                "turn_gaps_s": gaps,
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    assert len(gaps) == 12
    assert all(gap >= 0.5 for gap in gaps)


@pytest.mark.component_integration
@pytest.mark.parametrize(
    ("limits", "expected_count"),
    [
        param("--request-count 7", 7, id="request-cap-mid-session"),
        param("--num-sessions 6", 18, id="session-cap-with-prefill-backpressure"),
        param("--num-sessions 6 --num-warmup-sessions 2", 18, id="warmup-to-profiling"),
        param("--benchmark-duration 0.5 --benchmark-grace-period 2", None, id="duration-cutoff"),
    ],
)  # fmt: skip
def test_direct_continuations_respect_limits(
    cli: AIPerfCLI, limits: str, expected_count: int | None
) -> None:
    """Root continuations retain request, session, prefill and phase gates."""
    result = cli.run_sync(
        f"aiperf profile --model {defaults.model} --streaming {limits} "
        f"--session-turns-mean 3 --session-turns-stddev 0 "
        f"--session-turn-delay-mean 10 --request-rate 100 "
        f"--continuation-dispatch direct --concurrency 2 --prefill-concurrency 1 "
        f"--isl 16 --osl 10 --extra-inputs ignore_eos:true --ui {defaults.ui}",
        timeout=30.0,
    )
    if expected_count is not None:
        assert result.request_count == expected_count
    else:
        assert result.request_count > 0
    flow = CreditFlowAnalyzer(result.runner_result)
    concurrency = ConcurrencyAnalyzer(result)
    assert flow.credits_balanced()
    assert flow.turn_indices_sequential()
    assert concurrency.concurrency_within_limit(2)
    assert concurrency.prefill_concurrency_within_limit(1)
