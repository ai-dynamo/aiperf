# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine the static scenario-lock outcome with runtime threshold checks.

The validator-side outcome (``ScenarioOutcome`` from ``apply_scenario``) covers
static config violations -- invariant-lock conflicts and ``--unsafe-override``.
This helper folds in runtime-only signals that are only knowable post-run: the
context-overflow response rate (InferenceX AgentX spec §7) and early
cancellation.
"""

from aiperf.common.environment import Environment

CONTEXT_OVERFLOW_REASON = "context_overflow_rate_exceeded"
RUN_CANCELLED_REASON = "run_cancelled"


def compute_submission_outcome(
    *,
    scenario_name: str | None,
    validator_submission_valid: bool | None,
    validator_reasons: list[str] | None = None,
    total_responses: int = 0,
    context_overflow_count: int = 0,
    was_cancelled: bool = False,
) -> tuple[bool | None, list[str]]:
    """Combine validator outcome with runtime threshold checks into a verdict.

    The validator-side outcome covers static config violations (handled by
    ``apply_scenario`` and stored on ``run.resolved.scenario_outcome``). This
    helper folds in runtime-only signals that are only knowable post-run --
    the >1% context-overflow rate per the InferenceX AgentX spec §7, and early
    cancellation (Ctrl+C):
    a cancelled run produces partial metrics and is never a valid submission.

    Rate semantics: strictly greater than
    ``Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT`` (default 0.01 per the
    InferenceX AgentX spec §7, override via
    ``AIPERF_AGENTX_CONTEXT_OVERFLOW_RATE_LIMIT``) flips
    ``submission_valid`` to False; equal-to is accepted (boundary behavior
    pinned by tests). When ``total_responses == 0`` the rate is treated as 0
    (undefined / no successful responses), so the overflow rule does not flip
    submission validity in that case.

    When ``scenario_name`` is None this is a no-scenario run and the function
    returns ``(None, [])`` -- callers should drop the ``submission_valid``
    field from the output entirely.

    Args:
        scenario_name: Active scenario, or None for a non-scenario run.
        validator_submission_valid: Outcome from ``apply_scenario`` -- True if
            the static lock was satisfied, False under ``--unsafe-override``
            with violations, None for a non-scenario run.
        validator_reasons: Reason codes already collected by the validator
            (e.g. ``"unsafe_override"``).
        total_responses: Total responses received during the run (successes +
            overflow + other failures).
        context_overflow_count: Count of context-overflow responses during the
            run.
        was_cancelled: Whether the run was cancelled early (graceful Ctrl+C).
            True flips ``submission_valid`` to False with reason
            ``"run_cancelled"``.

    Returns:
        A ``(submission_valid, reasons)`` tuple. ``submission_valid`` is
        ``None`` when ``scenario_name`` is None.
    """
    if scenario_name is None:
        return None, []

    reasons: list[str] = list(validator_reasons or [])
    valid: bool = (
        bool(validator_submission_valid)
        if validator_submission_valid is not None
        else True
    )

    if total_responses > 0:
        rate = context_overflow_count / total_responses
        if rate > Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT:
            valid = False
            if CONTEXT_OVERFLOW_REASON not in reasons:
                reasons.append(CONTEXT_OVERFLOW_REASON)

    if was_cancelled:
        valid = False
        if RUN_CANCELLED_REASON not in reasons:
            reasons.append(RUN_CANCELLED_REASON)

    return valid, reasons
