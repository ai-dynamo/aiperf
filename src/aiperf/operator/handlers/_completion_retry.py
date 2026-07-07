# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transient-fetch-failure retry gate for the completion handler.

Closes the ``CompletedBeforeMonitor -> ResultsFetchFailed`` race documented
in ``tests/kubernetes/audit/cases.py``. Sub-second benchmarks let the
controller's post-export shutdown race the operator's HTTP fetch — the
readiness marker and key files exist on the controller PVC, but the
operator hits a connection-refused or empty results listing as the
controller container terminates.

Strategy: when the fetch result has the race signature (``has_error`` set,
no key result files), and the completion claim is still fresh, raise
``kopf.TemporaryError`` BEFORE the caller writes terminal status. The
orphan-claim recovery path
(``monitor.py::_recover_orphaned_completion_claim``) re-runs
``handle_completion`` on the next monitor tick because the CR remains
non-terminal but the claim annotation is durable. The retry is bounded by
the ``RESULTS.TRANSIENT_FETCH_RETRY_BUDGET_SEC`` setting (wall-clock from
the claim timestamp) so a permanently-broken controller still progresses
to ``Phase.FAILED``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import kopf

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.status import parse_timestamp

if TYPE_CHECKING:
    from aiperf.operator.handlers.completion import _ResultFlags
    from aiperf.operator.models import ControllerFetchResult

logger = logging.getLogger(__name__)

__all__ = ["maybe_raise_for_transient_fetch_failure"]


def _claim_age_seconds(body: dict[str, Any]) -> float | None:
    """Return seconds since ``Annotations.COMPLETION_CLAIMED`` was stamped.

    Returns None when the annotation is absent or unparsable so the caller
    can fall back to the legacy fail-fast path rather than retrying forever
    on a malformed timestamp.
    """
    annotations = body.get("metadata", {}).get("annotations") or {}
    claim_ts = annotations.get(Annotations.COMPLETION_CLAIMED)
    if not claim_ts:
        return None
    try:
        claimed_at = parse_timestamp(claim_ts)
    except (ValueError, TypeError):
        return None
    return (datetime.now(UTC) - claimed_at).total_seconds()


def maybe_raise_for_transient_fetch_failure(
    *,
    body: dict[str, Any],
    namespace: str,
    job_id: str,
    result: ControllerFetchResult,
    flags: _ResultFlags,
) -> None:
    """Raise ``kopf.TemporaryError`` if the fetch failure looks transient.

    Callers MUST invoke this BEFORE writing terminal status (failed phase
    or completion event); otherwise the retry observes an already-Failed
    CR and short-circuits.

    Gate signals (all must hold):
        1. Key export files are still missing, and the fetch looks transient:
           either ``flags.has_error`` is set OR we got partial progress
           (metrics and/or non-key artifacts) without the authoritative
           exports.
        2. ``RESULTS.TRANSIENT_FETCH_RETRY_BUDGET_SEC > 0`` — set 0 to disable.
        3. Parseable ``Annotations.COMPLETION_CLAIMED`` timestamp on body.
        4. Wall-clock claim age below the budget.
    """
    # Cheap pre-check on the result shape avoids reading env settings at
    # all on the happy path.
    has_partial_progress = bool(result.metrics) or bool(result.downloaded)
    if flags.has_files or (not flags.has_error and not has_partial_progress):
        return
    budget = OperatorEnvironment.RESULTS.TRANSIENT_FETCH_RETRY_BUDGET_SEC
    if budget <= 0:
        return
    age = _claim_age_seconds(body)
    if age is None or age >= budget:
        return
    delay = OperatorEnvironment.RESULTS.TRANSIENT_FETCH_RETRY_DELAY_SEC
    msg = (
        f"transient results fetch failure for {namespace}/{job_id} "
        f"(claim age {age:.1f}s of {budget:.0f}s budget): "
        f"{result.error or 'no detail'}; "
        "retrying via orphan-claim recovery on next monitor tick"
    )
    logger.warning(msg)
    raise kopf.TemporaryError(msg, delay=delay)
