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
from aiperf.kubernetes.phase import parse_timestamp
from aiperf.operator.environment import OperatorEnvironment

if TYPE_CHECKING:
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers.completion import _ResultFlags

logger = logging.getLogger(__name__)

__all__ = ["maybe_raise_for_transient_fetch_failure"]


def _coerce_settings_float(value: Any, *, default: float = 0.0) -> float:
    """Read a numeric setting defensively.

    ``OperatorEnvironment`` declares both retry settings as floats, so real
    configuration always coerces. Tests routinely stub ``RESULTS`` with a
    partial mock, though, and an auto-created attribute is a mock rather than a
    number: comparing it raises TypeError from inside a kopf completion
    handler, on the results-fetch failure path, and the CR then retries forever
    instead of surfacing the real fetch error.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _claim_age_seconds(
    body: dict[str, Any], namespace: str, job_id: str
) -> float | None:
    """Return seconds since ``Annotations.COMPLETION_CLAIMED`` was stamped.

    Reads the body snapshot first, then falls back to the claim timestamp
    this process latched in ``client_cache``. The fallback matters because
    kopf hands handlers a read-only ``kopf.Body``, so a claim won during this
    same tick cannot be written back into the snapshot.

    Returns None when no timestamp is available from either source, or when
    it is unparsable, so the caller falls back to the legacy fail-fast path
    rather than retrying forever on a malformed timestamp.
    """
    from aiperf.operator.client_cache import get_claim_timestamp, job_key

    annotations = body.get("metadata", {}).get("annotations") or {}
    claim_ts = annotations.get(Annotations.COMPLETION_CLAIMED) or get_claim_timestamp(
        job_key(
            namespace,
            job_id,
            str(body["metadata"]["uid"])
            if (body.get("metadata") or {}).get("uid") is not None
            else None,
        )
    )
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
        3. Parseable ``Annotations.COMPLETION_CLAIMED`` timestamp, from the
           body snapshot or the process-local claim registry.
        4. Wall-clock claim age below the budget.
    """
    # Cheap pre-check on the result shape avoids reading env settings at
    # all on the happy path.
    has_partial_progress = bool(result.metrics) or bool(result.downloaded)
    if flags.has_files or (not flags.has_error and not has_partial_progress):
        return
    budget = _coerce_settings_float(
        OperatorEnvironment.RESULTS.TRANSIENT_FETCH_RETRY_BUDGET_SEC
    )
    if budget <= 0:
        return
    age = _claim_age_seconds(body, namespace, job_id)
    if age is None or age >= budget:
        return
    # A None/mock delay would read to kopf as 'retry immediately', turning a
    # paced retry into a hot loop against the apiserver.
    delay = _coerce_settings_float(
        OperatorEnvironment.RESULTS.TRANSIENT_FETCH_RETRY_DELAY_SEC, default=5.0
    )
    msg = (
        f"transient results fetch failure for {namespace}/{job_id} "
        f"(claim age {age:.1f}s of {budget:.0f}s budget): "
        f"{result.error or 'no detail'}; "
        "retrying via orphan-claim recovery on next monitor tick"
    )
    logger.warning(msg)
    raise kopf.TemporaryError(msg, delay=delay)
