# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.handlers._completion_retry``.

The retry gate raises ``kopf.TemporaryError`` when a results-fetch failure
looks transient and the completion claim is still inside the budget. Tests
exercise:

- pre-check shape: returns silently when has_files OR (no error AND no partial).
- partial-progress short-circuit: metrics-only / downloaded-only must enter the
  budget check, not skip it.
- budget gating: 0 disables; over budget → silent return; within budget → raise.
- claim age parsing: missing / unparsable annotation → silent return.
- error-message content: the ``TemporaryError`` message must name the namespace,
  job_id, claim age, budget, and the upstream error string for ops debuggability.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import kopf
import pytest

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.environment import _ResultsSettings
from aiperf.operator.handlers._completion_retry import (
    maybe_raise_for_transient_fetch_failure,
)
from aiperf.operator.handlers.completion import _ResultFlags
from aiperf.operator.models import ControllerFetchResult


def _now_iso(offset_seconds: float = 0.0) -> str:
    """ISO timestamp; offset_seconds<0 → in the past (older claim)."""
    return (datetime.now(UTC) + timedelta(seconds=offset_seconds)).isoformat()


def _body(claim_ts: str | None) -> dict[str, Any]:
    annotations: dict[str, str] = {}
    if claim_ts is not None:
        annotations[Annotations.COMPLETION_CLAIMED] = claim_ts
    return {"metadata": {"annotations": annotations}}


def _flags(
    *,
    has_metrics: bool = False,
    has_files: bool = False,
    has_error: bool = True,
    success: bool = False,
) -> _ResultFlags:
    return _ResultFlags(
        has_metrics=has_metrics,
        has_files=has_files,
        has_error=has_error,
        success=success,
    )


def _result(
    *,
    error: str = "connection refused",
    metrics: dict | None = None,
    downloaded: list[str] | None = None,
) -> ControllerFetchResult:
    return ControllerFetchResult(
        metrics=metrics,
        downloaded=downloaded or [],
        error=error,
    )


@pytest.fixture
def env_with_budget():
    """Patch RESULTS at the ``_completion_retry`` import site (matching the
    pattern used by ``test_completion_handler.py``).

    Mutating the live ``OperatorEnvironment.RESULTS`` Pydantic instance via
    ``monkeypatch.setattr`` is unreliable under xdist when sibling tests use
    ``unittest.mock.patch(... .OperatorEnvironment.RESULTS ...)`` — patch
    teardown can land back on a stale view of the singleton. Patching the
    module-local import binding sidesteps that entirely.
    """
    patches: list[Any] = []

    def _set(budget: float = 60.0, delay: float = 5.0) -> None:
        p = patch(
            "aiperf.operator.handlers._completion_retry.OperatorEnvironment.RESULTS",
            new=_ResultsSettings(
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=budget,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=delay,
            ),
        )
        p.start()
        patches.append(p)

    yield _set

    for p in patches:
        p.stop()


class TestPreCheckShortCircuits:
    def test_has_files_returns_silently_even_with_error(self, env_with_budget) -> None:
        env_with_budget()
        # has_files=True overrides has_error — caller is presumed to have
        # written terminal status already.
        maybe_raise_for_transient_fetch_failure(
            body=_body(_now_iso(-5)),
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(has_files=True, has_error=True),
        )

    def test_no_error_and_no_partial_progress_returns_silently(
        self, env_with_budget
    ) -> None:
        env_with_budget()
        maybe_raise_for_transient_fetch_failure(
            body=_body(_now_iso(-5)),
            namespace="ns",
            job_id="job",
            result=_result(error=""),
            flags=_flags(has_error=False),
        )


class TestPartialProgressEntersBudgetCheck:
    def test_metrics_present_no_error_still_retries(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        with pytest.raises(kopf.TemporaryError):
            maybe_raise_for_transient_fetch_failure(
                body=_body(_now_iso(-5)),
                namespace="ns",
                job_id="job",
                result=_result(error="", metrics={"some": "data"}),
                flags=_flags(has_metrics=True, has_error=False),
            )

    def test_downloaded_present_no_error_still_retries(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        with pytest.raises(kopf.TemporaryError):
            maybe_raise_for_transient_fetch_failure(
                body=_body(_now_iso(-5)),
                namespace="ns",
                job_id="job",
                result=_result(error="", downloaded=["checkpoint-1.json"]),
                flags=_flags(has_error=False),
            )


class TestBudgetGating:
    def test_zero_budget_disables_retry(self, env_with_budget) -> None:
        env_with_budget(budget=0.0)
        # Despite has_error and a fresh claim, no raise.
        maybe_raise_for_transient_fetch_failure(
            body=_body(_now_iso(-1)),
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(),
        )

    def test_within_budget_raises_temporary_error(self, env_with_budget) -> None:
        env_with_budget(budget=60.0, delay=5.0)
        with pytest.raises(kopf.TemporaryError) as exc_info:
            maybe_raise_for_transient_fetch_failure(
                body=_body(_now_iso(-10)),
                namespace="ns",
                job_id="job",
                result=_result(),
                flags=_flags(),
            )
        assert exc_info.value.delay == 5.0

    def test_over_budget_returns_silently(self, env_with_budget) -> None:
        env_with_budget(budget=10.0)
        maybe_raise_for_transient_fetch_failure(
            body=_body(_now_iso(-30)),
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(),
        )


class TestClaimAgeParsing:
    def test_missing_annotation_returns_silently(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        maybe_raise_for_transient_fetch_failure(
            body=_body(claim_ts=None),
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(),
        )

    def test_unparsable_annotation_returns_silently(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        maybe_raise_for_transient_fetch_failure(
            body=_body(claim_ts="not-a-timestamp"),
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(),
        )

    def test_empty_annotation_returns_silently(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        maybe_raise_for_transient_fetch_failure(
            body=_body(claim_ts=""),
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(),
        )

    def test_missing_metadata_section_returns_silently(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        maybe_raise_for_transient_fetch_failure(
            body={},
            namespace="ns",
            job_id="job",
            result=_result(),
            flags=_flags(),
        )


class TestErrorMessageContent:
    def test_message_includes_namespace_job_age_budget_error(
        self, env_with_budget
    ) -> None:
        env_with_budget(budget=60.0)
        with pytest.raises(kopf.TemporaryError) as exc_info:
            maybe_raise_for_transient_fetch_failure(
                body=_body(_now_iso(-15)),
                namespace="prod-ns",
                job_id="bench-7f2a",
                result=_result(error="connection refused on :19090"),
                flags=_flags(),
            )
        msg = str(exc_info.value)
        assert "prod-ns/bench-7f2a" in msg
        assert "60s budget" in msg
        assert "connection refused on :19090" in msg

    def test_message_when_error_is_empty(self, env_with_budget) -> None:
        env_with_budget(budget=60.0)
        with pytest.raises(kopf.TemporaryError) as exc_info:
            maybe_raise_for_transient_fetch_failure(
                body=_body(_now_iso(-5)),
                namespace="ns",
                job_id="job",
                result=_result(error="", metrics={"x": 1}),
                flags=_flags(has_metrics=True, has_error=False),
            )
        assert "no detail" in str(exc_info.value)
