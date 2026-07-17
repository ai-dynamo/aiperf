# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the InferenceX AgentX context-overflow submission contract.

Covers the runtime classifier (``is_context_overflow_response``), the
``context_overflow_count`` aggregate metric, and the
``compute_submission_outcome`` fold that combines the static scenario-lock
outcome with the runtime overflow rate. Uses REAL config objects (no MagicMock)
so attribute-path drift on Environment / RequestRecord / ScenarioOutcome cannot
hide behind auto-created mock attributes.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.common.scenario import (
    CONTEXT_OVERFLOW_REASON,
    compute_submission_outcome,
    is_context_overflow_response,
)

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        param("This exceeds the maximum context length of 8192", True, id="raw_substr"),
        param("error: prompt is too long for the model", True, id="raw_prompt_long"),
        param('{"error": {"message": "context_length_exceeded"}}', True, id="openai"),
        param('{"error": "maximum context reached"}', True, id="openai_str_error"),
        param('{"detail": "context length exceeded"}', True, id="vllm_detail_raw"),
        param("Internal server error", False, id="normal_error"),
        param('{"error": {"message": "rate limit exceeded"}}', False, id="other_429"),
        param("", False, id="empty"),
        param(None, False, id="none"),
    ],
)  # fmt: skip
def test_is_context_overflow_response_classifies(body, expected):
    """Default allowlist matches overflow bodies and rejects normal errors."""
    assert is_context_overflow_response(body=body) is expected


def test_is_context_overflow_response_case_insensitive():
    """Matching is case-insensitive against both body and nested message."""
    assert is_context_overflow_response(body="MAXIMUM CONTEXT length") is True
    assert (
        is_context_overflow_response(body='{"error": {"message": "Context Length"}}')
        is True
    )


def test_is_context_overflow_response_bytes_body():
    """Bytes bodies are decoded (utf-8, replace) before matching."""
    assert is_context_overflow_response(body=b"maximum context exceeded") is True


def test_is_context_overflow_response_explicit_substrings_override():
    """An explicit substrings arg overrides the env allowlist entirely."""
    assert (
        is_context_overflow_response(
            body="custom overflow marker", substrings=["custom overflow"]
        )
        is True
    )
    # Default allowlist would not match this body.
    assert is_context_overflow_response(body="custom overflow marker") is False


def test_is_context_overflow_response_empty_allowlist_disables(monkeypatch):
    """An empty allowlist disables runtime detection (always False)."""
    monkeypatch.setattr(Environment.AGENTX, "CONTEXT_OVERFLOW_SUBSTRINGS", [])
    assert is_context_overflow_response(body="maximum context length") is False


def test_is_context_overflow_response_env_extension(monkeypatch):
    """Extending the env allowlist enables a new server vocabulary."""
    monkeypatch.setattr(
        Environment.AGENTX,
        "CONTEXT_OVERFLOW_SUBSTRINGS",
        ["token limit reached"],
    )
    assert is_context_overflow_response(body="token limit reached") is True
    assert is_context_overflow_response(body="maximum context length") is False


# ---------------------------------------------------------------------------
# compute_submission_outcome -- runtime rate fold
# ---------------------------------------------------------------------------


def test_compute_submission_outcome_over_threshold_flips_invalid():
    """> threshold overflow rate flips submission_valid=False + reason."""
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=True,
        total_responses=100,
        context_overflow_count=2,
    )
    assert valid is False
    assert CONTEXT_OVERFLOW_REASON in reasons


def test_compute_submission_outcome_under_threshold_unaffected():
    """< threshold overflow rate leaves the lock outcome unaffected."""
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=True,
        total_responses=1000,
        context_overflow_count=5,
    )
    assert valid is True
    assert reasons == []


def test_compute_submission_outcome_boundary_equal_accepted():
    """Rate exactly equal to the limit is accepted (strict greater-than)."""
    # 1/100 == 0.01 == default limit -> accepted.
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=True,
        total_responses=100,
        context_overflow_count=1,
    )
    assert valid is True
    assert reasons == []


def test_compute_submission_outcome_zero_responses_no_flip():
    """Zero total responses treats the rate as 0 -- no overflow flip."""
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=True,
        total_responses=0,
        context_overflow_count=0,
    )
    assert valid is True
    assert reasons == []


def test_compute_submission_outcome_preserves_lock_reasons():
    """The lock-only outcome (unsafe_override) is preserved and merged."""
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=False,
        validator_reasons=["unsafe_override"],
        total_responses=100,
        context_overflow_count=5,
    )
    assert valid is False
    assert "unsafe_override" in reasons
    assert CONTEXT_OVERFLOW_REASON in reasons


def test_compute_submission_outcome_lock_only_under_threshold_stays_invalid():
    """Lock-invalid + under-threshold overflow stays invalid, no overflow reason."""
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=False,
        validator_reasons=["unsafe_override"],
        total_responses=100,
        context_overflow_count=0,
    )
    assert valid is False
    assert reasons == ["unsafe_override"]


def test_compute_submission_outcome_no_scenario_returns_none():
    """No scenario -> (None, []); callers drop the field."""
    valid, reasons = compute_submission_outcome(
        scenario_name=None,
        validator_submission_valid=None,
        total_responses=100,
        context_overflow_count=50,
    )
    assert valid is None
    assert reasons == []


def test_compute_submission_outcome_cancelled_flips_invalid():
    """A cancelled run is never a valid submission."""
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=True,
        total_responses=100,
        context_overflow_count=0,
        was_cancelled=True,
    )
    assert valid is False
    assert "run_cancelled" in reasons


def test_compute_submission_outcome_threshold_override(monkeypatch):
    """A raised env threshold tolerates a higher overflow rate."""
    monkeypatch.setattr(Environment.AGENTX, "CONTEXT_OVERFLOW_RATE_LIMIT", 0.5)
    # 10% overflow now under the 50% limit.
    valid, reasons = compute_submission_outcome(
        scenario_name="s",
        validator_submission_valid=True,
        total_responses=100,
        context_overflow_count=10,
    )
    assert valid is True
    assert reasons == []


# ---------------------------------------------------------------------------
# context_overflow_count metric
# ---------------------------------------------------------------------------


def test_context_overflow_count_metric_registered():
    """The metric registers under its tag with ERROR_ONLY flag."""
    from aiperf.common.enums import MetricFlags
    from aiperf.metrics import MetricRegistry

    cls = MetricRegistry.get_class("context_overflow_count")
    assert cls is not None
    assert cls.tag == "context_overflow_count"
    assert MetricFlags.ERROR_ONLY in cls.flags
    assert MetricFlags.NO_INDIVIDUAL_RECORDS in cls.flags


def test_context_overflow_count_metric_counts_flagged_records():
    """_parse_record returns 1 only when request.context_overflow is True."""
    from types import SimpleNamespace

    from aiperf.metrics.metric_dicts import MetricRecordDict
    from aiperf.metrics.types.context_overflow_count_metric import (
        ContextOverflowCountMetric,
    )

    metric = ContextOverflowCountMetric()
    overflow_rec = SimpleNamespace(request=SimpleNamespace(context_overflow=True))
    normal_rec = SimpleNamespace(request=SimpleNamespace(context_overflow=False))
    assert metric._parse_record(overflow_rec, MetricRecordDict()) == 1
    assert metric._parse_record(normal_rec, MetricRecordDict()) == 0
