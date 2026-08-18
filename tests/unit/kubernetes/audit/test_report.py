# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``tests/kubernetes/audit/report.py``."""

from __future__ import annotations

import json

import pytest

from tests.kubernetes.audit.diff import AuditFindings, Finding
from tests.kubernetes.audit.report import render_json, render_markdown


@pytest.fixture
def findings() -> AuditFindings:
    return AuditFindings(
        case_id="baseline-chat",
        findings=[
            Finding(
                bucket="exact",
                field="request_count",
                expected=64,
                actual=63,
                reason="off by one",
            ),
            Finding(
                bucket="tolerance",
                field="request_latency.p99",
                expected=200.0,
                actual=350.0,
                reason="75% > 25%",
            ),
            Finding(
                bucket="structural",
                field="profile_export_records.csv",
                expected="present",
                actual="missing",
                reason="missing on operator",
            ),
        ],
    )


def test_render_json_round_trips(findings: AuditFindings) -> None:
    text = render_json(findings)
    payload = json.loads(text)
    assert payload["case_id"] == "baseline-chat"
    assert len(payload["findings"]) == 3
    assert payload["findings"][0]["bucket"] == "exact"
    assert payload["findings"][0]["field"] == "request_count"


def test_render_markdown_contains_each_bucket_and_pass_fail_header(
    findings: AuditFindings,
) -> None:
    text = render_markdown(findings)
    assert "FAIL" in text
    assert "## Exact" in text
    assert "## Tolerance" in text
    assert "## Structural" in text
    assert "request_count" in text
    assert "p99" in text


def test_render_markdown_pass_when_empty() -> None:
    f = AuditFindings(case_id="baseline-chat", findings=[])
    text = render_markdown(f)
    assert "PASS" in text


class TestPassedMeansGating:
    """`passed` must mean what the test actually asserts.

    It reported `findings.empty`, so every stored audit read passed=false over
    tolerance drift the harness deliberately does not gate on -- making a real
    exact/structural regression indistinguishable from a busy machine. All
    four checked-in baselines had 1,208 findings between them and zero gating
    ones.
    """

    def _payload(self, buckets: list[str]) -> dict:
        import orjson

        from tests.kubernetes.audit.diff import AuditFindings, Finding
        from tests.kubernetes.audit.report import render_json

        findings = AuditFindings(
            case_id="c",
            findings=[
                Finding(bucket=b, field="f", expected=1, actual=2, reason="r")
                for b in buckets
            ],
        )
        return orjson.loads(render_json(findings))

    def test_tolerance_only_is_advisory(self) -> None:
        payload = self._payload(["tolerance", "tolerance"])
        assert payload["passed"] is True
        assert payload["advisory_only"] is True
        assert len(payload["findings"]) == 2

    def test_exact_finding_fails(self) -> None:
        payload = self._payload(["tolerance", "exact"])
        assert payload["passed"] is False
        assert payload["advisory_only"] is False

    def test_structural_finding_fails(self) -> None:
        assert self._payload(["structural"])["passed"] is False

    def test_index_consistency_finding_fails(self) -> None:
        assert self._payload(["index_consistency"])["passed"] is False

    def test_clean_run_is_not_advisory(self) -> None:
        payload = self._payload([])
        assert payload["passed"] is True
        assert payload["advisory_only"] is False
