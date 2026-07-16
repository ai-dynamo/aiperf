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
