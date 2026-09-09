# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderers for ``AuditFindings`` -> markdown / JSON."""

from __future__ import annotations

import orjson

from tests.kubernetes.audit.diff import GATING_BUCKETS, AuditFindings, Finding


def render_json(findings: AuditFindings) -> str:
    gating = [f for f in findings.findings if f.bucket in GATING_BUCKETS]
    payload = {
        "case_id": findings.case_id,
        # "passed" must mean what the test asserts. Reporting `empty` marked
        # every run failed over environment-dependent timing drift the test
        # deliberately does not gate on, so a genuine exact/structural
        # regression was indistinguishable from a busy machine.
        "passed": not gating,
        "advisory_only": bool(findings.findings) and not gating,
        "findings": [
            {
                "bucket": f.bucket,
                "field": f.field,
                "expected": f.expected,
                "actual": f.actual,
                "reason": f.reason,
            }
            for f in findings.findings
        ],
    }
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()


def _render_section(title: str, items: list[Finding]) -> list[str]:
    lines = [f"## {title}", ""]
    if not items:
        lines.append("_(no findings)_")
        lines.append("")
        return lines
    lines.append("| field | expected | actual | reason |")
    lines.append("| --- | --- | --- | --- |")
    for f in items:
        lines.append(f"| `{f.field}` | `{f.expected}` | `{f.actual}` | {f.reason} |")
    lines.append("")
    return lines


def render_markdown(findings: AuditFindings) -> str:
    status = "PASS" if findings.empty else "FAIL"
    lines: list[str] = [
        f"# Audit Report: `{findings.case_id}` — {status}",
        "",
    ]
    for bucket_title, bucket_key in (
        ("Exact", "exact"),
        ("Tolerance", "tolerance"),
        ("Structural", "structural"),
    ):
        items = [f for f in findings.findings if f.bucket == bucket_key]
        lines.extend(_render_section(bucket_title, items))
    return "\n".join(lines)
