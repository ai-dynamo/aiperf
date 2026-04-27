# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``tests/kubernetes/audit/diff.py``.

Builds synthetic artifact trees with known divergences and asserts the
expected ``Finding`` objects come out of each bucket.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.audit.diff import (
    AuditFindings,
    Finding,
    diff_exact,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _make_tree(root: Path, request_count: int, errors: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs.json").write_text(
        json.dumps(
            {"endpoint_type": "chat", "concurrency": 4, "request_count": request_count}
        )
    )
    rows = [{"request_index": i, "error": (i < errors)} for i in range(request_count)]
    _write_jsonl(root / "profile_export.jsonl", rows)
    _write_csv(
        root / "profile_export_records.csv",
        header=["request_index", "ttft_ms"],
        rows=[[str(i), "10.0"] for i in range(request_count)],
    )


@pytest.fixture
def case() -> AuditCase:
    return AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        num_conversations=5,
    )


def test_diff_exact_matching_trees_returns_no_findings(
    tmp_path: Path, case: AuditCase
) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _make_tree(op, request_count=10)
    _make_tree(bare, request_count=10)

    findings = diff_exact(operator_dir=op, bare_dir=bare, case=case)

    assert findings == []


def test_diff_exact_request_count_mismatch_is_reported(
    tmp_path: Path, case: AuditCase
) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _make_tree(op, request_count=9)
    _make_tree(bare, request_count=10)

    findings = diff_exact(operator_dir=op, bare_dir=bare, case=case)

    assert len(findings) == 1
    f = findings[0]
    assert f.bucket == "exact"
    assert f.field == "request_count"
    assert f.expected == 10
    assert f.actual == 9


def test_diff_exact_error_count_nonzero_is_reported(
    tmp_path: Path, case: AuditCase
) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _make_tree(op, request_count=10, errors=2)
    _make_tree(bare, request_count=10, errors=0)

    findings = diff_exact(operator_dir=op, bare_dir=bare, case=case)

    assert any(f.field == "error_count" and f.actual == 2 for f in findings)


def test_audit_findings_empty_property() -> None:
    f = AuditFindings(case_id="x", findings=[])
    assert f.empty is True
    f2 = AuditFindings(
        case_id="x",
        findings=[Finding(bucket="exact", field="x", expected=1, actual=2, reason="r")],
    )
    assert f2.empty is False
