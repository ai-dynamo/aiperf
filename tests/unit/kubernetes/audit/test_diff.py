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


from tests.kubernetes.audit.diff import diff_tolerance  # noqa: E402


def _write_summary(
    root: Path, *, mean: float, p50: float, p99: float, throughput: float
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "request_latency": {"avg": mean, "p50": p50, "p99": p99},
                "request_throughput": {"avg": throughput},
            }
        )
    )


def test_diff_tolerance_within_band_returns_no_findings(
    tmp_path: Path, case: AuditCase
) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _write_summary(op, mean=100.0, p50=95.0, p99=200.0, throughput=50.0)
    _write_summary(bare, mean=105.0, p50=99.0, p99=220.0, throughput=51.0)

    findings = diff_tolerance(operator_dir=op, bare_dir=bare, case=case)

    assert findings == []


def test_diff_tolerance_mean_out_of_band_is_reported(
    tmp_path: Path, case: AuditCase
) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _write_summary(op, mean=200.0, p50=95.0, p99=200.0, throughput=50.0)
    _write_summary(bare, mean=100.0, p50=95.0, p99=200.0, throughput=50.0)

    findings = diff_tolerance(operator_dir=op, bare_dir=bare, case=case)

    assert any("avg" in f.field and f.bucket == "tolerance" for f in findings)


def test_diff_tolerance_per_case_override_relaxes_band(tmp_path: Path) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _write_summary(op, mean=100.0, p50=95.0, p99=350.0, throughput=50.0)
    _write_summary(bare, mean=100.0, p50=95.0, p99=222.0, throughput=50.0)

    case = AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        metric_tolerance_overrides={"p99": 0.60},
    )
    findings = diff_tolerance(operator_dir=op, bare_dir=bare, case=case)

    assert findings == []


from tests.kubernetes.audit.diff import diff_structural  # noqa: E402


def test_diff_structural_missing_expected_artifact_is_reported(tmp_path: Path) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    op.mkdir()
    bare.mkdir()
    (bare / "profile_export.jsonl").write_text("")
    case = AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        expected_artifacts=("profile_export.jsonl",),
    )

    findings = diff_structural(operator_dir=op, bare_dir=bare, case=case)

    assert any(
        f.field == "profile_export.jsonl" and "operator" in f.reason for f in findings
    )


def test_diff_structural_csv_header_mismatch_is_reported(tmp_path: Path) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    op.mkdir()
    bare.mkdir()
    (op / "profile_export_records.csv").write_text("a,b,c\n1,2,3\n")
    (bare / "profile_export_records.csv").write_text("a,b\n1,2\n")
    case = AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        expected_artifacts=("profile_export_records.csv",),
    )

    findings = diff_structural(operator_dir=op, bare_dir=bare, case=case)

    assert any(
        f.field.endswith("profile_export_records.csv") and "header" in f.reason
        for f in findings
    )


def test_diff_structural_json_top_level_keyset_mismatch_is_reported(
    tmp_path: Path,
) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    op.mkdir()
    bare.mkdir()
    (op / "inputs.json").write_text(json.dumps({"a": 1, "b": 2}))
    (bare / "inputs.json").write_text(json.dumps({"a": 1, "b": 2, "c": 3}))
    case = AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        expected_artifacts=("inputs.json",),
    )

    findings = diff_structural(operator_dir=op, bare_dir=bare, case=case)

    assert any(
        f.field.endswith("inputs.json") and "key" in f.reason.lower() for f in findings
    )
