# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-bucket diff for the K8s-vs-local audit suite.

Each bucket is a pure function over two artifact directory trees plus an
``AuditCase``. Functions return a list of ``Finding``s; an empty list means
no divergence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tests.kubernetes.audit.cases import AuditCase

Bucket = Literal["exact", "tolerance", "structural"]


@dataclass(frozen=True)
class Finding:
    bucket: Bucket
    field: str
    expected: Any
    actual: Any
    reason: str


@dataclass(frozen=True)
class AuditFindings:
    case_id: str
    findings: list[Finding]

    @property
    def empty(self) -> bool:
        return not self.findings


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def _file_set(root: Path) -> set[str]:
    """Top-level filenames in ``root`` (non-recursive)."""
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_file()}


def _dataset_hash(root: Path) -> str | None:
    """SHA-256 of profile_export.jsonl payload field, sorted for stability."""
    rows = _read_jsonl(root / "profile_export.jsonl")
    if not rows:
        return None
    payloads = sorted(json.dumps(r.get("payload", {}), sort_keys=True) for r in rows)
    h = hashlib.sha256()
    for p in payloads:
        h.update(p.encode())
    return h.hexdigest()


def _record_count(root: Path) -> int:
    """Prefer JSONL record count; fall back to records.csv if JSONL absent."""
    jsonl = root / "profile_export.jsonl"
    if jsonl.exists():
        return len(_read_jsonl(jsonl))
    return _csv_row_count(root / "profile_export_records.csv")


def _error_count(root: Path) -> int:
    rows = _read_jsonl(root / "profile_export.jsonl")
    return sum(1 for r in rows if r.get("error"))


def _inputs_args(root: Path) -> dict[str, Any]:
    p = root / "inputs.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def diff_exact(
    *,
    operator_dir: Path,
    bare_dir: Path,
    case: AuditCase,
) -> list[Finding]:
    """Bucket 1: fields that must match byte-for-byte."""
    findings: list[Finding] = []

    op_count = _record_count(operator_dir)
    bare_count = _record_count(bare_dir)
    counts_differ = op_count != bare_count
    if counts_differ:
        findings.append(
            Finding(
                bucket="exact",
                field="request_count",
                expected=bare_count,
                actual=op_count,
                reason="record count differs between operator and bare-pod runs",
            )
        )

    op_errors = _error_count(operator_dir)
    bare_errors = _error_count(bare_dir)
    if op_errors != bare_errors:
        findings.append(
            Finding(
                bucket="exact",
                field="error_count",
                expected=bare_errors,
                actual=op_errors,
                reason="error rows present or counts differ",
            )
        )

    op_args = _inputs_args(operator_dir)
    bare_args = _inputs_args(bare_dir)
    # When record counts already differ, skip downstream config-echo findings
    # for `request_count` since they restate the same divergence.
    skip_inputs = {"request_count"} if counts_differ else set()
    for key in ("endpoint_type", "concurrency", "request_count"):
        if key in skip_inputs:
            continue
        if op_args.get(key) != bare_args.get(key):
            findings.append(
                Finding(
                    bucket="exact",
                    field=f"inputs.{key}",
                    expected=bare_args.get(key),
                    actual=op_args.get(key),
                    reason="configured-args echo differs",
                )
            )

    # Dataset hash is only meaningful when both sides have the same record count;
    # different counts already produce a primary finding.
    if not counts_differ:
        op_hash = _dataset_hash(operator_dir)
        bare_hash = _dataset_hash(bare_dir)
        if op_hash is not None and bare_hash is not None and op_hash != bare_hash:
            findings.append(
                Finding(
                    bucket="exact",
                    field="dataset_hash",
                    expected=bare_hash,
                    actual=op_hash,
                    reason="seeded dataset payloads diverged between modes",
                )
            )

    op_files = _file_set(operator_dir)
    bare_files = _file_set(bare_dir)
    if op_files != bare_files:
        findings.append(
            Finding(
                bucket="exact",
                field="file_set",
                expected=sorted(bare_files),
                actual=sorted(op_files),
                reason="exporter file set differs",
            )
        )

    return findings
