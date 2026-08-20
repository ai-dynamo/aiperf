# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-bucket diff for the K8s-vs-local audit suite.

Each bucket is a pure function over two artifact directory trees plus an
``AuditCase``. Functions return a list of ``Finding``s; an empty list means
no divergence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tests.kubernetes.audit.cases import AuditCase

Bucket = Literal["exact", "tolerance", "structural", "index_consistency"]


@dataclass(frozen=True)
class Finding:
    bucket: Bucket
    field: str
    expected: Any
    actual: Any
    reason: str


GATING_BUCKETS: frozenset[str] = frozenset({"exact", "structural", "index_consistency"})
"""Buckets that make an audit a failure.

``tolerance`` is deliberately excluded: it compares absolute timing between
two different topologies -- the operator spreads concurrency across worker
pods over ZMQ while the bare run is a single process -- so on a contended
single-node kind cluster it drifts by 50-80% with nothing wrong. The test
gates on these buckets and offers --audit-strict-tolerance for a controlled
machine. See ``test_audit.py``.
"""


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

    # File-set equality intentionally NOT checked: operator and bare emit
    # legitimately different supersets (operator wraps in extras like
    # server_metrics_export.parquet and job_spec.json). The structural bucket
    # already enforces presence of every file in `case.expected_artifacts` on
    # both sides — that is the right invariant.

    return findings


_DEFAULT_BANDS: dict[str, float] = {
    "avg": 0.10,
    "mean": 0.10,
    "p50": 0.10,
    "median": 0.10,
    "p90": 0.25,
    "p95": 0.25,
    "p99": 0.25,
    "throughput": 0.10,
    "min": 0.25,
    "max": 0.25,
    "std": 0.50,
}
_EPS = 1e-9


def _band_for(stat_key: str, case: AuditCase) -> float:
    if stat_key in case.metric_tolerance_overrides:
        return case.metric_tolerance_overrides[stat_key]
    return _DEFAULT_BANDS.get(stat_key, 0.10)


def _summary_path(root: Path) -> Path | None:
    """Return the canonical summary JSON path, if present."""
    if not root.exists():
        return None
    candidates = sorted(root.glob("profile_export_*.json"))
    for c in candidates:
        if "_partial" in c.name or "_timeslices" in c.name:
            continue
        return c
    return None


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten ``{metric: {stat: value}}`` into ``{metric.stat: value}``."""
    out: dict[str, float] = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, prefix=f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), _EPS)
    return abs(a - b) / denom


def diff_tolerance(
    *,
    operator_dir: Path,
    bare_dir: Path,
    case: AuditCase,
) -> list[Finding]:
    """Bucket 2: numeric stats compared with per-stat-suffix relative bands."""
    findings: list[Finding] = []

    op_path = _summary_path(operator_dir)
    bare_path = _summary_path(bare_dir)
    if op_path is None or bare_path is None:
        return findings

    op = _flatten_metrics(json.loads(op_path.read_text()))
    bare = _flatten_metrics(json.loads(bare_path.read_text()))

    for field_name, bare_value in bare.items():
        if field_name not in op:
            continue
        op_value = op[field_name]
        stat_key = field_name.rsplit(".", 1)[-1].lower()
        band = _band_for(stat_key, case)
        rel = _relative_diff(op_value, bare_value)
        if rel > band:
            findings.append(
                Finding(
                    bucket="tolerance",
                    field=field_name,
                    expected=bare_value,
                    actual=op_value,
                    reason=f"relative diff {rel:.1%} exceeds band {band:.1%} for stat '{stat_key}'",
                )
            )

    return findings


_NARROW_METRICS = (
    "request_throughput",
    "request_latency",
    "time_to_first_token",
    "output_token_throughput",
    "output_token_throughput_per_user",
    "inter_token_latency",
)
_NARROW_STATS = ("avg", "p50", "p99")
_INDEX_FLOAT_TOL = 1e-9


def diff_index_consistency(
    *,
    operator_dir: Path,
    index_row: dict[str, Any] | None,
) -> list[Finding]:
    """Bucket 4: runs-index row vs on-disk summary, narrow metric columns only.

    The index is a cache mirroring disk: every flat column in the ``runs``
    table must equal the matching ``profile_export_aiperf.json`` stat. Any
    drift is silent corruption of the read API (leaderboard, history,
    compare) so the audit always gates on this bucket.

    ``index_row`` is the JSON payload returned by
    ``GET /admin/index/run/{ns}/{job_id}`` (the narrow-column projection).
    Pass ``None`` to record a finding when the index has no row at all.
    """
    findings: list[Finding] = []

    if index_row is None:
        findings.append(
            Finding(
                bucket="index_consistency",
                field="row",
                expected="present",
                actual="missing",
                reason="runs_index has no row for this operator-side run",
            )
        )
        return findings

    summary_path = _summary_path(operator_dir)
    if summary_path is None:
        findings.append(
            Finding(
                bucket="index_consistency",
                field="profile_export_aiperf.json",
                expected="present",
                actual="missing",
                reason="operator-side summary JSON missing; cannot verify index row",
            )
        )
        return findings

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        findings.append(
            Finding(
                bucket="index_consistency",
                field=summary_path.name,
                expected="parseable JSON",
                actual=f"decode error: {exc}",
                reason="cannot decode operator-side summary; index check skipped",
            )
        )
        return findings

    for metric in _NARROW_METRICS:
        m = summary.get(metric) or {}
        for stat in _NARROW_STATS:
            disk_val = m.get(stat)
            row_val = index_row.get(f"{metric}_{stat}")
            if disk_val is None and row_val is None:
                continue
            if disk_val is None or row_val is None:
                findings.append(
                    Finding(
                        bucket="index_consistency",
                        field=f"{metric}.{stat}",
                        expected=disk_val,
                        actual=row_val,
                        reason="one side has a value and the other is null",
                    )
                )
                continue
            if abs(float(disk_val) - float(row_val)) > _INDEX_FLOAT_TOL:
                findings.append(
                    Finding(
                        bucket="index_consistency",
                        field=f"{metric}.{stat}",
                        expected=disk_val,
                        actual=row_val,
                        reason=(
                            f"runs index row diverges from disk by "
                            f"{abs(float(disk_val) - float(row_val)):.3e}"
                        ),
                    )
                )

    return findings


def _csv_header(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open() as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _json_keyset_depth2(path: Path) -> set[str] | None:
    """Top-level + depth-1 keys, joined with '.'. Returns None on read failure.

    ``input_config.*`` is intentionally excluded: that subtree carries
    config-construction metadata (``runtime.api_port``, ``logging.level``,
    ``metrics``, ``variables``) that legitimately differs between the bare
    CLI path and the operator's spec_converter path. The audit's invariant
    is metric-schema parity, not config-wrapper parity.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return set()
    keys: set[str] = {k for k in payload if k != "input_config"}
    for k, v in payload.items():
        if k == "input_config":
            continue
        if isinstance(v, dict):
            keys.update(f"{k}.{kk}" for kk in v)
    return keys


def diff_structural(
    *,
    operator_dir: Path,
    bare_dir: Path,
    case: AuditCase,
) -> list[Finding]:
    """Bucket 3: file presence + per-file schema (CSV header / JSON key set)."""
    findings: list[Finding] = []

    for filename in case.expected_artifacts:
        op_path = operator_dir / filename
        bare_path = bare_dir / filename

        op_present = op_path.exists()
        bare_present = bare_path.exists()
        if not op_present:
            findings.append(
                Finding(
                    bucket="structural",
                    field=filename,
                    expected="present",
                    actual="missing",
                    reason=f"expected artifact missing on operator side: {filename}",
                )
            )
        if not bare_present:
            findings.append(
                Finding(
                    bucket="structural",
                    field=filename,
                    expected="present",
                    actual="missing",
                    reason=f"expected artifact missing on bare side: {filename}",
                )
            )
        if not (op_present and bare_present):
            continue

        if filename.endswith(".csv"):
            op_header = _csv_header(op_path)
            bare_header = _csv_header(bare_path)
            if op_header != bare_header:
                findings.append(
                    Finding(
                        bucket="structural",
                        field=f"schema:{filename}",
                        expected=bare_header,
                        actual=op_header,
                        reason=f"CSV header set differs in {filename}",
                    )
                )
        elif filename.endswith(".json"):
            op_keys = _json_keyset_depth2(op_path)
            bare_keys = _json_keyset_depth2(bare_path)
            if op_keys is not None and bare_keys is not None and op_keys != bare_keys:
                findings.append(
                    Finding(
                        bucket="structural",
                        field=f"schema:{filename}",
                        expected=sorted(bare_keys),
                        actual=sorted(op_keys),
                        reason=f"JSON depth-2 key set differs in {filename}",
                    )
                )

    return findings
