# K8s-vs-Local Correctness Audit Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in pytest suite (`tests/kubernetes/audit/`) that runs each of three single-run workflow cases twice — once via the operator + `aiperf kube results` download path, once via a bare `batch/v1.Job` running `aiperf profile` — and diffs the artifact trees through three buckets (exact / tolerance / structural).

**Scope note:** The spec lists five cases. Two — `multi-epoch` and `small-sweep` — depend on operator-side helpers (`AIPerfJobConfig.epochs`, an `AIPerfSweep` runner) that do not exist yet. Those cases are deferred to a followup plan once the helpers land. The audit harness (deployers, diff, report) ships now with the three single-run cases; adding the deferred cases later is a `cases.py` change plus, where needed, helper extensions.

**Architecture:** Reuses `tests/kubernetes/conftest.py` cluster + image fixtures. New module under `tests/kubernetes/audit/` with five small files: `cases.py`, `bare_pod.py`, `operator_runner.py`, `diff.py`, `report.py`, plus `conftest.py` and `test_audit.py`. The bare-pod side is the oracle; any divergence beyond bucket tolerances fails the operator-side run. Pure-Python diff/report logic is unit-tested independently of the cluster.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, kubectl shellouts, kubernetes_asyncio (already used by helpers), orjson, the existing `aiperf:local` image, the existing kind harness.

**Spec:** `docs/superpowers/specs/2026-04-26-k8s-vs-local-audit-design.md`

---

## File Structure

```
tests/kubernetes/audit/
  __init__.py             # empty package marker
  conftest.py             # k8s_audit marker, --audit-repeats option, audit fixtures
  cases.py                # AuditCase dataclass + AUDIT_CASES list (5 cases)
  diff.py                 # Finding, AuditFindings, diff_exact/tolerance/structural
  report.py               # render_markdown, render_json
  bare_pod.py             # BarePodDeployer.run(case) -> Path
  operator_runner.py      # OperatorAuditRunner.run(case) -> Path
  test_audit.py           # one parametrized test using AUDIT_CASES
tests/unit/kubernetes/audit/
  __init__.py
  test_diff.py            # synthetic-tree unit tests for the three buckets
  test_report.py          # rendering tests
```

Modified:

- `pyproject.toml`: register `k8s_audit` marker; add it to the default-deselect list in `addopts`.
- `CLAUDE.md` (and the two sync files): one-line reference to the new audit module under the Kubernetes section.

---

## Task 1: Register the `k8s_audit` marker

**Files:**
- Modify: `pyproject.toml:182-210` (markers list + `addopts`)

- [ ] **Step 1: Add the marker**

In `pyproject.toml`, inside `[tool.pytest.ini_options].markers` (the list that already contains `"integration: marks tests as integration tests"`), add a new entry:

```toml
    "k8s_audit: marks Kubernetes operator-vs-bare-pod correctness audit tests",
```

- [ ] **Step 2: Add to default-deselect**

In the same file, append `and not k8s_audit` to the `addopts` `-m` expression. Final form:

```toml
addopts = "--strict-markers -m 'not performance and not ffmpeg and not stress and not statistical and not component_integration and not integration and not server_unit and not gpu and not vllm and not dynamo and not trtllm and not sglang and not k8s and not fern and not e2e and not k8s_audit'"
```

- [ ] **Step 3: Verify pytest accepts it**

Run: `uv run pytest --collect-only -m k8s_audit tests/ 2>&1 | tail -5`
Expected: `0 tests collected` with no `Unknown marker` warning.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -s -m "test(k8s-audit): register k8s_audit pytest marker"
```

---

## Task 2: Audit module skeleton + conftest

**Files:**
- Create: `tests/kubernetes/audit/__init__.py`
- Create: `tests/kubernetes/audit/conftest.py`
- Create: `tests/unit/kubernetes/audit/__init__.py`

- [ ] **Step 1: Create empty package markers**

```python
# tests/kubernetes/audit/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

```python
# tests/unit/kubernetes/audit/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Write the audit conftest**

Create `tests/kubernetes/audit/conftest.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures and options for the K8s-vs-local audit suite.

The audit suite reuses the cluster, image, and mock-server fixtures from
``tests/kubernetes/conftest.py`` and adds:

- ``--audit-repeats N`` (default 1): per-side repeat count. When N > 1, each
  side runs N times and per-metric medians are diffed; useful locally when
  investigating a divergence.
- ``audit_artifacts_dir`` fixture: per-test directory under
  ``tests/_artifacts/audit/<case-id>/`` where both modes' artifacts and the
  rendered report are written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--audit-repeats",
        action="store",
        type=int,
        default=1,
        help="Per-side repeat count for the k8s_audit suite (default 1).",
    )


@pytest.fixture(scope="session")
def audit_repeats(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--audit-repeats"))


@pytest.fixture
def audit_artifacts_dir(request: pytest.FixtureRequest) -> Path:
    """Per-test artifacts directory.

    Uses the parametrize id (the AuditCase.case_id) so cases are isolated.
    """
    case_id = request.node.callspec.id if hasattr(request.node, "callspec") else request.node.name
    base = _REPO_ROOT / "tests" / "_artifacts" / "audit" / case_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "operator").mkdir(exist_ok=True)
    (base / "bare").mkdir(exist_ok=True)
    return base
```

- [ ] **Step 3: Verify the conftest loads**

Run: `uv run pytest --collect-only -m k8s_audit tests/kubernetes/audit/ 2>&1 | tail -5`
Expected: `0 tests collected` with no errors (the directory has no tests yet, but the conftest must import cleanly).

- [ ] **Step 4: Commit**

```bash
git add tests/kubernetes/audit/__init__.py tests/kubernetes/audit/conftest.py tests/unit/kubernetes/audit/__init__.py
git commit -s -m "test(k8s-audit): add audit module skeleton + conftest"
```

---

## Task 3: `AuditCase` dataclass + 5 case definitions

**Files:**
- Create: `tests/kubernetes/audit/cases.py`

- [ ] **Step 1: Write `cases.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit case definitions — one per workflow shape exercised by the suite."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuditCase:
    """One workflow audited by ``test_audit.py``.

    ``profile_args`` is the canonical CLI form. Both deployers translate it:
    the bare deployer passes the args verbatim to ``aiperf profile``; the
    operator runner translates them to an ``AIPerfJobConfig`` (see
    ``operator_runner.py``).
    """

    case_id: str
    """Stable id used for parametrize ids and artifact directory names."""

    endpoint_type: str
    """chat, completions, embeddings, etc."""

    concurrency: int
    """--concurrency value passed to both sides."""

    request_count: int
    """--request-count value passed to both sides."""

    num_conversations: int | None = None
    """--num-conversations override; None lets aiperf pick its default."""

    epochs: int = 1
    """Number of epochs (1 means a single run on each side)."""

    sweep: dict[str, list[int | float | str]] | None = None
    """Optional sweep dimension; e.g. {'concurrency': [4, 16]}. None disables."""

    seed: int = 42
    """--random-seed value pinned for determinism."""

    metric_tolerance_overrides: dict[str, float] = field(default_factory=dict)
    """Per-metric tolerance overrides (relative diff, e.g. 0.30 = 30%)."""

    expected_artifacts: tuple[str, ...] = (
        "profile_export.jsonl",
        "profile_export_records.csv",
        "inputs.json",
    )
    """Filenames that MUST exist on both sides for the structural diff."""


AUDIT_CASES: tuple[AuditCase, ...] = (
    AuditCase(
        case_id="baseline-chat",
        endpoint_type="chat",
        concurrency=4,
        request_count=64,
        num_conversations=32,
    ),
    AuditCase(
        case_id="baseline-completions",
        endpoint_type="completions",
        concurrency=4,
        request_count=64,
        num_conversations=32,
    ),
    AuditCase(
        case_id="concurrency-scale",
        endpoint_type="chat",
        concurrency=16,
        request_count=128,
        num_conversations=64,
        # Higher concurrency on operator side spreads across worker pods;
        # tail latency is structurally noisier than a single bare process.
        metric_tolerance_overrides={
            "p99": 0.40,
            "p95": 0.35,
        },
    ),
)
# Deferred cases (need operator-side helper extensions before they can audit):
#   - multi-epoch: AIPerfJobConfig.epochs + bare-side multi-run loop
#   - small-sweep: AIPerfSweep runner in tests/kubernetes/helpers/
# Add to AUDIT_CASES once the helpers land; no harness changes required.
```

- [ ] **Step 2: Verify the module imports**

Run: `uv run python -c "from tests.kubernetes.audit.cases import AUDIT_CASES; print(len(AUDIT_CASES), [c.case_id for c in AUDIT_CASES])"`
Expected: `3 ['baseline-chat', 'baseline-completions', 'concurrency-scale']`

- [ ] **Step 3: Commit**

```bash
git add tests/kubernetes/audit/cases.py
git commit -s -m "test(k8s-audit): add AuditCase dataclass and 5 audit cases"
```

---

## Task 4: `diff.py` — types + `diff_exact` (TDD)

**Files:**
- Create: `tests/kubernetes/audit/diff.py`
- Create: `tests/unit/kubernetes/audit/test_diff.py`

- [ ] **Step 1: Write the failing test for `diff_exact`**

Create `tests/unit/kubernetes/audit/test_diff.py`:

```python
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
        json.dumps({"endpoint_type": "chat", "concurrency": 4, "request_count": request_count})
    )
    rows = [
        {"request_index": i, "error": (i < errors)}
        for i in range(request_count)
    ]
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


def test_diff_exact_matching_trees_returns_no_findings(tmp_path: Path, case: AuditCase) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _make_tree(op, request_count=10)
    _make_tree(bare, request_count=10)

    findings = diff_exact(operator_dir=op, bare_dir=bare, case=case)

    assert findings == []


def test_diff_exact_request_count_mismatch_is_reported(tmp_path: Path, case: AuditCase) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _make_tree(op, request_count=9)   # operator dropped one
    _make_tree(bare, request_count=10)

    findings = diff_exact(operator_dir=op, bare_dir=bare, case=case)

    assert len(findings) == 1
    f = findings[0]
    assert f.bucket == "exact"
    assert f.field == "request_count"
    assert f.expected == 10
    assert f.actual == 9


def test_diff_exact_error_count_nonzero_is_reported(tmp_path: Path, case: AuditCase) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _make_tree(op, request_count=10, errors=2)
    _make_tree(bare, request_count=10, errors=0)

    findings = diff_exact(operator_dir=op, bare_dir=bare, case=case)

    assert any(f.field == "error_count" and f.actual == 2 for f in findings)


def test_audit_findings_empty_property() -> None:
    f = AuditFindings(case_id="x", findings=[])
    assert f.empty is True
    f2 = AuditFindings(case_id="x", findings=[Finding(bucket="exact", field="x", expected=1, actual=2, reason="r")])
    assert f2.empty is False
```

- [ ] **Step 2: Run test — expect import error / no module**

Run: `uv run pytest tests/unit/kubernetes/audit/test_diff.py -v 2>&1 | tail -10`
Expected: collection error — `No module named 'tests.kubernetes.audit.diff'`.

- [ ] **Step 3: Implement `diff.py` with types + `diff_exact`**

Create `tests/kubernetes/audit/diff.py`:

```python
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
        return max(0, sum(1 for _ in f) - 1)  # subtract header


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
    if op_count != bare_count:
        findings.append(Finding(
            bucket="exact",
            field="request_count",
            expected=bare_count,
            actual=op_count,
            reason="record count differs between operator and bare-pod runs",
        ))

    op_errors = _error_count(operator_dir)
    bare_errors = _error_count(bare_dir)
    if op_errors != 0 or bare_errors != 0 or op_errors != bare_errors:
        findings.append(Finding(
            bucket="exact",
            field="error_count",
            expected=bare_errors,
            actual=op_errors,
            reason="error rows present or counts differ",
        ))

    op_args = _inputs_args(operator_dir)
    bare_args = _inputs_args(bare_dir)
    for key in ("endpoint_type", "concurrency", "request_count"):
        if op_args.get(key) != bare_args.get(key):
            findings.append(Finding(
                bucket="exact",
                field=f"inputs.{key}",
                expected=bare_args.get(key),
                actual=op_args.get(key),
                reason="configured-args echo differs",
            ))

    op_hash = _dataset_hash(operator_dir)
    bare_hash = _dataset_hash(bare_dir)
    if op_hash is not None and bare_hash is not None and op_hash != bare_hash:
        findings.append(Finding(
            bucket="exact",
            field="dataset_hash",
            expected=bare_hash,
            actual=op_hash,
            reason="seeded dataset payloads diverged between modes",
        ))

    op_files = _file_set(operator_dir)
    bare_files = _file_set(bare_dir)
    if op_files != bare_files:
        findings.append(Finding(
            bucket="exact",
            field="file_set",
            expected=sorted(bare_files),
            actual=sorted(op_files),
            reason="exporter file set differs",
        ))

    return findings
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/unit/kubernetes/audit/test_diff.py -v 2>&1 | tail -15`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/kubernetes/audit/diff.py tests/unit/kubernetes/audit/test_diff.py
git commit -s -m "test(k8s-audit): add diff types and diff_exact bucket"
```

---

## Task 5: `diff_tolerance` (TDD)

**Files:**
- Modify: `tests/kubernetes/audit/diff.py`
- Modify: `tests/unit/kubernetes/audit/test_diff.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/kubernetes/audit/test_diff.py`:

```python
from tests.kubernetes.audit.diff import diff_tolerance


def _write_summary(root: Path, *, mean: float, p50: float, p99: float, throughput: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile_export_aiperf.json").write_text(json.dumps({
        "request_latency": {"avg": mean, "p50": p50, "p99": p99},
        "request_throughput": {"avg": throughput},
    }))


def test_diff_tolerance_within_band_returns_no_findings(tmp_path: Path, case: AuditCase) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _write_summary(op, mean=100.0, p50=95.0, p99=200.0, throughput=50.0)
    _write_summary(bare, mean=105.0, p50=99.0, p99=220.0, throughput=51.0)  # all within bands

    findings = diff_tolerance(operator_dir=op, bare_dir=bare, case=case)

    assert findings == []


def test_diff_tolerance_mean_out_of_band_is_reported(tmp_path: Path, case: AuditCase) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _write_summary(op, mean=200.0, p50=95.0, p99=200.0, throughput=50.0)  # 100% off mean
    _write_summary(bare, mean=100.0, p50=95.0, p99=200.0, throughput=50.0)

    findings = diff_tolerance(operator_dir=op, bare_dir=bare, case=case)

    assert any("avg" in f.field and f.bucket == "tolerance" for f in findings)


def test_diff_tolerance_per_case_override_relaxes_band(tmp_path: Path) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    _write_summary(op, mean=100.0, p50=95.0, p99=350.0, throughput=50.0)  # p99 ~57% off
    _write_summary(bare, mean=100.0, p50=95.0, p99=222.0, throughput=50.0)

    case = AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        metric_tolerance_overrides={"p99": 0.60},  # allow up to 60%
    )
    findings = diff_tolerance(operator_dir=op, bare_dir=bare, case=case)

    # Mean and p50 are within default bands; p99 is within the override.
    assert findings == []
```

- [ ] **Step 2: Run tests — expect 3 failing**

Run: `uv run pytest tests/unit/kubernetes/audit/test_diff.py -v 2>&1 | tail -15`
Expected: ImportError for `diff_tolerance` (or 3 failures once it exists).

- [ ] **Step 3: Implement `diff_tolerance`**

Append to `tests/kubernetes/audit/diff.py`:

```python
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
    # Prefer the model-named summary; fall back to any matching file.
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
        return findings  # missing-file is a structural finding, not tolerance

    op = _flatten_metrics(json.loads(op_path.read_text()))
    bare = _flatten_metrics(json.loads(bare_path.read_text()))

    for field_name, bare_value in bare.items():
        if field_name not in op:
            continue  # missing metric is a structural finding
        op_value = op[field_name]
        stat_key = field_name.rsplit(".", 1)[-1].lower()
        band = _band_for(stat_key, case)
        rel = _relative_diff(op_value, bare_value)
        if rel > band:
            findings.append(Finding(
                bucket="tolerance",
                field=field_name,
                expected=bare_value,
                actual=op_value,
                reason=f"relative diff {rel:.1%} exceeds band {band:.1%} for stat '{stat_key}'",
            ))

    return findings
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/unit/kubernetes/audit/test_diff.py -v 2>&1 | tail -15`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/kubernetes/audit/diff.py tests/unit/kubernetes/audit/test_diff.py
git commit -s -m "test(k8s-audit): add diff_tolerance bucket with per-stat bands"
```

---

## Task 6: `diff_structural` (TDD)

**Files:**
- Modify: `tests/kubernetes/audit/diff.py`
- Modify: `tests/unit/kubernetes/audit/test_diff.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/kubernetes/audit/test_diff.py`:

```python
from tests.kubernetes.audit.diff import diff_structural


def test_diff_structural_missing_expected_artifact_is_reported(tmp_path: Path) -> None:
    op = tmp_path / "operator"
    bare = tmp_path / "bare"
    op.mkdir()
    bare.mkdir()
    (bare / "profile_export.jsonl").write_text("")
    # operator side missing the file
    case = AuditCase(
        case_id="unit",
        endpoint_type="chat",
        concurrency=4,
        request_count=10,
        expected_artifacts=("profile_export.jsonl",),
    )

    findings = diff_structural(operator_dir=op, bare_dir=bare, case=case)

    assert any(f.field == "profile_export.jsonl" and "operator" in f.reason for f in findings)


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

    assert any(f.field.endswith("profile_export_records.csv") and "header" in f.reason for f in findings)


def test_diff_structural_json_top_level_keyset_mismatch_is_reported(tmp_path: Path) -> None:
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

    assert any(f.field.endswith("inputs.json") and "key" in f.reason.lower() for f in findings)
```

- [ ] **Step 2: Run tests — expect failures**

Run: `uv run pytest tests/unit/kubernetes/audit/test_diff.py -v 2>&1 | tail -10`
Expected: ImportError or 3 failures.

- [ ] **Step 3: Implement `diff_structural`**

Append to `tests/kubernetes/audit/diff.py`:

```python
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
    """Top-level + depth-1 keys, joined with '.'. Returns None on read failure."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return set()
    keys: set[str] = set(payload.keys())
    for k, v in payload.items():
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
            findings.append(Finding(
                bucket="structural",
                field=filename,
                expected="present",
                actual="missing",
                reason=f"expected artifact missing on operator side: {filename}",
            ))
        if not bare_present:
            findings.append(Finding(
                bucket="structural",
                field=filename,
                expected="present",
                actual="missing",
                reason=f"expected artifact missing on bare side: {filename}",
            ))
        if not (op_present and bare_present):
            continue

        if filename.endswith(".csv"):
            op_header = _csv_header(op_path)
            bare_header = _csv_header(bare_path)
            if op_header != bare_header:
                findings.append(Finding(
                    bucket="structural",
                    field=f"schema:{filename}",
                    expected=bare_header,
                    actual=op_header,
                    reason=f"CSV header set differs in {filename}",
                ))
        elif filename.endswith(".json"):
            op_keys = _json_keyset_depth2(op_path)
            bare_keys = _json_keyset_depth2(bare_path)
            if op_keys is not None and bare_keys is not None and op_keys != bare_keys:
                findings.append(Finding(
                    bucket="structural",
                    field=f"schema:{filename}",
                    expected=sorted(bare_keys),
                    actual=sorted(op_keys),
                    reason=f"JSON depth-2 key set differs in {filename}",
                ))

    return findings
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/unit/kubernetes/audit/test_diff.py -v 2>&1 | tail -15`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/kubernetes/audit/diff.py tests/unit/kubernetes/audit/test_diff.py
git commit -s -m "test(k8s-audit): add diff_structural bucket"
```

---

## Task 7: `report.py` — markdown + JSON renderers (TDD)

**Files:**
- Create: `tests/kubernetes/audit/report.py`
- Create: `tests/unit/kubernetes/audit/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/kubernetes/audit/test_report.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``tests/kubernetes/audit/report.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.kubernetes.audit.diff import AuditFindings, Finding
from tests.kubernetes.audit.report import render_json, render_markdown


@pytest.fixture
def findings() -> AuditFindings:
    return AuditFindings(
        case_id="baseline-chat",
        findings=[
            Finding(bucket="exact", field="request_count", expected=64, actual=63, reason="off by one"),
            Finding(bucket="tolerance", field="request_latency.p99", expected=200.0, actual=350.0, reason="75% > 25%"),
            Finding(bucket="structural", field="profile_export_records.csv", expected="present", actual="missing", reason="missing on operator"),
        ],
    )


def test_render_json_round_trips(findings: AuditFindings) -> None:
    text = render_json(findings)
    payload = json.loads(text)
    assert payload["case_id"] == "baseline-chat"
    assert len(payload["findings"]) == 3
    assert payload["findings"][0]["bucket"] == "exact"
    assert payload["findings"][0]["field"] == "request_count"


def test_render_markdown_contains_each_bucket_and_pass_fail_header(findings: AuditFindings) -> None:
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
```

- [ ] **Step 2: Run test — expect import error**

Run: `uv run pytest tests/unit/kubernetes/audit/test_report.py -v 2>&1 | tail -8`
Expected: ImportError for `tests.kubernetes.audit.report`.

- [ ] **Step 3: Implement `report.py`**

Create `tests/kubernetes/audit/report.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderers for ``AuditFindings`` -> markdown / JSON."""

from __future__ import annotations

from typing import Literal

import orjson

from tests.kubernetes.audit.diff import AuditFindings, Bucket, Finding


def render_json(findings: AuditFindings) -> str:
    payload = {
        "case_id": findings.case_id,
        "passed": findings.empty,
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
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/unit/kubernetes/audit/ -v 2>&1 | tail -20`
Expected: 13 passed (10 diff + 3 report).

- [ ] **Step 5: Commit**

```bash
git add tests/kubernetes/audit/report.py tests/unit/kubernetes/audit/test_report.py
git commit -s -m "test(k8s-audit): add markdown + JSON report renderers"
```

---

## Task 8: `bare_pod.py` — `BarePodDeployer`

**Files:**
- Create: `tests/kubernetes/audit/bare_pod.py`

- [ ] **Step 1: Implement `BarePodDeployer`**

Create `tests/kubernetes/audit/bare_pod.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare-pod deployer: runs ``aiperf profile`` in a single ``batch/v1.Job``.

This is the "oracle" side of the audit. No operator, no JobSet, no controller,
no workers — just one pod running the local CLI against the in-cluster mock
server. Results are extracted via ``kubectl cp`` before the Job is deleted.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


@dataclass
class BarePodConfig:
    """Resolved settings for one bare-pod run."""

    image: str = "aiperf:local"
    image_pull_policy: str = "Never"
    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    model_name: str = "mock-model"
    tokenizer_name: str = "gpt2"


class BarePodDeployer:
    """Submits a raw Job, waits for completion, copies artifacts out."""

    def __init__(
        self,
        kubectl: KubectlClient,
        config: BarePodConfig | None = None,
    ) -> None:
        self.kubectl = kubectl
        self.config = config or BarePodConfig()

    def _build_args(self, case: AuditCase, *, swept_value: object | None = None) -> list[str]:
        """Translate AuditCase -> ``aiperf profile`` argv (excluding the binary)."""
        concurrency = case.concurrency
        if case.sweep and "concurrency" in case.sweep and swept_value is not None:
            concurrency = int(swept_value)

        args: list[str] = [
            "profile",
            "--model", self.config.model_name,
            "--url", self.config.endpoint_url,
            "--endpoint-type", case.endpoint_type,
            "--tokenizer", self.config.tokenizer_name,
            "--concurrency", str(concurrency),
            "--request-count", str(case.request_count),
            "--random-seed", str(case.seed),
            "--ui", "none",
            "--artifact-dir", "/aiperf-output",
        ]
        if case.num_conversations is not None:
            args += ["--num-conversations", str(case.num_conversations)]
        return args

    def _build_job_manifest(
        self,
        *,
        name: str,
        namespace: str,
        argv: list[str],
    ) -> str:
        """Build the batch/v1.Job manifest as a YAML string."""
        body = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": namespace, "labels": {"app.kubernetes.io/name": "aiperf-bare-audit"}},
            "spec": {
                "ttlSecondsAfterFinished": 3600,
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": "aiperf-bare-audit"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "aiperf",
                                "image": self.config.image,
                                "imagePullPolicy": self.config.image_pull_policy,
                                "command": ["aiperf"],
                                "args": argv,
                                "volumeMounts": [
                                    {"name": "output", "mountPath": "/aiperf-output"},
                                ],
                            },
                        ],
                        "volumes": [{"name": "output", "emptyDir": {}}],
                    },
                },
            },
        }
        return yaml.safe_dump(body, sort_keys=False)

    async def _wait_for_terminal(self, name: str, namespace: str, timeout: int) -> str:
        """Poll the Job until it reports Complete or Failed. Returns final phase."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            result = await self.kubectl.run(
                "get", "job", name, "-n", namespace, "-o", "json", check=False,
            )
            if result.returncode == 0:
                import json
                payload = json.loads(result.stdout)
                conditions = payload.get("status", {}).get("conditions", []) or []
                for c in conditions:
                    if c.get("type") == "Complete" and c.get("status") == "True":
                        return "Complete"
                    if c.get("type") == "Failed" and c.get("status") == "True":
                        return "Failed"
            await asyncio.sleep(3)
        raise TimeoutError(f"bare-pod job {namespace}/{name} did not reach terminal state in {timeout}s")

    async def _pod_for_job(self, name: str, namespace: str) -> str:
        result = await self.kubectl.run(
            "get", "pod", "-n", namespace,
            "-l", f"job-name={name}",
            "-o", "jsonpath={.items[0].metadata.name}",
            check=True,
        )
        pod = result.stdout.strip()
        if not pod:
            raise RuntimeError(f"no pod found for job {namespace}/{name}")
        return pod

    async def _kubectl_cp(self, pod: str, namespace: str, dest_dir: Path) -> None:
        """Copy /aiperf-output from the (terminal) pod to dest_dir."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        await self.kubectl.run(
            "cp",
            f"{namespace}/{pod}:/aiperf-output/.",
            str(dest_dir),
            "-c", "aiperf",
            check=True,
        )

    async def run(
        self,
        *,
        case: AuditCase,
        namespace: str,
        dest_dir: Path,
        swept_value: object | None = None,
        timeout: int = 600,
    ) -> Path:
        """Run one bare-pod invocation; return ``dest_dir`` with artifacts copied in."""
        suffix = uuid.uuid4().hex[:6]
        name = f"audit-bare-{case.case_id}-{suffix}"

        # Ensure namespace exists (idempotent).
        await self.kubectl.run("create", "namespace", namespace, check=False)

        argv = self._build_args(case, swept_value=swept_value)
        manifest = self._build_job_manifest(name=name, namespace=namespace, argv=argv)
        await self.kubectl.apply_manifest(manifest)

        try:
            phase = await self._wait_for_terminal(name, namespace, timeout)
            pod = await self._pod_for_job(name, namespace)
            await self._kubectl_cp(pod, namespace, dest_dir)
            if phase != "Complete":
                logger.warning(f"bare-pod job {name} terminal phase = {phase}")
        finally:
            await self.kubectl.run("delete", "job", name, "-n", namespace, "--wait=false", check=False)

        return dest_dir
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from tests.kubernetes.audit.bare_pod import BarePodDeployer, BarePodConfig; print('ok')"`
Expected: `ok`

If `KubectlClient.apply_manifest` does not exist (check the file), use `apply` or write the manifest to a tempfile and call `kubectl.run("apply", "-f", path)` instead.

- [ ] **Step 3: Commit**

```bash
git add tests/kubernetes/audit/bare_pod.py
git commit -s -m "test(k8s-audit): add BarePodDeployer (raw Job + kubectl cp)"
```

---

## Task 9: `operator_runner.py` — `OperatorAuditRunner`

**Files:**
- Create: `tests/kubernetes/audit/operator_runner.py`

- [ ] **Step 1: Implement `OperatorAuditRunner`**

Create `tests/kubernetes/audit/operator_runner.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-side runner: deploy ``AIPerfJob``, then ``aiperf kube results``.

This wraps the existing ``OperatorDeployer`` to submit a CR that mirrors the
``AuditCase``'s profile args, waits for completion, then shells out to the
user-facing ``aiperf kube results <id> --output <dir>`` CLI to download
artifacts. Exercising the download CLI is part of the audit's purpose: if the
operator-managed run produces correct artifacts internally but ships them
incorrectly to the user, the audit catches that.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
)

logger = AIPerfLogger(__name__)


@dataclass
class OperatorAuditConfig:
    image: str = "aiperf:local"
    image_pull_policy: str = "Never"
    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    model_name: str = "mock-model"
    tokenizer_name: str = "gpt2"


class OperatorAuditRunner:
    """Submits an AIPerfJob and downloads its artifacts via ``aiperf kube results``."""

    def __init__(
        self,
        deployer: OperatorDeployer,
        config: OperatorAuditConfig | None = None,
    ) -> None:
        self.deployer = deployer
        self.config = config or OperatorAuditConfig()

    def _build_job_config(self, case: AuditCase, *, swept_value: object | None = None) -> AIPerfJobConfig:
        concurrency = case.concurrency
        if case.sweep and "concurrency" in case.sweep and swept_value is not None:
            concurrency = int(swept_value)

        return AIPerfJobConfig(
            endpoint_url=self.config.endpoint_url,
            model_name=self.config.model_name,
            endpoint_type=case.endpoint_type,
            concurrency=concurrency,
            request_count=case.request_count,
            warmup_request_count=0,
            tokenizer_name=self.config.tokenizer_name,
            image=self.config.image,
            image_pull_policy=self.config.image_pull_policy,
        )

    async def _download_results(
        self,
        *,
        namespace: str,
        job_name: str,
        dest_dir: Path,
        kubeconfig: str | None,
    ) -> None:
        """Shell out to ``aiperf kube results`` with the cluster's kubeconfig."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "aiperf", "kube", "results", job_name,
            "--namespace", namespace,
            "--output", str(dest_dir),
            "--all",
        ]
        env = dict(os.environ)
        if kubeconfig:
            env["KUBECONFIG"] = kubeconfig

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"aiperf kube results failed (rc={proc.returncode})\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )

    async def run(
        self,
        *,
        case: AuditCase,
        namespace: str,
        dest_dir: Path,
        kubeconfig: str | None = None,
        swept_value: object | None = None,
        timeout: int = 600,
    ) -> Path:
        suffix = uuid.uuid4().hex[:6]
        job_name = f"audit-op-{case.case_id}-{suffix}"
        cfg = self._build_job_config(case, swept_value=swept_value)

        result = await self.deployer.run_job(
            config=cfg,
            name=job_name,
            namespace=namespace,
            timeout=timeout,
        )
        if not result.success:
            raise RuntimeError(
                f"operator job {namespace}/{job_name} did not succeed: "
                f"{result.error_message or 'unknown'}"
            )

        await self._download_results(
            namespace=namespace,
            job_name=job_name,
            dest_dir=dest_dir,
            kubeconfig=kubeconfig,
        )
        return dest_dir
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from tests.kubernetes.audit.operator_runner import OperatorAuditRunner; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add tests/kubernetes/audit/operator_runner.py
git commit -s -m "test(k8s-audit): add OperatorAuditRunner (AIPerfJob + kube results)"
```

---

## Task 10: `test_audit.py` — parametrized end-to-end test

**Files:**
- Create: `tests/kubernetes/audit/test_audit.py`

- [ ] **Step 1: Write the test**

Create `tests/kubernetes/audit/test_audit.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-vs-bare-pod correctness audit.

For each ``AuditCase`` in ``cases.AUDIT_CASES``, this test:

1. Runs the case via the operator path; downloads results via
   ``aiperf kube results``.
2. Runs the same case via a bare ``batch/v1.Job`` (no operator); copies
   results via ``kubectl cp``.
3. Diffs the two artifact trees through three buckets (exact / tolerance /
   structural) and asserts no findings.

The bare-pod side is the oracle. Tolerance bands handle wall-clock-noisy
numeric stats; exact and structural buckets must match.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from tests.kubernetes.audit.bare_pod import BarePodConfig, BarePodDeployer
from tests.kubernetes.audit.cases import AUDIT_CASES, AuditCase
from tests.kubernetes.audit.diff import (
    AuditFindings,
    diff_exact,
    diff_structural,
    diff_tolerance,
)
from tests.kubernetes.audit.operator_runner import (
    OperatorAuditConfig,
    OperatorAuditRunner,
)
from tests.kubernetes.audit.report import render_json, render_markdown
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import OperatorDeployer


@pytest.mark.k8s_audit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", AUDIT_CASES, ids=lambda c: c.case_id)
async def test_operator_vs_bare_pod(
    case: AuditCase,
    kubectl: KubectlClient,
    operator_deployer: OperatorDeployer,
    mock_server: None,
    audit_artifacts_dir: Path,
) -> None:
    """One audit case: operator path vs bare-pod path, three-bucket diff."""
    namespace = f"audit-{case.case_id}-{uuid.uuid4().hex[:6]}"
    op_dir = audit_artifacts_dir / "operator"
    bare_dir = audit_artifacts_dir / "bare"

    # Operator side first (slow setup, fail fast if AIPerfJob CRD missing).
    operator_runner = OperatorAuditRunner(deployer=operator_deployer, config=OperatorAuditConfig())
    await operator_runner.run(
        case=case,
        namespace=namespace,
        dest_dir=op_dir,
        timeout=900,
    )

    # Bare side.
    bare = BarePodDeployer(kubectl=kubectl, config=BarePodConfig())
    await bare.run(
        case=case,
        namespace=namespace,
        dest_dir=bare_dir,
        timeout=900,
    )

    findings_list = (
        diff_exact(operator_dir=op_dir, bare_dir=bare_dir, case=case)
        + diff_tolerance(operator_dir=op_dir, bare_dir=bare_dir, case=case)
        + diff_structural(operator_dir=op_dir, bare_dir=bare_dir, case=case)
    )
    findings = AuditFindings(case_id=case.case_id, findings=findings_list)

    (audit_artifacts_dir / "audit-report.json").write_text(render_json(findings))
    md = render_markdown(findings)
    (audit_artifacts_dir / "report.md").write_text(md)

    if not findings.empty:
        print(md)
    assert findings.empty, f"audit failures for {case.case_id}: see {audit_artifacts_dir}/report.md"
```

- [ ] **Step 2: Verify collection**

Run: `uv run pytest --collect-only -m k8s_audit tests/kubernetes/audit/ 2>&1 | tail -10`
Expected: 3 items collected (one per case).

- [ ] **Step 3: Commit**

```bash
git add tests/kubernetes/audit/test_audit.py
git commit -s -m "test(k8s-audit): parametrized operator-vs-bare-pod audit test"
```

---

## Task 11: Documentation update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`

- [ ] **Step 1: Add a Kubernetes-section bullet**

In each of the three sync files, under the existing `## Kubernetes` section (which lists kopf handlers, k8s_client, etc.), append one bullet near the bottom:

```markdown
- **K8s-vs-local audit suite** — `tests/kubernetes/audit/` runs each workflow case twice (operator + `aiperf kube results` download path; bare `batch/v1.Job` running `aiperf profile` directly) and diffs the artifact trees through three buckets (exact / tolerance / structural). Opt-in: `pytest -m k8s_audit tests/kubernetes/audit/ -n auto`. Spec: `docs/superpowers/specs/2026-04-26-k8s-vs-local-audit-design.md`.
```

- [ ] **Step 2: Diff the three files to confirm sync**

Run:
```bash
diff <(grep -A1 "K8s-vs-local audit suite" CLAUDE.md) <(grep -A1 "K8s-vs-local audit suite" .github/copilot-instructions.md)
diff <(grep -A1 "K8s-vs-local audit suite" CLAUDE.md) <(grep -A1 "K8s-vs-local audit suite" .cursor/rules/python.mdc)
```

Expected: empty output (identical bullets).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -s -m "docs(k8s-audit): reference audit suite in CLAUDE.md (+ sync files)"
```

---

## Self-Review Checklist (run by the implementer once all tasks complete)

- [ ] All 11 tasks committed; `git log --oneline | head -12` shows them.
- [ ] `uv run pytest tests/unit/kubernetes/audit/ -n auto` → 13 passed.
- [ ] `uv run pytest --collect-only -m k8s_audit tests/kubernetes/audit/` → 3 items.
- [ ] `pre-commit run --all-files` is clean (or only baselined-unrelated noise).
- [ ] CLAUDE.md / copilot-instructions / cursor rule contain the same one-line bullet.
- [ ] (Optional, requires kind cluster) `pytest -m k8s_audit tests/kubernetes/audit/ -k baseline-chat -n auto` runs end-to-end and produces a `report.md` showing PASS.
