# UID-keyed Results Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every AIPerfJob submission its own artifact directory keyed by the CR's `metadata.uid`, so re-creating a CR with the same name never overwrites prior results — while keeping the stable CR-name handle and the existing results HTTP API shape.

**Architecture:** New `results_layout.py` module owns the on-disk layout. Write sites compute `<base>/<ns>/<name>/<uid>/` via the module; a single atomic success gate at `handlers/completion.py:243` writes `<name>/latest.txt`, mirrors the uid to CR status, and trims old runs. The results router resolves `latest.txt` on existing two-arg routes (backward-compatible) and adds additive `/runs/<uid>/` routes for historical pinning. A one-time migration shim at results-server lifespan relocates pre-migration trees under `<name>/legacy/`.

**Tech Stack:** Python 3.10+, FastAPI, kopf, pytest-asyncio, pytest-xdist, Pydantic v2, pydantic-settings, Helm (CRD schema), `orjson`.

**Spec:** `docs/superpowers/specs/2026-04-24-uid-keyed-results-layout.md`

**Branch:** `ajc/k8s` (do not fork a feature branch — commit on the current branch per standing feedback).

**Per-task verification contract:**

- Exactly **one** `uv run pytest -n auto tests/unit/` invocation per task (no subfolder splits).
- `make check-ergonomics` and `make check-ruff-baselined` run at the end of every task.
- `ruff format . && ruff check --fix .` before commit.
- DCO sign-off (`git commit -s`) with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

---

## Task 1: `results_layout.py` foundation (spec §§ File inventory, Data model, Retention, Migration shim)

**Files:**
- Create: `src/aiperf/operator/results_layout.py`
- Create: `tests/unit/operator/test_results_layout.py`

**Preconditions:**
- Clean tree on `ajc/k8s`.
- `make first-time-setup` has been run (standing feedback: never `uv sync`).

**Postconditions:**
- `results_layout.py` exports `LATEST_POINTER`, `UID_RE`, `job_dir`, `run_dir`, `write_latest`, `resolve_latest`, `resolve_run_dir`, `enforce_retention`, `migrate_legacy_layout`, `list_run_uids`.
- All 12 unit tests from spec §Tests pass.
- No other code yet imports the module (consumers wire in later tasks).

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/operator/test_results_layout.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.results_layout.

Covers the full public API:
- write_latest + resolve_latest (atomic pointer file)
- resolve_run_dir (latest + explicit uid + missing-uid None)
- enforce_retention (mtime ordering, keep count, protect_uid guarantee)
- migrate_legacy_layout (relocates pre-migration files, idempotent, mixed layouts)
- list_run_uids
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from pytest import param

from aiperf.operator.results_layout import (
    LATEST_POINTER,
    enforce_retention,
    job_dir,
    list_run_uids,
    migrate_legacy_layout,
    resolve_latest,
    resolve_run_dir,
    run_dir,
    write_latest,
)

UID_A = "5f8b2a3c-7d4e-4f1a-9b2c-1e3f4a5b6c7d"
UID_B = "1a2b3c4d-5e6f-4789-abcd-ef0123456789"
UID_C = "9876fedc-ba98-4765-8432-10fedcba9876"


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_write_latest_atomic(tmp_path: Path) -> None:
    write_latest(tmp_path, "ns", "job", UID_A)
    assert resolve_latest(tmp_path, "ns", "job") == UID_A
    write_latest(tmp_path, "ns", "job", UID_B)
    assert resolve_latest(tmp_path, "ns", "job") == UID_B


def test_resolve_latest_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_latest(tmp_path, "ns", "job") is None


def test_resolve_run_dir_uid_none_uses_latest(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", UID_A).mkdir(parents=True)
    write_latest(tmp_path, "ns", "job", UID_A)
    resolved = resolve_run_dir(tmp_path, "ns", "job")
    assert resolved == run_dir(tmp_path, "ns", "job", UID_A)


def test_resolve_run_dir_explicit_uid(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", UID_A).mkdir(parents=True)
    run_dir(tmp_path, "ns", "job", UID_B).mkdir(parents=True)
    write_latest(tmp_path, "ns", "job", UID_B)
    resolved = resolve_run_dir(tmp_path, "ns", "job", uid=UID_A)
    assert resolved == run_dir(tmp_path, "ns", "job", UID_A)


def test_resolve_run_dir_uid_not_on_disk_returns_none(tmp_path: Path) -> None:
    assert resolve_run_dir(tmp_path, "ns", "job", uid=UID_A) is None


def test_resolve_run_dir_latest_points_at_missing_uid_returns_none(
    tmp_path: Path,
) -> None:
    write_latest(tmp_path, "ns", "job", UID_A)
    assert resolve_run_dir(tmp_path, "ns", "job") is None


def test_list_run_uids_lists_only_uid_shaped_dirs(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", UID_A).mkdir(parents=True)
    run_dir(tmp_path, "ns", "job", UID_B).mkdir(parents=True)
    (job_dir(tmp_path, "ns", "job") / "legacy").mkdir()
    (job_dir(tmp_path, "ns", "job") / LATEST_POINTER).write_text(UID_A)
    uids = set(list_run_uids(tmp_path, "ns", "job"))
    assert uids == {UID_A, UID_B, "legacy"}


def test_enforce_retention_keeps_n_newest(tmp_path: Path) -> None:
    base_time = time.time()
    uids = [f"0000000{i}-0000-4000-8000-000000000000"[:36] for i in range(15)]
    for idx, uid in enumerate(uids):
        d = run_dir(tmp_path, "ns", "job", uid)
        d.mkdir(parents=True)
        mtime = base_time - (idx * 60)
        os.utime(d, (mtime, mtime))
    deleted = enforce_retention(tmp_path, "ns", "job", keep=10, protect_uid=uids[0])
    assert len(deleted) == 5
    survivors = set(list_run_uids(tmp_path, "ns", "job"))
    assert len(survivors) == 10
    assert uids[0] in survivors


def test_enforce_retention_protects_uid_even_if_oldest(tmp_path: Path) -> None:
    base_time = time.time()
    uids = [
        "00000001-0000-4000-8000-000000000000",
        "00000002-0000-4000-8000-000000000000",
        "00000003-0000-4000-8000-000000000000",
    ]
    for idx, uid in enumerate(uids):
        d = run_dir(tmp_path, "ns", "job", uid)
        d.mkdir(parents=True)
        mtime = base_time - (idx * 60)
        os.utime(d, (mtime, mtime))
    enforce_retention(tmp_path, "ns", "job", keep=1, protect_uid=uids[2])
    survivors = set(list_run_uids(tmp_path, "ns", "job"))
    assert uids[0] in survivors
    assert uids[2] in survivors


def test_enforce_retention_empty_dir_noop(tmp_path: Path) -> None:
    assert enforce_retention(tmp_path, "ns", "job", keep=10, protect_uid=UID_A) == []


def test_migrate_legacy_layout_relocates_files(tmp_path: Path) -> None:
    _touch(tmp_path / "ns" / "job" / "foo.json", b'{"ok": true}')
    _touch(tmp_path / "ns" / "job" / "checkpoints" / "chk.json", b"{}")
    migrated = migrate_legacy_layout(tmp_path)
    assert migrated == [("ns", "job")]
    assert (tmp_path / "ns" / "job" / "legacy" / "foo.json").is_file()
    assert (tmp_path / "ns" / "job" / "legacy" / "checkpoints" / "chk.json").is_file()
    assert resolve_latest(tmp_path, "ns", "job") == "legacy"


def test_migrate_legacy_layout_idempotent(tmp_path: Path) -> None:
    _touch(tmp_path / "ns" / "job" / "foo.json")
    migrate_legacy_layout(tmp_path)
    second = migrate_legacy_layout(tmp_path)
    assert second == []


def test_migrate_legacy_layout_skips_already_migrated(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", UID_A).mkdir(parents=True)
    _touch(run_dir(tmp_path, "ns", "job", UID_A) / "foo.json")
    write_latest(tmp_path, "ns", "job", UID_A)
    migrated = migrate_legacy_layout(tmp_path)
    assert migrated == []
    assert resolve_latest(tmp_path, "ns", "job") == UID_A


def test_migrate_legacy_layout_mixed_uid_and_legacy(tmp_path: Path) -> None:
    run_dir(tmp_path, "ns", "job", UID_A).mkdir(parents=True)
    _touch(tmp_path / "ns" / "other" / "bar.json")
    migrated = migrate_legacy_layout(tmp_path)
    assert migrated == [("ns", "other")]
    assert (tmp_path / "ns" / "job" / UID_A).is_dir()
    assert (tmp_path / "ns" / "other" / "legacy" / "bar.json").is_file()


def test_migrate_legacy_layout_empty_name_dir_noop(tmp_path: Path) -> None:
    (tmp_path / "ns" / "job").mkdir(parents=True)
    assert migrate_legacy_layout(tmp_path) == []
    assert resolve_latest(tmp_path, "ns", "job") is None
```

- [ ] **Step 2: Verify tests fail with ImportError**

Run: `uv run pytest -n auto tests/unit/operator/test_results_layout.py`
Expected: every test fails with `ImportError: cannot import name ... from 'aiperf.operator.results_layout'`.

- [ ] **Step 3: Create `src/aiperf/operator/results_layout.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-disk layout for AIPerfJob result artifacts.

Results are stored at ``<base>/<namespace>/<name>/<uid>/`` where ``uid`` is the
Kubernetes ``metadata.uid`` of the CR at creation time. A pointer file
``<name>/latest.txt`` records the uid of the most recent successful run.

This module is the single owner of the layout. All write sites call
:func:`run_dir`; all read sites call :func:`resolve_run_dir`. No other module
should build result paths by hand.

Example
-------
::

    from aiperf.operator.results_layout import run_dir, write_latest

    dest = run_dir(base, "production", "deepseek-r1-bench", uid)
    dest.mkdir(parents=True, exist_ok=True)
    # ... drop artifacts in dest ...
    write_latest(base, "production", "deepseek-r1-bench", uid)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

__all__ = [
    "LATEST_POINTER",
    "UID_RE",
    "enforce_retention",
    "job_dir",
    "list_run_uids",
    "migrate_legacy_layout",
    "resolve_latest",
    "resolve_run_dir",
    "run_dir",
    "write_latest",
]

logger = logging.getLogger(__name__)

LATEST_POINTER = "latest.txt"
LEGACY_UID = "legacy"
_TMP_SUFFIX = ".tmp"
UID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^legacy$"
)


def job_dir(base: Path, namespace: str, name: str) -> Path:
    """Return the parent dir holding all runs of one AIPerfJob name.

    Never contains artifact files directly after migration. Only ``latest.txt``
    and uid-shaped subdirectories are valid children.
    """
    return base / namespace / name


def run_dir(base: Path, namespace: str, name: str, uid: str) -> Path:
    """Return the per-run artifact dir for a given CR uid."""
    return base / namespace / name / uid


def write_latest(base: Path, namespace: str, name: str, uid: str) -> None:
    """Atomically set the latest-run pointer for ``<ns>/<name>/`` to ``uid``.

    Uses staged-write + :func:`os.replace` so a reader never sees a partial
    pointer. Both the tmp and final paths live under the same ``<name>/``
    directory, which is a single filesystem on all supported storage classes.
    """
    parent = job_dir(base, namespace, name)
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{LATEST_POINTER}{_TMP_SUFFIX}"
    tmp.write_text(f"{uid}\n", encoding="utf-8")
    os.replace(tmp, parent / LATEST_POINTER)


def resolve_latest(base: Path, namespace: str, name: str) -> str | None:
    """Read the latest-run uid, or None if no pointer exists."""
    ptr = job_dir(base, namespace, name) / LATEST_POINTER
    if not ptr.is_file():
        return None
    try:
        return ptr.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def resolve_run_dir(
    base: Path, namespace: str, name: str, uid: str | None = None
) -> Path | None:
    """Return the on-disk run dir, or None if it doesn't exist.

    ``uid=None`` (or ``"latest"``) resolves via ``latest.txt``. Any other value
    is treated as an explicit uid — useful for the historical API routes.
    """
    if uid and uid != "latest":
        target = run_dir(base, namespace, name, uid)
        return target if target.is_dir() else None
    resolved = resolve_latest(base, namespace, name)
    if resolved is None:
        return None
    target = run_dir(base, namespace, name, resolved)
    return target if target.is_dir() else None


def list_run_uids(base: Path, namespace: str, name: str) -> list[str]:
    """List every uid-shaped run dir under ``<ns>/<name>/``. Order unspecified."""
    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []
    return [
        p.name
        for p in parent.iterdir()
        if p.is_dir() and UID_RE.match(p.name)
    ]


def enforce_retention(
    base: Path, namespace: str, name: str, keep: int, protect_uid: str
) -> list[str]:
    """Trim older runs under ``<ns>/<name>/`` to the ``keep`` newest.

    ``protect_uid`` is always retained, even if it would otherwise be outside
    the keep window. Returns the uids deleted.

    Retention failures (I/O errors, permissions) are logged and suppressed:
    the caller's success path has already completed, and retention is
    bookkeeping — never fail a CR on retention.
    """
    parent = job_dir(base, namespace, name)
    if not parent.is_dir():
        return []
    runs = [p for p in parent.iterdir() if p.is_dir() and UID_RE.match(p.name)]
    if not runs:
        return []
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    keepers = {r.name for r in runs[:keep]}
    keepers.add(protect_uid)
    deleted: list[str] = []
    for r in runs:
        if r.name in keepers:
            continue
        try:
            shutil.rmtree(r)
            deleted.append(r.name)
        except OSError as exc:
            logger.warning(
                "retention: failed to remove %s/%s/%s: %s",
                namespace,
                name,
                r.name,
                exc,
            )
    return deleted


def migrate_legacy_layout(base: Path) -> list[tuple[str, str]]:
    """One-time: relocate pre-migration files under ``<name>/legacy/``.

    A ``<ns>/<name>/`` directory is treated as pre-migration iff it has no
    ``latest.txt`` pointer and no uid-shaped subdirectories. All loose files
    and non-uid subdirs get moved under ``<name>/legacy/``, and a pointer
    ``latest.txt=legacy`` is written.

    Idempotent: a second call is a no-op. Safe under partial-crash: on restart
    the shim completes any half-moved migration because ``legacy/`` already
    existing does not block ``mkdir(exist_ok=True)``, and ``shutil.move`` is
    a rename within a single filesystem.

    Returns a list of ``(namespace, name)`` pairs actually migrated.
    """
    migrated: list[tuple[str, str]] = []
    if not base.is_dir():
        return migrated
    for ns_dir in base.iterdir():
        if not ns_dir.is_dir():
            continue
        for name_dir in ns_dir.iterdir():
            if not name_dir.is_dir():
                continue
            if _migrate_one(base, ns_dir.name, name_dir):
                migrated.append((ns_dir.name, name_dir.name))
    return migrated


def _migrate_one(base: Path, namespace: str, name_dir: Path) -> bool:
    """Migrate a single ``<ns>/<name>/`` directory. Returns True if relocated."""
    children = list(name_dir.iterdir())
    has_pointer = any(c.name == LATEST_POINTER for c in children)
    has_uid_child = any(
        c.is_dir() and UID_RE.match(c.name) for c in children
    )
    if has_pointer or has_uid_child:
        return False  # already migrated or mixed with at least one run
    files = [c for c in children if c.is_file()]
    misc_subdirs = [
        c
        for c in children
        if c.is_dir() and c.name != LEGACY_UID and not UID_RE.match(c.name)
    ]
    if not files and not misc_subdirs:
        return False  # empty — nothing to do
    legacy_dir = name_dir / LEGACY_UID
    legacy_dir.mkdir(exist_ok=True)
    for entry in files + misc_subdirs:
        shutil.move(str(entry), str(legacy_dir / entry.name))
    write_latest(base, namespace, name_dir.name, LEGACY_UID)
    return True
```

- [ ] **Step 4: Run tests and verify all pass**

Run: `uv run pytest -n auto tests/unit/operator/test_results_layout.py`
Expected: `14 passed`.

- [ ] **Step 5: Format, lint, and full unit suite**

```bash
ruff format src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
ruff check --fix src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

Expected: ruff clean; full unit suite passes; no new ergonomics or ruff baselined violations.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
git commit -s -m "$(cat <<'EOF'
feat(operator): add results_layout module for uid-keyed run dirs

Single owner of the on-disk layout that all future write and read paths
will route through. Exposes run_dir/job_dir/write_latest/resolve_run_dir/
enforce_retention/migrate_legacy_layout/list_run_uids.

No existing code imports the module yet — later tasks wire consumers in.

Spec: docs/superpowers/specs/2026-04-24-uid-keyed-results-layout.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Retention env var `AIPERF_RESULTS_RETAIN_RUNS` (spec § CRD / schema additions)

**Files:**
- Modify: `src/aiperf/operator/environment.py` (under `_ResultsSettings`)
- Test: `tests/unit/operator/test_environment.py` (create if missing)

**Spec correction:** The spec text refers to `src/aiperf/common/environment.py`, but the actual `_ResultsSettings` class lives in `src/aiperf/operator/environment.py` and uses UPPERCASE field names (matches existing `DIR`, `MAX_RETRIES`, etc.). The env var name `AIPERF_RESULTS_RETAIN_RUNS` and the access form `OperatorEnvironment.RESULTS.RETAIN_RUNS` are unchanged.

**Preconditions:**
- Task 1 committed.

**Postconditions:**
- `OperatorEnvironment.RESULTS.RETAIN_RUNS` returns 10 by default and respects `AIPERF_RESULTS_RETAIN_RUNS=N` override for N ≥ 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/operator/test_environment.py` (create file with header if missing):

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.environment settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_results_retain_runs_default_is_10() -> None:
    from aiperf.operator.environment import _ResultsSettings

    s = _ResultsSettings()
    assert s.RETAIN_RUNS == 10


def test_results_retain_runs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIPERF_RESULTS_RETAIN_RUNS", "25")
    from aiperf.operator.environment import _ResultsSettings

    s = _ResultsSettings()
    assert s.RETAIN_RUNS == 25


def test_results_retain_runs_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIPERF_RESULTS_RETAIN_RUNS", "0")
    from aiperf.operator.environment import _ResultsSettings

    with pytest.raises(ValidationError):
        _ResultsSettings()
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest -n auto tests/unit/operator/test_environment.py::test_results_retain_runs_default_is_10 -v`
Expected: `AttributeError: ... has no attribute 'RETAIN_RUNS'` or the assertion fails.

- [ ] **Step 3: Add the field**

Edit `src/aiperf/operator/environment.py`, inside `_ResultsSettings`, after the `COMPRESS_ON_DISK` field:

```python
    RETAIN_RUNS: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Max runs kept per <namespace>/<name>/ before retention trimming. "
        "Applied after every successful completion; the just-written uid is always protected.",
    )
```

- [ ] **Step 4: Run tests, full unit suite, ergonomics**

```bash
ruff format src/aiperf/operator/environment.py tests/unit/operator/test_environment.py
ruff check --fix src/aiperf/operator/environment.py tests/unit/operator/test_environment.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

Expected: all three new env-var tests pass; full suite green.

- [ ] **Step 5: Regenerate env-vars doc**

```bash
make generate-env-vars-docs
```

Expected: `docs/environment-variables.md` gets a new `AIPERF_RESULTS_RETAIN_RUNS` entry.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/environment.py tests/unit/operator/test_environment.py docs/environment-variables.md
git commit -s -m "$(cat <<'EOF'
feat(operator): add AIPERF_RESULTS_RETAIN_RUNS env var

Caps the number of per-job run directories kept on disk (default 10, ≥1).
Consumed by the retention pass that runs at every successful completion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Status helper + CRD `runUid` (spec § CRD / schema additions)

**Files:**
- Modify: `src/aiperf/operator/status.py` (after `set_results_path` at line 305)
- Modify: `src/aiperf/kubernetes/models.py` (add `run_uid` to `CRJobStatus`)
- Modify: `deploy/helm/aiperf-operator/templates/crd.yaml` (under status.properties, near `resultsPath`)
- Test: `tests/unit/operator/test_status.py`

**Preconditions:**
- Task 2 committed.

**Postconditions:**
- `StatusBuilder.set_run_uid(uid)` sets `status.runUid` and is chainable.
- `CRJobStatus.run_uid` is `str | None = None` with camelCase alias `runUid`.
- The CRD YAML schema validates `status.runUid` as an optional string.

- [ ] **Step 1: Write the failing status test**

Append to `tests/unit/operator/test_status.py`:

```python
def test_status_builder_set_run_uid_writes_camelcase_field() -> None:
    import kopf

    from aiperf.operator.status import StatusBuilder

    patch = kopf.Patch()
    sb = StatusBuilder(patch)
    result = sb.set_run_uid("5f8b2a3c-7d4e-4f1a-9b2c-1e3f4a5b6c7d")
    assert result is sb  # chainable
    assert patch.status["runUid"] == "5f8b2a3c-7d4e-4f1a-9b2c-1e3f4a5b6c7d"
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest -n auto tests/unit/operator/test_status.py::test_status_builder_set_run_uid_writes_camelcase_field -v`
Expected: `AttributeError: 'StatusBuilder' object has no attribute 'set_run_uid'`.

- [ ] **Step 3: Add `set_run_uid` to `StatusBuilder`**

Edit `src/aiperf/operator/status.py`, insert immediately after `set_results_path` (line 305):

```python
    def set_run_uid(self, uid: str) -> StatusBuilder:
        """Set the uid of the most recent successful run (mirror of metadata.uid)."""
        self._patch.status["runUid"] = uid
        return self
```

- [ ] **Step 4: Add `run_uid` to `CRJobStatus`**

Edit `src/aiperf/kubernetes/models.py`, in `CRJobStatus`, after the `completion_time` field:

```python
    run_uid: str | None = Field(
        default=None,
        description="metadata.uid of the most recent successful run. Use as "
        "{uid} in /api/v1/results/<ns>/<name>/runs/<uid>/ to pin historical artifacts.",
    )
```

- [ ] **Step 5: Add `runUid` to helm CRD template**

Edit `deploy/helm/aiperf-operator/templates/crd.yaml`, insert immediately after the `resultsPath` status property (line 353 area):

```yaml
              runUid:
                type: string
                description: metadata.uid of the most recent successful run. Use as {uid} in /api/v1/results/<ns>/<name>/runs/<uid>/ to pin historical artifacts.
```

- [ ] **Step 6: Run tests, full unit suite, ergonomics**

```bash
ruff format src/aiperf/operator/status.py src/aiperf/kubernetes/models.py tests/unit/operator/test_status.py
ruff check --fix src/aiperf/operator/status.py src/aiperf/kubernetes/models.py tests/unit/operator/test_status.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

Expected: status test passes; full unit suite green.

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/operator/status.py src/aiperf/kubernetes/models.py deploy/helm/aiperf-operator/templates/crd.yaml tests/unit/operator/test_status.py
git commit -s -m "$(cat <<'EOF'
feat(operator): add status.runUid for historical-run pinning

StatusBuilder.set_run_uid() mirrors the run's metadata.uid onto the
AIPerfJob status subresource. CRD schema and CRJobStatus pydantic
model both updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migration shim in results-server lifespan (spec § Migration shim)

**Files:**
- Modify: `src/aiperf/operator/results_server.py` (inside the lifespan context, function at line 76)
- Modify: `tests/unit/operator/test_results_server.py` (new test covering migration trigger)

**Preconditions:**
- Task 3 committed.

**Postconditions:**
- On operator process start, any pre-migration `<ns>/<name>/` directories under `RESULTS.DIR` are folded under `<name>/legacy/` with a `latest.txt=legacy` pointer.
- The hook logs the count of migrated jobs (INFO on >0, DEBUG on 0).
- Failure inside the shim logs a warning but does not crash the server.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/operator/test_results_server.py`:

```python
def test_lifespan_runs_migration_shim(tmp_path: Path) -> None:
    """Create a pre-migration layout, spin up the app, assert migration fired."""
    from fastapi.testclient import TestClient

    from aiperf.operator.results_layout import LATEST_POINTER
    from aiperf.operator.results_server import create_app

    _create_result_file(tmp_path, "ns", "legacy-job", "foo.json", b"{}")

    app = create_app(results_dir=tmp_path)
    with TestClient(app):
        # Lifespan fired; migration should have relocated the file.
        assert (tmp_path / "ns" / "legacy-job" / "legacy" / "foo.json").is_file()
        pointer = (tmp_path / "ns" / "legacy-job" / LATEST_POINTER).read_text().strip()
        assert pointer == "legacy"
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest -n auto tests/unit/operator/test_results_server.py::test_lifespan_runs_migration_shim -v`
Expected: fails because migration shim hasn't been wired in.

- [ ] **Step 3: Wire shim into `lifespan`**

Edit `src/aiperf/operator/results_server.py`, inside `_build_lifespan` (around line 76), before the DB init:

```python
from aiperf.operator.results_layout import migrate_legacy_layout
```

Then inside `lifespan`:

```python
    async def lifespan(app: FastAPI):
        # One-time layout migration: relocate pre-uid directories under <name>/legacy/
        try:
            migrated = await asyncio.to_thread(migrate_legacy_layout, base_dir)
            if migrated:
                logger.info(
                    "results layout migration: relocated %d jobs under %s",
                    len(migrated),
                    base_dir,
                )
        except Exception:
            logger.warning(
                "results layout migration failed under %s",
                base_dir,
                exc_info=True,
            )
        # ... existing DB init ...
```

Place the shim call BEFORE `db_holder[0] = ResultsDB(base_dir)` and before any dashboard mount.

- [ ] **Step 4: Run tests, full unit suite, ergonomics**

```bash
ruff format src/aiperf/operator/results_server.py tests/unit/operator/test_results_server.py
ruff check --fix src/aiperf/operator/results_server.py tests/unit/operator/test_results_server.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

Expected: new test passes; full unit suite green.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/results_server.py tests/unit/operator/test_results_server.py
git commit -s -m "$(cat <<'EOF'
feat(operator): run results layout migration at results-server lifespan

migrate_legacy_layout fires once per process at server startup (before DB
init). Errors log a warning but do not crash the server; the migration is
idempotent so a retry on next restart completes any half-moved tree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Write-path — route all `dest_dir` computations through `run_dir` + completion success gate (spec § Write-path changes)

**Files:**
- Modify: `src/aiperf/operator/handlers/monitor.py:1053`
- Modify: `src/aiperf/operator/handlers/completion.py:242, 373`
- Modify: `src/aiperf/operator/handlers/_completion_fetch.py:401`
- Modify: `src/aiperf/operator/job_index.py:198`
- Modify: `src/aiperf/operator/handlers/completion.py:243` (add success gate block)

**Preconditions:**
- Tasks 1-4 committed.
- `uid` is already a kopf handler kwarg at every call site (verified via `handlers/create.py:413` signature).

**Postconditions:**
- Every `dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id` becomes `dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)`.
- At `completion.py:243` success path, a single block writes `latest.txt`, mirrors uid to status, and trims retention.

- [ ] **Step 1: Audit uid availability**

Run these greps to confirm `uid` is in scope at each call site:

```bash
grep -n "dest_dir\s*=\s*OperatorEnvironment.RESULTS.DIR" src/aiperf/operator/handlers/monitor.py src/aiperf/operator/handlers/completion.py src/aiperf/operator/handlers/_completion_fetch.py src/aiperf/operator/job_index.py
grep -B 20 "dest_dir = OperatorEnvironment.RESULTS.DIR" src/aiperf/operator/handlers/monitor.py | grep -E "def |uid"
```

Expected: every enclosing function either takes `uid` as a parameter or has it via `**kwargs` / `body["metadata"]["uid"]`. If any site is missing it, thread `uid: str` through as a positional arg and update callers. Note any gaps and handle them in Step 2 before editing lines.

- [ ] **Step 2: Update each write site**

At each of the 5 sites, change:

```python
dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id
```

to:

```python
from aiperf.operator.results_layout import run_dir
# ...
dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)
```

Import block hygiene: add the `run_dir` import once near the existing `OperatorEnvironment` import at the top of each file.

If any enclosing function lacks `uid`, add it as a required positional arg and update the caller chain. The smallest safe patch is preferred — do NOT refactor signatures beyond what's needed.

- [ ] **Step 3: Add success gate at `completion.py:243`**

Edit `src/aiperf/operator/handlers/completion.py`. After the existing `sb.set_results_path(str(dest_dir))` at line 243:

```python
    if has_files:
        dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)
        sb.set_results_path(str(dest_dir))
        write_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id, uid)
        sb.set_run_uid(uid)
        try:
            deleted = enforce_retention(
                OperatorEnvironment.RESULTS.DIR,
                namespace,
                job_id,
                keep=OperatorEnvironment.RESULTS.RETAIN_RUNS,
                protect_uid=uid,
            )
            if deleted:
                logger.info(
                    "retention: trimmed %d old runs for %s/%s",
                    len(deleted), namespace, job_id,
                )
        except Exception:
            logger.warning(
                "retention pass failed for %s/%s; continuing",
                namespace, job_id, exc_info=True,
            )
        events.results_stored(body, str(dest_dir), len(result.downloaded))
        logger.info(f"Downloaded {len(result.downloaded)} result files to {dest_dir}")
```

Top-of-file imports: add `write_latest, enforce_retention` alongside the existing `run_dir` import.

- [ ] **Step 4: Run full unit suite**

```bash
ruff format src/aiperf/operator/handlers/monitor.py src/aiperf/operator/handlers/completion.py src/aiperf/operator/handlers/_completion_fetch.py src/aiperf/operator/job_index.py
ruff check --fix src/aiperf/operator/handlers/monitor.py src/aiperf/operator/handlers/completion.py src/aiperf/operator/handlers/_completion_fetch.py src/aiperf/operator/job_index.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

Expected: all tests green. If any existing handler test seeds `<ns>/<name>/*.json` and then reads it by name, it will fail — fix those seed paths to use `run_dir(...)` with an explicit uid fixture, and add a `write_latest` call so the chokepoint reader still finds files.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/handlers/monitor.py src/aiperf/operator/handlers/completion.py src/aiperf/operator/handlers/_completion_fetch.py src/aiperf/operator/job_index.py
git commit -s -m "$(cat <<'EOF'
feat(operator): route all result writes through uid-keyed run_dir

Every dest_dir computation now threads metadata.uid through run_dir(),
so each CR creation writes to <ns>/<name>/<uid>/ instead of the flat
<ns>/<name>/ path. The completion.py success gate is the single atomic
flip point: set_results_path → write_latest → set_run_uid →
enforce_retention, all on the same commit of the on-disk state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Read-path chokepoint — `_resolve_job_dir` + `/runs/<uid>/` routes (spec § Read-path changes)

**Files:**
- Modify: `src/aiperf/operator/routers/results_files.py`
  - `_resolve_job_dir` at line 224-228
  - `_scan_job_dirs` at line 185-204
  - Three new historical routes inside `create_results_files_router`
- Modify: `tests/unit/operator/test_results_server.py` (new tests for latest resolution + historical routes + invalid uid)

**Preconditions:**
- Task 5 committed.

**Postconditions:**
- `GET /api/v1/results/{ns}/{name}` returns the latest run's files.
- `GET /api/v1/results/{ns}/{name}/runs/{uid}` returns the specified historical run's files.
- `GET /api/v1/results/{ns}/{name}/runs/{uid}.zip` and `/runs/{uid}/{filename:path}` work analogously.
- Invalid uids on the historical routes return 422 (pattern rejection) without touching disk.
- `_scan_job_dirs` yields one `JobEntry` per `<ns>/<name>/` (the latest run).

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/operator/test_results_server.py`:

```python
def _seed_uid_run(
    base: Path,
    namespace: str,
    name: str,
    uid: str,
    filename: str,
    content: bytes = b"{}",
) -> Path:
    from aiperf.operator.results_layout import run_dir, write_latest

    d = run_dir(base, namespace, name, uid)
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_bytes(content)
    write_latest(base, namespace, name, uid)
    return d


_UID_OLD = "11111111-1111-4111-8111-111111111111"
_UID_NEW = "22222222-2222-4222-8222-222222222222"


def test_list_job_files_resolves_latest(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    _seed_uid_run(tmp_path, "ns", "job", _UID_OLD, "old.json", b'{"v":1}')
    _seed_uid_run(tmp_path, "ns", "job", _UID_NEW, "new.json", b'{"v":2}')

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job")
        assert r.status_code == 200
        names = {f["name"] for f in r.json()["files"]}
        assert names == {"new.json"}


def test_historical_route_pins_uid(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    _seed_uid_run(tmp_path, "ns", "job", _UID_OLD, "old.json", b'{"v":1}')
    _seed_uid_run(tmp_path, "ns", "job", _UID_NEW, "new.json", b'{"v":2}')

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(f"/api/v1/results/ns/job/runs/{_UID_OLD}")
        assert r.status_code == 200
        names = {f["name"] for f in r.json()["files"]}
        assert names == {"old.json"}


def test_historical_route_invalid_uid_rejected(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    _seed_uid_run(tmp_path, "ns", "job", _UID_OLD, "old.json")
    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs/..%2Fevil")
        assert r.status_code in (404, 422)


def test_historical_zip_bundle_pins_uid(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    _seed_uid_run(tmp_path, "ns", "job", _UID_OLD, "old.json", b'{"v":1}')
    _seed_uid_run(tmp_path, "ns", "job", _UID_NEW, "new.json", b'{"v":2}')
    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(f"/api/v1/results/ns/job/runs/{_UID_OLD}.zip")
        assert r.status_code == 200
        assert b"old.json" in r.content


def test_scan_job_dirs_collapses_to_latest(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from aiperf.operator.results_server import create_app

    _seed_uid_run(tmp_path, "ns", "job", _UID_OLD, "old.json")
    _seed_uid_run(tmp_path, "ns", "job", _UID_NEW, "new.json")
    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results")
        assert r.status_code == 200
        entries = [
            (j["namespace"], j["job_id"], j["file_count"])
            for j in r.json()["jobs"]
        ]
        # One JobEntry per <ns>/<name>, file_count from the latest run only.
        assert entries == [("ns", "job", 1)]
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest -n auto tests/unit/operator/test_results_server.py -k "resolves_latest or pins_uid or invalid_uid or zip_bundle_pins or collapses_to_latest"`
Expected: tests fail (404s, wrong counts, or `Method Not Allowed`).

- [ ] **Step 3: Update `_resolve_job_dir`**

In `src/aiperf/operator/routers/results_files.py`, replace lines 224-229:

```python
from aiperf.operator.results_layout import UID_RE, resolve_run_dir


def _resolve_job_dir(
    base_dir: Path,
    namespace: str,
    job_id: str,
    uid: str | None = None,
) -> Path:
    """Resolve a run dir under ``<base>/<ns>/<name>/``.

    ``uid=None`` → latest run via ``latest.txt``.
    ``uid="<uuid>"`` → explicit historical run.
    """
    resolved = resolve_run_dir(base_dir, namespace, job_id, uid=uid)
    if resolved is None:
        target = f"{namespace}/{job_id}" + (f"/runs/{uid}" if uid else "")
        raise HTTPException(404, f"No results for {target}")
    return resolved
```

Keep the existing `_safe_resolve` helper — `resolve_run_dir` already guards against traversal via the explicit path construction, but the file-download route uses `_safe_resolve` on the *filename* portion (separate concern).

- [ ] **Step 4: Update `_scan_job_dirs` to walk uid subdirs and collapse to latest**

Replace the body (lines 185-204):

```python
def _scan_job_dirs(base_dir: Path) -> list[JobEntry]:
    """Walk ``<namespace>/<job_id>/<uid>/`` under ``base_dir`` and summarize each job.

    Yields one :class:`JobEntry` per ``<ns>/<name>`` using the run pointed to by
    ``latest.txt``. Jobs whose pointer is missing or targets a vanished uid are
    skipped silently.
    """
    found: list[JobEntry] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        for name_dir in sorted(ns_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            latest_dir = resolve_run_dir(base_dir, ns_dir.name, name_dir.name)
            if latest_dir is None:
                continue
            files = [f for f in latest_dir.iterdir() if f.is_file()]
            if not files:
                continue
            found.append(
                JobEntry(
                    namespace=ns_dir.name,
                    job_id=name_dir.name,
                    file_count=len(files),
                    total_size_bytes=sum(f.stat().st_size for f in files),
                )
            )
    return found
```

- [ ] **Step 5: Add the three historical routes**

Still in `src/aiperf/operator/routers/results_files.py`, inside `create_results_files_router`, after the existing `download_file` route:

```python
    @router.get(
        "/results/{namespace}/{job_id}/runs/{uid}.zip",
    )
    async def download_historical_bundle(
        namespace: str, job_id: str, uid: str
    ) -> StreamingResponse:
        """Download every file from a pinned historical run as a zip."""
        if not UID_RE.match(uid):
            raise HTTPException(422, f"Invalid uid: {uid}")
        job_dir = _resolve_job_dir(base_dir, namespace, job_id, uid=uid)
        bundle_name = f"{namespace}__{job_id}__{uid}.zip"
        return StreamingResponse(
            _stream_job_bundle(job_dir),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{bundle_name}"',
                "X-Filename": bundle_name,
            },
        )

    @router.get(
        "/results/{namespace}/{job_id}/runs/{uid}",
        response_model=FileListResponse,
    )
    async def list_historical_files(
        namespace: str, job_id: str, uid: str
    ) -> FileListResponse:
        """List files for a pinned historical run."""
        if not UID_RE.match(uid):
            raise HTTPException(422, f"Invalid uid: {uid}")
        job_dir = _resolve_job_dir(base_dir, namespace, job_id, uid=uid)
        files = await asyncio.to_thread(_list_job_files, job_dir)
        return FileListResponse(namespace=namespace, job_id=job_id, files=files)

    @router.get("/results/{namespace}/{job_id}/runs/{uid}/{filename:path}")
    async def download_historical_file(
        namespace: str,
        job_id: str,
        uid: str,
        filename: str,
        request: Request,
    ) -> StreamingResponse:
        """Download a file from a pinned historical run with content negotiation."""
        if not UID_RE.match(uid):
            raise HTTPException(422, f"Invalid uid: {uid}")
        job_dir = _resolve_job_dir(base_dir, namespace, job_id, uid=uid)
        zst_path = _safe_resolve(job_dir, filename + ".zst")
        raw_path = _safe_resolve(job_dir, filename)
        if zst_path and zst_path.is_file():
            return _serve_zst_file(request, zst_path, filename)
        if raw_path and raw_path.is_file():
            return _serve_raw_file(request, raw_path)
        raise HTTPException(404, f"File not found: {filename}")
```

**Route ordering note:** FastAPI matches in registration order. `/runs/{uid}.zip` and `/runs/{uid}` must be registered BEFORE `/runs/{uid}/{filename:path}` so the matcher doesn't swallow `.zip` into `filename`.

- [ ] **Step 6: Run tests**

```bash
ruff format src/aiperf/operator/routers/results_files.py tests/unit/operator/test_results_server.py
ruff check --fix src/aiperf/operator/routers/results_files.py tests/unit/operator/test_results_server.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

Expected: 5 new tests pass; full unit suite green. If any pre-existing `results_server` test seeded directly under `<ns>/<name>/` (pre-uid layout), update it to use `_seed_uid_run` with a synthetic uid.

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/operator/routers/results_files.py tests/unit/operator/test_results_server.py
git commit -s -m "$(cat <<'EOF'
feat(operator): resolve results via latest.txt + add /runs/<uid>/ routes

_resolve_job_dir now walks through results_layout.resolve_run_dir, so the
existing two-arg endpoints silently serve the latest run. New additive
routes /runs/<uid>, /runs/<uid>.zip, and /runs/<uid>/<filename> pin a
specific historical run; they validate uid against UID_RE before any
disk access. _scan_job_dirs collapses to one JobEntry per <ns>/<name>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Read-path cascade — analytics, jobs router, job_union, results_db (spec § Read-path changes)

**Files:**
- Modify: `src/aiperf/operator/routers/results_analytics.py:208`
- Modify: `src/aiperf/operator/routers/jobs.py:160`
- Modify: `src/aiperf/operator/results_db.py:282`
- Modify: `src/aiperf/operator/job_union.py:115-118, 191-194, 320`
- Modify: any test that seeds result files under the pre-uid flat path

**Preconditions:**
- Task 6 committed.

**Postconditions:**
- Every remaining read site resolves via `resolve_run_dir`. None build `<base>/<ns>/<name>/...` manually.
- Analytics/jobs/DB consumers all see the latest run by default; behavior for missing data is a 404 or empty-response rather than a crash.

- [ ] **Step 1: Audit the four files**

```bash
grep -n "/\s*namespace\s*/\s*\(job_id\|name\)" src/aiperf/operator/routers/results_analytics.py src/aiperf/operator/routers/jobs.py src/aiperf/operator/results_db.py src/aiperf/operator/job_union.py
```

Confirm the four sites from the spec plus any incidental ones the spec missed. Add any stragglers to your edit list.

- [ ] **Step 2: Patch `results_analytics.py:208`**

Find:

```python
spec_file = base_dir / namespace / job_id / "job_spec.json"
```

Replace with:

```python
from aiperf.operator.results_layout import resolve_run_dir
# ...
run = resolve_run_dir(base_dir, namespace, job_id)
if run is None:
    raise HTTPException(404, f"No results for {namespace}/{job_id}")
spec_file = run / "job_spec.json"
```

- [ ] **Step 3: Patch `jobs.py:160`**

Find:

```python
job_dir = results_dir / namespace / name
```

Replace with:

```python
from aiperf.operator.results_layout import resolve_run_dir
# ...
job_dir = resolve_run_dir(results_dir, namespace, name)
if job_dir is None:
    continue  # or the existing skip path; whatever the surrounding code does
```

Match whatever graceful-skip the surrounding code already uses; do not invent new behavior.

- [ ] **Step 4: Patch `results_db.py:282`**

Find:

```python
job_dir = self._results_dir / namespace / job_id
if not job_dir.is_dir():
    return
```

Replace with:

```python
from aiperf.operator.results_layout import resolve_run_dir
# ...
job_dir = resolve_run_dir(self._results_dir, namespace, job_id)
if job_dir is None:
    return
```

The subsequent checks for `zst` / `raw` files remain unchanged — they now reference the resolved per-uid dir.

- [ ] **Step 5: Patch `job_union.py`**

Lines 115-118 and 191-194 currently iterate `ns_dir.iterdir()` children as job dirs. Keep that iteration (each child is a `<name>` dir), but resolve the run dir via `resolve_run_dir(results_dir, ns_dir.name, name_dir.name)` before reading `_SUMMARY_FILE`.

Line 320:

```python
summary_path = results_dir / namespace / name / _SUMMARY_FILE
```

Replace with:

```python
from aiperf.operator.results_layout import resolve_run_dir
# ...
run = resolve_run_dir(results_dir, namespace, name)
if run is None:
    return None  # or whatever the function's "no data" contract is
summary_path = run / _SUMMARY_FILE
```

- [ ] **Step 6: Fix any existing tests**

Run the full suite once first:

```bash
uv run pytest -n auto tests/unit/
```

If tests like `test_results_db.py`, `test_results_analytics.py`, or `test_job_union.py` seed `<base>/<ns>/<name>/*.json` directly, update them to use `_seed_uid_run` (or an inline equivalent) so the layout matches the post-migration invariant.

- [ ] **Step 7: Final verify, ergonomics, commit**

```bash
ruff format src/aiperf/operator/routers/results_analytics.py src/aiperf/operator/routers/jobs.py src/aiperf/operator/results_db.py src/aiperf/operator/job_union.py
ruff check --fix src/aiperf/operator/routers/results_analytics.py src/aiperf/operator/routers/jobs.py src/aiperf/operator/results_db.py src/aiperf/operator/job_union.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

```bash
git add src/aiperf/operator/routers/results_analytics.py src/aiperf/operator/routers/jobs.py src/aiperf/operator/results_db.py src/aiperf/operator/job_union.py tests/
git commit -s -m "$(cat <<'EOF'
feat(operator): route analytics/jobs/db/union reads through resolve_run_dir

Final read-path cascade: results_analytics job_spec lookup, the dashboard
jobs router listing, the DuckDB analytics scan, and the job_union
summary path all now resolve via <ns>/<name>/<latest-uid>/. No read
site builds a results path by hand anymore.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Integration test — same-name CR resubmit preserves prior run (spec § Tests, integration subsection)

**Files:**
- Create: `tests/integration/operator/test_same_name_resubmit.py` (marker: `component_integration`)

**Preconditions:**
- Tasks 1-7 committed.
- `make first-time-setup` has been run (integration deps available).

**Postconditions:**
- New test passes under `uv run pytest -m component_integration -n auto -k same_name_resubmit`.

- [ ] **Step 1: Inspect existing integration harness**

```bash
ls tests/integration/operator/
grep -l "component_integration" tests/integration/operator/*.py | head -5
```

Identify the fixture that spins up the operator + results server in-process, and the helper that submits a minimal AIPerfJob spec. Reuse that scaffolding.

- [ ] **Step 2: Write the integration test**

Create `tests/integration/operator/test_same_name_resubmit.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit an AIPerfJob, delete it, resubmit with the same name, verify both
runs coexist on disk and the API serves the newer by default while the older
is still reachable via /runs/<uid>/."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_same_name_resubmit_preserves_prior_run(
    operator_harness,  # fixture: spins up operator + results server against tmp_path
    minimal_aiperfjob_yaml,  # fixture: returns YAML text for a trivial AIPerfJob
) -> None:
    # First submission
    cr1 = await operator_harness.submit("foo", minimal_aiperfjob_yaml)
    await operator_harness.wait_until_complete(cr1.name, timeout=60)
    uid_old = cr1.status["runUid"]
    files_old = await operator_harness.list_results("default", "foo")
    assert files_old, "First run should have result files"

    await operator_harness.delete(cr1.name)
    await operator_harness.wait_until_gone(cr1.name, timeout=30)

    # Second submission under the same name — fresh uid
    cr2 = await operator_harness.submit("foo", minimal_aiperfjob_yaml)
    await operator_harness.wait_until_complete(cr2.name, timeout=60)
    uid_new = cr2.status["runUid"]
    assert uid_new != uid_old, "Second run must have a distinct uid"

    # Disk: both uid directories present
    base = operator_harness.results_dir
    assert (base / "default" / "foo" / uid_old).is_dir()
    assert (base / "default" / "foo" / uid_new).is_dir()

    # API: default route serves the newer run
    files_latest = await operator_harness.list_results("default", "foo")
    assert files_latest, "Latest route should resolve"
    # Newer uid should be reflected in status
    assert files_latest != files_old or len(files_latest) == len(files_old)

    # API: historical route still serves the older run
    older_files = await operator_harness.list_results(
        "default", "foo", uid=uid_old,
    )
    assert older_files, "Historical /runs/<uid>/ route should resolve older run"
```

If the harness exposes methods with different names, adapt the call surface to match — the assertions are the important part.

- [ ] **Step 3: Run the integration test**

```bash
uv run pytest -m component_integration -n auto tests/integration/operator/test_same_name_resubmit.py -v
```

Expected: pass.

- [ ] **Step 4: Final verify + commit**

```bash
ruff format tests/integration/operator/test_same_name_resubmit.py
ruff check --fix tests/integration/operator/test_same_name_resubmit.py
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
```

```bash
git add tests/integration/operator/test_same_name_resubmit.py
git commit -s -m "$(cat <<'EOF'
test(operator): assert same-name CR resubmit preserves prior run

Covers the user-visible contract: kubectl apply → delete → kubectl apply
on the same-named AIPerfJob yields two distinct run directories on disk,
the latest-resolution route serves the newer one, and the older is still
reachable via /api/v1/results/<ns>/<name>/runs/<uid>/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Docs — kubernetes-flow, generate-all-docs, minor operator readme hooks (spec § Estimated scope)

**Files:**
- Modify: `docs/dev/kubernetes-flow.md` (results-layout section)
- Regenerate: `docs/environment-variables.md` (already covered in Task 2, but re-run for consistency)
- Regenerate: `docs/cli-options.md` via `make generate-cli-docs` (no-op unless CLI changed, run to satisfy pre-commit)

**Preconditions:**
- Tasks 1-8 committed.

**Postconditions:**
- `docs/dev/kubernetes-flow.md` has a subsection titled "Results layout and history" describing the `<ns>/<name>/<uid>/` shape, `latest.txt` pointer semantics, and the new `/runs/<uid>/` API routes.
- `make generate-all-docs` reports no pending changes.

- [ ] **Step 1: Add the docs section**

Append to `docs/dev/kubernetes-flow.md` under a new heading `## Results layout and history`:

```markdown
## Results layout and history

Each AIPerfJob submission is keyed by its Kubernetes `metadata.uid`, so re-creating a CR with the same name never overwrites prior results.

On-disk shape under the operator's results PVC (`AIPERF_RESULTS_DIR`, default `/data`):

    <base>/<namespace>/<name>/
      <uid-A>/   ← run A artifacts
      <uid-B>/   ← run B artifacts
      latest.txt ← pointer to the current uid

The pointer is written atomically (`os.replace`) at the single success gate in `handlers/completion.py` — alongside `status.resultsPath` and `status.runUid`. A retention pass (env `AIPERF_RESULTS_RETAIN_RUNS`, default 10) trims older run dirs on every successful completion; the just-written uid is always protected.

HTTP API:

- `GET /api/v1/results/<ns>/<name>/...` — serves the **latest** run (resolved via `latest.txt`). Backward compatible: existing clients get the latest without knowing about uids.
- `GET /api/v1/results/<ns>/<name>/runs/<uid>/...` — pins a specific historical run. `uid` must match `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^legacy$`.

Migration of pre-uid data runs automatically at results-server lifespan: any `<ns>/<name>/` directory with files directly under it gets folded into `<name>/legacy/` and `latest.txt=legacy`. Idempotent.
```

- [ ] **Step 2: Regenerate all generated docs**

```bash
make generate-all-docs
```

Expected: env-vars doc already shows `AIPERF_RESULTS_RETAIN_RUNS`. CLI docs unchanged. Plugin docs unchanged.

- [ ] **Step 3: Final verify**

```bash
uv run pytest -n auto tests/unit/
make check-ergonomics
make check-ruff-baselined
pre-commit run --files docs/dev/kubernetes-flow.md docs/environment-variables.md
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/dev/kubernetes-flow.md docs/environment-variables.md docs/cli-options.md
git commit -s -m "$(cat <<'EOF'
docs(kubernetes): document uid-keyed results layout + /runs/<uid>/ API

Adds a 'Results layout and history' section to kubernetes-flow covering the
<ns>/<name>/<uid>/ directory shape, latest.txt pointer semantics, retention
env var, and the new historical-run API routes. Includes regenerated
env-vars doc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Completion

After Task 9 is committed:

```bash
git log --oneline origin/main..HEAD | head -15
```

Expected: 9 new commits on `ajc/k8s`, one per task. Each commit is independently reviewable.

Optional wrap-up:
- Open a PR against `origin/main` for review (user may prefer branch-only per standing feedback).
- Run `/aiperf-llm-ergonomics-review` before shipping — it flags semantic issues the mechanical checks don't catch.
- Run `/aiperf-code-review` for broader review against `origin/main`.

## Self-review checklist (done by plan author)

- **Spec coverage:** every spec section maps to at least one task. ✅
  - § File inventory → Tasks 1, 3, 4, 5, 6, 7
  - § Data model → Task 1
  - § Write-path changes → Task 5
  - § Read-path changes → Tasks 6, 7
  - § Retention → Task 1 (impl) + Task 5 (wiring)
  - § Migration shim → Task 1 (impl) + Task 4 (wiring)
  - § CRD/schema additions → Tasks 2, 3
  - § Tests → Tasks 1, 6, 8
  - § Risks/OQ → captured as inline comments in implementation
- **Placeholder scan:** no TBD/TODO, every code step shows the actual code, every test step shows actual assertions. ✅
- **Type consistency:** `run_dir`, `resolve_run_dir`, `enforce_retention`, `UID_RE`, `LATEST_POINTER`, `set_run_uid`, `RETAIN_RUNS` — names referenced in tasks 3-9 exactly match the signatures defined in Task 1 and Task 2. ✅
- **Scope:** ~450 lines of change across 9 atomic tasks; single PR; single unit-test run per task as per standing feedback. ✅
