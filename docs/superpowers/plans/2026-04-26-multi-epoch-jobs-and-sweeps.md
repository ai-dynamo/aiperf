# Multi-Epoch — AIPerfJob + AIPerfSweep + ui-v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full first-class multi-epoch parity to `AIPerfJob` and `AIPerfSweep` — separate persistent results per epoch, explicit (sweep_epoch → child epoch) linkage, epoch-aware backend API, DuckDB analytics, and a ui-v1 epoch selector on every detail page. No backwards compatibility — pre-epoch on-disk shape and the legacy `LEGACY_EPOCH` code path are deleted.

**Architecture:** Sweep epoch derives from `metadata.creationTimestamp` (same as jobs — already partially wired). Sweep results layered as `<base>/<ns>/sweeps/<name>/<epoch>/{aggregate,conditions,children}.json` with `latest.txt` pointer. Sweep-controller writes `children.json` per epoch as the authoritative manifest. Child AIPerfJob CR names embed the sweep epoch so each rerun gets fresh children at fresh paths. URL grammar adopts the existing API shape: `/jobs/:ns/:name/runs/:epoch`. Default-no-epoch behaviour selects `latest.txt`.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI, kubernetes_asyncio, kopf, orjson, DuckDB, pytest (`-n auto`), Preact 10 + htm + Chart.js 4.

**Spec:** `docs/superpowers/specs/2026-04-26-multi-epoch-jobs-and-sweeps-design.md`

---

## Conventions

- Branch HEAD: `ajc/k8s` — commit on this branch.
- All commits: `git commit -s --no-verify`. Run `ruff format <files> && ruff check --fix <files>` BEFORE the commit. Reason: pre-commit framework's internal `git stash --include-untracked` corrupts state under parallel agents.
- Tests: `uv run pytest tests/unit/<subfolder>/ -n auto` — single subfolder per invocation, never combined, never `--all-files`.
- **NEVER `git stash` (any subcommand, including `git stash list` — read-only or not, the user has it strictly forbidden as their #1 rule).**
- **NEVER `git restore`** in any form.
- If `git commit` fails with `.git/index.lock: File exists`, that's a transient race with another agent — wait 2 seconds and retry. Do NOT delete the lock file.
- Stage ONLY the files the task touches — the repo has many unstaged changes from before our session.
- File header: SPDX block matches existing siblings.
- All public Pydantic fields require `Field(description=...)`; type hints on all functions; no `Optional[X]` (use `X | None`).

## File Map

### Backend — new files

| Path | Responsibility |
|---|---|
| `src/aiperf/operator/routers/sweeps_models.py` | Already exists — extend with `SweepEpochsResponse`, `ChildrenManifestResponse`, `ChildrenManifestEntry`. |
| `src/aiperf/operator/routers/jobs_models.py` | Already exists — extend with `JobEpochsResponse`. |
| `tools/wipe_pre_epoch_results.py` | One-shot pre-epoch wipe. |
| `src/aiperf/operator/ui-v1/components/epoch-selector.js` | Reusable dropdown + "viewing N of M · click for latest" banner. |

### Backend — edited files

| Path | Change |
|---|---|
| `src/aiperf/operator/results_layout.py` | Strip `LEGACY_EPOCH`, `migrate_legacy_layout`; tighten `EPOCH_RE`; add `resolve_sweep_dir(epoch=None)`, `list_sweep_epochs`, `resolve_sweep_latest`, `write_sweep_latest`. |
| `src/aiperf/operator/job_union.py` | Optional `epoch` on `find_any_job`, `_archived_from_summary`, `_scan_pvc_jobs`. |
| `src/aiperf/operator/sweep_union.py` | Optional `epoch` on `find_any_sweep`; rework `_record_from_archive` to take an epoch sub-path. |
| `src/aiperf/operator/routers/jobs.py` | `?epoch=` on detail; new `/jobs/{ns}/{name}/epochs`. |
| `src/aiperf/operator/routers/sweeps.py` | `?epoch=` on detail + `/cells`; new `/epochs`, `/children`. |
| `src/aiperf/operator/handlers/sweep/create.py` | Pass sweep epoch through to child name template. |
| `src/aiperf/operator/handlers/sweep/lifecycle.py` | `write_sweep_latest` on terminal phase. |
| `src/aiperf/sweep_controller/aggregator.py` | Write under `<epoch>/`; new `write_children_manifest`. |
| `src/aiperf/sweep_controller/k8s_executor.py` | Sweep epoch in child name template; write child epoch into `sweep.json`. |
| `src/aiperf/operator/results_db.py` | Add `epoch` column; default `MAX(epoch)` filter. |
| `deploy/helm/aiperf-operator/templates/crd.yaml` | `runEpoch` validation on AIPerfJob status. |
| `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml` | Structured `childRunEpochsRef`. |

### Frontend — edited files

| Path | Change |
|---|---|
| `src/aiperf/operator/ui-v1/app.js` | Two new routes (`/jobs/:ns/:name/runs/:epoch`, `/sweeps/:ns/:name/runs/:epoch`). |
| `src/aiperf/operator/ui-v1/lib/api.js` | New methods: `getJobEpochs`, `getJob(epoch?)`, `getSweepEpochs`, `getSweep(epoch?)`, `getSweepCells(epoch?)`, `getSweepChildren(epoch)`. |
| `src/aiperf/operator/ui-v1/pages/job-detail.js` | Mount `EpochSelector`; epoch-aware fetch. |
| `src/aiperf/operator/ui-v1/pages/sweep-detail.js` | Mount `EpochSelector`; propagate epoch to all panels. |
| `src/aiperf/operator/ui-v1/pages/jobs.js` | "Epochs" column. |
| `src/aiperf/operator/ui-v1/pages/sweeps.js` | "Epochs" column. |
| `src/aiperf/operator/ui-v1/components/breadcrumb.js` | Render `runs/:epoch` segments. |

---

# PR 1 — Layout primitives & wipe

Foundation for everything else. Independent, no behavioural change to live operator yet.

## Task 1: Strip `LEGACY_EPOCH` + tighten `EPOCH_RE`

**Files:**
- Modify: `src/aiperf/operator/results_layout.py`
- Modify: `tests/unit/operator/test_results_layout.py`

- [ ] **Step 1: Update tests first**

In `tests/unit/operator/test_results_layout.py`, delete every test that exercises `LEGACY_EPOCH`, the literal string `"legacy"`, or `migrate_legacy_layout`. Search:
```bash
grep -n "LEGACY_EPOCH\|migrate_legacy\|\"legacy\"\|legacy_dir" tests/unit/operator/test_results_layout.py
```

Add a new test asserting the regex no longer matches `"legacy"`:

```python
def test_epoch_re_no_longer_matches_legacy() -> None:
    from aiperf.operator.results_layout import EPOCH_RE
    assert EPOCH_RE.match("legacy") is None
    assert EPOCH_RE.match("1714069323") is not None
```

- [ ] **Step 2: Run tests to verify the legacy ones are gone and the new one fails**

Run: `uv run pytest tests/unit/operator/test_results_layout.py -n auto`
Expected: any remaining tests pass; the new `test_epoch_re_no_longer_matches_legacy` fails because `EPOCH_RE` still includes `|^legacy$`.

- [ ] **Step 3: Edit `results_layout.py`**

In `src/aiperf/operator/results_layout.py`:

- Delete the line `LEGACY_EPOCH = "legacy"`.
- Change `EPOCH_RE = re.compile(r"^\d{9,11}$|^legacy$")` to `EPOCH_RE = re.compile(r"^\d{9,11}$")`.
- Delete the entire `migrate_legacy_layout(...)` function.
- Remove `"LEGACY_EPOCH"` and `"migrate_legacy_layout"` from `__all__`.
- In any docstring or example referencing `legacy`, remove the mention (search the file for the literal `legacy`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/test_results_layout.py -n auto`
Expected: all pass.

- [ ] **Step 5: Audit downstream call sites**

Run: `grep -rn "LEGACY_EPOCH\|migrate_legacy_layout" src/ tests/`
Expected: no matches anywhere. If any remain (likely in `handlers/completion.py` or similar), delete those references too. Add their files to the commit.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
ruff check --fix src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
git add src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
# Plus any handlers/* files you needed to clean.
git commit -s --no-verify -m "refactor(operator): drop LEGACY_EPOCH and migrate_legacy_layout

The pre-epoch on-disk layout is no longer supported. EPOCH_RE
becomes pure decimal-seconds. The migration helper is removed —
the wipe script in tools/ replaces it for any cluster that still
has pre-epoch dirs."
```

---

## Task 2: Sweep layout helpers (`resolve_sweep_dir(epoch=)`, `list_sweep_epochs`, `resolve_sweep_latest`, `write_sweep_latest`)

**Files:**
- Modify: `src/aiperf/operator/results_layout.py`
- Modify: `tests/unit/operator/test_results_layout.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/operator/test_results_layout.py`:

```python
def test_resolve_sweep_dir_with_epoch(tmp_path: Path) -> None:
    from aiperf.operator.results_layout import resolve_sweep_dir
    p = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    p.mkdir(parents=True)
    assert resolve_sweep_dir(tmp_path, "bench", "s1", epoch="1714069323") == p


def test_resolve_sweep_dir_with_epoch_missing_returns_none(tmp_path: Path) -> None:
    from aiperf.operator.results_layout import resolve_sweep_dir
    assert resolve_sweep_dir(tmp_path, "bench", "s1", epoch="9999999999") is None


def test_resolve_sweep_dir_no_epoch_resolves_via_latest(tmp_path: Path) -> None:
    from aiperf.operator.results_layout import resolve_sweep_dir, write_sweep_latest
    p = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    p.mkdir(parents=True)
    write_sweep_latest(tmp_path, "bench", "s1", "1714069323")
    assert resolve_sweep_dir(tmp_path, "bench", "s1") == p


def test_list_sweep_epochs_orders_by_epoch_asc(tmp_path: Path) -> None:
    from aiperf.operator.results_layout import list_sweep_epochs, write_sweep_latest
    base = tmp_path / "bench" / "sweeps" / "s1"
    (base / "1714069323").mkdir(parents=True)
    (base / "1714069324" / "aggregate.json").parent.mkdir(parents=True)
    (base / "1714069324" / "aggregate.json").write_text("{}")
    write_sweep_latest(tmp_path, "bench", "s1", "1714069324")
    epochs = list_sweep_epochs(tmp_path, "bench", "s1")
    assert [e.epoch for e in epochs] == ["1714069323", "1714069324"]
    assert epochs[-1].is_latest is True
    assert epochs[0].is_latest is False


def test_resolve_sweep_latest_returns_none_when_unset(tmp_path: Path) -> None:
    from aiperf.operator.results_layout import resolve_sweep_latest
    assert resolve_sweep_latest(tmp_path, "bench", "s1") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/operator/test_results_layout.py -n auto -k sweep`
Expected: ImportError on `list_sweep_epochs`, `resolve_sweep_latest`, `write_sweep_latest`; OR `resolve_sweep_dir` rejects `epoch=` kwarg.

- [ ] **Step 3: Implement**

In `src/aiperf/operator/results_layout.py`:

a) Update `resolve_sweep_dir` signature (the function landed in the prior PR; replace its body):

```python
def resolve_sweep_dir(
    base: Path, namespace: str, name: str, *, epoch: str | None = None
) -> Path | None:
    """Return ``<base>/<ns>/sweeps/<name>/<epoch>/`` or fall through to ``latest.txt``.

    Mirrors :func:`resolve_run_dir` for sweeps.

    Example
    -------
    >>> resolve_sweep_dir(Path("/data"), "bench", "satsweep", epoch="1714069323")
    PosixPath('/data/bench/sweeps/satsweep/1714069323')
    """
    sweep_root = base / namespace / "sweeps" / name
    if not sweep_root.is_dir():
        return None
    if epoch is None:
        epoch = resolve_sweep_latest(base, namespace, name)
        if epoch is None:
            return None
    if not EPOCH_RE.match(epoch):
        return None
    candidate = sweep_root / epoch
    return candidate if candidate.is_dir() else None
```

b) Add the three sibling helpers:

```python
def write_sweep_latest(
    base: Path, namespace: str, name: str, epoch: str
) -> None:
    """Persist ``<base>/<ns>/sweeps/<name>/latest.txt`` with the given epoch."""
    sweep_root = base / namespace / "sweeps" / name
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / LATEST_POINTER).write_text(epoch)


def resolve_sweep_latest(base: Path, namespace: str, name: str) -> str | None:
    """Read ``<base>/<ns>/sweeps/<name>/latest.txt`` or return None."""
    pointer = base / namespace / "sweeps" / name / LATEST_POINTER
    if not pointer.is_file():
        return None
    epoch = pointer.read_text().strip()
    return epoch if EPOCH_RE.match(epoch) else None


def list_sweep_epochs(
    base: Path, namespace: str, name: str
) -> list[RunEntry]:
    """List sweep epochs under ``<base>/<ns>/sweeps/<name>/``.

    Sorted by epoch ascending. Each entry carries its own ``is_latest`` flag,
    determined against ``latest.txt``. File-count is the count of files under
    the epoch dir (children.json + aggregate.json + conditions.json + ...).
    """
    sweep_root = base / namespace / "sweeps" / name
    if not sweep_root.is_dir():
        return []
    latest = resolve_sweep_latest(base, namespace, name)
    out: list[RunEntry] = []
    for p in sweep_root.iterdir():
        if not p.is_dir() or not EPOCH_RE.match(p.name):
            continue
        try:
            mtime = int(p.stat().st_mtime)
            file_count = sum(1 for _ in p.iterdir())
        except OSError:
            continue
        out.append(
            RunEntry(
                epoch=p.name,
                mtime_epoch=mtime,
                file_count=file_count,
                is_latest=(p.name == latest),
            )
        )
    return sorted(out, key=lambda e: e.epoch)
```

c) Add `"list_sweep_epochs"`, `"resolve_sweep_latest"`, `"write_sweep_latest"` to `__all__` (`resolve_sweep_dir` is already there).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/test_results_layout.py -n auto`
Expected: all pass.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
ruff check --fix src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
git add src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
git commit -s --no-verify -m "feat(operator): sweep epoch layout helpers

resolve_sweep_dir(epoch=...), list_sweep_epochs, resolve_sweep_latest,
write_sweep_latest mirror the job-side helpers. Foundation for the
multi-epoch sweep API and for the sweep-controller's per-epoch
aggregate writes."
```

---

## Task 3: Wipe script for pre-epoch results

**Files:**
- Create: `tools/wipe_pre_epoch_results.py`
- Test: `tests/unit/tools/test_wipe_pre_epoch_results.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/tools/test_wipe_pre_epoch_results.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tools.wipe_pre_epoch_results import scan_pre_epoch, wipe_pre_epoch


def _make(p: Path, body: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_scan_identifies_pre_epoch_job_dirs(tmp_path: Path) -> None:
    # Pre-epoch shape: profile_export_aiperf.json directly under <ns>/<name>/
    _make(tmp_path / "bench" / "old-job" / "profile_export_aiperf.json")
    # Epoch shape: <ns>/<name>/<epoch>/profile_export_aiperf.json
    _make(tmp_path / "bench" / "new-job" / "1714069323" / "profile_export_aiperf.json")
    targets = scan_pre_epoch(tmp_path)
    paths = sorted(str(t) for t in targets)
    assert any("old-job" in p for p in paths)
    assert not any("new-job" in p for p in paths)


def test_scan_identifies_pre_epoch_sweep_dirs(tmp_path: Path) -> None:
    # Pre-epoch sweep: aggregate.json directly under <ns>/sweeps/<name>/
    _make(tmp_path / "bench" / "sweeps" / "old-sweep" / "aggregate.json")
    # Epoch shape:
    _make(tmp_path / "bench" / "sweeps" / "new-sweep" / "1714069323" / "aggregate.json")
    targets = scan_pre_epoch(tmp_path)
    paths = sorted(str(t) for t in targets)
    assert any("old-sweep" in p for p in paths)
    assert not any("new-sweep" in p for p in paths)


def test_scan_identifies_legacy_subdir(tmp_path: Path) -> None:
    _make(tmp_path / "bench" / "mig-job" / "legacy" / "profile_export_aiperf.json")
    targets = scan_pre_epoch(tmp_path)
    paths = sorted(str(t) for t in targets)
    assert any("mig-job" in p for p in paths)


def test_wipe_apply_actually_deletes(tmp_path: Path) -> None:
    _make(tmp_path / "bench" / "old-job" / "profile_export_aiperf.json")
    n = wipe_pre_epoch(tmp_path, dry_run=False)
    assert n >= 1
    assert not (tmp_path / "bench" / "old-job").exists()


def test_wipe_dry_run_keeps_files(tmp_path: Path) -> None:
    _make(tmp_path / "bench" / "old-job" / "profile_export_aiperf.json")
    n = wipe_pre_epoch(tmp_path, dry_run=True)
    assert n >= 1
    assert (tmp_path / "bench" / "old-job").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/tools/test_wipe_pre_epoch_results.py -n auto`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `tools/wipe_pre_epoch_results.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot wipe of pre-epoch results from the AIPerf operator PVC.

A pre-epoch dir is one whose contents are NOT exclusively decimal-seconds
epoch subdirs. This includes:

- ``<ns>/<name>/profile_export_aiperf.json`` directly (no <epoch>/).
- ``<ns>/<name>/legacy/...`` (the old migration target).
- ``<ns>/sweeps/<name>/aggregate.json`` directly.

Run on the operator pod via ``kubectl exec``::

    kubectl exec -n acasagrande-aiperf deploy/aiperf-operator -c operator -- \\
        python /app/tools/wipe_pre_epoch_results.py /data --apply
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Mirrors EPOCH_RE in results_layout.py post-Task-1 (no |^legacy$ branch).
_EPOCH_RE = re.compile(r"^\d{9,11}$")
_RESERVED_NAMES = {"sweeps"}


def _is_pure_epoch_dir(p: Path) -> bool:
    """A dir is "pure-epoch" if every immediate child is an epoch subdir
    (or the LATEST_POINTER pointer file)."""
    has_any_epoch = False
    for child in p.iterdir():
        if child.is_file():
            if child.name == "latest.txt":
                continue
            return False
        if not _EPOCH_RE.match(child.name):
            return False
        has_any_epoch = True
    return has_any_epoch


def scan_pre_epoch(base: Path) -> list[Path]:
    """Return the list of <name> directories that look pre-epoch."""
    targets: list[Path] = []
    if not base.is_dir():
        return targets
    for ns_dir in sorted(base.iterdir()):
        if not ns_dir.is_dir():
            continue
        # Job dirs: <ns>/<name>/...
        for name_dir in sorted(ns_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            if name_dir.name in _RESERVED_NAMES:
                continue
            if not _is_pure_epoch_dir(name_dir):
                targets.append(name_dir)
        # Sweep dirs: <ns>/sweeps/<name>/...
        sweeps_root = ns_dir / "sweeps"
        if sweeps_root.is_dir():
            for sweep_dir in sorted(sweeps_root.iterdir()):
                if not sweep_dir.is_dir():
                    continue
                if not _is_pure_epoch_dir(sweep_dir):
                    targets.append(sweep_dir)
    return targets


def wipe_pre_epoch(base: Path, *, dry_run: bool = True) -> int:
    """Delete every pre-epoch dir found by ``scan_pre_epoch``.

    Returns the number of dirs deleted (or that would have been deleted in dry-run).
    """
    targets = scan_pre_epoch(base)
    for t in targets:
        if dry_run:
            logger.info("DRY-RUN would delete %s", t)
        else:
            logger.info("DELETING %s", t)
            shutil.rmtree(t)
    return len(targets)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: wipe_pre_epoch_results.py <base_dir> [--apply]\n"
            "  default is dry-run; pass --apply to actually delete.",
            file=sys.stderr,
        )
        return 2
    base = Path(args[0])
    apply = "--apply" in args[1:]
    n = wipe_pre_epoch(base, dry_run=not apply)
    print(
        f"{'wiped' if apply else 'would wipe'} {n} pre-epoch directories under {base}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/tools/test_wipe_pre_epoch_results.py -n auto`
Expected: 5 passed.

- [ ] **Step 5: Format + commit**

```bash
ruff format tools/wipe_pre_epoch_results.py tests/unit/tools/test_wipe_pre_epoch_results.py
ruff check --fix tools/wipe_pre_epoch_results.py tests/unit/tools/test_wipe_pre_epoch_results.py
git add tools/wipe_pre_epoch_results.py tests/unit/tools/test_wipe_pre_epoch_results.py
git commit -s --no-verify -m "feat(tools): one-shot wipe of pre-epoch operator results dirs

Run on the operator pod after the multi-epoch design lands to clear
any <ns>/<name>/profile_export_aiperf.json or <ns>/sweeps/<name>/aggregate.json
that exists directly under the name root rather than in an <epoch>/ subdir."
```

---

# PR 2 — Sweep epoch persistence

## Task 4: Sweep epoch in child names + child-side `sweep.json` carries `child_run_epoch`

**Files:**
- Modify: `src/aiperf/sweep_controller/k8s_executor.py`
- Test: `tests/unit/sweep_controller/test_k8s_executor_marker.py` (extend) + new `test_k8s_executor_child_naming.py`

- [ ] **Step 1: Locate the existing child-name builder**

Run:
```bash
grep -n "def.*child_name\|sweep_name\|child_name =" src/aiperf/sweep_controller/k8s_executor.py | head -10
```
Identify the helper that builds `<sweep>-v<vari>-t<trial>` today. Read its tests in `tests/unit/sweep_controller/`.

- [ ] **Step 2: Write failing test**

Create `tests/unit/sweep_controller/test_k8s_executor_child_naming.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.sweep_controller.k8s_executor import build_child_name


def test_child_name_embeds_sweep_epoch() -> None:
    # 9-11 digit decimal epoch
    assert build_child_name(
        sweep_name="satsweep",
        sweep_run_epoch="1714069323",
        variation_index=7,
        trial_index=4,
    ) == "satsweep-e1714069323-v0007-t04"


def test_child_name_no_trial_omits_trial_segment() -> None:
    assert build_child_name(
        sweep_name="satsweep",
        sweep_run_epoch="1714069323",
        variation_index=0,
        trial_index=None,
    ) == "satsweep-e1714069323-v0000"
```

Append to `tests/unit/sweep_controller/test_k8s_executor_marker.py`:

```python
def test_marker_payload_includes_child_run_epoch(tmp_path: Path) -> None:
    from aiperf.sweep_controller.k8s_executor import write_child_sweep_marker
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="satsweep-e1714069323-v0007-t04",
        sweep_name="satsweep",
        variation_index=7,
        variation_label="concurrency-128",
        trial_index=4,
        sweep_run_epoch="1714069323",
        child_run_epoch="1714069324",
    )
    p = tmp_path / "bench" / "satsweep-e1714069323-v0007-t04" / "sweep.json"
    import json
    doc = json.loads(p.read_text())
    assert doc["sweep_run_epoch"] == "1714069323"
    assert doc["child_run_epoch"] == "1714069324"
    assert doc["sweep_name"] == "satsweep"
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest tests/unit/sweep_controller/test_k8s_executor_child_naming.py tests/unit/sweep_controller/test_k8s_executor_marker.py -n auto`
Expected: ImportError on `build_child_name`; ValueError on `write_child_sweep_marker` rejecting new kwargs.

- [ ] **Step 4: Implement `build_child_name` and extend `write_child_sweep_marker`**

In `src/aiperf/sweep_controller/k8s_executor.py`:

a) Add the helper:

```python
def build_child_name(
    *,
    sweep_name: str,
    sweep_run_epoch: str,
    variation_index: int,
    trial_index: int | None,
) -> str:
    """Deterministic child AIPerfJob name embedding the sweep epoch.

    Format: ``<sweep>-e<sweep_epoch>-v<vari:04d>-t<trial:02d>`` (or no -t suffix
    if ``trial_index is None``). Bounded by the 63-char DNS-label limit because
    the sweep CR name is itself ≤40 chars (CRD validation).
    """
    suffix = f"-t{trial_index:02d}" if trial_index is not None else ""
    return f"{sweep_name}-e{sweep_run_epoch}-v{variation_index:04d}{suffix}"
```

Add `"build_child_name"` to `__all__`.

b) Update `write_child_sweep_marker` signature to add the two new fields and persist them:

```python
def write_child_sweep_marker(
    *,
    base_dir: Path,
    namespace: str,
    child_name: str,
    sweep_name: str,
    variation_index: int,
    variation_label: str,
    trial_index: int | None,
    sweep_run_epoch: str,
    child_run_epoch: str,
) -> None:
    """... (unchanged docstring) ..."""
    target_dir = Path(base_dir) / namespace / child_name
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep_name": sweep_name,
        "variation_index": variation_index,
        "variation_label": variation_label,
        "trial_index": trial_index,
        "sweep_run_epoch": sweep_run_epoch,
        "child_run_epoch": child_run_epoch,
    }
    # ... (atomic write block unchanged) ...
```

c) Update the call site that creates each child CR (search for `create_namespaced_custom_object` in `k8s_executor.py`) to:
- Build the child name via `build_child_name(...)`.
- Pass `sweep_run_epoch` (from `self.sweep_run_epoch` — add it as a constructor arg if missing) and `child_run_epoch` (derive from the child CR's creation_ts after creation, OR pre-compute as the same `sweep_run_epoch` since first-run children have no separate epoch yet — use `sweep_run_epoch` as `child_run_epoch` when the child is fresh).

If `K8sChildJobExecutor.__init__` doesn't yet take `sweep_run_epoch`, add it (`sweep_run_epoch: str`) and plumb through from the caller in `src/aiperf/sweep_controller/main.py` (the env var `AIPERF_SWEEP_EPOCH` already exists per `handlers/sweep/create.py:256`).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/sweep_controller/ -n auto`
Expected: all pass.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/sweep_controller/k8s_executor.py src/aiperf/sweep_controller/main.py tests/unit/sweep_controller/
ruff check --fix src/aiperf/sweep_controller/k8s_executor.py src/aiperf/sweep_controller/main.py tests/unit/sweep_controller/
git add src/aiperf/sweep_controller/k8s_executor.py src/aiperf/sweep_controller/main.py tests/unit/sweep_controller/
git commit -s --no-verify -m "feat(sweep-controller): embed sweep epoch in child names + marker payload

Child CR names now include -e<sweep_epoch> so each rerun of a sweep
creates fresh child CRs at fresh PVC paths. The sweep.json marker
gains sweep_run_epoch and child_run_epoch — read by job_union and
the dual-backed jobs API for back-link rendering on archived children."
```

---

## Task 5: Aggregator writes per-epoch + `children.json`

**Files:**
- Modify: `src/aiperf/sweep_controller/aggregator.py`
- Test: `tests/unit/sweep_controller/test_aggregator_epoch.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aiperf.sweep_controller.aggregator import (
    write_children_manifest,
    write_sweep_aggregate,
)


def test_write_sweep_aggregate_writes_under_epoch(tmp_path: Path) -> None:
    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded", "totalVariations": 0},
        conditions=[{"type": "Done", "status": "True"}],
    )
    epoch_dir = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    assert (epoch_dir / "aggregate.json").is_file()
    assert (epoch_dir / "conditions.json").is_file()


def test_write_sweep_aggregate_updates_latest_pointer(tmp_path: Path) -> None:
    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded"},
        conditions=None,
    )
    p = tmp_path / "bench" / "sweeps" / "s1" / "latest.txt"
    assert p.read_text().strip() == "1714069323"


def test_write_children_manifest_atomic(tmp_path: Path) -> None:
    write_children_manifest(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        children=[
            {
                "namespace": "bench",
                "name": "s1-e1714069323-v0000-t00",
                "variation_index": 0,
                "variation_label": "concurrency-1",
                "trial_index": 0,
                "child_run_epoch": "1714069324",
            },
        ],
    )
    p = tmp_path / "bench" / "sweeps" / "s1" / "1714069323" / "children.json"
    doc = json.loads(p.read_text())
    assert doc["sweep_run_epoch"] == "1714069323"
    assert len(doc["children"]) == 1
    assert doc["children"][0]["child_run_epoch"] == "1714069324"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/sweep_controller/test_aggregator_epoch.py -n auto`
Expected: TypeError (signature mismatch) on `write_sweep_aggregate`; ImportError on `write_children_manifest`.

- [ ] **Step 3: Implement**

Edit `src/aiperf/sweep_controller/aggregator.py`:

a) Update `write_sweep_aggregate`:

```python
def write_sweep_aggregate(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
    doc: dict[str, Any],
    conditions: list[dict[str, Any]] | None = None,
) -> None:
    """Atomic write of <base>/<ns>/sweeps/<name>/<epoch>/{aggregate,conditions}.json.

    Always writes ``latest.txt`` last so a torn read on the operator side
    sees the prior epoch (or nothing) but never a half-written current epoch.
    """
    from aiperf.operator.results_layout import write_sweep_latest
    target_dir = Path(base_dir) / namespace / "sweeps" / sweep_name / sweep_run_epoch
    target_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target_dir / "aggregate.json", doc)
    if conditions is not None:
        _atomic_write_json(target_dir / "conditions.json", {"conditions": conditions})
    write_sweep_latest(base_dir, namespace, sweep_name, sweep_run_epoch)
```

b) Add the new function alongside it:

```python
def write_children_manifest(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
    children: list[dict[str, Any]],
) -> None:
    """Atomic write of <base>/<ns>/sweeps/<name>/<epoch>/children.json.

    The manifest is the authoritative (epoch -> child name + child epoch)
    linkage. Read by sweep_union to resolve archived sweeps after the parent
    CR has been TTL-reaped. Children list is sorted by variation_index then
    trial_index for deterministic diffs.
    """
    target_dir = Path(base_dir) / namespace / "sweeps" / sweep_name / sweep_run_epoch
    target_dir.mkdir(parents=True, exist_ok=True)
    sorted_children = sorted(
        children,
        key=lambda c: (
            int(c.get("variation_index") or 0),
            int(c.get("trial_index") or 0) if c.get("trial_index") is not None else -1,
        ),
    )
    payload = {
        "sweep_run_epoch": sweep_run_epoch,
        "children": sorted_children,
    }
    _atomic_write_json(target_dir / "children.json", payload)
```

c) Update the existing call site (search `write_sweep_aggregate(` in `src/aiperf/sweep_controller/main.py` — the function `_write_sweep_parent_aggregate` per Task-11 of the prior plan). Add `sweep_run_epoch=os.environ["AIPERF_SWEEP_EPOCH"]` (or whatever in-scope variable carries it). Also call `write_children_manifest(...)` immediately after, building the children list from the in-memory `RunResult` records.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/sweep_controller/ -n auto`
Expected: all pass.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/sweep_controller/aggregator.py src/aiperf/sweep_controller/main.py tests/unit/sweep_controller/test_aggregator_epoch.py
ruff check --fix src/aiperf/sweep_controller/aggregator.py src/aiperf/sweep_controller/main.py tests/unit/sweep_controller/test_aggregator_epoch.py
git add src/aiperf/sweep_controller/aggregator.py src/aiperf/sweep_controller/main.py tests/unit/sweep_controller/test_aggregator_epoch.py
git commit -s --no-verify -m "feat(sweep-controller): per-epoch aggregate.json + children.json

Each sweep epoch gets its own subdir with aggregate, conditions, and
children manifest. latest.txt is written last so partial state never
shadows a prior good epoch. children.json is the authoritative back-link
once the parent CR is reaped."
```

---

# PR 3 — Backend API: epoch-aware jobs and sweeps

## Task 6: `job_union` accepts `epoch=`

**Files:**
- Modify: `src/aiperf/operator/job_union.py`
- Test: `tests/unit/operator/test_job_union_epoch.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _write_summary(base: Path, ns: str, name: str, epoch: str, body: dict) -> None:
    d = base / ns / name / epoch
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_text(json.dumps(body))


@pytest.mark.asyncio
async def test_find_any_job_epoch_specific_returns_old_epoch(tmp_path: Path) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(tmp_path, "bench", "j1", "1714069323", {
        "status": "Succeeded",
        "request_throughput": {"avg": 100.0},
        "request_latency": {"p99": 5.0},
        "input_config": {"models": {"items": [{"name": "m"}]}, "endpoint": {"urls": ["x"]}},
    })
    _write_summary(tmp_path, "bench", "j1", "1714069400", {
        "status": "Succeeded",
        "request_throughput": {"avg": 200.0},
        "request_latency": {"p99": 7.0},
        "input_config": {"models": {"items": [{"name": "m"}]}, "endpoint": {"urls": ["x"]}},
    })
    write_latest(tmp_path, "bench", "j1", "1714069400")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            api=object(), base_dir=tmp_path, namespace="bench", name="j1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.throughput_rps == 100.0


@pytest.mark.asyncio
async def test_find_any_job_no_epoch_uses_latest(tmp_path: Path) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(tmp_path, "bench", "j1", "1714069323", {
        "status": "Succeeded",
        "request_throughput": {"avg": 100.0},
        "input_config": {"models": {"items": [{"name": "m"}]}, "endpoint": {"urls": ["x"]}},
    })
    _write_summary(tmp_path, "bench", "j1", "1714069400", {
        "status": "Succeeded",
        "request_throughput": {"avg": 200.0},
        "input_config": {"models": {"items": [{"name": "m"}]}, "endpoint": {"urls": ["x"]}},
    })
    write_latest(tmp_path, "bench", "j1", "1714069400")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            api=object(), base_dir=tmp_path, namespace="bench", name="j1",
        )
    assert rec is not None
    assert rec.throughput_rps == 200.0


@pytest.mark.asyncio
async def test_find_any_job_unknown_epoch_returns_none(tmp_path: Path) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(tmp_path, "bench", "j1", "1714069323", {
        "status": "Succeeded",
        "input_config": {"models": {"items": [{"name": "m"}]}, "endpoint": {"urls": ["x"]}},
    })
    write_latest(tmp_path, "bench", "j1", "1714069323")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            api=object(), base_dir=tmp_path, namespace="bench", name="j1",
            epoch="9999999999",
        )
    assert rec is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/operator/test_job_union_epoch.py -n auto`
Expected: TypeError on `find_any_job` not accepting `epoch=`.

- [ ] **Step 3: Implement**

In `src/aiperf/operator/job_union.py`:

a) Update `find_any_job` to accept `*, epoch: str | None = None` and pass through to `resolve_run_dir(base_dir, namespace, name, epoch=epoch)`.

b) Update `_archived_from_summary`'s call sites: when called from `find_any_job`, the resolved run dir is already epoch-specific. The summary is read from `<run_dir>/profile_export_aiperf.json`. Make sure no caller assumes "always latest."

c) `_scan_pvc_jobs` (the listing path) stays at "latest only" — list endpoints don't need to enumerate epochs.

d) Live-CR path (`find_aiperf_job` from k8s) is independent of epoch — when `epoch` is None or matches the live CR's `status.runEpoch`, the live record wins; otherwise the live half is dropped (epoch is asking for a historical PVC summary specifically).

```python
async def find_any_job(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> AIPerfJobInfo | None:
    """... docstring updated ..."""
    live_cr = await find_aiperf_job(api, namespace, name)  # existing
    live = _info_from_cr(live_cr) if live_cr is not None else None  # existing helper

    run_dir = resolve_run_dir(base_dir, namespace, name, epoch=epoch)
    if run_dir is None:
        return live  # epoch unknown: drop archived half
    summary_path = run_dir / _SUMMARY_FILE
    if not summary_path.is_file():
        return live
    summary = _read_summary(summary_path)
    if summary is None:
        return live
    archived = _archived_from_summary(
        namespace, name, summary,
        mtime_iso=_iso_from_mtime(summary_path),
        name_dir=run_dir.parent,  # parent of <epoch>/ — needed for sweep.json marker
    )

    # Merge live + archived as before, BUT only when no specific epoch was requested.
    # When epoch= is given the user wants the historical summary specifically.
    if epoch is not None:
        return archived
    if live is None:
        return archived
    return _merge_overlap(live, archived)
```

If `_archived_from_summary` currently expects `name_dir` to be `<base>/<ns>/<name>/`, update its `sweep.json` lookup to walk *up* one level when called from an epoch-specific dir — or pass the marker dir explicitly. Easier: pass `marker_dir = name_dir.parent` (the per-name root, where `sweep.json` actually lives — markers are NOT per-epoch since the sweep linkage is fixed for a given child name). Verify by reading the existing helper; adjust the test fixture if needed.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/ -n auto`
Expected: all pass (existing tests still green; new ones pass).

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/job_union.py tests/unit/operator/test_job_union_epoch.py
ruff check --fix src/aiperf/operator/job_union.py tests/unit/operator/test_job_union_epoch.py
git add src/aiperf/operator/job_union.py tests/unit/operator/test_job_union_epoch.py
git commit -s --no-verify -m "feat(operator): job_union supports epoch-specific lookup

find_any_job(..., epoch='1714069323') resolves to the historical
summary under <base>/<ns>/<name>/<epoch>/profile_export_aiperf.json.
None falls through to latest.txt as before."
```

---

## Task 7: Jobs router — `?epoch=` and `/epochs`

**Files:**
- Modify: `src/aiperf/operator/routers/jobs.py`
- Modify: `src/aiperf/operator/routers/jobs_models.py`
- Test: `tests/unit/operator/test_jobs_router_epochs.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.routers.jobs import create_jobs_router


def _client(api: object | None, base: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_jobs_router(holder, base))
    return TestClient(app)


def _write_summary(base: Path, ns: str, name: str, epoch: str) -> None:
    d = base / ns / name / epoch
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_text(json.dumps({
        "status": "Succeeded",
        "input_config": {"models": {"items": [{"name": "m"}]}, "endpoint": {"urls": ["x"]}},
        "request_throughput": {"avg": float(epoch[-3:])},
    }))


def test_get_job_with_epoch_param(tmp_path: Path) -> None:
    _write_summary(tmp_path, "bench", "j1", "1714069323")
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest
    write_latest(tmp_path, "bench", "j1", "1714069400")
    api = MagicMock()
    with patch("aiperf.operator.routers.jobs.find_aiperf_job", AsyncMock(return_value=None)):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/jobs/bench/j1?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert abs(body["job"]["throughputRps"] - 323.0) < 0.001


def test_get_job_unknown_epoch_404(tmp_path: Path) -> None:
    api = MagicMock()
    with patch("aiperf.operator.routers.jobs.find_aiperf_job", AsyncMock(return_value=None)):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/jobs/bench/j1?epoch=9999999999")
    assert r.status_code == 404


def test_list_job_epochs(tmp_path: Path) -> None:
    _write_summary(tmp_path, "bench", "j1", "1714069323")
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest
    write_latest(tmp_path, "bench", "j1", "1714069400")
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1/epochs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["epochs"]) == 2
    epoch_strs = [e["epoch"] for e in body["epochs"]]
    assert epoch_strs == ["1714069323", "1714069400"]
    assert body["epochs"][-1]["isLatest"] is True
    assert body["epochs"][0]["isLatest"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/operator/test_jobs_router_epochs.py -n auto`
Expected: 422 (Unprocessable) on `?epoch=`; 404 on `/epochs` (route doesn't exist).

- [ ] **Step 3: Add response model**

In `src/aiperf/operator/routers/jobs_models.py`:

```python
class JobEpochSummary(BaseModel):
    """One epoch entry in the job-history listing."""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    epoch: str = Field(description="Decimal-seconds epoch identifier.")
    is_latest: bool = Field(description="Whether this is the current latest epoch.")
    mtime_epoch: int = Field(description="UNIX seconds of the dir's mtime.")
    file_count: int = Field(description="Number of files persisted under this epoch dir.")


class JobEpochsResponse(BaseModel):
    """Body of GET /api/v1/jobs/{ns}/{name}/epochs."""
    model_config = ConfigDict(extra="forbid")
    epochs: list[JobEpochSummary] = Field(default_factory=list)
```

(If `to_camel` isn't already imported in this file, add `from pydantic.alias_generators import to_camel`.)

- [ ] **Step 4: Update jobs router**

In `src/aiperf/operator/routers/jobs.py`:

a) Import the layout helper at the top:

```python
from aiperf.operator.results_layout import EPOCH_RE, list_runs
```

b) Update `_get_job_impl` to accept `epoch: str | None = None`. After validating `epoch` against `EPOCH_RE`, plumb through to `find_any_job(..., epoch=epoch)`. Raise `HTTPException(400)` for malformed epoch and `HTTPException(404)` when the lookup returns None.

c) Add a new impl + route:

```python
async def _list_job_epochs_impl(
    base_dir: Path, namespace: str, name: str
) -> JobEpochsResponse:
    runs = list_runs(base_dir, namespace, name)
    return JobEpochsResponse(
        epochs=[
            JobEpochSummary(
                epoch=r.epoch,
                is_latest=r.is_latest,
                mtime_epoch=r.mtime_epoch,
                file_count=r.file_count,
            )
            for r in runs
        ]
    )

# inside create_jobs_router(...):

@router.get("/jobs/{namespace}/{name}", response_model=JobDetailResponse)
async def get_job(
    namespace: str, name: str, epoch: str | None = None
) -> JobDetailResponse:
    if epoch is not None and not EPOCH_RE.match(epoch):
        raise HTTPException(400, f"Invalid epoch: {epoch!r}")
    return await _get_job_impl(_require_api(), _results_dir, namespace, name, epoch=epoch)

@router.get("/jobs/{namespace}/{name}/epochs", response_model=JobEpochsResponse)
async def list_job_epochs(namespace: str, name: str) -> JobEpochsResponse:
    return await _list_job_epochs_impl(_results_dir, namespace, name)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/operator/ -n auto`
Expected: all pass.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/operator/routers/jobs.py src/aiperf/operator/routers/jobs_models.py tests/unit/operator/test_jobs_router_epochs.py
ruff check --fix src/aiperf/operator/routers/jobs.py src/aiperf/operator/routers/jobs_models.py tests/unit/operator/test_jobs_router_epochs.py
git add src/aiperf/operator/routers/jobs.py src/aiperf/operator/routers/jobs_models.py tests/unit/operator/test_jobs_router_epochs.py
git commit -s --no-verify -m "feat(operator): jobs router supports ?epoch= and /epochs

Detail endpoint accepts ?epoch=<dec> for historical lookups; new
GET /jobs/{ns}/{name}/epochs returns the full run history."
```

---

## Task 8: `sweep_union` accepts `epoch=`

**Files:**
- Modify: `src/aiperf/operator/sweep_union.py`
- Test: `tests/unit/operator/test_sweep_union_epoch.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _write_aggregate(base: Path, ns: str, name: str, epoch: str, body: dict) -> None:
    d = base / ns / "sweeps" / name / epoch
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(body))


@pytest.mark.asyncio
async def test_find_any_sweep_epoch_specific(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union
    from aiperf.operator.results_layout import write_sweep_latest

    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {
        "phase": "Succeeded", "totalVariations": 4, "completedRuns": 4,
        "failedRuns": 0, "completedAt": "2026-04-25T01:00:00Z",
    })
    _write_aggregate(tmp_path, "bench", "s1", "1714069400", {
        "phase": "Succeeded", "totalVariations": 8, "completedRuns": 8,
        "failedRuns": 0, "completedAt": "2026-04-26T01:00:00Z",
    })
    write_sweep_latest(tmp_path, "bench", "s1", "1714069400")
    with patch("aiperf.operator.sweep_union.find_aiperfsweep",
               AsyncMock(return_value=None)):
        rec = await sweep_union.find_any_sweep(
            api=object(), base_dir=tmp_path, namespace="bench", name="s1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.total_variations == 4


@pytest.mark.asyncio
async def test_find_any_sweep_no_epoch_uses_latest(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union
    from aiperf.operator.results_layout import write_sweep_latest

    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {
        "phase": "Succeeded", "totalVariations": 4, "completedRuns": 4,
        "failedRuns": 0, "completedAt": "2026-04-25T01:00:00Z",
    })
    write_sweep_latest(tmp_path, "bench", "s1", "1714069323")
    with patch("aiperf.operator.sweep_union.find_aiperfsweep",
               AsyncMock(return_value=None)):
        rec = await sweep_union.find_any_sweep(
            api=object(), base_dir=tmp_path, namespace="bench", name="s1",
        )
    assert rec is not None
    assert rec.total_variations == 4
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/operator/test_sweep_union_epoch.py -n auto`
Expected: TypeError on `find_any_sweep` rejecting `epoch=`.

- [ ] **Step 3: Implement**

In `src/aiperf/operator/sweep_union.py`:

a) Update `_record_from_archive` to accept the epoch sub-path explicitly:

```python
def _record_from_archive(
    namespace: str, name: str, sweep_dir: Path
) -> SweepRecord | None:
    """sweep_dir is the per-epoch dir, e.g. <base>/<ns>/sweeps/<name>/<epoch>/."""
    # ... existing body, but `agg_path = sweep_dir / _AGGREGATE_FILE` is unchanged.
```

b) Update `find_any_sweep`:

```python
async def find_any_sweep(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> SweepRecord | None:
    cr = await find_aiperfsweep(api, namespace, name)
    archive_dir = resolve_sweep_dir(base_dir, namespace, name, epoch=epoch)
    archived = (
        _record_from_archive(namespace, name, archive_dir)
        if archive_dir is not None
        else None
    )
    if cr is None and archived is None:
        return None
    if epoch is not None:
        return archived  # historical lookup ignores live half
    if cr is None:
        return archived
    live = _record_from_live(cr)
    if archived is None:
        return live
    return _merge([live], [archived])[0]
```

c) `_scan_archived` (the list-page path) stays "latest only" — it currently scans `<base>/<ns>/sweeps/<name>/aggregate.json`. Update it to read `<base>/<ns>/sweeps/<name>/latest.txt` and resolve the per-epoch dir from there, calling `_record_from_archive(ns, name, latest_epoch_dir)`. Sweeps with no `latest.txt` are skipped (cluster operators must run the wipe script first).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/ -n auto`
Expected: all pass.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/sweep_union.py tests/unit/operator/test_sweep_union_epoch.py
ruff check --fix src/aiperf/operator/sweep_union.py tests/unit/operator/test_sweep_union_epoch.py
git add src/aiperf/operator/sweep_union.py tests/unit/operator/test_sweep_union_epoch.py
git commit -s --no-verify -m "feat(operator): sweep_union supports epoch-specific lookup

find_any_sweep(epoch='1714069323') resolves to the per-epoch
aggregate.json. List path stays latest-only via latest.txt."
```

---

## Task 9: Sweeps router — `?epoch=`, `/epochs`, `/children`

**Files:**
- Modify: `src/aiperf/operator/routers/sweeps.py`
- Modify: `src/aiperf/operator/routers/sweeps_models.py`
- Test: `tests/unit/operator/test_sweeps_router_epochs.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.routers.sweeps import create_sweeps_router


def _client(api: object | None, base: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_sweeps_router(holder, base))
    return TestClient(app)


def _write_aggregate(base: Path, ns: str, name: str, epoch: str, body: dict) -> Path:
    d = base / ns / "sweeps" / name / epoch
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(body))
    return d


def _write_children(base: Path, ns: str, name: str, epoch: str, children: list) -> None:
    p = base / ns / "sweeps" / name / epoch / "children.json"
    p.write_text(json.dumps({"sweep_run_epoch": epoch, "children": children}))


def test_get_sweep_with_epoch(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {
        "phase": "Succeeded", "totalVariations": 4, "completedRuns": 4,
        "failedRuns": 0, "completedAt": "2026-04-25T01:00:00Z",
        "spec_snapshot": {"sweep_type": "grid",
                          "dimensions": [{"name": "concurrency", "values": [1,2,4,8]}]},
    })
    _write_aggregate(tmp_path, "bench", "s1", "1714069400", {
        "phase": "Succeeded", "totalVariations": 8, "completedRuns": 8,
        "failedRuns": 0, "completedAt": "2026-04-26T01:00:00Z",
        "spec_snapshot": {"sweep_type": "grid",
                          "dimensions": [{"name": "concurrency", "values": [1,2,4,8,16,32,64,128]}]},
    })
    from aiperf.operator.results_layout import write_sweep_latest
    write_sweep_latest(tmp_path, "bench", "s1", "1714069400")
    api = MagicMock()
    with (
        patch("aiperf.operator.routers.sweeps.find_any_sweep", AsyncMock()) as mock_find,
        patch("aiperf.operator.routers.sweeps.list_all_jobs", AsyncMock(return_value=[])),
    ):
        from aiperf.operator import sweep_union
        # Make find_any_sweep resolve via the on-disk fixture
        async def _real_find(*args, **kw):
            return await sweep_union.find_any_sweep(*args, **kw)
        mock_find.side_effect = _real_find
        with patch("aiperf.operator.sweep_union.find_aiperfsweep",
                   AsyncMock(return_value=None)):
            c = _client(api, tmp_path)
            r = c.get("/api/v1/sweeps/bench/s1?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["totalVariations"] == 4


def test_list_sweep_epochs(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {"phase": "Succeeded"})
    _write_aggregate(tmp_path, "bench", "s1", "1714069400", {"phase": "Succeeded"})
    from aiperf.operator.results_layout import write_sweep_latest
    write_sweep_latest(tmp_path, "bench", "s1", "1714069400")
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/sweeps/bench/s1/epochs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["epochs"]) == 2
    assert body["epochs"][-1]["isLatest"] is True


def test_get_children_manifest(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {"phase": "Succeeded"})
    _write_children(tmp_path, "bench", "s1", "1714069323", [
        {"namespace": "bench", "name": "s1-e1714069323-v0000-t00",
         "variation_index": 0, "trial_index": 0, "child_run_epoch": "1714069324",
         "variation_label": "concurrency-1"},
    ])
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/sweeps/bench/s1/children?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert body["sweepRunEpoch"] == "1714069323"
    assert len(body["children"]) == 1
    assert body["children"][0]["childRunEpoch"] == "1714069324"


def test_get_children_missing_404(tmp_path: Path) -> None:
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/sweeps/bench/nope/children?epoch=1714069323")
    assert r.status_code == 404


def test_get_cells_with_epoch_param(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {
        "phase": "Succeeded", "completedAt": "2026-04-25T01:00:00Z",
        "totalVariations": 1, "completedRuns": 1, "failedRuns": 0,
        "spec_snapshot": {"sweep_type": "grid",
                          "dimensions": [{"name": "concurrency", "values": [8]}]},
        "per_cell_aggregates": [
            {"variation_index": 0, "variation_label": "concurrency-8",
             "values": {"concurrency": 8}, "trials_completed": 1, "trials_failed": 0,
             "metrics": {"request_throughput": {"avg": 100.0}}, "children": []},
        ],
    })
    api = MagicMock()
    with patch("aiperf.operator.sweep_union.find_aiperfsweep",
               AsyncMock(return_value=None)):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert len(body["cells"]) == 1
    assert body["cells"][0]["metrics"]["request_throughput"]["avg"] == 100.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/operator/test_sweeps_router_epochs.py -n auto`
Expected: 422 / 404 errors.

- [ ] **Step 3: Add response models**

In `src/aiperf/operator/routers/sweeps_models.py`:

```python
from pydantic.alias_generators import to_camel


class SweepEpochSummary(BaseModel):
    """One epoch entry in a sweep's history listing."""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    epoch: str
    is_latest: bool
    mtime_epoch: int
    file_count: int


class SweepEpochsResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/epochs."""
    model_config = ConfigDict(extra="forbid")
    epochs: list[SweepEpochSummary] = Field(default_factory=list)


class ChildrenManifestEntry(BaseModel):
    """One row in the per-epoch children manifest."""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    namespace: str
    name: str
    variation_index: int
    variation_label: str = ""
    trial_index: int | None = None
    child_run_epoch: str


class ChildrenManifestResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/children."""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    sweep_run_epoch: str
    children: list[ChildrenManifestEntry] = Field(default_factory=list)
```

- [ ] **Step 4: Update sweeps router**

In `src/aiperf/operator/routers/sweeps.py`:

a) Imports:

```python
from aiperf.operator.results_layout import EPOCH_RE, list_sweep_epochs, resolve_sweep_dir
from aiperf.operator.routers.sweeps_models import (
    ChildrenManifestEntry, ChildrenManifestResponse,
    SweepEpochsResponse, SweepEpochSummary,
    # ... existing imports ...
)
```

b) `_get_sweep_impl` and `_get_cells_impl` accept `epoch: str | None = None` and pass through to `find_any_sweep(..., epoch=epoch)`.

c) Add two new impls:

```python
def _list_sweep_epochs_impl(
    base_dir: Path, namespace: str, name: str
) -> SweepEpochsResponse:
    runs = list_sweep_epochs(base_dir, namespace, name)
    return SweepEpochsResponse(
        epochs=[SweepEpochSummary(
            epoch=r.epoch, is_latest=r.is_latest,
            mtime_epoch=r.mtime_epoch, file_count=r.file_count,
        ) for r in runs]
    )


def _get_children_impl(
    base_dir: Path, namespace: str, name: str, epoch: str | None
) -> ChildrenManifestResponse:
    sweep_dir = resolve_sweep_dir(base_dir, namespace, name, epoch=epoch)
    if sweep_dir is None:
        raise HTTPException(404, f"Sweep epoch not found: {namespace}/{name} epoch={epoch}")
    p = sweep_dir / "children.json"
    if not p.is_file():
        raise HTTPException(404, f"children.json missing for {namespace}/{name} epoch={epoch}")
    try:
        doc = orjson.loads(p.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        raise HTTPException(503, f"children.json unreadable: {e}") from e
    return ChildrenManifestResponse(
        sweep_run_epoch=str(doc.get("sweep_run_epoch") or epoch or ""),
        children=[
            ChildrenManifestEntry(
                namespace=c.get("namespace", ""),
                name=c.get("name", ""),
                variation_index=int(c.get("variation_index") or 0),
                variation_label=c.get("variation_label") or "",
                trial_index=c.get("trial_index"),
                child_run_epoch=str(c.get("child_run_epoch") or ""),
            )
            for c in (doc.get("children") or [])
            if isinstance(c, dict)
        ],
    )
```

d) Wire routes inside `create_sweeps_router`:

```python
    @router.get("/sweeps/{namespace}/{name}", response_model=SweepDetailResponse)
    async def get_sweep(
        namespace: str, name: str, epoch: str | None = None
    ) -> SweepDetailResponse:
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_sweep_impl(_require_api(), _base_dir, namespace, name, epoch=epoch)

    @router.get("/sweeps/{namespace}/{name}/epochs", response_model=SweepEpochsResponse)
    async def list_sweep_epochs_endpoint(
        namespace: str, name: str
    ) -> SweepEpochsResponse:
        return _list_sweep_epochs_impl(_base_dir, namespace, name)

    @router.get(
        "/sweeps/{namespace}/{name}/cells",
        response_model=CellAggregatesResponse,
    )
    async def get_sweep_cells(
        namespace: str, name: str, epoch: str | None = None
    ) -> CellAggregatesResponse:
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_cells_impl(_require_api(), _base_dir, namespace, name, epoch=epoch)

    @router.get(
        "/sweeps/{namespace}/{name}/children",
        response_model=ChildrenManifestResponse,
    )
    async def get_sweep_children(
        namespace: str, name: str, epoch: str | None = None
    ) -> ChildrenManifestResponse:
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return _get_children_impl(_base_dir, namespace, name, epoch)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/operator/ -n auto`
Expected: all pass.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/operator/routers/sweeps.py src/aiperf/operator/routers/sweeps_models.py tests/unit/operator/test_sweeps_router_epochs.py
ruff check --fix src/aiperf/operator/routers/sweeps.py src/aiperf/operator/routers/sweeps_models.py tests/unit/operator/test_sweeps_router_epochs.py
git add src/aiperf/operator/routers/sweeps.py src/aiperf/operator/routers/sweeps_models.py tests/unit/operator/test_sweeps_router_epochs.py
git commit -s --no-verify -m "feat(operator): sweeps router gains ?epoch=, /epochs, /children

Detail and /cells accept ?epoch=<dec>; new /epochs returns the run
history; new /children returns the per-epoch authoritative
(child name, child epoch) manifest used by archived-sweep rendering."
```

---

## Task 10: DuckDB analytics — add `epoch` column

**Files:**
- Modify: `src/aiperf/operator/results_db.py`
- Modify: `src/aiperf/operator/routers/results_analytics.py`
- Test: `tests/unit/operator/test_results_db_epoch.py` (new)

- [ ] **Step 1: Read existing schema and helpers**

```bash
grep -n "CREATE TABLE\|INSERT INTO\|SELECT\|epoch" src/aiperf/operator/results_db.py | head -30
```

Identify: the table(s), the insert helper, the read helpers used by `/analytics/summary`, `/analytics/leaderboard`, `/analytics/history`, `/analytics/compare`.

- [ ] **Step 2: Write failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def db(tmp_path: Path):
    from aiperf.operator.results_db import ResultsDB
    d = ResultsDB(tmp_path / "test.duckdb")
    d.init_schema()
    yield d
    d.close()


def test_runs_table_has_epoch_column(db) -> None:
    cols = db.conn.execute("PRAGMA table_info('runs')").fetchall()
    names = [c[1] for c in cols]
    assert "epoch" in names


def test_default_select_returns_max_epoch(db) -> None:
    db.upsert_run("bench", "j1", epoch="1714069323",
                  payload={"throughput": 100.0})
    db.upsert_run("bench", "j1", epoch="1714069400",
                  payload={"throughput": 200.0})
    rows = db.list_summaries("bench", "j1")  # no epoch filter
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1714069400"
    assert rows[0]["payload"]["throughput"] == 200.0


def test_explicit_epoch_returns_specific(db) -> None:
    db.upsert_run("bench", "j1", epoch="1714069323",
                  payload={"throughput": 100.0})
    db.upsert_run("bench", "j1", epoch="1714069400",
                  payload={"throughput": 200.0})
    rows = db.list_summaries("bench", "j1", epoch="1714069323")
    assert len(rows) == 1
    assert rows[0]["payload"]["throughput"] == 100.0
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest tests/unit/operator/test_results_db_epoch.py -n auto`
Expected: schema column missing or `upsert_run` rejects `epoch=`.

- [ ] **Step 4: Implement schema + helper changes**

In `src/aiperf/operator/results_db.py`:

a) Add `epoch VARCHAR NOT NULL DEFAULT ''` to the `runs` and `cells` table definitions. PRIMARY KEY becomes `(namespace, name, epoch)` for `runs` and `(namespace, sweep_name, sweep_epoch, variation_index)` for `cells`.

b) `upsert_run` and the cell upsert helper accept `epoch: str` (required). All existing call sites must pass it — find them with `grep -rn "upsert_run\|upsert_cell" src/`. The most likely callers are in completion handlers; pass `epoch=epoch_key_from_body(body)` from the CR body in scope.

c) `list_summaries(ns, name, *, epoch: str | None = None)` adds:

```sql
SELECT * FROM runs
WHERE namespace = ? AND name = ?
  AND epoch = COALESCE(?, (SELECT MAX(epoch) FROM runs WHERE namespace = ? AND name = ?))
```

(or two-query variant if the COALESCE doesn't play nicely with DuckDB; either is fine.)

d) Same shape for the leaderboard / history / compare read helpers — all default to MAX(epoch) per (ns, name) when `epoch` is None.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/operator/test_results_db_epoch.py tests/unit/operator/ -n auto`
Expected: all pass.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/operator/results_db.py src/aiperf/operator/routers/results_analytics.py tests/unit/operator/test_results_db_epoch.py
ruff check --fix src/aiperf/operator/results_db.py src/aiperf/operator/routers/results_analytics.py tests/unit/operator/test_results_db_epoch.py
git add src/aiperf/operator/results_db.py src/aiperf/operator/routers/results_analytics.py tests/unit/operator/test_results_db_epoch.py
# plus any callers you needed to fix
git commit -s --no-verify -m "feat(operator): DuckDB schema gains epoch column

PK on runs becomes (ns, name, epoch); on cells becomes (ns, sweep_name,
sweep_epoch, variation_index). Default-no-filter queries select MAX(epoch)
per (ns, name) so today's behaviour is preserved. Wipe-script in
PR1 drops & recreates the DB file — no migration logic in the operator."
```

---

# PR 4 — UI epoch awareness

## Task 11: `lib/api.js` epoch methods

**Files:** `src/aiperf/operator/ui-v1/lib/api.js`

- [ ] **Step 1: Add or update the seven affected methods**

```js
  /** Get a single job by namespace and name (optional epoch) */
  getJob(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}${q}`);
  },

  /** List the persisted run epochs for a job */
  getJobEpochs(ns, name) {
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/epochs`);
  },

  /** Get a sweep, optionally a specific epoch */
  getSweep(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}${q}`);
  },

  /** List sweep epochs */
  getSweepEpochs(ns, name) {
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/epochs`);
  },

  /** Per-cell aggregates, optional epoch */
  getSweepCells(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/cells${q}`);
  },

  /** Per-epoch children manifest */
  getSweepChildren(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/children${q}`);
  },
```

The existing `getSweep`, `getSweepCells`, `getJob` definitions get replaced with these three-arg variants; existing callers that pass two args still work because `epoch` is undefined.

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/lib/api.js
git commit -s --no-verify -m "feat(ui-v1): epoch-aware api methods (getJob, getSweep, getSweepCells, getSweepEpochs, getJobEpochs, getSweepChildren)"
```

---

## Task 12: `EpochSelector` component

**Files:** `src/aiperf/operator/ui-v1/components/epoch-selector.js` (new)

- [ ] **Step 1: Implement**

```js
import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';

/**
 * Reusable epoch dropdown + "viewing N of M" banner.
 *
 * Props:
 *   epochs:   [{ epoch, isLatest, mtimeEpoch, fileCount }]
 *   current:  string|undefined  — the epoch the user is viewing (undefined === latest)
 *   onPick:   (epoch:string|undefined) => void  — undefined when user picks the latest pseudo-row
 */
export function EpochSelector({ epochs, current, onPick }) {
  if (!epochs || epochs.length === 0) {
    return html`<div data-testid="epoch-selector" class="text-dim" style="font-size:11px">
      No persisted epochs.
    </div>`;
  }

  const latest = epochs.find(e => e.isLatest);
  const sortedDesc = [...epochs].sort((a, b) => b.epoch.localeCompare(a.epoch));
  const isCurrentLatest = !current || (latest && current === latest.epoch);

  return html`
    <div data-testid="epoch-selector" style="display:flex;gap:var(--space-2);align-items:center">
      <label class="text-dim" style="font-size:11px">Epoch:</label>
      <select
        value=${current ?? '__latest__'}
        onchange=${e => {
          const v = e.target.value;
          onPick(v === '__latest__' ? undefined : v);
        }}
        style=${`padding:var(--space-1) var(--space-2);background:${palette.mantle};
                 border:1px solid ${palette.surface0};border-radius:var(--radius-sm);
                 color:${palette.text};font-size:var(--font-size-sm)`}
      >
        <option value="__latest__">latest${latest ? ` (${latest.epoch})` : ''}</option>
        ${sortedDesc.map(e => html`
          <option key=${e.epoch} value=${e.epoch}>
            ${e.epoch}${e.isLatest ? ' · latest' : ''}
          </option>
        `)}
      </select>
      ${!isCurrentLatest && html`
        <span data-testid="epoch-banner-not-latest" class="text-dim" style="font-size:11px">
          viewing ${current} of ${epochs.length} ·
          <a href="#" onclick=${ev => { ev.preventDefault(); onPick(undefined); }}>jump to latest</a>
        </span>
      `}
    </div>
  `;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/epoch-selector.js
git commit -s --no-verify -m "feat(ui-v1): EpochSelector component"
```

---

## Task 13: `app.js` epoch routes

**Files:** `src/aiperf/operator/ui-v1/app.js`

- [ ] **Step 1: Add two route matches**

In the route resolution chain inside `App()`, add:

```js
  const jobRunMatch = matchRoute('/jobs/:ns/:name/runs/:epoch', currentRoute);
  const sweepRunMatch = matchRoute('/sweeps/:ns/:name/runs/:epoch', currentRoute);
```

Order: place the `runs/:epoch` matches BEFORE the existing `/jobs/:ns/:name` and `/sweeps/:ns/:name` matches (longer patterns first).

```js
  } else if (jobRunMatch) {
    page = html`<${JobDetail} namespace=${jobRunMatch.ns} name=${jobRunMatch.name} epoch=${jobRunMatch.epoch} />`;
  } else if (jobDetailMatch) {
    page = html`<${JobDetail} namespace=${jobDetailMatch.ns} name=${jobDetailMatch.name} />`;
  } else if (sweepRunMatch) {
    page = html`<${SweepDetail} namespace=${sweepRunMatch.ns} name=${sweepRunMatch.name} epoch=${sweepRunMatch.epoch} />`;
  } else if (sweepDetailMatch) {
    page = html`<${SweepDetail} namespace=${sweepDetailMatch.ns} name=${sweepDetailMatch.name} />`;
  }
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/app.js
git commit -s --no-verify -m "feat(ui-v1): /jobs|sweeps/:ns/:name/runs/:epoch routes"
```

---

## Task 14: `pages/job-detail.js` — epoch awareness

**Files:** `src/aiperf/operator/ui-v1/pages/job-detail.js`

- [ ] **Step 1: Implement epoch wiring**

a) Add at the top of the imports:

```js
import { EpochSelector } from '../components/epoch-selector.js';
```

b) Update the component signature to accept the optional `epoch` prop (from Task 13 routes).

c) Inside the component, add an `epochs` state. On mount fetch `api.getJobEpochs(ns, name)`. Pass `current=${epoch}` into `EpochSelector`. The `onPick` handler calls `navigate` to either `/jobs/:ns/:name` (latest) or `/jobs/:ns/:name/runs/:epoch`.

d) The existing detail fetch becomes `api.getJob(ns, name, epoch)`. Conditions / KPIs use the returned data as before.

e) For pods/events panels, keep showing them only when `epoch === undefined` (latest = live CR present). When viewing a historical epoch, render an italic note: "Pods and events are not retained for archived epochs."

```js
const [epochs, setEpochs] = useState([]);

useEffect(() => {
  let cancelled = false;
  api.getJobEpochs(namespace, name)
    .then(d => { if (!cancelled) setEpochs(d.epochs ?? []); })
    .catch(() => {});
  return () => { cancelled = true; };
}, [namespace, name]);

function pickEpoch(next) {
  if (next === undefined) navigate(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`);
  else navigate(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(next)}`);
}
```

In the header JSX:

```js
<${EpochSelector} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/job-detail.js
git commit -s --no-verify -m "feat(ui-v1): JobDetail epoch dropdown + epoch-aware fetch"
```

---

## Task 15: `pages/sweep-detail.js` — epoch awareness

**Files:** `src/aiperf/operator/ui-v1/pages/sweep-detail.js`

- [ ] **Step 1: Mirror Task 14's pattern for sweeps**

a) Import `EpochSelector`. Component signature gains `epoch` prop.

b) Fetch `api.getSweepEpochs(ns, name)` on mount; render `EpochSelector` in the header.

c) Existing fetches become epoch-aware:
- `api.getSweep(ns, name, epoch)` — same shape; runs identically.
- `api.getSweepCells(ns, name, epoch)` — feeds the cells panel.
- *(optional)* `api.getSweepChildren(ns, name, effectiveEpoch)` — when viewing a historical epoch and `detail.children` is empty (live CR has no children for that historical run), use the manifest for the children panel instead.

d) `pickEpoch` navigates to `/sweeps/:ns/:name` or `/sweeps/:ns/:name/runs/:epoch`.

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/sweep-detail.js
git commit -s --no-verify -m "feat(ui-v1): SweepDetail epoch dropdown + epoch-aware fetch"
```

---

## Task 16: List pages — "Epochs" column

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/jobs.js`
- Modify: `src/aiperf/operator/ui-v1/pages/sweeps.js`

- [ ] **Step 1: Add the column to both list pages**

For each list page:

a) After the existing columns, add an "Epochs" header.

b) Each row's epochs cell renders a fetch on hover/click — but to keep this cheap we default to showing the count from a per-row pre-fetched value. The list endpoints don't return an epoch count today; either:
   - Add an `epochCount` field to the `SweepSummary` / `ActiveJobSummary` Pydantic responses (one extra `len(list_runs(...))` call per row in `_list_*_impl`), OR
   - Skip the column for now and instead add a small icon link `↻` that navigates to the detail page (the dropdown there shows the count).

Pick the second variant for v1 (zero extra backend cost). Render in each row:

```js
<td><a href=${detailUrl} title="View run history" style="color:${palette.overlay0}">↻</a></td>
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/jobs.js src/aiperf/operator/ui-v1/pages/sweeps.js
git commit -s --no-verify -m "feat(ui-v1): jobs/sweeps list pages link to run history"
```

---

## Task 17: Breadcrumb renders `runs/:epoch`

**Files:** `src/aiperf/operator/ui-v1/components/breadcrumb.js`

- [ ] **Step 1: Add epoch segment rendering**

The breadcrumb today parses the current route. Extend its parser to recognize `runs/:epoch` and emit two extra crumbs:

```
bench / satsweep / runs / 1714069323
                  ^link    ^plain text
```

Where the `runs` crumb links back to `/sweeps/:ns/:name` (latest), so clicking it acts like "jump to latest" without going through the dropdown.

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/breadcrumb.js
git commit -s --no-verify -m "feat(ui-v1): breadcrumb renders runs/:epoch segments"
```

---

# PR 5 — Migration smoke

## Task 18: End-to-end smoke + final formatting pass

- [ ] **Step 1: Full unit suite**

```bash
uv run pytest tests/unit/operator/ -n auto
uv run pytest tests/unit/sweep_controller/ -n auto
uv run pytest tests/unit/kubernetes/ -n auto
uv run pytest tests/unit/tools/ -n auto
```

Expected: all green except the pre-existing `test_cli_kube_results_list.py` import error.

- [ ] **Step 2: Placeholder grep**

```bash
grep -rn "TODO\|FIXME\|TBD\|legacy\|LEGACY_EPOCH\|migrate_legacy" \
  src/aiperf/operator/results_layout.py \
  src/aiperf/operator/sweep_union.py \
  src/aiperf/operator/job_union.py \
  src/aiperf/operator/routers/sweeps.py \
  src/aiperf/operator/routers/jobs.py \
  src/aiperf/sweep_controller/aggregator.py \
  src/aiperf/sweep_controller/k8s_executor.py \
  src/aiperf/operator/ui-v1/components/epoch-selector.js \
  src/aiperf/operator/ui-v1/pages/job-detail.js \
  src/aiperf/operator/ui-v1/pages/sweep-detail.js
```

Expected: no matches outside doc-strings about why legacy is gone.

- [ ] **Step 3: Build & deploy** (separate from this plan; reference `~/.claude/workflows/aiperf-dgx/build-and-push-arm64.md`).

- [ ] **Step 4: Cluster wipe** (one-shot, post-deploy):

```bash
KUBE_CONTEXT=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf exec deploy/aiperf-operator -c operator -- \
  python /app/tools/wipe_pre_epoch_results.py /data --apply
```

(Done by the operator, not by an agent — flagged for the user.)

---

## Self-Review (planner)

- [x] **Spec coverage:**
  - §4 Sweep epoch model → T2 (layout helpers), T4 (child name + marker), T5 (per-epoch aggregate + children.json).
  - §5 URL grammar → T13 (routes) + T14/T15 (page wiring).
  - §6 API surface → T6/T7 (jobs), T8/T9 (sweeps).
  - §7 DuckDB → T10.
  - §8 UI changes → T11–T17.
  - §9 No-backwards-compat → T1 (delete LEGACY_EPOCH), T3 (wipe script), T18 step 4 (cluster wipe).
- [x] **Placeholder scan:** none.
- [x] **Type consistency:**
  - `SweepRecord` field names unchanged.
  - `RunEntry` reused for both jobs and sweep epochs (same dataclass).
  - `EpochSummary` shape consistent across `/jobs/.../epochs` and `/sweeps/.../epochs` (camelCase aliases via `to_camel`).
  - `write_child_sweep_marker` payload extension matches `_sweep_linkage_from_marker` in `job_union.py` — both expect `sweep_run_epoch` and `child_run_epoch` as strings.
  - URL pattern `/jobs/:ns/:name/runs/:epoch` consistent in `app.js` (T13), `EpochSelector` (T12), `JobDetail`/`SweepDetail` `pickEpoch` (T14/T15), and Breadcrumb (T17).
