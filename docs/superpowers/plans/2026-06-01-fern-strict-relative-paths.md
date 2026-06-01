# Fern Local Strict Relative-Path / Broken-Link Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `make test-fern-docs` run Fern's strict broken-link / relative-path checks against md_to_mdx-converted docs locally, mirroring CI.

**Architecture:** Add a session-scoped pytest fixture to `tests/unit/fern/test_fern_docs.py` that copies `fern/` + `docs/` into a temp tree and runs `fern/md_to_mdx.py` (reproducing CI / `make fern-preview` staging). Re-point the two existing tests at that staged tree, and add two new tests that run `fern check --warnings --strict-broken-links` and `fern docs broken-links` there. No production code changes.

**Tech Stack:** Python 3.10+, pytest, `subprocess`, `shutil`, the `fern` CLI (skipped when not installed), `fern/md_to_mdx.py`.

---

## Background the implementer needs

- The `fern` CLI is discovered from the current working directory. Running `fern check` with `cwd` set to a directory containing `fern.config.json` + `docs.yml` validates that config. In the staged tree we point `cwd` at the staged `fern/` directory.
- Raw `docs/*.md` files use GitHub-flavored markdown (HTML comments `<!-- ... -->`, `> [!NOTE]` callouts). Fern's MDX parser rejects HTML comments, which makes the `valid-markdown-links` rule fail to *initialize* — a false error unrelated to any real link. `fern/md_to_mdx.py` converts these to MDX (`{/* ... */}`, `<Note>`), which is why checks must run against converted content.
- `fern/docs.yml` references docs via `path: ../docs/index.yml`. In the staged layout (`<tmp>/fern` and `<tmp>/docs` side by side) that relative path resolves correctly, so no `docs.yml` transformation is needed (CI transforms paths only because the docs-website branch uses a different layout).
- `fern/md_to_mdx.py` converts in place and only touches `*.md` files (via `rglob("*.md")`); `index.yml` is left alone.
- These tests carry `pytest.mark.fern` and are skipped when the `fern` CLI is absent. They run via `make test-fern-docs` (`pytest tests/unit/fern/ -m fern`), not the default unit-test job.

## File Structure

- **Modify:** `tests/unit/fern/test_fern_docs.py` — add imports, add `staged_fern_docs` fixture, re-point `test_fern_check` and `test_fern_docs_dev_starts`, add `test_fern_check_strict` and `test_fern_broken_links`.
- **Modify:** `Makefile:310` — extend the `test-fern-docs` target comment to mention the broken-link checks.

Current relevant state of `tests/unit/fern/test_fern_docs.py`:
- Imports (lines 12-20): `re`, `shutil`, `socket`, `subprocess`, `threading`, `pytest`. (`shutil` present; `pathlib.Path` is NOT imported.)
- `pytestmark` skip-if-not-installed (lines 36-39).
- `test_fern_check` (lines 42-53): runs `fern check` from implicit repo root.
- `test_fern_docs_dev_starts` (lines 56-113): runs `fern docs dev --port N` from implicit repo root.

---

### Task 1: Add the staged-conversion fixture

**Files:**
- Modify: `tests/unit/fern/test_fern_docs.py` (imports near lines 12-20; new fixture after the `pytestmark` block ~line 39)

- [ ] **Step 1: Add the `pathlib.Path` import**

In the import block, add `Path` from `pathlib`. After the change the stdlib imports read:

```python
from __future__ import annotations

import re
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest
```

- [ ] **Step 2: Add a repo-root constant and the fixture**

Insert immediately after the `pytestmark = [...]` block (after line 39):

```python
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def staged_fern_docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage and convert docs the way CI and ``make fern-preview`` do.

    Copies ``fern/`` and ``docs/`` into a temp tree, runs ``md_to_mdx.py`` to
    convert GitHub Markdown to Fern MDX, then returns the staged ``fern/``
    directory. Fern link validation must run against converted content: raw
    ``docs/`` contains HTML comments that Fern's MDX parser rejects, which
    breaks the link-check rules with a false error.
    """
    staged = tmp_path_factory.mktemp("fern-docs")
    shutil.copytree(
        _REPO_ROOT / "fern",
        staged / "fern",
        ignore=shutil.ignore_patterns(".local-preview"),
    )
    shutil.copytree(_REPO_ROOT / "docs", staged / "docs")
    subprocess.run(
        [
            "python3",
            str(staged / "fern" / "md_to_mdx.py"),
            "--dir",
            str(staged / "docs"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return staged / "fern"
```

- [ ] **Step 3: Sanity-check the fixture builds (temporary inline test)**

Add this throwaway test at the end of the file to confirm staging works, then run it:

```python
def test_staged_fern_docs_builds(staged_fern_docs: Path) -> None:
    assert (staged_fern_docs / "docs.yml").is_file()
    assert (staged_fern_docs.parent / "docs" / "index.md").is_file()
```

- [ ] **Step 4: Run the sanity check**

Run: `uv run pytest tests/unit/fern/test_fern_docs.py::test_staged_fern_docs_builds -m fern -v`
Expected: PASS. If it reports SKIP, the `fern` CLI is not installed — install it (see "Notes for the executor") and re-run to get a real PASS; the fixture itself does not need `fern`, but the `pytestmark` skip gates the whole module.

- [ ] **Step 5: Remove the throwaway test**

Delete `test_staged_fern_docs_builds` — the real tests in Task 2/3 exercise the fixture. Do not commit it.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/fern/test_fern_docs.py
git commit -s -m "test(fern): add staged md_to_mdx conversion fixture"
```

---

### Task 2: Re-point existing tests at the staged tree

**Files:**
- Modify: `tests/unit/fern/test_fern_docs.py` (`test_fern_check` ~lines 42-53; `test_fern_docs_dev_starts` ~lines 56-113)

- [ ] **Step 1: Re-point `test_fern_check`**

Replace the function with:

```python
def test_fern_check(staged_fern_docs: Path) -> None:
    """Validate the Fern definition (converted content) has no errors."""
    result = subprocess.run(
        ["fern", "check"],
        cwd=staged_fern_docs,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fern check failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
```

- [ ] **Step 2: Re-point `test_fern_docs_dev_starts`**

Change the signature to accept the fixture and set `cwd` on the `Popen` call. The signature becomes:

```python
def test_fern_docs_dev_starts(staged_fern_docs: Path) -> None:
```

and the `Popen` call becomes:

```python
    proc = subprocess.Popen(
        ["fern", "docs", "dev", "--port", str(port)],
        cwd=staged_fern_docs,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
```

Leave the rest of the function body unchanged.

- [ ] **Step 3: Run both re-pointed tests**

Run: `uv run pytest "tests/unit/fern/test_fern_docs.py::test_fern_check" "tests/unit/fern/test_fern_docs.py::test_fern_docs_dev_starts" -m fern -v`
Expected: both PASS (requires `fern` installed; otherwise SKIP).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/fern/test_fern_docs.py
git commit -s -m "test(fern): run existing checks against converted content"
```

---

### Task 3: Add the strict and broken-links tests

**Files:**
- Modify: `tests/unit/fern/test_fern_docs.py` (append two tests)

- [ ] **Step 1: Add `test_fern_check_strict`**

Append:

```python
def test_fern_check_strict(staged_fern_docs: Path) -> None:
    """Strict validation: broken or relative markdown links must fail.

    The auth-skipped redirects warning and accent-contrast warning remain as
    warnings (not errors) under ``--strict-broken-links``; broken/relative
    links are promoted to errors.
    """
    result = subprocess.run(
        ["fern", "check", "--warnings", "--strict-broken-links"],
        cwd=staged_fern_docs,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fern check --strict-broken-links failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
```

- [ ] **Step 2: Add `test_fern_broken_links`**

Append:

```python
def test_fern_broken_links(staged_fern_docs: Path) -> None:
    """Verify Fern finds no broken links in the converted content."""
    result = subprocess.run(
        ["fern", "docs", "broken-links"],
        cwd=staged_fern_docs,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"fern docs broken-links failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
```

- [ ] **Step 3: Run both new tests — expect PASS**

Run: `uv run pytest "tests/unit/fern/test_fern_docs.py::test_fern_check_strict" "tests/unit/fern/test_fern_docs.py::test_fern_broken_links" -m fern -v`
Expected: both PASS (requires `fern` installed).

- [ ] **Step 4: Prove the strict test actually catches a broken relative link (red check)**

Temporarily inject a broken relative link into a real source doc, then confirm the strict test fails:

```bash
printf '\n[broken](./this-page-does-not-exist.md)\n' >> docs/index.md
uv run pytest "tests/unit/fern/test_fern_docs.py::test_fern_check_strict" -m fern -v
```
Expected: FAIL, with the assertion message showing a `valid-markdown-links` error for the missing page.

- [ ] **Step 5: Revert the injected link**

```bash
git checkout -- docs/index.md
```

Re-run to confirm green again:

Run: `uv run pytest "tests/unit/fern/test_fern_docs.py::test_fern_check_strict" -m fern -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/fern/test_fern_docs.py
git commit -s -m "test(fern): add strict broken-link and relative-path checks"
```

---

### Task 4: Update the Makefile target comment

**Files:**
- Modify: `Makefile:310`

- [ ] **Step 1: Update the comment**

Change:

```make
test-fern-docs: #? validate Fern documentation (check, strict check, dev server).
```

to:

```make
test-fern-docs: #? validate Fern documentation (check, strict broken-link + broken-links checks, dev server).
```

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -s -m "build: note broken-link checks in test-fern-docs help"
```

---

### Task 5: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole fern suite**

Run: `make test-fern-docs`
Expected: all four tests PASS (`test_fern_check`, `test_fern_check_strict`, `test_fern_broken_links`, `test_fern_docs_dev_starts`) and the closing "Fern documentation checks passed!" line.

- [ ] **Step 2: Run pre-commit on changed files**

Run: `pre-commit run --files tests/unit/fern/test_fern_docs.py Makefile`
Expected: all hooks Pass or Skip.

---

## Notes for the executor

- If `fern` is not installed locally, the tests SKIP rather than fail. Install with `npm i -g fern-api@<version-from-fern/fern.config.json>` to actually exercise them. The red-check in Task 3 Step 4 only proves anything with `fern` installed.
- Do not add a new Makefile target or duplicate the staging shell from `fern-preview`; the fixture is the single source of staging for tests.
- No documentation table entry is required: this is test-infra only, not a user-facing feature, CLI option, env var, plugin, or service.
