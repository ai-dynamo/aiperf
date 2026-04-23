# Pytest Memory Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-test virtual-address-space cap (default 4 GiB) to the in-process `tests/unit/` and `tests/component_integration/` suites so runaway-allocation tests fail as attributable `MemoryError` pytest failures instead of OOM-killing xdist workers.

**Architecture:** A single autouse function-scoped fixture in `tests/conftest.py` sets `resource.setrlimit(RLIMIT_AS, ...)` before each qualifying test and restores the original limit in teardown. Opt-out via `@pytest.mark.no_memory_limit`, per-test override via `@pytest.mark.memory_limit(mb=N)`, global override/disable via `AIPERF_TEST_MEMORY_LIMIT_MB` env var. Integration/k8s/e2e suites and platforms without `RLIMIT_AS` (Windows, possibly macOS) skip the guard entirely.

**Tech Stack:** Python 3.10+, stdlib `resource` module, pytest, pytest-xdist.

**Spec:** `docs/superpowers/specs/2026-04-23-pytest-memory-guard-design.md`

---

## File Structure

- **`pyproject.toml`** — register two new markers (`memory_limit`, `no_memory_limit`) in the existing `[tool.pytest.ini_options]` `markers` list.
- **`tests/conftest.py`** — add the autouse fixture, module-level platform detection, original-limit capture, and helper to resolve the effective cap. Export the import-time original soft limit as a public constant for the verification tests.
- **`tests/unit/test_memory_limit_guard.py`** (new) — four verification tests covering default cap, marker override, opt-out, and teardown restoration.
- **`CLAUDE.md`**, **`.github/copilot-instructions.md`**, **`.cursor/rules/python.mdc`** — short subsection under `## Testing Conventions` documenting the guard. Three-File Sync Rule applies.

No changes to `docs/dev/patterns.md` (this is test infra, not a code pattern).

---

### Task 1: Register pytest markers

**Files:**
- Modify: `pyproject.toml:171-189` (the `markers = [...]` list)

- [ ] **Step 1: Add the two markers to the list**

Open `pyproject.toml` and insert two lines into the `markers = [...]` list (order irrelevant; add at the end before the closing `]`):

```toml
    "memory_limit(mb=N): cap this test's virtual address space at N MiB (default 4096)",
    "no_memory_limit: disable the per-test virtual address space cap entirely",
```

The resulting tail of the list should look like:

```toml
    "e2e: marks tests as browser-based end-to-end UI tests (requires playwright chromium, deselected by default)",
    "memory_limit(mb=N): cap this test's virtual address space at N MiB (default 4096)",
    "no_memory_limit: disable the per-test virtual address space cap entirely",
]
```

- [ ] **Step 2: Verify pytest recognizes the markers**

Run: `uv run pytest --markers | grep -E "memory_limit|no_memory_limit"`
Expected output: two lines, one per marker, each showing the description we added.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -s -m "test: register memory_limit and no_memory_limit pytest markers"
```

---

### Task 2: Write failing verification tests

**Files:**
- Create: `tests/unit/test_memory_limit_guard.py`

These tests define the contract; they will fail until Task 3 lands.

- [ ] **Step 1: Create the test file with all four tests**

Write the following exact content to `tests/unit/test_memory_limit_guard.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification tests for the per-test memory guard in tests/conftest.py.

These tests exercise the guard itself: marker-based cap, marker-based opt-out,
and the default 4 GiB cap in the unit suite. Teardown restoration is verified
implicitly: the no_memory_limit test asserts the soft limit is the original,
which only holds if the preceding marker-based test restored it on teardown.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    ORIGINAL_RLIMIT_AS_SOFT,
    RLIMIT_AS_SUPPORTED,
)

pytestmark = pytest.mark.skipif(
    not RLIMIT_AS_SUPPORTED,
    reason="RLIMIT_AS not supported on this platform",
)


def _current_soft_limit() -> int:
    import resource

    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    return soft


@pytest.mark.memory_limit(mb=256)
def test_memory_limit_marker_caps_allocation() -> None:
    """A 256 MiB marker must cap allocations and raise MemoryError."""
    chunks: list[bytearray] = []
    with pytest.raises(MemoryError):
        # Allocate up to 2 GiB in 32 MiB chunks; the 256 MiB cap must fire.
        for _ in range(64):
            chunks.append(bytearray(32 * 1024 * 1024))


@pytest.mark.no_memory_limit
def test_no_memory_limit_marker_disables_guard() -> None:
    """no_memory_limit must skip the guard; soft limit equals the original.

    This also verifies teardown: if the preceding marker-based test did not
    restore the original limit, this assertion would fail.
    """
    assert _current_soft_limit() == ORIGINAL_RLIMIT_AS_SOFT


def test_default_cap_applied_in_unit_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any marker and with env var unset, the 4 GiB default applies.

    The monkeypatch.delenv only affects future env reads; the guard for THIS
    test has already resolved its cap before the test body runs. So the
    assertion checks the soft limit is the default 4 GiB. monkeypatch.delenv
    is defensive in case a future refactor re-reads the env mid-test.
    """
    monkeypatch.delenv("AIPERF_TEST_MEMORY_LIMIT_MB", raising=False)
    expected = 4096 * 1024 * 1024
    assert _current_soft_limit() == expected
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `uv run pytest tests/unit/test_memory_limit_guard.py -n auto -v`

Expected: all three tests fail with `ImportError` (the constants `ORIGINAL_RLIMIT_AS_SOFT` and `RLIMIT_AS_SUPPORTED` do not yet exist in `tests/conftest.py`). This is the expected failing state before Task 3.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/unit/test_memory_limit_guard.py
git commit -s -m "test(memory-guard): add verification tests (red)"
```

---

### Task 3: Implement the memory guard fixture

**Files:**
- Modify: `tests/conftest.py` (add at end of file, after the existing `pytest_configure` function)

- [ ] **Step 1: Add the memory-guard code to `tests/conftest.py`**

Append the following to `tests/conftest.py`. Do **not** touch the existing `_PATH_MARKER_MAP` block or `pytest_configure` — add after them.

```python
# ---------------------------------------------------------------------------
# Per-test virtual-address-space guard.
#
# Caps RLIMIT_AS for tests under tests/unit/ and tests/component_integration/
# so a runaway allocation loop fails with MemoryError instead of triggering
# the OS OOM-killer and crashing an xdist worker. See
# docs/superpowers/specs/2026-04-23-pytest-memory-guard-design.md.
# ---------------------------------------------------------------------------

import os  # noqa: E402

try:
    import resource  # noqa: E402

    RLIMIT_AS_SUPPORTED = hasattr(resource, "RLIMIT_AS")
except ImportError:  # pragma: no cover - Windows lacks the resource module
    resource = None  # type: ignore[assignment]
    RLIMIT_AS_SUPPORTED = False

_DEFAULT_MEMORY_CAP_MB = 4096
_MEMORY_LIMIT_ENV_VAR = "AIPERF_TEST_MEMORY_LIMIT_MB"
_MEMORY_GUARD_PATH_PREFIXES = (
    "tests/unit/",
    "tests/component_integration/",
)

if RLIMIT_AS_SUPPORTED:
    ORIGINAL_RLIMIT_AS_SOFT, ORIGINAL_RLIMIT_AS_HARD = resource.getrlimit(
        resource.RLIMIT_AS
    )
else:
    ORIGINAL_RLIMIT_AS_SOFT = -1
    ORIGINAL_RLIMIT_AS_HARD = -1


def _node_in_guarded_suite(nodeid: str) -> bool:
    """Return True if the test's node id starts with a guarded path prefix."""
    return any(nodeid.startswith(p) for p in _MEMORY_GUARD_PATH_PREFIXES)


def _resolve_memory_cap_mb(request: pytest.FixtureRequest) -> int | None:
    """Resolve the effective memory cap in MiB, or None to skip the guard.

    Precedence (first match wins):
      1. @pytest.mark.no_memory_limit  -> None (skip)
      2. AIPERF_TEST_MEMORY_LIMIT_MB=0 -> None (skip)
      3. @pytest.mark.memory_limit(mb=N) -> N
      4. AIPERF_TEST_MEMORY_LIMIT_MB=N (N > 0) -> N
      5. default 4096
    """
    if request.node.get_closest_marker("no_memory_limit") is not None:
        return None

    env_raw = os.environ.get(_MEMORY_LIMIT_ENV_VAR)
    env_cap: int | None = None
    if env_raw is not None:
        try:
            env_cap = int(env_raw)
        except ValueError:
            env_cap = None
        else:
            if env_cap == 0:
                return None

    marker = request.node.get_closest_marker("memory_limit")
    if marker is not None:
        mb = marker.kwargs.get("mb")
        if mb is None and marker.args:
            mb = marker.args[0]
        if isinstance(mb, int) and mb > 0:
            return mb

    if env_cap is not None and env_cap > 0:
        return env_cap

    return _DEFAULT_MEMORY_CAP_MB


@pytest.fixture(autouse=True)
def _memory_limit_guard(request: pytest.FixtureRequest):
    """Cap virtual address space for unit / component_integration tests.

    Best-effort: any OSError/ValueError from setrlimit is logged and the
    test runs unbounded rather than failing on guard setup.
    """
    if not RLIMIT_AS_SUPPORTED:
        yield
        return

    if not _node_in_guarded_suite(request.node.nodeid):
        yield
        return

    cap_mb = _resolve_memory_cap_mb(request)
    if cap_mb is None:
        yield
        return

    cap_bytes = cap_mb * 1024 * 1024
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (cap_bytes, ORIGINAL_RLIMIT_AS_HARD)
        )
    except (OSError, ValueError):
        # Guard setup failed (e.g. sandbox policy); run the test unbounded.
        yield
        return

    try:
        yield
    finally:
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (ORIGINAL_RLIMIT_AS_SOFT, ORIGINAL_RLIMIT_AS_HARD),
            )
        except (OSError, ValueError):
            pass
```

- [ ] **Step 2: Verify the verification tests pass**

Run: `uv run pytest tests/unit/test_memory_limit_guard.py -n auto -v`
Expected: all three tests PASS.

Notes on what each test proves at this step:
- `test_memory_limit_marker_caps_allocation` — guard fires under marker override (256 MiB cap trips before 2 GiB of chunks are allocated).
- `test_no_memory_limit_marker_disables_guard` — opt-out marker skips the guard; the soft limit is the original value captured at import time. This also proves teardown restored correctly after the preceding capped test.
- `test_default_cap_applied_in_unit_suite` — default 4 GiB cap is active in `tests/unit/` when no marker and no env override are present.

- [ ] **Step 3: Run the broader unit suite to catch regressions**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all tests pass (or fail for pre-existing unrelated reasons). No new `MemoryError` failures should appear — the audit in the spec confirmed no unit test allocates near 4 GiB.

If a previously-passing test now fails with `MemoryError`, STOP and investigate. It is either (a) a genuine runaway the guard correctly caught, or (b) a legitimately-heavy test that needs `@pytest.mark.memory_limit(mb=N)` with N > 4096. Do not silently add the marker — surface the finding for review.

- [ ] **Step 4: Run the component_integration suite**

Run: `uv run pytest tests/component_integration/ -n auto`
Expected: all tests pass (same caveat as Step 3).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -s -m "test(memory-guard): cap RLIMIT_AS for unit + component_integration (green)"
```

---

### Task 4: Document the guard in the three sync files

**Files:**
- Modify: `CLAUDE.md` (insert under `## Testing Conventions`, after the existing auto-fixtures bullet at line 125)
- Modify: `.github/copilot-instructions.md` (same insertion, mirror location)
- Modify: `.cursor/rules/python.mdc` (same insertion, mirror location)

The Three-File Sync Rule requires identical content across all three (only headers/frontmatter differ).

- [ ] **Step 1: Insert the documentation block in `CLAUDE.md`**

In `CLAUDE.md`, find the `## Testing Conventions` section. After the existing bullet line that ends with `Auto-fixtures (always active): asyncio.sleep runs instantly, RNG=42, singletons reset between tests`, add the following content:

```markdown
- Memory guard (unit + component_integration only): each test runs under a 4 GiB `RLIMIT_AS` cap so runaway allocations fail with `MemoryError` instead of OOM-killing the worker. Override per test with `@pytest.mark.memory_limit(mb=N)`; fully opt out with `@pytest.mark.no_memory_limit`. Disable globally with `AIPERF_TEST_MEMORY_LIMIT_MB=0`, or raise/lower the default with `AIPERF_TEST_MEMORY_LIMIT_MB=N`. Integration/k8s/e2e suites and non-Linux platforms skip the guard.
```

- [ ] **Step 2: Insert the same block in `.github/copilot-instructions.md`**

Make the identical insertion at the matching location in `.github/copilot-instructions.md` (same `## Testing Conventions` section, after the auto-fixtures bullet).

- [ ] **Step 3: Insert the same block in `.cursor/rules/python.mdc`**

Make the identical insertion at the matching location in `.cursor/rules/python.mdc`. Preserve the file's existing `alwaysApply: true` frontmatter.

- [ ] **Step 4: Diff the three files to confirm content sync**

Run:

```bash
diff <(sed -n '/## Testing Conventions/,/^## /p' CLAUDE.md) \
     <(sed -n '/## Testing Conventions/,/^## /p' .github/copilot-instructions.md)
diff <(sed -n '/## Testing Conventions/,/^## /p' CLAUDE.md) \
     <(sed -n '/## Testing Conventions/,/^## /p' .cursor/rules/python.mdc)
```

Expected: both diffs empty (or frontmatter-only differences for the `.mdc` file, which should not appear inside `## Testing Conventions`).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -s -m "docs: document pytest memory guard in sync files"
```

---

### Task 5: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite fresh**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all tests pass.

- [ ] **Step 2: Run the full component_integration suite fresh**

Run: `uv run pytest tests/component_integration/ -n auto`
Expected: all tests pass.

- [ ] **Step 3: Sanity-check that an integration test is unaffected**

Run: `uv run pytest tests/integration/ -n auto -k "not very_slow" --collect-only | head -5`
Expected: collection succeeds. (We are not running them — just confirming the guard's path gating did not break collection of non-guarded suites.)

- [ ] **Step 4: Run pre-commit on all staged hooks for the changed files**

Run: `pre-commit run --files pyproject.toml tests/conftest.py tests/unit/test_memory_limit_guard.py CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc`
Expected: all hooks pass (no formatting drift, license headers present, etc.).

No commit in this task — verification only. The previous four tasks already committed all changes.
