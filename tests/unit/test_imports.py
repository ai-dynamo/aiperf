# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that all Python modules can be imported.

Imports leaf modules first (deepest nesting) to catch import errors early,
before parent modules potentially mask issues through __init__.py re-exports.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.harness.optional_deps import unsupported_test_module_names

# Root directories
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

# Dependencies declared in pyproject with an environment marker, hence
# intentionally absent on some platforms (no Windows-on-ARM wheel for pyarrow
# or datasets). A module that fails to import *only* because one of these is
# missing is an expected environmental skip, not a code defect, so the sweep
# treats it as a skip rather than a failure. Modules needing the non-importable
# soundfile/plotly native stacks self-skip via ``pytest.importorskip`` /
# module-level ``pytest.skip`` and are caught by the ``pytest.skip.Exception``
# branch below.
_MARKER_GATED_DEPS = {"datasets", "pyarrow"}


def discover_modules(
    base_dir: Path,
    package_dir: Path,
    *,
    exclude_names: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> list[tuple[str, int]]:
    """Discover all Python modules in a directory with their depth.

    Args:
        base_dir: The directory to use as the import root (e.g., src/ or tests/).
        package_dir: The package directory to scan (e.g., src/aiperf or tests/unit).
        exclude_names: Set of filenames to exclude (e.g., {"conftest.py"}).
        exclude_dirs: Set of directory names to exclude (e.g., {"ci"}).

    Returns:
        List of (module_path, depth) tuples, where depth is the nesting level.
        Deeper modules (leaves) have higher depth values.
    """
    exclude_names = exclude_names or set()
    exclude_dirs = exclude_dirs or set()
    modules: list[tuple[str, int]] = []

    for py_file in package_dir.rglob("*.py"):
        # Skip __pycache__ directories
        if "__pycache__" in py_file.parts:
            continue

        # Skip excluded filenames
        if py_file.name in exclude_names:
            continue

        # Skip excluded directories
        if exclude_dirs & set(py_file.parts):
            continue

        # Convert path to module name
        relative = py_file.relative_to(base_dir)
        parts = list(relative.parts)

        # Replace filename with stem (removes .py extension)
        parts[-1] = py_file.stem

        # Skip __init__ files - they're imported when importing the package
        if parts[-1] == "__init__":
            # For __init__.py, the module is the parent package
            parts = parts[:-1]
            if not parts:
                continue

        module_path = ".".join(parts)
        depth = len(parts)
        modules.append((module_path, depth))

    return modules


def sorted_leaves_first(modules: list[tuple[str, int]]) -> list[str]:
    """Return module paths sorted with leaves (deepest) first.

    This ordering ensures we catch import errors in leaf modules
    before parent __init__.py files potentially mask the issues.
    """
    # Sort by depth descending (deepest first), then alphabetically for stability
    return [m[0] for m in sorted(modules, key=lambda x: (-x[1], x[0]))]


def sorted_roots_first(modules: list[tuple[str, int]]) -> list[str]:
    """Return module paths sorted with roots (shallowest) first.

    This ordering tests the import hierarchy top-down.
    """
    # Sort by depth ascending (shallowest first), then alphabetically
    return [m[0] for m in sorted(modules, key=lambda x: (x[1], x[0]))]


def import_modules(modules: list[str]) -> dict[str, Exception]:
    """Import all modules and return dict of failures.

    Args:
        modules: List of module paths to import.

    Returns:
        Dict mapping failed module paths to their exceptions. Modules that
        call ``pytest.importorskip`` at top level (soft optional deps like
        ``optuna``, ``botorch``, ``torch``) are treated as a successful skip,
        not a failure — they raise ``pytest.skip.Exception`` (a
        ``BaseException``) which is intentionally not caught by ``Exception``.
    """
    failures: dict[str, Exception] = {}
    for module_path in modules:
        try:
            importlib.import_module(module_path)
        except pytest.skip.Exception:
            continue
        except ModuleNotFoundError as e:
            # Expected when an environment-marker-gated optional dep is absent
            # on this platform (e.g. datasets/pyarrow on Windows-on-ARM).
            if (e.name or "").split(".")[0] in _MARKER_GATED_DEPS:
                continue
            failures[module_path] = e
        except Exception as e:
            failures[module_path] = e
    return failures


# =============================================================================
# Module discovery (done once at collection time)
# =============================================================================

_AIPERF_MODULES_WITH_DEPTH = discover_modules(SRC_DIR, SRC_DIR / "aiperf")
# Use REPO_ROOT as base to get "tests.unit.xxx" paths matching pytest's import style
_TEST_MODULES_WITH_DEPTH = discover_modules(
    REPO_ROOT,
    TESTS_DIR,
    exclude_names={"conftest.py", "test_imports.py"},
    # aiperf_mock_amdsmi is the fake amdsmi bindings package; it raises OSError
    # on import unless AIPERF_MOCK_AMDSMI is set (the dormancy gate), so it is
    # intentionally non-importable here. It has dedicated coverage in
    # tests/unit/gpu_telemetry/test_fake_amdsmi_module.py.
    exclude_dirs={"ci", "aiperf_mock_amdsmi"},
)

AIPERF_MODULES = sorted_leaves_first(_AIPERF_MODULES_WITH_DEPTH)
# Test modules that directly import a no-Windows-on-ARM-wheel native dep are
# excluded from the sweep: the soundfile importers raise OSError (not caught by
# the ModuleNotFoundError rule above), and the rest are already known-skipped.
_UNSUPPORTED_TEST_MODULES = unsupported_test_module_names(TESTS_DIR, "tests")
TEST_MODULES = [
    m
    for m in sorted_leaves_first(_TEST_MODULES_WITH_DEPTH)
    if m not in _UNSUPPORTED_TEST_MODULES
]


# =============================================================================
# Import tests
# =============================================================================


def test_all_aiperf_modules_can_be_imported() -> None:
    """Test that all modules in src/aiperf can be imported without errors.

    This test catches:
    - Syntax errors
    - Missing dependencies
    - Circular import issues
    - Name errors at module level
    - Invalid relative imports after __init__.py cleanup
    """
    failures = import_modules(AIPERF_MODULES)
    if failures:
        messages = [f"  {mod}: {err!r}" for mod, err in failures.items()]
        pytest.fail(
            f"Failed to import {len(failures)}/{len(AIPERF_MODULES)} modules:\n"
            + "\n".join(messages)
        )


def test_all_test_modules_can_be_imported() -> None:
    """Test that all modules in tests/ can be imported without errors.

    This test catches:
    - Syntax errors
    - Missing dependencies
    - Circular import issues
    - Name errors at module level
    """
    failures = import_modules(TEST_MODULES)
    if failures:
        messages = [f"  {mod}: {err!r}" for mod, err in failures.items()]
        pytest.fail(
            f"Failed to import {len(failures)}/{len(TEST_MODULES)} modules:\n"
            + "\n".join(messages)
        )


# =============================================================================
# Import ordering verification
# =============================================================================


class TestImportOrder:
    """Verify the import ordering logic."""

    def test_leaves_first_ordering(self) -> None:
        """Verify that deeper modules come before shallower ones."""
        modules = sorted_leaves_first(_AIPERF_MODULES_WITH_DEPTH)

        # Check that we have modules at different depths
        depths = [m.count(".") for m in modules]
        assert max(depths) > min(depths), "Expected modules at different depths"

        # Verify ordering: each module should have depth >= next module's depth
        for i in range(len(modules) - 1):
            current_depth = modules[i].count(".")
            next_depth = modules[i + 1].count(".")
            assert current_depth >= next_depth, (
                f"Ordering violated: {modules[i]} (depth {current_depth}) "
                f"came before {modules[i + 1]} (depth {next_depth})"
            )

    def test_roots_first_ordering(self) -> None:
        """Verify that shallower modules come before deeper ones."""
        modules = sorted_roots_first(_AIPERF_MODULES_WITH_DEPTH)

        # Verify ordering: each module should have depth <= next module's depth
        for i in range(len(modules) - 1):
            current_depth = modules[i].count(".")
            next_depth = modules[i + 1].count(".")
            assert current_depth <= next_depth, (
                f"Ordering violated: {modules[i]} (depth {current_depth}) "
                f"came before {modules[i + 1]} (depth {next_depth})"
            )

    def test_aiperf_modules_found(self) -> None:
        """Verify we're finding a reasonable number of aiperf modules."""
        assert len(_AIPERF_MODULES_WITH_DEPTH) > 100, (
            f"Expected >100 modules, found {len(_AIPERF_MODULES_WITH_DEPTH)}"
        )

    def test_test_modules_found(self) -> None:
        """Verify we're finding a reasonable number of test modules."""
        assert len(_TEST_MODULES_WITH_DEPTH) > 50, (
            f"Expected >50 test modules, found {len(_TEST_MODULES_WITH_DEPTH)}"
        )

    def test_no_pycache_modules(self) -> None:
        """Verify we're not including __pycache__ files."""
        for modules in [_AIPERF_MODULES_WITH_DEPTH, _TEST_MODULES_WITH_DEPTH]:
            for module, _ in modules:
                assert "__pycache__" not in module, f"Found pycache in: {module}"
