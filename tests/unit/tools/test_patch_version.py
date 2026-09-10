# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nightly version patcher used by update-pyproject-version."""

from pathlib import Path

import pytest
from pytest import param

from tools.patch_version import PatchError, main, patch_version

_PYPROJECT = """\
[project]
name = "aiperf"
version = "0.13.0"
dependencies = ["aioitertools"]
"""

# Mirrors uv's layout: a same-versioned dependency before the root, and an inline
# `{ name = "aiperf", ... }` requires-dist reference after it, neither of which
# may be touched.
_LOCK = """\
version = 1
revision = 3

[[package]]
name = "aioitertools"
version = "0.13.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "aiperf"
version = "0.13.0"
source = { editable = "." }
dependencies = [
    { name = "aioitertools" },
]

[package.metadata]
requires-dist = [
    { name = "aiperf", extras = ["botorch"], marker = "extra == 'optuna'" },
]

[[package]]
name = "aiperf-mock-server"
version = "0.13.0"
source = { editable = "tests/aiperf_mock_server" }
"""


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal pyproject.toml and uv.lock pair in a temporary directory."""
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_text(_PYPROJECT)
    lock.write_text(_LOCK)
    return pyproject, lock


@pytest.mark.parametrize(
    "version",
    [
        param("0.13.0.dev20260910", id="scheduled-nightly"),
        param("0.13.0.dev20260910+manual.34398837728", id="manual-dispatch"),
        param("0.14.0rc1", id="release-candidate"),
    ],
)  # fmt: skip
def test_patch_version_valid_version_updates_only_root_entries(
    tree: tuple[Path, Path], version: str
) -> None:
    pyproject, lock = tree

    assert patch_version(version, pyproject, lock) == "aiperf"

    assert f'version = "{version}"' in pyproject.read_text()
    expected_lock = _LOCK.replace(
        'name = "aiperf"\nversion = "0.13.0"', f'name = "aiperf"\nversion = "{version}"'
    )
    assert lock.read_text() == expected_lock


def test_patch_version_same_version_is_a_no_op(tree: tuple[Path, Path]) -> None:
    pyproject, lock = tree

    patch_version("0.13.0", pyproject, lock)

    assert pyproject.read_text() == _PYPROJECT
    assert lock.read_text() == _LOCK


@pytest.mark.parametrize(
    "version",
    [
        param("not-a-version", id="garbage"),
        param("1.2", id="two-component"),
        param("0.13.0+bad_local", id="underscore-in-local"),
    ],
)  # fmt: skip
def test_patch_version_invalid_version_raises_before_writing(
    tree: tuple[Path, Path], version: str
) -> None:
    pyproject, lock = tree

    with pytest.raises(PatchError, match="Invalid PEP 440"):
        patch_version(version, pyproject, lock)

    assert pyproject.read_text() == _PYPROJECT
    assert lock.read_text() == _LOCK


def test_patch_version_missing_lock_raises(tree: tuple[Path, Path]) -> None:
    pyproject, lock = tree
    lock.unlink()

    with pytest.raises(PatchError, match="uv.lock not found"):
        patch_version("0.13.0.dev20260910", pyproject, lock)

    assert pyproject.read_text() == _PYPROJECT


def test_patch_version_lock_without_root_entry_raises_and_writes_nothing(
    tree: tuple[Path, Path],
) -> None:
    pyproject, lock = tree
    lock.write_text(
        _LOCK.replace('source = { editable = "." }', 'source = { editable = "src" }')
    )
    before = lock.read_text()

    with pytest.raises(PatchError, match="editable root package entry"):
        patch_version("0.13.0.dev20260910", pyproject, lock)

    assert pyproject.read_text() == _PYPROJECT
    assert lock.read_text() == before


def test_patch_version_pyproject_without_version_raises(
    tree: tuple[Path, Path],
) -> None:
    pyproject, lock = tree
    pyproject.write_text('[project]\nname = "aiperf"\n')

    with pytest.raises(PatchError, match="Could not find 'version"):
        patch_version("0.13.0.dev20260910", pyproject, lock)


def test_main_success_prints_both_files_and_returns_zero(
    tree: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject, lock = tree

    rc = main(
        ["0.13.0.dev20260910", "--pyproject", str(pyproject), "--lock", str(lock)]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert f"Patched {pyproject}" in out and f"Patched {lock}" in out


def test_main_failure_emits_github_error_annotation(
    tree: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject, lock = tree
    lock.unlink()

    rc = main(
        ["0.13.0.dev20260910", "--pyproject", str(pyproject), "--lock", str(lock)]
    )

    assert rc == 1
    assert capsys.readouterr().err.startswith("::error::")
