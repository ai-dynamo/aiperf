# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nightly version patcher used by update-pyproject-version."""

import shutil
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest import param

import tools.patch_version as patch_version_module
from tools.patch_version import PatchError, main, patch_version

_REPO_ROOT = Path(__file__).resolve().parents[3]

# A table with its own `name` precedes [project] so an unscoped regex would
# pick up "pytorch-cpu" instead of the project name.
_PYPROJECT = """\
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"

[project]
name = "aiperf"
version = "0.13.0"
dependencies = ["aioitertools"]

[tool.uv]
required-environments = ["sys_platform == 'linux'"]
"""

# Mirrors uv's layout: a same-versioned dependency before the root and an inline
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
"""

_NEW = "0.13.0.dev20260910"


def _expected_lock(version: str) -> str:
    return _LOCK.replace(
        'name = "aiperf"\nversion = "0.13.0"', f'name = "aiperf"\nversion = "{version}"'
    )


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
def test_patch_version_valid_version_updates_only_project_and_root_entries(
    tree: tuple[Path, Path], version: str
) -> None:
    pyproject, lock = tree

    assert patch_version(version, pyproject, lock) == "aiperf"

    project = tomllib.loads(pyproject.read_text())["project"]
    assert project == {
        "name": "aiperf",
        "version": version,
        "dependencies": ["aioitertools"],
    }
    assert 'name = "pytorch-cpu"' in pyproject.read_text()
    assert lock.read_text() == _expected_lock(version)


def test_patch_version_preserves_crlf_line_endings(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_bytes(_PYPROJECT.replace("\n", "\r\n").encode())
    lock.write_bytes(_LOCK.replace("\n", "\r\n").encode())

    patch_version(_NEW, pyproject, lock)

    assert pyproject.read_bytes() == (
        _PYPROJECT.replace('version = "0.13.0"', f'version = "{_NEW}"')
        .replace("\n", "\r\n")
        .encode()
    )
    assert lock.read_bytes() == _expected_lock(_NEW).replace("\n", "\r\n").encode()


@pytest.mark.parametrize(
    "version",
    [
        param("not-a-version", id="garbage"),
        param("1.2", id="two-component"),
        param("0.13.0+bad_local", id="underscore-in-local"),
        param("0.13.0.dev20260910\n", id="trailing-newline"),
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


def test_patch_version_normalizes_project_name_for_lock_lookup(tmp_path: Path) -> None:
    """uv writes the PEP 503 name into uv.lock even when pyproject spells it differently."""
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_text(_PYPROJECT.replace('name = "aiperf"', 'name = "AI_Perf.Tool"'))
    lock.write_text(_LOCK.replace('name = "aiperf"', 'name = "ai-perf-tool"'))

    assert patch_version(_NEW, pyproject, lock) == "AI_Perf.Tool"

    assert f'name = "ai-perf-tool"\nversion = "{_NEW}"' in lock.read_text()


def test_patch_version_missing_lock_raises_with_remedy(tree: tuple[Path, Path]) -> None:
    pyproject, lock = tree
    lock.unlink()

    with pytest.raises(PatchError, match=r"uv.lock not found; run `uv lock`"):
        patch_version(_NEW, pyproject, lock)

    assert pyproject.read_text() == _PYPROJECT


@pytest.mark.parametrize(
    ("lock_text", "message"),
    [
        param(
            _LOCK.replace('source = { editable = "." }', 'source = { editable = "src" }'),
            "exactly one editable root",
            id="no-root-entry",
        ),
        param(
            _LOCK + '\n[[package]]\nname = "aiperf"\nversion = "0.13.0"\nsource = { editable = "." }\n',
            "does not carry",
            id="two-root-entries",
        ),
    ],
)  # fmt: skip
def test_patch_version_bad_lock_raises_and_writes_nothing(
    tree: tuple[Path, Path], lock_text: str, message: str
) -> None:
    pyproject, lock = tree
    lock.write_text(lock_text)

    with pytest.raises(PatchError, match=message):
        patch_version(_NEW, pyproject, lock)

    assert pyproject.read_text() == _PYPROJECT
    assert lock.read_text() == lock_text


@pytest.mark.parametrize(
    ("pyproject_text", "message"),
    [
        param('[tool.poetry]\nname = "aiperf"\nversion = "0.13.0"\n', r"no \[project\] table", id="poetry-only"),
        param('[project]\nname = "aiperf"\n', r'no version = "..." line', id="no-version"),
        param('[project]\nversion = "0.13.0"\n', "no name", id="no-name"),
        param('[project\nname = "aiperf"\n', "not valid TOML", id="invalid-toml"),
    ],
)  # fmt: skip
def test_patch_version_bad_pyproject_raises_and_writes_nothing(
    tree: tuple[Path, Path], pyproject_text: str, message: str
) -> None:
    pyproject, lock = tree
    pyproject.write_text(pyproject_text)

    with pytest.raises(PatchError, match=message):
        patch_version(_NEW, pyproject, lock)

    assert pyproject.read_text() == pyproject_text
    assert lock.read_text() == _LOCK


def test_patch_version_against_repository_files_changes_exactly_two_lines(
    tmp_path: Path,
) -> None:
    """Bind the textual patterns to the real pyproject.toml and uv.lock layout."""
    pyproject = shutil.copy(_REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    lock = shutil.copy(_REPO_ROOT / "uv.lock", tmp_path / "uv.lock")
    original_pyproject = (_REPO_ROOT / "pyproject.toml").read_text().splitlines()
    original_lock = (_REPO_ROOT / "uv.lock").read_text().splitlines()
    current = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    # The nightly runs this suite on an already-patched tree, so derive a target
    # that can never equal the current version (dev numbers are dates).
    target = f"{current.split('.dev')[0].split('+')[0]}.dev99999999"

    assert patch_version(target, Path(pyproject), Path(lock)) == "aiperf"

    changed_pyproject = [
        (a, b)
        for a, b in zip(
            original_pyproject, Path(pyproject).read_text().splitlines(), strict=True
        )
        if a != b
    ]
    changed_lock = [
        (a, b)
        for a, b in zip(original_lock, Path(lock).read_text().splitlines(), strict=True)
        if a != b
    ]
    assert changed_pyproject == [(f'version = "{current}"', f'version = "{target}"')]
    assert changed_lock == [(f'version = "{current}"', f'version = "{target}"')]
    assert tomllib.loads(Path(pyproject).read_text())["project"]["version"] == target


def test_main_success_prints_both_files_and_returns_zero(
    tree: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject, lock = tree

    rc = main([_NEW, "--pyproject", str(pyproject), "--lock", str(lock)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"Patched {pyproject}" in out and f"Patched {lock}" in out


@pytest.mark.parametrize(
    ("break_tree", "message"),
    [
        param(lambda pyproject, lock: lock.unlink(), "not found", id="missing-lock"),
        param(lambda pyproject, lock: (lock.unlink(), lock.mkdir()), "not a regular file", id="lock-is-a-directory"),
    ],
)  # fmt: skip
def test_main_failure_emits_github_error_annotation(
    tree: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    break_tree: Callable[[Path, Path], object],
    message: str,
) -> None:
    pyproject, lock = tree
    break_tree(pyproject, lock)

    rc = main([_NEW, "--pyproject", str(pyproject), "--lock", str(lock)])

    assert rc == 1
    assert message in capsys.readouterr().err
    assert pyproject.read_text() == _PYPROJECT


def test_main_write_failure_reports_error_and_leaves_both_files_untouched(
    tree: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OSError while staging the lock must not leave pyproject.toml already patched."""
    pyproject, lock = tree
    real_stage = patch_version_module._stage

    def failing_stage(path: Path, text: str) -> Path:
        if path == lock:
            raise PermissionError(13, "Permission denied", str(path))
        return real_stage(path, text)

    monkeypatch.setattr(patch_version_module, "_stage", failing_stage)

    rc = main([_NEW, "--pyproject", str(pyproject), "--lock", str(lock)])

    assert rc == 1
    assert capsys.readouterr().err.startswith("::error::")
    assert pyproject.read_text() == _PYPROJECT
    assert lock.read_text() == _LOCK
    assert not (pyproject.parent / "pyproject.toml.tmp").exists()
