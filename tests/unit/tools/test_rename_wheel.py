# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tools/rename_wheel.py``."""

from __future__ import annotations

import importlib.util
import stat
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_rename_wheel():
    spec = importlib.util.spec_from_file_location(
        "rename_wheel", _REPO_ROOT / "tools" / "rename_wheel.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rename_wheel = _load_rename_wheel()

_EXEC_ATTR = (stat.S_IFREG | 0o755) << 16
_SCRIPT = "aiperf-0.11.0.data/scripts/aiperf"


def _wheel_with_executable_script(tmp_path: Path) -> Path:
    wheel = tmp_path / "aiperf-0.11.0-py3-none-manylinux_2_39_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("aiperf/__init__.py", "__version__ = '0.11.0'\n")
        info = zipfile.ZipInfo(_SCRIPT)
        info.external_attr = _EXEC_ATTR
        info.create_system = 3
        zf.writestr(info, b"\x7fELF-stub")
        zf.writestr(
            "aiperf-0.11.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: aiperf\n"
        )
        zf.writestr(
            "aiperf-0.11.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            "Tag: py3-none-manylinux_2_39_x86_64\n",
        )
        zf.writestr(
            "aiperf-0.11.0.dist-info/RECORD", "aiperf-0.11.0.dist-info/RECORD,,\n"
        )
    return wheel


def _rename(monkeypatch: pytest.MonkeyPatch, wheel: Path, new_name: str) -> Path:
    monkeypatch.setattr(
        "sys.argv", ["rename_wheel.py", str(wheel), "--new-name", new_name]
    )
    assert rename_wheel.main() == 0
    escaped = rename_wheel.escape_name(new_name)
    return next(wheel.parent.glob(f"{escaped}-*.whl"))


def test_rename_preserves_the_script_executable_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    renamed = _rename(
        monkeypatch, _wheel_with_executable_script(tmp_path), "aiperf-nightly"
    )

    with zipfile.ZipFile(renamed) as zf:
        script = next(n for n in zf.namelist() if n.endswith(".data/scripts/aiperf"))
        mode = zf.getinfo(script).external_attr >> 16
        assert stat.S_ISREG(mode), f"mode {mode:o} lost its regular-file type bits"
        assert mode & 0o111, f"mode {mode:o} is not executable"
        # Non-executable payload keeps its own mode rather than a blanket 0755.
        init_mode = zf.getinfo("aiperf/__init__.py").external_attr >> 16
        assert not init_mode & 0o111, f"mode {init_mode:o} gained an executable bit"


def test_rename_moves_the_data_dir_to_the_new_distribution_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    renamed = _rename(
        monkeypatch, _wheel_with_executable_script(tmp_path), "aiperf-nightly"
    )

    with zipfile.ZipFile(renamed) as zf:
        names = zf.namelist()
        assert "aiperf_nightly-0.11.0.data/scripts/aiperf" in names
        assert not any(n.startswith("aiperf-0.11.0.data/") for n in names)
        record = zf.read("aiperf_nightly-0.11.0.dist-info/RECORD").decode()
    listed = [line.split(",")[0] for line in record.splitlines() if line]
    assert sorted(listed) == sorted(names), "RECORD must list every member exactly once"


def test_rename_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    once = _rename(
        monkeypatch, _wheel_with_executable_script(tmp_path), "aiperf-nightly"
    )
    twice = _rename(monkeypatch, once, "aiperf-nightly-again")

    with zipfile.ZipFile(twice) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        script = "aiperf_nightly_again-0.11.0.data/scripts/aiperf"
        assert script in names
        assert not any(n.startswith("aiperf-0.11.0.data/") for n in names)
        assert not any(n.startswith("aiperf_nightly-0.11.0.data/") for n in names)
        assert zf.getinfo(script).external_attr >> 16 & 0o111
