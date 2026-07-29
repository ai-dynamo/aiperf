# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tools/wheel_repack.py``."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_wheel_repack():
    """Import tools/wheel_repack.py by path (tools/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        "wheel_repack", _REPO_ROOT / "tools" / "wheel_repack.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wheel_repack = _load_wheel_repack()


def test_manylinux_tag_formats_glibc_floor():
    assert wheel_repack.manylinux_tag((2, 39), "x86_64") == "manylinux_2_39_x86_64"
    assert wheel_repack.manylinux_tag((2, 34), "aarch64") == "manylinux_2_34_aarch64"


@pytest.mark.skipif(sys.platform != "linux", reason="ELF scan is Linux-only")
def test_glibc_versions_reads_a_real_elf():
    # The running interpreter is always a glibc-linked ELF on Linux and always
    # needs at least GLIBC_2.2.5 (the base version on x86-64/aarch64).
    versions = wheel_repack.glibc_versions(Path(sys.executable))
    assert versions, "expected at least one GLIBC_ version need"
    assert max(versions) >= (2, 2)


@pytest.mark.skipif(sys.platform != "linux", reason="ELF scan is Linux-only")
def test_platform_tag_for_composes_floor_and_machine():
    tag = wheel_repack.platform_tag_for(Path(sys.executable))
    assert tag.startswith("manylinux_2_")
    assert tag.endswith(("_x86_64", "_aarch64"))


def test_glibc_versions_rejects_a_non_elf(tmp_path: Path):
    not_elf = tmp_path / "notelf.bin"
    not_elf.write_bytes(b"#!/bin/sh\necho hi\n")
    with pytest.raises(ValueError, match="not an ELF"):
        wheel_repack.glibc_versions(not_elf)


def test_rewrite_wheel_tag_replaces_every_tag_line():
    body = (
        "Wheel-Version: 1.0\n"
        "Generator: hatchling 1.27.0\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )
    out = wheel_repack.rewrite_wheel_tag(body, "py3-none-manylinux_2_39_x86_64")
    assert "Tag: py3-none-manylinux_2_39_x86_64\n" in out
    assert "py3-none-any" not in out
    assert "Root-Is-Purelib: false\n" in out
    assert "Root-Is-Purelib: true" not in out
    # Exactly one Tag line survives.
    assert sum(1 for line in out.splitlines() if line.startswith("Tag:")) == 1


def _make_stub_wheel(tmp_path: Path, name: str) -> Path:
    """A minimal hatchling-shaped wheel: one module, METADATA, WHEEL, RECORD."""
    wheel = tmp_path / name
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("aiperf/__init__.py", "__version__ = '0.11.0'\n")
        zf.writestr("aiperf-0.11.0.dist-info/METADATA", "Name: aiperf\n")
        zf.writestr(
            "aiperf-0.11.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: stub\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        zf.writestr(
            "aiperf-0.11.0.dist-info/RECORD",
            "aiperf/__init__.py,sha256=x,26\naiperf-0.11.0.dist-info/RECORD,,\n",
        )
    return wheel


@pytest.mark.skipif(sys.platform != "linux", reason="needs a real ELF to inject")
def test_repack_injects_binary_rewrites_tag_and_renames(tmp_path: Path):
    wheel = _make_stub_wheel(tmp_path, "aiperf-0.11.0-py3-none-any.whl")
    binary = Path(sys.executable)

    out = wheel_repack.repack(wheel, binary)

    expected_tag = f"py3-none-{wheel_repack.platform_tag_for(binary)}"
    assert out.name == f"aiperf-0.11.0-{expected_tag}.whl"
    assert not wheel.exists(), "the py3-none-any input should have been renamed away"

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        script = "aiperf-0.11.0.data/scripts/aiperf"
        assert script in names
        assert zf.read(script) == binary.read_bytes()

        wheel_text = zf.read("aiperf-0.11.0.dist-info/WHEEL").decode()
        assert f"Tag: {expected_tag}\n" in wheel_text
        assert "Root-Is-Purelib: false\n" in wheel_text

        record = zf.read("aiperf-0.11.0.dist-info/RECORD").decode()
        # Both rewritten entries are re-hashed; RECORD lists itself last (PEP 376).
        assert f"{script},sha256=" in record
        assert "aiperf-0.11.0.dist-info/WHEEL,sha256=" in record
        assert record.rstrip("\n").endswith("aiperf-0.11.0.dist-info/RECORD,,")
        # The script keeps its executable bit for pip's zip_item_is_executable.
        mode = zf.getinfo(script).external_attr >> 16
        assert mode & 0o111, f"script mode {mode:o} is not executable"


@pytest.mark.skipif(sys.platform != "linux", reason="needs a real ELF to inject")
def test_repack_is_idempotent(tmp_path: Path):
    wheel = _make_stub_wheel(tmp_path, "aiperf-0.11.0-py3-none-any.whl")
    binary = Path(sys.executable)

    first = wheel_repack.repack(wheel, binary)
    second = wheel_repack.repack(first, binary)

    assert second == first
    with zipfile.ZipFile(second) as zf:
        script = "aiperf-0.11.0.data/scripts/aiperf"
        assert zf.namelist().count(script) == 1
        record = zf.read("aiperf-0.11.0.dist-info/RECORD").decode()
        assert record.count(script) == 1
        assert record.count("dist-info/WHEEL") == 1
