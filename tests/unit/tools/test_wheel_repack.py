# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tools/wheel_repack.py``."""

from __future__ import annotations

import importlib.util
import sys
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
