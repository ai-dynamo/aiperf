# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weka-backed public datasets must declare that they carry timing data.

These corpora are replayed by ``WekaTraceLoader``, which reads a per-request
timestamp out of ``requests[].t``. The resolver cannot see that by probing the
first record, so it relies on ``metadata.has_timing_data``. Without the flag,
``--fixed-schedule`` is rejected outright on every one of them.

Scope note: this is purely permissive. Auto-promotion is gated on
``cli.input_file``, which ``--public-dataset`` never sets, so these corpora
still require an explicit ``--fixed-schedule`` -- the flag unblocks that, it
does not make them auto-promote.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

PLUGINS = pathlib.Path(__file__).resolve().parents[3] / "src/aiperf/plugin/plugins.yaml"


def _public_dataset_loaders() -> dict:
    return yaml.safe_load(PLUGINS.read_text())["public_dataset_loader"]


def _weka_entries() -> list[str]:
    return sorted(n for n in _public_dataset_loaders() if "weka" in n)


def test_there_are_weka_public_datasets_to_check() -> None:
    """Guards the parametrize below from silently covering nothing."""
    assert len(_weka_entries()) >= 10


@pytest.mark.parametrize("name", _weka_entries())
def test_weka_public_dataset_declares_timing_data(name: str) -> None:
    entry = _public_dataset_loaders()[name] or {}
    metadata = entry.get("metadata") or {}
    assert metadata.get("has_timing_data") is True, (
        f"public dataset '{name}' is replayed by WekaTraceLoader, which emits "
        "per-request timestamps, but does not declare has_timing_data, so "
        "--fixed-schedule is rejected for it at config resolution."
    )
