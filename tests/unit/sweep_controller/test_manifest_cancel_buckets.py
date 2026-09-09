# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""manifest.json and the parent aggregate must agree on cancelled children.

_write_aggregate_manifest counted every non-success as failed while
_write_sweep_parent_aggregate, twenty lines below, used the three-way
succeeded/failed/cancelled split. Two artifacts generated from the same result
list disagreed, and the manifest also contradicted the live rollup.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson

from aiperf.sweep_controller.main import _child_status, _write_aggregate_manifest


def _result(
    label: str, *, success: bool, error: str = "", was_cancelled: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        success=success,
        error=error,
        artifacts_path=None,
        was_cancelled=was_cancelled,
    )


def _manifest(tmp_path: Path, results: list) -> dict:
    _write_aggregate_manifest(
        tmp_path,
        {"metadata": {"name": "s", "namespace": "ns", "uid": "u"}, "status": {}},
        results,
        SimpleNamespace(configs=[1, 2, 3]),
    )
    return orjson.loads((tmp_path / "manifest.json").read_bytes())


def test_cancelled_child_is_not_counted_as_failed(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            _result("v0", success=True),
            _result("v1", success=False, error="boom"),
            _result("v2", success=False, error="cancelled", was_cancelled=True),
        ],
    )
    assert manifest["completed_runs"] == 1
    assert manifest["failed_runs"] == 1
    assert manifest["cancelled_runs"] == 1


def test_child_status_strings_match_the_buckets(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            _result("v0", success=True),
            _result("v1", success=False, error="boom"),
            _result("v2", success=False, error="cancelled", was_cancelled=True),
        ],
    )
    assert [c["status"] for c in manifest["child_runs"]] == [
        "Succeeded",
        "Failed",
        "Cancelled",
    ]


def test_child_status_helper() -> None:
    assert _child_status(_result("a", success=True)) == "Succeeded"
    assert _child_status(_result("b", success=False, error="x")) == "Failed"
    assert (
        _child_status(_result("c", success=False, error="x", was_cancelled=True))
        == "Cancelled"
    )
