# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aiperf.sweep_controller.k8s_executor import write_child_sweep_marker


def test_write_child_sweep_marker_creates_file(tmp_path: Path) -> None:
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="ch-0007-04",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        trial_index=4,
        sweep_run_epoch="1714069323",
        child_run_epoch="1714069323",
    )
    p = tmp_path / "bench" / "ch-0007-04" / "sweep.json"
    assert p.is_file()
    doc = json.loads(p.read_text())
    assert doc == {
        "sweep_name": "saturation-sweep",
        "variation_index": 7,
        "variation_label": "concurrency-128-rate-50",
        "trial_index": 4,
        "sweep_run_epoch": "1714069323",
        "child_run_epoch": "1714069323",
    }


def test_write_child_sweep_marker_is_atomic_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "bench" / "ch-0007-04" / "sweep.json"
    p.parent.mkdir(parents=True)
    p.write_text("stale content")
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="ch-0007-04",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        trial_index=4,
        sweep_run_epoch="1714069323",
        child_run_epoch="1714069323",
    )
    doc = json.loads(p.read_text())
    assert doc["sweep_name"] == "saturation-sweep"


def test_write_child_sweep_marker_no_trial_index(tmp_path: Path) -> None:
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="ch-0007",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        trial_index=None,
        sweep_run_epoch="1714069323",
        child_run_epoch="1714069323",
    )
    doc = json.loads((tmp_path / "bench" / "ch-0007" / "sweep.json").read_text())
    assert "trial_index" in doc
    assert doc["trial_index"] is None


def test_marker_payload_includes_child_run_epoch(tmp_path: Path) -> None:
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="satsweep-v07-t4",
        sweep_name="satsweep",
        variation_index=7,
        variation_label="concurrency-128",
        trial_index=4,
        sweep_run_epoch="1714069323",
        child_run_epoch="1714069324",
    )
    p = tmp_path / "bench" / "satsweep-v07-t4" / "sweep.json"
    doc = json.loads(p.read_text())
    assert doc["sweep_run_epoch"] == "1714069323"
    assert doc["child_run_epoch"] == "1714069324"
    assert doc["sweep_name"] == "satsweep"
