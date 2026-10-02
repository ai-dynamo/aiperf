# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aiperf.sweep_controller.aggregator import (
    write_children_manifest,
    write_sweep_aggregate,
)


def test_write_sweep_aggregate_writes_under_epoch(tmp_path: Path) -> None:
    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded", "totalVariations": 0},
        conditions=[{"type": "Done", "status": "True"}],
    )
    epoch_dir = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    assert (epoch_dir / "aggregate.json").is_file()
    assert (epoch_dir / "conditions.json").is_file()


def test_write_sweep_aggregate_updates_latest_pointer(tmp_path: Path) -> None:
    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded"},
        conditions=None,
    )
    p = tmp_path / "bench" / "sweeps" / "s1" / "latest.txt"
    assert p.read_text().strip() == "1714069323"


def test_write_children_manifest_atomic(tmp_path: Path) -> None:
    write_children_manifest(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        children=[
            {
                "namespace": "bench",
                "name": "s1-v00-t0",
                "variation_index": 0,
                "variation_label": "concurrency-1",
                "trial_index": 0,
                "child_run_epoch": "1714069324",
            },
        ],
    )
    p = tmp_path / "bench" / "sweeps" / "s1" / "1714069323" / "children.json"
    doc = json.loads(p.read_text())
    assert doc["sweep_run_epoch"] == "1714069323"
    assert len(doc["children"]) == 1
    assert doc["children"][0]["child_run_epoch"] == "1714069324"
