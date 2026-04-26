# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from tools.backfill_sweep_markers import backfill_sweep_markers


def test_backfill_writes_marker_for_each_child(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "child_runs": [
                    {
                        "namespace": "bench",
                        "name": "ch-0-0",
                        "variation_index": 0,
                        "variation_label": "concurrency-8",
                        "trial_index": 0,
                    },
                    {
                        "namespace": "bench",
                        "name": "ch-1-0",
                        "variation_index": 1,
                        "variation_label": "concurrency-32",
                        "trial_index": 0,
                    },
                ],
            }
        )
    )
    (tmp_path / "bench" / "ch-0-0").mkdir(parents=True)
    (tmp_path / "bench" / "ch-1-0").mkdir(parents=True)

    backfill_sweep_markers(tmp_path)

    m0 = json.loads((tmp_path / "bench" / "ch-0-0" / "sweep.json").read_text())
    m1 = json.loads((tmp_path / "bench" / "ch-1-0" / "sweep.json").read_text())
    assert m0["sweep_name"] == "s1"
    assert m0["variation_index"] == 0
    assert m1["variation_index"] == 1


def test_backfill_skips_children_without_results_dir(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "child_runs": [
                    {
                        "namespace": "bench",
                        "name": "ch-only-on-disk",
                        "variation_index": 0,
                        "variation_label": "x",
                        "trial_index": 0,
                    },
                    {
                        "namespace": "bench",
                        "name": "ch-no-disk",
                        "variation_index": 1,
                        "variation_label": "y",
                        "trial_index": 0,
                    },
                ],
            }
        )
    )
    (tmp_path / "bench" / "ch-only-on-disk").mkdir(parents=True)
    backfill_sweep_markers(tmp_path)
    assert (tmp_path / "bench" / "ch-only-on-disk" / "sweep.json").is_file()
    assert not (tmp_path / "bench" / "ch-no-disk").exists()
