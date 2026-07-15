# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson


def test_aggregator_writes_aggregate_json(tmp_path: Path) -> None:
    from aiperf.sweep_controller.aggregator import write_sweep_aggregate

    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="saturation-sweep",
        sweep_run_epoch="1714069323",
        doc={
            "phase": "Succeeded",
            "totalVariations": 2,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [8, 32]}],
            },
            "model": "m",
            "per_cell_aggregates": [],
            "child_runs": [],
        },
        conditions=[{"type": "Done", "status": "True"}],
    )
    epoch_dir = tmp_path / "bench" / "sweeps" / "saturation-sweep" / "1714069323"
    assert (epoch_dir / "aggregate.json").is_file()
    assert (epoch_dir / "conditions.json").is_file()
    doc = orjson.loads((epoch_dir / "aggregate.json").read_bytes())
    assert doc["phase"] == "Succeeded"
    cond = orjson.loads((epoch_dir / "conditions.json").read_bytes())
    assert cond == {"conditions": [{"type": "Done", "status": "True"}]}


def test_aggregator_skips_conditions_when_none(tmp_path: Path) -> None:
    from aiperf.sweep_controller.aggregator import write_sweep_aggregate

    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded"},
        conditions=None,
    )
    epoch_dir = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    assert (epoch_dir / "aggregate.json").is_file()
    assert not (epoch_dir / "conditions.json").exists()


def test_aggregator_atomic_no_tmp_leftover(tmp_path: Path) -> None:
    """Successful write leaves no .tmp sibling files in the sweep dir."""
    from aiperf.sweep_controller.aggregator import write_sweep_aggregate

    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded"},
        conditions=[{"type": "Done", "status": "True"}],
    )
    epoch_dir = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    leftovers = [p.name for p in epoch_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_aggregator_overwrites_existing(tmp_path: Path) -> None:
    """Re-calling the writer replaces the existing aggregate.json atomically."""
    from aiperf.sweep_controller.aggregator import write_sweep_aggregate

    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Running"},
    )
    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="s1",
        sweep_run_epoch="1714069323",
        doc={"phase": "Succeeded"},
    )
    epoch_dir = tmp_path / "bench" / "sweeps" / "s1" / "1714069323"
    doc = orjson.loads((epoch_dir / "aggregate.json").read_bytes())
    assert doc["phase"] == "Succeeded"
