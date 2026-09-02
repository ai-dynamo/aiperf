# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _write_aggregate(base: Path, ns: str, name: str, epoch: str, body: dict) -> None:
    d = base / ns / "sweeps" / name / epoch
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(body))


@pytest.mark.asyncio
async def test_find_any_sweep_epoch_specific(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union
    from aiperf.operator.results_layout import write_sweep_latest

    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        "1714069323",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
        },
    )
    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        "1714069400",
        {
            "phase": "Succeeded",
            "totalVariations": 8,
            "completedRuns": 8,
            "failedRuns": 0,
            "completedAt": "2026-04-26T01:00:00Z",
        },
    )
    write_sweep_latest(tmp_path, "bench", "s1", "1714069400")
    with patch(
        "aiperf.operator.sweep_union.find_aiperfsweep",
        AsyncMock(return_value=None),
    ):
        rec = await sweep_union.find_any_sweep(
            api=object(),
            base_dir=tmp_path,
            namespace="bench",
            name="s1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.total_variations == 4


@pytest.mark.asyncio
async def test_find_any_sweep_no_epoch_uses_latest(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union
    from aiperf.operator.results_layout import write_sweep_latest

    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        "1714069323",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
        },
    )
    write_sweep_latest(tmp_path, "bench", "s1", "1714069323")
    with patch(
        "aiperf.operator.sweep_union.find_aiperfsweep",
        AsyncMock(return_value=None),
    ):
        rec = await sweep_union.find_any_sweep(
            api=object(),
            base_dir=tmp_path,
            namespace="bench",
            name="s1",
        )
    assert rec is not None
    assert rec.total_variations == 4
