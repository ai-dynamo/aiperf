# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aiperf.cli_runner._aggregation_dispatch import aggregate_plan_results


class _Strategy:
    def get_aggregate_path(self, base_dir: Path) -> Path:
        return base_dir / "aggregate"


@pytest.fixture
def aggregation_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, int]:
    from aiperf.cli_runner import _aggregate, _sweep_aggregate

    calls = {"single": 0, "per_variation": 0, "sweep": 0}
    sweep_dir = tmp_path / "sweep_aggregate"

    async def _single(*_args: Any, **_kwargs: Any) -> None:
        calls["single"] += 1

    async def _per_variation(*_args: Any, **_kwargs: Any) -> list[Path]:
        calls["per_variation"] += 1
        return []

    async def _sweep(*_args: Any, **_kwargs: Any) -> Path:
        calls["sweep"] += 1
        return sweep_dir

    monkeypatch.setattr(_aggregate, "aggregate_and_export", _single)
    monkeypatch.setattr(
        _sweep_aggregate, "aggregate_per_variation_and_export", _per_variation
    )
    monkeypatch.setattr(_sweep_aggregate, "aggregate_sweep_and_export", _sweep)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_sweep", "is_adaptive_search"),
    [
        pytest.param(True, False, id="grid-sweep"),
        pytest.param(False, True, id="adaptive-search"),
    ],
)  # fmt: skip
async def test_aggregate_plan_results_uses_sweep_stack_for_multi_config_plans(
    tmp_path: Path,
    aggregation_calls: dict[str, int],
    is_sweep: bool,
    is_adaptive_search: bool,
) -> None:
    result_dir = await aggregate_plan_results(
        [],
        SimpleNamespace(
            is_sweep=is_sweep,
            is_adaptive_search=is_adaptive_search,
        ),
        strategy=_Strategy(),
        base_dir=tmp_path,
        logger=SimpleNamespace(),
    )

    assert aggregation_calls == {"single": 0, "per_variation": 1, "sweep": 1}
    assert result_dir == tmp_path / "sweep_aggregate"


@pytest.mark.asyncio
async def test_aggregate_plan_results_uses_confidence_stack_for_single_config(
    tmp_path: Path, aggregation_calls: dict[str, int]
) -> None:
    result_dir = await aggregate_plan_results(
        [],
        SimpleNamespace(is_sweep=False, is_adaptive_search=False),
        strategy=_Strategy(),
        base_dir=tmp_path,
        logger=SimpleNamespace(),
    )

    assert aggregation_calls == {"single": 1, "per_variation": 0, "sweep": 0}
    assert result_dir == tmp_path / "aggregate"


@pytest.mark.asyncio
async def test_aggregate_plan_results_falls_back_when_sweep_export_has_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    aggregation_calls: dict[str, int],
) -> None:
    from aiperf.cli_runner import _sweep_aggregate

    async def _no_output(*_args: Any, **_kwargs: Any) -> None:
        aggregation_calls["sweep"] += 1

    monkeypatch.setattr(_sweep_aggregate, "aggregate_sweep_and_export", _no_output)

    result_dir = await aggregate_plan_results(
        [],
        SimpleNamespace(is_sweep=True, is_adaptive_search=False),
        strategy=_Strategy(),
        base_dir=tmp_path,
        logger=SimpleNamespace(),
    )

    assert result_dir == tmp_path / "aggregate"
