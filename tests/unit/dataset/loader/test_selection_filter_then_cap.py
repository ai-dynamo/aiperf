# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-only filter-then-cap selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest
from pytest import param

from aiperf.dataset.loader.selection import SelectionStats, filter_then_cap
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from tests.unit.dataset.loader.conftest import make_weka_run

_MAX_CTX = 1000
_NUM_ENTRIES = 10
_OVER_LIMIT_COUNT = 8
_TOTAL = 20
_OVER_INPUT = 2000
_UNDER_INPUT = 100
_OUTPUT = 10


def _is_over_limit(index: int) -> bool:
    return index < _OVER_LIMIT_COUNT


def test_filter_then_cap_skips_rejects_until_n_eligible() -> None:
    """Eight over-limit entries first, then eligibles; cap=10 keeps 10."""
    candidates = (
        (
            f"t{i:02d}",
            _OVER_INPUT + _OUTPUT if _is_over_limit(i) else _UNDER_INPUT + _OUTPUT,
        )
        for i in range(_TOTAL)
    )
    kept, stats = filter_then_cap(
        candidates,
        num_dataset_entries=_NUM_ENTRIES,
        max_context_length=_MAX_CTX,
    )
    assert kept == [
        f"t{i:02d}" for i in range(_OVER_LIMIT_COUNT, _OVER_LIMIT_COUNT + _NUM_ENTRIES)
    ]
    assert stats.rejected_by_maxctx == _OVER_LIMIT_COUNT
    assert stats.eligible == _NUM_ENTRIES
    assert stats.loaded == _NUM_ENTRIES
    assert stats.largest_observed == _OVER_INPUT + _OUTPUT


def test_filter_then_cap_both_none_keeps_scan_order() -> None:
    candidates = ((i, 1) for i in range(5))
    kept, stats = filter_then_cap(
        candidates, num_dataset_entries=None, max_context_length=None
    )
    assert kept == [0, 1, 2, 3, 4]
    assert stats == SelectionStats(scanned=5, eligible=5, loaded=5, largest_observed=1)


def _weka_trace_dict(index: int) -> dict:
    input_length = _OVER_INPUT if _is_over_limit(index) else _UNDER_INPUT
    return {
        "id": f"trace-{index:02d}",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "m",
                "in": input_length,
                "out": _OUTPUT,
                "hash_ids": [1, 2],
            }
        ],
    }


def _write_weka_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(_TOTAL):
        (root / f"t{index:02d}.json").write_bytes(orjson.dumps(_weka_trace_dict(index)))
    return root


def _make_loader(
    filename: Path, run, monkeypatch: pytest.MonkeyPatch
) -> WekaTraceLoader:
    loader = WekaTraceLoader(filename=str(filename), run=run)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"p-{r.key}" for r in rs},
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    return loader


@pytest.mark.parametrize(
    "num_entries,max_ctx,expected_ids",
    [
        param(
            _NUM_ENTRIES,
            _MAX_CTX,
            [
                f"trace-{i:02d}"
                for i in range(_OVER_LIMIT_COUNT, _OVER_LIMIT_COUNT + _NUM_ENTRIES)
            ],
            id="filter_then_cap_n_eligible",
        ),
        param(
            None,
            None,
            [f"trace-{i:02d}" for i in range(_TOTAL)],
            id="both_none_loads_all",
        ),
    ],
)  # fmt: skip
def test_weka_file_loader_filter_then_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_entries: int | None,
    max_ctx: int | None,
    expected_ids: list[str],
) -> None:
    root = _write_weka_dir(tmp_path / "weka")
    run = make_weka_run(
        model_names=["m"],
        tokenizer_name="t",
        max_context_length=max_ctx,
        entries=num_entries,
    )
    loader = _make_loader(root, run, monkeypatch)
    data = loader.load_dataset()
    if num_entries is not None or max_ctx is not None:
        data = loader._select_traces_filter_then_cap(
            data,
            num_dataset_entries=num_entries,
            max_context_length=max_ctx,
        )
    assert list(data.keys()) == expected_ids
