# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A warm mmap cache must not undo "fixed schedule wins over ignore-trace-delays".

The weka_trace loader lets fixed-schedule replay keep its timestamps even when
`--ignore-trace-delays` is set. That decision bakes into the cached Turn
timestamps, but the cache key used the raw flag -- and a plain `--fixed-schedule`
run sets no start/end offsets, so its key matched a non-fixed run with the same
flag exactly. Either direction serves the wrong mode from a warm entry: fixed
scheduling fails on stripped timestamps, or a non-fixed run silently keeps
delays it asked to drop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.environment import Environment
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset import mmap_cache
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep this module off the shared on-disk cache that other tests assert on."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", True)
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path / "mmap")


def _trace(tmp_path: Path) -> Path:
    path = tmp_path / "weka.jsonl"
    path.write_text('{"timestamp": 0, "input_length": 8, "output_length": 4}\n')
    return path


def _key(trace: Path, *, dataset_type, ignore: bool, fixed: bool) -> str | None:
    # Keep artifacts out of the repo-root `artifacts/` these runs would
    # otherwise resolve to, which is shared across xdist workers.
    kwargs: dict = {
        "artifact_directory": str(trace.parent / "artifacts"),
        "model_names": ["test-model"],
        "tokenizer_name": "test-tokenizer",
        "input_file": str(trace),
        "custom_dataset_type": dataset_type,
        "ignore_trace_delays": ignore,
    }
    if not fixed:
        kwargs["disable_auto_fixed_schedule"] = True
    return mmap_cache.compute_cache_key_from_run(make_run_from_cli(CLIConfig(**kwargs)))


def test_fixed_schedule_weka_does_not_share_a_key_with_non_fixed(tmp_path) -> None:
    """weka_trace auto-promotes to fixed schedule, so the contrast needs
    --disable-auto-fixed-schedule on the other side."""
    trace = _trace(tmp_path)
    fixed = _key(
        trace, dataset_type=CustomDatasetType.WEKA_TRACE, ignore=True, fixed=True
    )
    non_fixed = _key(
        trace, dataset_type=CustomDatasetType.WEKA_TRACE, ignore=True, fixed=False
    )
    assert fixed is not None and non_fixed is not None
    assert fixed != non_fixed


def test_fixed_schedule_weka_matches_the_mode_it_actually_runs(tmp_path) -> None:
    """With the flag suppressed, it must key like a run that never passed it."""
    trace = _trace(tmp_path)
    fixed_with_flag = _key(
        trace, dataset_type=CustomDatasetType.WEKA_TRACE, ignore=True, fixed=True
    )
    fixed_without_flag = _key(
        trace, dataset_type=CustomDatasetType.WEKA_TRACE, ignore=False, fixed=True
    )
    assert fixed_with_flag == fixed_without_flag


def test_non_weka_keys_are_unchanged(tmp_path) -> None:
    """Scoping matters: no other dataset's warm cache may be cold-started."""
    trace = _trace(tmp_path)
    a = _key(
        trace, dataset_type=CustomDatasetType.MOONCAKE_TRACE, ignore=True, fixed=False
    )
    b = _key(
        trace, dataset_type=CustomDatasetType.MOONCAKE_TRACE, ignore=True, fixed=True
    )
    assert a is not None and b is not None
    assert a == b
