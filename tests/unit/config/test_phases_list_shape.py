# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the list-of-named-phases schema (post-refactor)."""

from __future__ import annotations

import pytest

from aiperf.config.config import BenchmarkConfig

_BASE: dict = {
    "models": "mock",
    "endpoint": {"urls": ["http://x:8000/v1/chat/completions"], "streaming": True},
    "datasets": [{"name": "main", "type": "synthetic"}],
}


def _cfg(phases):
    return BenchmarkConfig.model_validate({**_BASE, "phases": phases})


def test_phases_accepts_list_with_name_field():
    cfg = _cfg(
        [
            {
                "name": "warmup",
                "type": "concurrency",
                "requests": 10,
                "concurrency": 2,
                "exclude_from_results": True,
            },
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 100,
                "concurrency": 4,
            },
        ]
    )
    assert isinstance(cfg.phases, list)
    assert [p.name for p in cfg.phases] == ["warmup", "profiling"]


def test_phases_preserves_input_order_warmup_first():
    cfg = _cfg(
        [
            {
                "name": "warmup",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
                "exclude_from_results": True,
            },
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            },
        ]
    )
    assert cfg.phases[0].name == "warmup"
    assert cfg.phases[1].name == "profiling"


def test_phases_preserves_input_order_profiling_first():
    cfg = _cfg(
        [
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            },
            {
                "name": "warmup",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
                "exclude_from_results": True,
            },
        ]
    )
    assert cfg.phases[0].name == "profiling"
    assert cfg.phases[1].name == "warmup"


def test_phases_rejects_dict_shape():
    with pytest.raises(ValueError, match="phases must be a list"):
        _cfg({"warmup": {"type": "concurrency", "requests": 1, "concurrency": 1}})


def test_phases_rejects_missing_name():
    with pytest.raises(ValueError, match="name"):
        _cfg([{"type": "concurrency", "requests": 1, "concurrency": 1}])


def test_phases_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate phase name"):
        _cfg(
            [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 2,
                },
            ]
        )


def test_phases_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 1 item"):
        _cfg([])
