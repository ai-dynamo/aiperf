# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the list-of-named-datasets schema (post-refactor)."""

from __future__ import annotations

import pytest

from aiperf.config.config import BenchmarkConfig

_BASE: dict = {
    "models": "mock",
    "endpoint": {"urls": ["http://x:8000/v1/chat/completions"], "streaming": True},
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


def _cfg(datasets):
    return BenchmarkConfig.model_validate({**_BASE, "datasets": datasets})


def test_datasets_accepts_list_with_name_field():
    cfg = _cfg(
        [
            {"name": "main", "type": "synthetic", "prompts": {"isl": {"mean": 128}}},
            {"name": "eval", "type": "synthetic", "prompts": {"isl": {"mean": 64}}},
        ]
    )
    assert isinstance(cfg.datasets, list)
    assert [d.name for d in cfg.datasets] == ["main", "eval"]


def test_datasets_preserves_input_order():
    cfg = _cfg(
        [
            {"name": "zebra", "type": "synthetic"},
            {"name": "alpha", "type": "synthetic"},
        ]
    )
    # Insertion order — alphabetization would invert these.
    assert cfg.datasets[0].name == "zebra"
    assert cfg.datasets[1].name == "alpha"


def test_datasets_default_dataset_is_first_in_list():
    cfg = _cfg(
        [
            {"name": "primary", "type": "synthetic"},
            {"name": "fallback", "type": "synthetic"},
        ]
    )
    assert cfg.get_default_dataset_name() == "primary"


def test_datasets_rejects_dict_shape():
    with pytest.raises(ValueError, match="datasets must be a list"):
        _cfg({"main": {"type": "synthetic"}})


def test_datasets_rejects_missing_name():
    with pytest.raises(ValueError, match="name"):
        _cfg([{"type": "synthetic"}])


def test_datasets_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate dataset name"):
        _cfg(
            [
                {"name": "d", "type": "synthetic"},
                {"name": "d", "type": "synthetic"},
            ]
        )


def test_datasets_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 1 item"):
        _cfg([])


def test_phase_dataset_reference_resolves_by_name():
    """A phase referencing dataset='eval' must resolve to the eval entry by name lookup."""
    cfg = BenchmarkConfig.model_validate(
        {
            **{k: v for k, v in _BASE.items() if k != "phases"},
            "datasets": [
                {"name": "main", "type": "synthetic"},
                {"name": "eval", "type": "synthetic"},
            ],
            "phases": [
                {
                    "name": "p",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                    "dataset": "eval",
                }
            ],
        }
    )
    assert cfg.phases[0].dataset == "eval"


def test_public_dataset_uses_dataset_field_not_name():
    """PublicDataset.name was renamed to .dataset to free up `name` for the outer identifier."""
    cfg = _cfg(
        [
            {"name": "my_public", "type": "public", "dataset": "sharegpt"},
        ]
    )
    assert cfg.datasets[0].name == "my_public"
    assert cfg.datasets[0].dataset == "sharegpt"
