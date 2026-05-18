# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the datasets list shape.

The runtime currently loads exactly one dataset, so the schema enforces a
single-element list. The list shape exists to share the schema between YAML
and the AIPerfSweep CRD (which can't union string|object fields cleanly).
"""

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


def test_datasets_accepts_single_entry_list():
    cfg = _cfg(
        [
            {"name": "main", "type": "synthetic", "prompts": {"isl": {"mean": 128}}},
        ]
    )
    assert isinstance(cfg.datasets, list)
    assert [d.name for d in cfg.datasets] == ["main"]


def test_datasets_default_dataset_is_the_only_dataset():
    cfg = _cfg([{"name": "primary", "type": "synthetic"}])
    assert cfg.get_default_dataset_name() == "primary"


def test_datasets_rejects_dict_shape():
    with pytest.raises(ValueError, match="datasets must be a list"):
        _cfg({"main": {"type": "synthetic"}})


def test_datasets_rejects_missing_name():
    with pytest.raises(ValueError, match="name"):
        _cfg([{"type": "synthetic"}])


def test_datasets_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 1 item"):
        _cfg([])


def test_datasets_rejects_more_than_one_entry():
    """Runtime only loads the first dataset; reject multi-dataset configs."""
    with pytest.raises(ValueError, match="at most 1 item"):
        _cfg(
            [
                {"name": "a", "type": "synthetic"},
                {"name": "b", "type": "synthetic"},
            ]
        )


def test_public_dataset_uses_dataset_field_not_name():
    """PublicDataset.name was renamed to .dataset to free up `name` for the outer identifier."""
    cfg = _cfg(
        [
            {"name": "my_public", "type": "public", "dataset": "sharegpt"},
        ]
    )
    assert cfg.datasets[0].name == "my_public"
    assert cfg.datasets[0].dataset == "sharegpt"
