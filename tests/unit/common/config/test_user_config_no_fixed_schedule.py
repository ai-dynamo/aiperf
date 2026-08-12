# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace-delay flags (``--ignore-trace-delays`` / ``--use-think-time-only``) on the v2 dataset configs and their mutual-exclusivity validator."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.dataset.config import FileDataset, PublicDataset


def _file_dataset(**overrides) -> FileDataset:
    base = {"name": "main", "type": "file", "path": "/fake/trace.jsonl"}
    base.update(overrides)
    return FileDataset(**base)


def _public_dataset(**overrides) -> PublicDataset:
    base = {"name": "main", "type": "public", "dataset": "sharegpt"}
    base.update(overrides)
    return PublicDataset(**base)


class TestIgnoreTraceDelaysField:
    """``--ignore-trace-delays`` is settable on the dataset configs, defaults False."""

    @pytest.mark.parametrize(
        "factory, enabled, expected",
        [
            param(_file_dataset, False, False, id="default_false_file_dataset"),
            param(_file_dataset, True, True, id="can_be_enabled_file_dataset"),
            param(_public_dataset, False, False, id="default_false_public_dataset"),
            param(_public_dataset, True, True, id="can_be_enabled_public_dataset"),
        ],
    )  # fmt: skip
    def test_ignore_trace_delays_field(self, factory, enabled, expected) -> None:
        cfg = factory(ignore_trace_delays=enabled) if enabled else factory()
        assert cfg.ignore_trace_delays is expected


class TestUseThinkTimeOnlyField:
    """``--use-think-time-only`` is settable on the dataset configs, defaults False."""

    @pytest.mark.parametrize(
        "factory, enabled, expected",
        [
            param(_file_dataset, False, False, id="default_false_file_dataset"),
            param(_file_dataset, True, True, id="can_be_enabled_file_dataset"),
            param(_public_dataset, False, False, id="default_false_public_dataset"),
            param(_public_dataset, True, True, id="can_be_enabled_public_dataset"),
        ],
    )  # fmt: skip
    def test_use_think_time_only_field(self, factory, enabled, expected) -> None:
        cfg = factory(use_think_time_only=enabled) if enabled else factory()
        assert cfg.use_think_time_only is expected


class TestTraceDelayMutex:
    """``--ignore-trace-delays`` and ``--use-think-time-only`` are mutually exclusive."""

    @pytest.mark.parametrize(
        "factory",
        [
            param(_file_dataset, id="file_dataset"),
            param(_public_dataset, id="public_dataset"),
        ],
    )  # fmt: skip
    def test_mutex(self, factory) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            factory(ignore_trace_delays=True, use_think_time_only=True)
