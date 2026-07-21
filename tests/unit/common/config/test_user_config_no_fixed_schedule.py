# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace-delay flags on the v2 dataset configs (``--ignore-trace-delays`` /
``--use-think-time-only``) and their mutual-exclusivity validator.

REBASED from the v1 ``UserConfig`` / ``InputConfig`` ``--no-fixed-schedule``
suite. On v1 these were ``InputConfig`` fields plus the
``UserConfig._should_use_fixed_schedule_for_trace_dataset()`` auto-detection
method. On v2:

- the ``--no-fixed-schedule`` / auto-promotion-to-fixed-schedule logic moved
  into the CLI->YAML converter ``build_profiling()`` and is ALREADY covered by
  ``tests/unit/common/config/test_no_fixed_schedule.py`` (TestTraceAutoPromotion
  + TestSweepIncompatibleWithFixedSchedule). Those v1 tests are DROPPED here
  (dup) -- see that sibling file.
- the ``ignore_trace_delays`` / ``use_think_time_only`` fields and their
  "cannot be used together" mutex moved onto the v2 ``FileDataset`` /
  ``PublicDataset`` configs (``aiperf.config.dataset.config``). Those are the
  NET-NEW tests below.

DROPPED v1 tests (no v2 home -- v1 UserConfig method removed):
- TestDisableAutoFixedSchedule.test_disable_auto_skips_auto_detection_*
- TestDisableAutoFixedSchedule.test_default_keeps_auto_detection
- TestDisableAutoFixedSchedule.test_disable_auto_resolves_to_non_fixed_timing_mode
- TestDisableAutoFixedSchedule.test_explicit_fixed_schedule_with_disable_auto_raises
  All relied on ``UserConfig._should_use_fixed_schedule_for_trace_dataset()`` /
  ``InputConfig.disable_auto_fixed_schedule`` + ``fixed_schedule``, which v2
  reorganized into ``build_profiling()`` (covered by test_no_fixed_schedule.py).
"""

from __future__ import annotations

import pytest

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

    def test_default_false_file_dataset(self) -> None:
        assert _file_dataset().ignore_trace_delays is False

    def test_can_be_enabled_file_dataset(self) -> None:
        assert _file_dataset(ignore_trace_delays=True).ignore_trace_delays is True

    def test_default_false_public_dataset(self) -> None:
        assert _public_dataset().ignore_trace_delays is False

    def test_can_be_enabled_public_dataset(self) -> None:
        assert _public_dataset(ignore_trace_delays=True).ignore_trace_delays is True


class TestUseThinkTimeOnlyField:
    """``--use-think-time-only`` is settable on the dataset configs, defaults False."""

    def test_default_false_file_dataset(self) -> None:
        assert _file_dataset().use_think_time_only is False

    def test_can_be_enabled_file_dataset(self) -> None:
        assert _file_dataset(use_think_time_only=True).use_think_time_only is True

    def test_default_false_public_dataset(self) -> None:
        assert _public_dataset().use_think_time_only is False

    def test_can_be_enabled_public_dataset(self) -> None:
        assert _public_dataset(use_think_time_only=True).use_think_time_only is True


class TestTraceDelayMutex:
    """``--ignore-trace-delays`` and ``--use-think-time-only`` are mutually exclusive."""

    def test_mutex_file_dataset(self) -> None:
        with pytest.raises(ValueError, match="cannot be used together"):
            _file_dataset(ignore_trace_delays=True, use_think_time_only=True)

    def test_mutex_public_dataset(self) -> None:
        with pytest.raises(ValueError, match="cannot be used together"):
            _public_dataset(ignore_trace_delays=True, use_think_time_only=True)
