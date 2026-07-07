# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realtime-metrics interval resolution.

Ported from agentx onto v2, keeping both halves of the surface:

* ``REALTIME_METRICS_INTERVAL`` defaults to ``None`` and is resolved per UI
  type via ``Environment.UI.realtime_metrics_interval(ui_type)`` (5s under
  ``--ui dashboard``, 30s otherwise) — restored from agentx, G-C2.
* ``stats_interval`` lives on ``RuntimeConfig`` (main's #1035 surface) and
  writes through to ``Environment.UI.REALTIME_METRICS_INTERVAL`` when set,
  overriding the per-UIType auto-default.
"""

import os

import pytest

from aiperf.common.environment import Environment, _UISettings
from aiperf.config.runtime import RuntimeConfig
from aiperf.plugin.enums import UIType


@pytest.fixture(autouse=True)
def _reset_interval(monkeypatch):
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    monkeypatch.delenv("AIPERF_UI_REALTIME_METRICS_INTERVAL", raising=False)
    yield
    os.environ.pop("AIPERF_UI_REALTIME_METRICS_INTERVAL", None)


# ---------------------------------------------------------------------------
# v2 ported behavior: stats_interval write-through
# ---------------------------------------------------------------------------


def test_default_interval_is_unset() -> None:
    field = type(Environment.UI).model_fields["REALTIME_METRICS_INTERVAL"]
    assert field.default is None


def test_runtime_stats_interval_writes_through_env(monkeypatch) -> None:
    RuntimeConfig(stats_interval=7.0)
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 7.0


def test_runtime_stats_interval_zero_writes_through_env(monkeypatch) -> None:
    RuntimeConfig(stats_interval=0.0)
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 0.0


def test_runtime_unset_stats_interval_leaves_env_alone(monkeypatch) -> None:
    RuntimeConfig()
    assert Environment.UI.REALTIME_METRICS_INTERVAL is None


# ---------------------------------------------------------------------------
# stats_interval also writes os.environ so spawn children (which rebuild
# Environment from os.environ, and do not re-run model validators on unpickle)
# honor the flag.
# ---------------------------------------------------------------------------


def test_apply_stats_interval_writes_through_os_environ() -> None:
    RuntimeConfig(stats_interval=7.0)
    assert os.environ["AIPERF_UI_REALTIME_METRICS_INTERVAL"] == "7.0"


def test_apply_stats_interval_spawn_child_rebuild_honors_flag() -> None:
    RuntimeConfig(stats_interval=1.0)
    child = _UISettings()
    assert child.REALTIME_METRICS_INTERVAL == 1.0
    assert child.realtime_metrics_interval(UIType.DASHBOARD) == 1.0


def test_apply_stats_interval_zero_spawn_child_rebuild_disables_log_block() -> None:
    RuntimeConfig(stats_interval=0.0)
    child = _UISettings()
    assert child.realtime_metrics_interval(UIType.DASHBOARD) == 0.0


def test_apply_stats_interval_unset_leaves_os_environ_absent() -> None:
    RuntimeConfig()
    assert "AIPERF_UI_REALTIME_METRICS_INTERVAL" not in os.environ


# ---------------------------------------------------------------------------
# Per-UIType resolver method (restored from agentx, G-C2). With
# REALTIME_METRICS_INTERVAL unset (None), realtime_metrics_interval(ui_type)
# auto-defaults to 5s under --ui dashboard, 30s otherwise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ui_type, expected",
    [
        (UIType.DASHBOARD, 5.0),
        (UIType.SIMPLE, 30.0),
        (UIType.NONE, 30.0),
    ],
)
def test_per_ui_type_resolver_auto_default(ui_type, expected) -> None:
    assert Environment.UI.realtime_metrics_interval(ui_type) == expected


@pytest.mark.parametrize("ui_type", [UIType.DASHBOARD, UIType.SIMPLE, UIType.NONE])
def test_per_ui_type_resolver_explicit_value_wins(ui_type, monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 12.5)
    assert Environment.UI.realtime_metrics_interval(ui_type) == 12.5
