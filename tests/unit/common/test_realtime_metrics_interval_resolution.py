# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realtime-metrics interval resolution.

Both halves of the surface:

* ``REALTIME_METRICS_INTERVAL`` defaults to ``None`` and is resolved per UI
  type via ``Environment.UI.realtime_metrics_interval(ui_type)`` (5s under
  ``--ui dashboard``, 30s otherwise).
* ``stats_interval`` lives on ``RuntimeConfig`` and is resolved per config
  via ``RuntimeConfig.realtime_metrics_interval(ui_type)`` — it never writes
  process globals (``Environment`` singleton or ``os.environ``), so one
  config's override cannot leak into a later config in the same process.
"""

import os

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.runtime import RuntimeConfig
from aiperf.plugin.enums import UIType


@pytest.fixture(autouse=True)
def _reset_interval(monkeypatch):
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    monkeypatch.delenv("AIPERF_UI_REALTIME_METRICS_INTERVAL", raising=False)


# ---------------------------------------------------------------------------
# Config-scoped resolution: RuntimeConfig.realtime_metrics_interval
# ---------------------------------------------------------------------------


def test_default_interval_is_unset() -> None:
    field = type(Environment.UI).model_fields["REALTIME_METRICS_INTERVAL"]
    assert field.default is None


@pytest.mark.parametrize("ui_type", [UIType.DASHBOARD, UIType.SIMPLE, UIType.NONE])
def test_runtime_stats_interval_set_wins_over_defaults(ui_type) -> None:
    cfg = RuntimeConfig(stats_interval=7.0)
    assert cfg.realtime_metrics_interval(ui_type) == 7.0


def test_runtime_stats_interval_zero_resolves_to_zero() -> None:
    cfg = RuntimeConfig(stats_interval=0.0)
    assert cfg.realtime_metrics_interval(UIType.DASHBOARD) == 0.0


def test_runtime_stats_interval_set_wins_over_environment(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 12.5)
    cfg = RuntimeConfig(stats_interval=7.0)
    assert cfg.realtime_metrics_interval(UIType.SIMPLE) == 7.0


def test_runtime_stats_interval_unset_falls_back_to_environment(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 12.5)
    cfg = RuntimeConfig()
    assert cfg.realtime_metrics_interval(UIType.SIMPLE) == 12.5


@pytest.mark.parametrize(
    "ui_type, expected",
    [
        param(UIType.DASHBOARD, 5.0, id="dashboard"),
        param(UIType.SIMPLE, 30.0, id="simple"),
        param(UIType.NONE, 30.0, id="none"),
    ],
)  # fmt: skip
def test_runtime_stats_interval_unset_uses_per_ui_default(ui_type, expected) -> None:
    cfg = RuntimeConfig()
    assert cfg.realtime_metrics_interval(ui_type) == expected


# ---------------------------------------------------------------------------
# Regression (PR #1102 review): an explicit stats_interval must not leak
# into a later config constructed in the same process, and must not mutate
# the Environment singleton or os.environ.
# ---------------------------------------------------------------------------


def test_stats_interval_does_not_leak_into_later_config() -> None:
    first = RuntimeConfig(stats_interval=1.0)
    second = RuntimeConfig()
    assert first.realtime_metrics_interval(UIType.SIMPLE) == 1.0
    assert second.realtime_metrics_interval(UIType.SIMPLE) == 30.0
    assert second.realtime_metrics_interval(UIType.DASHBOARD) == 5.0


def test_stats_interval_does_not_mutate_process_globals() -> None:
    RuntimeConfig(stats_interval=1.0)
    assert Environment.UI.REALTIME_METRICS_INTERVAL is None
    assert "AIPERF_UI_REALTIME_METRICS_INTERVAL" not in os.environ
    assert Environment.UI.realtime_metrics_interval(UIType.SIMPLE) == 30.0


# ---------------------------------------------------------------------------
# Per-UIType resolver method on Environment.UI. With
# REALTIME_METRICS_INTERVAL unset (None), realtime_metrics_interval(ui_type)
# auto-defaults to 5s under --ui dashboard, 30s otherwise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ui_type, expected",
    [
        param(UIType.DASHBOARD, 5.0, id="dashboard"),
        param(UIType.SIMPLE, 30.0, id="simple"),
        param(UIType.NONE, 30.0, id="none"),
    ],
)  # fmt: skip
def test_per_ui_type_resolver_auto_default(ui_type, expected) -> None:
    assert Environment.UI.realtime_metrics_interval(ui_type) == expected


@pytest.mark.parametrize("ui_type", [UIType.DASHBOARD, UIType.SIMPLE, UIType.NONE])
def test_per_ui_type_resolver_explicit_value_wins(ui_type, monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 12.5)
    assert Environment.UI.realtime_metrics_interval(ui_type) == 12.5


def test_realtime_metrics_interval_description_uses_markdown_backticks() -> None:
    field = type(Environment.UI).model_fields["REALTIME_METRICS_INTERVAL"]
    assert "`realtime_metrics_interval(ui_type)`" in field.description
    assert "``realtime_metrics_interval(ui_type)``" not in field.description
