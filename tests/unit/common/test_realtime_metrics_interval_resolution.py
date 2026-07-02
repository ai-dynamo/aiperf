# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realtime-metrics interval resolution.

Rebased from agentx onto v2. On agentx the interval was resolved per UI type via
``Environment.UI.realtime_metrics_interval(ui_type)`` (5s dashboard / 30s otherwise,
``REALTIME_METRICS_INTERVAL`` defaulting to ``None``), and ``stats_interval`` lived
on ``ServiceConfig``. On v2 (main's #1035 surface, kept by the port):

* ``REALTIME_METRICS_INTERVAL`` is a plain field with a constant default of 5.0
  (no per-UIType resolver method).
* ``stats_interval`` lives on ``RuntimeConfig`` and writes through to
  ``Environment.UI.REALTIME_METRICS_INTERVAL``.

The write-through behavior IS ported and is validated below. The per-UIType
resolver method was NOT ported -- those tests are xfail-flagged as a port gap.
"""

import pytest

from aiperf.common.environment import Environment
from aiperf.config.runtime import RuntimeConfig
from aiperf.plugin.enums import UIType


@pytest.fixture(autouse=True)
def _reset_interval(monkeypatch):
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 5.0)
    yield


# ---------------------------------------------------------------------------
# v2 ported behavior: constant default + stats_interval write-through
# ---------------------------------------------------------------------------


def test_default_interval_is_5() -> None:
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 5.0


def test_runtime_stats_interval_writes_through_env(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 5.0)
    RuntimeConfig(stats_interval=7.0)
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 7.0


def test_runtime_stats_interval_zero_writes_through_env(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 5.0)
    RuntimeConfig(stats_interval=0.0)
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 0.0


def test_runtime_unset_stats_interval_leaves_env_alone(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 5.0)
    RuntimeConfig()
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 5.0


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
def test_per_ui_type_resolver_auto_default(ui_type, expected, monkeypatch) -> None:
    # Unset the explicit interval so the per-UIType auto-default fires.
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    assert Environment.UI.realtime_metrics_interval(ui_type) == expected
