# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.environment import Environment
from aiperf.plugin.enums import UIType


@pytest.fixture(autouse=True)
def _reset_interval(monkeypatch):
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    yield


def test_resolver_dashboard_unset_returns_5() -> None:
    assert Environment.UI.realtime_metrics_interval(UIType.DASHBOARD) == 5.0


def test_resolver_simple_unset_returns_30() -> None:
    assert Environment.UI.realtime_metrics_interval(UIType.SIMPLE) == 30.0


def test_resolver_none_unset_returns_30() -> None:
    assert Environment.UI.realtime_metrics_interval(UIType.NONE) == 30.0


def test_resolver_explicit_value_wins_over_dashboard_default(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 12.0)
    assert Environment.UI.realtime_metrics_interval(UIType.DASHBOARD) == 12.0


def test_resolver_explicit_value_wins_over_non_dashboard_default(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 12.0)
    assert Environment.UI.realtime_metrics_interval(UIType.NONE) == 12.0


def test_resolver_zero_is_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 0.0)
    assert Environment.UI.realtime_metrics_interval(UIType.DASHBOARD) == 0.0


def test_service_config_stats_interval_writes_through_env(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    ServiceConfig(stats_interval=7.0)  # type: ignore[call-arg]
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 7.0
    assert Environment.UI.realtime_metrics_interval(UIType.DASHBOARD) == 7.0
    assert Environment.UI.realtime_metrics_interval(UIType.NONE) == 7.0


def test_service_config_stats_interval_zero_writes_through_env(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    ServiceConfig(stats_interval=0.0)  # type: ignore[call-arg]
    assert Environment.UI.REALTIME_METRICS_INTERVAL == 0.0


def test_service_config_unset_stats_interval_leaves_env_alone(monkeypatch) -> None:
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", None)
    ServiceConfig()  # type: ignore[call-arg]
    assert Environment.UI.REALTIME_METRICS_INTERVAL is None
