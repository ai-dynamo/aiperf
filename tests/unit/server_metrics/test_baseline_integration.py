# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ServerMetricsManager adopts BaselineCollectorMixin for the phase baseline handshake."""

from aiperf.common.enums import ServiceCapability, make_result_producer_capability
from aiperf.common.mixins.baseline_collector_mixin import BaselineCollectorMixin
from aiperf.server_metrics.manager import ServerMetricsManager


def test_server_metrics_uses_mixin() -> None:
    assert issubclass(ServerMetricsManager, BaselineCollectorMixin)


def test_server_metrics_advertises_baseline_capability() -> None:
    assert (
        ServiceCapability.BASELINE_COLLECTOR in ServerMetricsManager.extra_capabilities
    )


def test_server_metrics_advertises_server_metrics_result_producer() -> None:
    assert (
        make_result_producer_capability("server_metrics")
        in ServerMetricsManager.extra_capabilities
    )


def test_server_metrics_implements_collect_baseline() -> None:
    assert "collect_baseline" in ServerMetricsManager.__dict__
