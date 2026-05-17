# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPUTelemetryManager adopts BaselineCollectorMixin for the phase baseline handshake."""

from aiperf.common.enums import ServiceCapability, make_result_producer_capability
from aiperf.common.mixins.baseline_collector_mixin import BaselineCollectorMixin
from aiperf.gpu_telemetry.manager import GPUTelemetryManager


def test_gpu_telemetry_uses_mixin() -> None:
    assert issubclass(GPUTelemetryManager, BaselineCollectorMixin)


def test_gpu_telemetry_advertises_baseline_capability() -> None:
    assert (
        ServiceCapability.BASELINE_COLLECTOR in GPUTelemetryManager.extra_capabilities
    )


def test_gpu_telemetry_advertises_telemetry_result_producer() -> None:
    assert (
        make_result_producer_capability("telemetry")
        in GPUTelemetryManager.extra_capabilities
    )


def test_gpu_telemetry_implements_collect_baseline() -> None:
    """The class must override the abstract method (be instantiable in principle)."""
    assert "collect_baseline" in GPUTelemetryManager.__dict__
