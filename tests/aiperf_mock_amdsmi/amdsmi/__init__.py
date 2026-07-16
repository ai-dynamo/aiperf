# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fake ``amdsmi`` bindings for testing AIPerf's AMD ROCm telemetry path.

This module masquerades as the real ``amdsmi`` Python bindings (which ship with
ROCm and require AMD hardware) so that ``aiperf.gpu_telemetry.AMDSMITelemetryCollector``
runs unchanged on non-AMD machines. It implements exactly the API surface that
collector consumes and reproduces the real-hardware quirks it defends against
(``'N/A'`` sentinels, EDGE-temperature exceptions, energy-field naming).

Dormancy gate: importing this module raises ``OSError`` unless
``AIPERF_MOCK_AMDSMI`` is truthy. This mirrors the real "wheel installed without a
working libamd_smi.so" failure that the collector already handles, so an
installed-but-unactivated fake degrades to "amdsmi unavailable" rather than
silently claiming AMD GPUs exist. Configure behavior with ``AIPERF_MOCK_AMDSMI_*``
(see the package README).
"""

import os
from typing import Any

from ._state import _Config, _Handle

__all__ = [
    "AmdSmiException",
    "AmdSmiLibraryException",
    "AmdSmiMemoryType",
    "AmdSmiTemperatureMetric",
    "AmdSmiTemperatureType",
    "amdsmi_get_energy_count",
    "amdsmi_get_gpu_activity",
    "amdsmi_get_gpu_board_info",
    "amdsmi_get_gpu_device_bdf",
    "amdsmi_get_gpu_device_uuid",
    "amdsmi_get_gpu_memory_usage",
    "amdsmi_get_gpu_metrics_info",
    "amdsmi_get_gpu_total_ecc_count",
    "amdsmi_get_power_info",
    "amdsmi_get_processor_handles",
    "amdsmi_get_temp_metric",
    "amdsmi_init",
    "amdsmi_shut_down",
]

_TRUTHY = {"1", "true", "yes", "on"}


def _activated() -> bool:
    return os.environ.get("AIPERF_MOCK_AMDSMI", "").strip().lower() in _TRUTHY


if not _activated():
    raise OSError(
        "aiperf-mock-amdsmi is dormant: set AIPERF_MOCK_AMDSMI=1 to activate the "
        "fake amdsmi bindings. (This mirrors a ROCm wheel installed without a "
        "working libamd_smi.so.)"
    )

# A modern (>= 26.x) binding version so the collector treats temperatures as
# Celsius (see AMDSMITelemetryCollector._amdsmi_returns_celsius).
__version__ = "26.2.1+aiperf-mock"

_NA = "N/A"

# Module-level handle set, established at amdsmi_init() and discarded at shutdown.
_handles: list[_Handle] = []
_initialized = False


class AmdSmiException(Exception):
    """Base exception type, mirroring ``amdsmi.AmdSmiException``."""


class AmdSmiLibraryException(AmdSmiException):
    """Library exception type, mirroring ``amdsmi.AmdSmiLibraryException``."""


class AmdSmiMemoryType:
    """Memory-type enum stand-in (only ``VRAM`` is consumed by the collector)."""

    VRAM = 0
    VIS_VRAM = 1
    GTT = 2


class AmdSmiTemperatureType:
    """Temperature-sensor enum stand-in."""

    EDGE = 0
    JUNCTION = 1
    HOTSPOT = 2
    VRAM = 3


class AmdSmiTemperatureMetric:
    """Temperature-metric enum stand-in (only ``CURRENT`` is consumed)."""

    CURRENT = 0


def amdsmi_init() -> None:
    """Initialize the fake, reading ``AIPERF_MOCK_AMDSMI_*`` config from the env."""
    global _initialized, _handles
    config = _Config.from_env()
    _handles = [_Handle(index=i, config=config) for i in range(config.num_gpus)]
    _initialized = True


def amdsmi_shut_down() -> None:
    """Tear down fake state."""
    global _initialized, _handles
    _initialized = False
    _handles = []


def amdsmi_get_processor_handles() -> list[_Handle]:
    """Return the enumerated GPU handles for this session."""
    return list(_handles)


def amdsmi_get_gpu_device_uuid(handle: _Handle) -> str:
    """Return a stable per-GPU UUID string."""
    return handle.uuid


def amdsmi_get_gpu_board_info(handle: _Handle) -> dict[str, Any]:
    """Return board info; only ``product_name`` is consumed by the collector."""
    return {"product_name": handle.config.spec.product_name}


def amdsmi_get_gpu_device_bdf(handle: _Handle) -> str:
    """Return the PCI bus/device/function string."""
    return handle.bdf


def amdsmi_get_power_info(handle: _Handle) -> dict[str, Any]:
    """Return power info. ``current_socket_power`` is in W; ``average`` is 'N/A'.

    On real MI300X/MI355X parts ``average_socket_power`` returns the literal
    ``'N/A'`` string, which the collector falls back from to ``current_socket_power``.
    """
    return {
        "current_socket_power": handle.config.power_w,
        "average_socket_power": _NA,
    }


def amdsmi_get_energy_count(handle: _Handle) -> dict[str, Any]:
    """Return a monotonically advancing energy accumulator and tick resolution."""
    return {
        "energy_accumulator": handle.next_energy_ticks(),
        "counter_resolution": handle.config.spec.counter_resolution_uj,
    }


def amdsmi_get_gpu_activity(handle: _Handle) -> dict[str, Any]:
    """Return engine activity. ``mm_activity`` is 'N/A' on Instinct parts."""
    return {
        "gfx_activity": handle.config.gfx_activity_pct,
        "umc_activity": handle.config.spec.umc_activity_pct,
        "mm_activity": _NA,
    }


def amdsmi_get_gpu_memory_usage(handle: _Handle, mem_type: int) -> int:
    """Return used memory in bytes for the requested memory type."""
    if mem_type != AmdSmiMemoryType.VRAM:
        raise AmdSmiException(f"Unsupported memory type: {mem_type}")
    spec = handle.config.spec
    return int(spec.vram_total_bytes * handle.config.vram_used_fraction)


def amdsmi_get_temp_metric(handle: _Handle, sensor: int, metric: int) -> float:
    """Return temperature in Celsius. EDGE is unsupported on Instinct (raises)."""
    if sensor == AmdSmiTemperatureType.EDGE:
        raise AmdSmiException("EDGE temperature not supported on this device")
    if sensor in (AmdSmiTemperatureType.JUNCTION, AmdSmiTemperatureType.HOTSPOT):
        return handle.config.junction_temp_c
    raise AmdSmiException(f"Unsupported temperature sensor: {sensor}")


def amdsmi_get_gpu_total_ecc_count(handle: _Handle) -> dict[str, Any]:
    """Return cumulative ECC error counts (zero on a healthy mock device)."""
    return {"correctable_count": 0, "uncorrectable_count": 0, "deferred_count": 0}


def amdsmi_get_gpu_metrics_info(handle: _Handle) -> dict[str, Any]:
    """Return metrics info; the collector reads throttle status from here."""
    return {"throttle_status": 0, "indep_throttle_status": 0}
