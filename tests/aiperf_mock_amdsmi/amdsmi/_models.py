# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static hardware specs for the AMD Instinct GPUs this fake can simulate.

The AMD analog of the mock server's ``GPU_CONFIGS`` in ``dcgm_faker.py``. Values
are nominal "under load" readings, chosen to land in plausible ranges for each
board; they do not vary with benchmark traffic (the fake is intentionally
static, see the package README).
"""

from dataclasses import dataclass

_GIB = 1024**3


@dataclass(frozen=True)
class AMDGpuSpec:
    """Nominal static readings for one AMD Instinct GPU model."""

    product_name: str
    """Board name as reported by ``amdsmi_get_gpu_board_info``'s ``product_name``."""

    vram_total_bytes: int
    """Total VRAM in bytes (amdsmi reports memory usage in bytes)."""

    power_w: float
    """Nominal socket power draw in watts (``current_socket_power``)."""

    junction_temp_c: float
    """Junction temperature in Celsius (modern amdsmi bindings return Celsius)."""

    gfx_activity_pct: float
    """Graphics/compute engine activity in percent (``gfx_activity``)."""

    umc_activity_pct: float
    """Unified memory controller activity in percent (``umc_activity``)."""

    counter_resolution_uj: float
    """Energy counter tick resolution in microjoules (``counter_resolution``)."""


# Specs are approximate vendor figures for simulation, not authoritative.
AMD_GPU_SPECS: dict[str, AMDGpuSpec] = {
    "mi300x": AMDGpuSpec("AMD Instinct MI300X OAM", 192 * _GIB, 600.0, 80.0, 85.0, 40.0, 15.3),
    "mi325x": AMDGpuSpec("AMD Instinct MI325X OAM", 256 * _GIB, 800.0, 82.0, 85.0, 45.0, 15.3),
    "mi355x": AMDGpuSpec("AMD Instinct MI355X OAM", 288 * _GIB, 1100.0, 85.0, 88.0, 50.0, 15.3),
    "mi250x": AMDGpuSpec("AMD Instinct MI250X", 128 * _GIB, 450.0, 75.0, 80.0, 35.0, 15.3),
}  # fmt: skip
"""Static specs keyed by the value of ``AIPERF_MOCK_AMDSMI_MODEL``."""

DEFAULT_MODEL = "mi300x"
"""Model used when ``AIPERF_MOCK_AMDSMI_MODEL`` is unset."""

DEFAULT_NUM_GPUS = 8
"""GPU count used when ``AIPERF_MOCK_AMDSMI_NUM_GPUS`` is unset."""

DEFAULT_VRAM_USED_FRACTION = 0.9
"""Fraction of VRAM reported as used when not overridden."""
