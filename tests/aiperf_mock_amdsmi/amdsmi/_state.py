# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-driven configuration and per-GPU static readings for the fake amdsmi.

Configuration is read from ``AIPERF_MOCK_AMDSMI_*`` environment variables at
``amdsmi_init()`` time (mirroring how real bindings read driver state at init),
not at import time. Readings are static per model; the lone exception is the
energy accumulator, which advances monotonically per call so the cumulative
``amd_energy_consumption`` delta metric (baseline -> final) is non-zero.
"""

import os
from dataclasses import dataclass, field

from ._models import (
    AMD_GPU_SPECS,
    DEFAULT_MODEL,
    DEFAULT_NUM_GPUS,
    DEFAULT_VRAM_USED_FRACTION,
    AMDGpuSpec,
)

# Energy ticks added to each GPU's accumulator on every read, so a benchmark's
# baseline and final scrapes differ. ~1e9 ticks * 15.3 uJ ~= 0.0153 MJ/read.
_ENERGY_TICKS_PER_READ = 1_000_000_000

# Base accumulator value (a plausible "since boot" energy total in ticks).
_ENERGY_BASE_TICKS = 40_000_000_000_000


def _env_float_or(name: str, default: float) -> float:
    """Parse an optional float env var, falling back to ``default`` when invalid."""
    import math

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _env_int(name: str, default: int) -> int:
    """Parse an optional int env var, falling back to ``default`` when invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class _Config:
    """Resolved fake-amdsmi configuration for one ``amdsmi_init()`` session."""

    num_gpus: int
    spec: AMDGpuSpec
    gfx_activity_pct: float
    power_w: float
    junction_temp_c: float
    vram_used_fraction: float

    @classmethod
    def from_env(cls) -> "_Config":
        """Build configuration from ``AIPERF_MOCK_AMDSMI_*`` environment variables."""
        model = (
            os.environ.get("AIPERF_MOCK_AMDSMI_MODEL", DEFAULT_MODEL).strip().lower()
        )
        spec = AMD_GPU_SPECS.get(model, AMD_GPU_SPECS[DEFAULT_MODEL])
        num_gpus = max(1, _env_int("AIPERF_MOCK_AMDSMI_NUM_GPUS", DEFAULT_NUM_GPUS))
        return cls(
            num_gpus=num_gpus,
            spec=spec,
            gfx_activity_pct=_env_float_or(
                "AIPERF_MOCK_AMDSMI_GFX_ACTIVITY", spec.gfx_activity_pct
            ),
            power_w=_env_float_or("AIPERF_MOCK_AMDSMI_POWER_W", spec.power_w),
            junction_temp_c=_env_float_or(
                "AIPERF_MOCK_AMDSMI_TEMP_C", spec.junction_temp_c
            ),
            vram_used_fraction=_env_float_or(
                "AIPERF_MOCK_AMDSMI_VRAM_USED_FRACTION", DEFAULT_VRAM_USED_FRACTION
            ),
        )


@dataclass
class _Handle:
    """Opaque per-GPU handle returned by ``amdsmi_get_processor_handles``.

    Stands in for the real amdsmi processor pointer. Carries the resolved config
    plus a mutable energy accumulator advanced on each energy read.
    """

    index: int
    config: _Config
    _energy_ticks: int = field(default=_ENERGY_BASE_TICKS)

    @property
    def uuid(self) -> str:
        return f"06ff74a1-0000-1000-806c-{self.index:012x}"

    @property
    def bdf(self) -> str:
        return f"0000:{self.index:02x}:00.0"

    def next_energy_ticks(self) -> int:
        """Return the current accumulator, then advance it for the next read."""
        current = self._energy_ticks
        self._energy_ticks += _ENERGY_TICKS_PER_READ
        return current
