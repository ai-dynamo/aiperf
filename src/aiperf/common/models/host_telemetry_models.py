# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models for host-level (non-accelerator) power and energy telemetry.

These mirror the GPU telemetry models rather than reusing them. `GpuMetadata`
requires `gpu_index`, `gpu_uuid`, `gpu_model_name` and `pci_bus_id`, none of
which a CPU package or a board power rail has, so `TelemetryRecord` cannot carry
a host sample.

The split mirrors the GPU side's metadata/metrics separation, but the shapes
are not identical and a unification would have to reconcile them: the GPU
record flattens metadata into the record through inheritance while this one
nests it as a field, and the energy units differ (NVML reports millijoules,
RAPL microjoules) with no shared naming convention yet.
"""

from __future__ import annotations

from typing import ClassVar

from aiperf.common.finite import FiniteFloat
from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class HostPowerDomainMetadata(AIPerfBaseModel):
    """Static identity of one host power domain.

    A domain is whatever the platform can meter separately: an x86 package, its
    `core` or `dram` subdomain, or a board power rail on an ARM SoC.
    """

    domain_index: int = Field(
        ge=0,
        description="Index of this domain on the node, used for stable display ordering",
    )
    domain_id: str = Field(
        description=(
            "Stable unique identifier for the domain, used as the primary key for its "
            "time series (e.g. 'intel-rapl:0', 'intel-rapl:0:1', 'pmic:VDD_CORE')"
        )
    )
    domain_name: str = Field(
        description="Human-readable name as the platform reports it (e.g. 'package-0', 'dram')"
    )
    parent_domain_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the enclosing domain when this one is a subdomain, so a consumer "
            "can avoid double-counting a package and its own core and dram children"
        ),
    )


class HostTelemetryMetrics(AIPerfBaseModel):
    """One sample of host power and energy for a single domain.

    Every field is optional because platforms expose different subsets. RAPL
    gives a cumulative energy counter and no instantaneous power; a board PMIC
    gives instantaneous power and no counter. A collector fills what it has.
    """

    energy_consumption_uj: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Cumulative energy for this domain in microjoules, counted from an arbitrary "
            "fixed point. Monotonic apart from hardware wraparound, which the collector "
            "is responsible for correcting before reporting."
        ),
    )
    power_usage_w: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Instantaneous power draw for this domain in watts. Platforms without a power "
            "register leave this unset rather than deriving it, so that a consumer can tell "
            "a measured value from a computed one."
        ),
    )
    energy_range_uj: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Value at which this domain's energy counter wraps, where the platform reports "
            "one. Carried so a consumer can audit the collector's wraparound handling."
        ),
    )


class HostTelemetryRecord(AIPerfBaseModel):
    """A single host telemetry data point for one power domain."""

    record_type: ClassVar[str] = "host_telemetry"

    timestamp_ns: int = Field(
        ge=0,
        description="Nanosecond wall-clock timestamp when the sample was taken (time_ns)",
    )
    telemetry_source_url: str = Field(
        description=(
            "Source identifier for the collector, matching the GPU side's convention "
            "(e.g. 'rapl://localhost', 'pmic://localhost')"
        )
    )
    domain: HostPowerDomainMetadata = Field(
        description="Identity of the power domain this sample belongs to"
    )
    telemetry_data: HostTelemetryMetrics = Field(
        description="Host power and energy snapshot collected at this timestamp"
    )
