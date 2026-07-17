# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration - Pydantic Models

Network Latency - TCP-handshake RTT calibration configuration. When enabled, AIPerf
probes the inference endpoint throughout the profiling phase, measures the network
round-trip time, and subtracts the mean RTT from request-start-anchored latency
metrics (emitted as separate ``network_adjusted_*`` metrics; raw metrics are
preserved). This makes runs taken from different network locations comparable.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig

__all__ = [
    "NetworkLatencyConfig",
]


def _default_ping_interval() -> float:
    # Imported lazily: aiperf.common.environment imports the config package, so a
    # module-level import here would create a circular import.
    from aiperf.common.environment import Environment

    return Environment.NETWORK_LATENCY.DEFAULT_PROBE_INTERVAL


class NetworkLatencyConfig(BaseConfig):
    """Network latency calibration configuration.

    When ``enabled``, AIPerf opens a fresh TCP connection to the endpoint
    host:port on an interval during profiling and records the handshake RTT.
    The mean RTT is subtracted from the request-start-anchored latency metrics
    (request_latency, time_to_first_token, time_to_first_output_token) to produce
    non-destructive ``network_adjusted_*``
    variants. ``mean_ms`` supplies a fixed mean RTT and skips active probing.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )

    enabled: Annotated[
        bool,
        Field(
            default=False,
            description="Enable network latency calibration. When true, AIPerf measures "
            "the endpoint RTT (or uses mean_ms) and subtracts it from latency "
            "metrics, emitted as separate network_adjusted_* metrics.",
        ),
    ]

    ping_interval: Annotated[
        float,
        Field(
            default_factory=_default_ping_interval,
            gt=0.0,
            description="Seconds between TCP-handshake RTT probes during profiling. "
            "Ignored when mean_ms is set.",
        ),
    ]

    mean_ms: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Fixed mean network RTT in milliseconds to subtract, bypassing "
            "active probing. When set, no probes are sent and this value is used directly.",
        ),
    ]

    @property
    def should_probe(self) -> bool:
        """True when active RTT probing should run (enabled and no manual mean)."""
        return self.enabled and self.mean_ms is None
