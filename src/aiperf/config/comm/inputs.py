# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Communication configuration models (IPC, TCP, DualBind).

Split out of ``models.py`` so the public module stays under the ergonomics
file-size cap. Re-exported via :mod:`aiperf.config.models`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from aiperf.common.enums import CommunicationType
from aiperf.config.base import BaseConfig


def _event_bus_proxy_default() -> TcpProxyConfig:
    """Default event-bus proxy ports, sourced from ``Environment.ZMQ``.

    Lazy import: ``Environment`` re-imports the config tree during bootstrap;
    a module-level import here would deadlock.
    """
    from aiperf.common.environment import Environment

    return TcpProxyConfig(
        frontend_port=Environment.ZMQ.EVENT_BUS_PROXY_FRONTEND_PORT,
        backend_port=Environment.ZMQ.EVENT_BUS_PROXY_BACKEND_PORT,
    )


class TcpProxyConfig(BaseConfig):
    """TCP proxy port configuration for a single ZMQ proxy."""

    model_config = ConfigDict(extra="forbid")

    frontend_port: Annotated[
        int,
        Field(ge=1, le=65535, description="TCP port for proxy frontend."),
    ]

    backend_port: Annotated[
        int,
        Field(ge=1, le=65535, description="TCP port for proxy backend."),
    ]


class IpcCommunicationConfig(BaseConfig):
    """
    IPC (Unix socket) communication configuration.

    For single-machine deployments with lowest latency.
    Uses Unix domain sockets for all inter-service communication.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[CommunicationType.IPC],
        Field(description="Communication type. Must be 'ipc'."),
    ]

    path: Annotated[
        str,
        Field(
            default="/tmp/aiperf",
            description="Directory for IPC socket files. "
            "AIPerf creates multiple socket files in this directory.",
        ),
    ]


class TcpCommunicationConfig(BaseConfig):
    """
    TCP socket communication configuration.

    For distributed deployments across machines.
    Provides detailed port configuration for all ZMQ proxies.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[CommunicationType.TCP],
        Field(description="Communication type. Must be 'tcp'."),
    ]

    host: Annotated[
        str,
        Field(
            default="127.0.0.1",
            description="Host address for TCP communication. "
            "Use 0.0.0.0 to listen on all interfaces.",
        ),
    ]

    # Core communication ports
    records_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            default=5557,
            description="Port for records push/pull communication.",
        ),
    ]

    credit_router_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            default=5564,
            description="Port for credit router (ROUTER-DEALER).",
        ),
    ]

    control_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            default=5667,
            description="Port for control channel (ROUTER-DEALER).",
        ),
    ]

    # Proxy configurations
    event_bus_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=_event_bus_proxy_default,
            description="Event bus proxy ports (XPUB/XSUB).",
        ),
    ]

    dataset_manager_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5661, backend_port=5662
            ),
            description="Dataset manager proxy ports (DEALER/ROUTER).",
        ),
    ]

    raw_inference_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5665, backend_port=5666
            ),
            description="Raw inference proxy ports (PUSH/PULL).",
        ),
    ]


class DualBindCommunicationConfig(BaseConfig):
    """
    Dual-bind (IPC + TCP) communication configuration.

    For Kubernetes deployments where controller services connect via IPC
    (co-located in same pod) and worker pods connect via TCP.

    When controller_host is None, services use IPC (local mode).
    When controller_host is set, services use TCP to that host (remote mode).
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[CommunicationType.DUAL],
        Field(description="Communication type. Must be 'dual'."),
    ]

    ipc_path: Annotated[
        str,
        Field(
            default="/tmp/aiperf",
            description="Directory for IPC socket files (local services).",
        ),
    ]

    tcp_host: Annotated[
        str,
        Field(
            default="0.0.0.0",
            description="TCP bind host for proxies. "
            "Use 0.0.0.0 to listen on all interfaces.",
        ),
    ]

    controller_host: Annotated[
        str | None,
        Field(
            default=None,
            description="Controller host for remote TCP connections. "
            "When set, services connect via TCP to this host instead of IPC. "
            "In Kubernetes, set via JobSet DNS (e.g., controller.namespace.svc).",
        ),
    ]

    # Core communication ports
    records_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            default=5557,
            description="Port for records push/pull communication.",
        ),
    ]

    credit_router_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            default=5564,
            description="Port for credit router (ROUTER-DEALER).",
        ),
    ]

    control_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            default=5667,
            description="Port for control channel (ROUTER-DEALER).",
        ),
    ]

    # Proxy configurations (TCP ports, IPC uses path-based naming)
    event_bus_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=_event_bus_proxy_default,
            description="Event bus proxy ports (XPUB/XSUB).",
        ),
    ]

    dataset_manager_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5661, backend_port=5662
            ),
            description="Dataset manager proxy ports (DEALER/ROUTER).",
        ),
    ]

    raw_inference_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5665, backend_port=5666
            ),
            description="Raw inference proxy ports (PUSH/PULL).",
        ),
    ]


# Union for communication configs using string discriminator
CommunicationConfig = Annotated[
    IpcCommunicationConfig | TcpCommunicationConfig | DualBindCommunicationConfig,
    Field(discriminator="type"),
]
