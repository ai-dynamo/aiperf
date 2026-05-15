# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ communication config builder.

Single source of truth for mapping user-facing communication config models
(IPC / TCP / DualBind) to the runtime ZMQ config objects consumed by services.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.enums import CommunicationType
from aiperf.config.comm.dual_bind import (
    ZMQDualBindConfig,
    ZMQDualBindProxyConfig,
)
from aiperf.config.comm.inputs import (
    DualBindCommunicationConfig,
    TcpCommunicationConfig,
)
from aiperf.config.comm.ipc import ZMQIPCConfig
from aiperf.config.comm.tcp import ZMQTCPConfig, ZMQTCPProxyConfig

if TYPE_CHECKING:
    from aiperf.config.comm.base import BaseZMQCommunicationConfig
    from aiperf.config.config import BenchmarkConfig


def _build_tcp(comm: TcpCommunicationConfig) -> ZMQTCPConfig:
    return ZMQTCPConfig(
        host=comm.host,
        records_push_pull_port=comm.records_port,
        credit_router_port=comm.credit_router_port,
        control_port=comm.control_port,
        event_bus_proxy_config=ZMQTCPProxyConfig(
            frontend_port=comm.event_bus_proxy.frontend_port,
            backend_port=comm.event_bus_proxy.backend_port,
        ),
        dataset_manager_proxy_config=ZMQTCPProxyConfig(
            frontend_port=comm.dataset_manager_proxy.frontend_port,
            backend_port=comm.dataset_manager_proxy.backend_port,
        ),
        raw_inference_proxy_config=ZMQTCPProxyConfig(
            frontend_port=comm.raw_inference_proxy.frontend_port,
            backend_port=comm.raw_inference_proxy.backend_port,
        ),
    )


def _build_dual(comm: DualBindCommunicationConfig) -> ZMQDualBindConfig:
    controller_host = comm.controller_host
    if controller_host is None:
        import os

        controller_host = os.environ.get("AIPERF_K8S_ZMQ_CONTROLLER_HOST")

    return ZMQDualBindConfig(
        ipc_path=Path(comm.ipc_path),
        tcp_host=comm.tcp_host,
        controller_host=controller_host,
        records_push_pull_tcp_port=comm.records_port,
        credit_router_tcp_port=comm.credit_router_port,
        control_tcp_port=comm.control_port,
        event_bus_proxy_config=ZMQDualBindProxyConfig(
            name="event_bus_proxy",
            tcp_frontend_port=comm.event_bus_proxy.frontend_port,
            tcp_backend_port=comm.event_bus_proxy.backend_port,
        ),
        dataset_manager_proxy_config=ZMQDualBindProxyConfig(
            name="dataset_manager_proxy",
            tcp_frontend_port=comm.dataset_manager_proxy.frontend_port,
            tcp_backend_port=comm.dataset_manager_proxy.backend_port,
        ),
        raw_inference_proxy_config=ZMQDualBindProxyConfig(
            name="raw_inference_proxy",
            tcp_frontend_port=comm.raw_inference_proxy.frontend_port,
            tcp_backend_port=comm.raw_inference_proxy.backend_port,
        ),
    )


def build_comm_config(config: BenchmarkConfig) -> BaseZMQCommunicationConfig:
    """Build a ZMQ communication config from a BenchmarkConfig.

    Called by:
    - BenchmarkHelpersMixin.comm_config property (mixed into BenchmarkConfig)
    - BenchmarkRun.comm_config property (orchestrator path)

    Handles complete field mapping (ports, proxy configs, control ports),
    K8s env-var auto-detection for dual-bind controller_host, and fallback
    to IPC when no communication config is set.

    Note: credit_return_router_port is not exposed on user-facing inputs and
    falls back to ZMQ defaults.
    """
    comm = config.runtime.communication
    if comm is None:
        return ZMQIPCConfig()

    if comm.type == CommunicationType.IPC:
        return ZMQIPCConfig(path=comm.path)

    if comm.type == CommunicationType.TCP:
        assert isinstance(comm, TcpCommunicationConfig)
        return _build_tcp(comm)

    if comm.type == CommunicationType.DUAL:
        assert isinstance(comm, DualBindCommunicationConfig)
        return _build_dual(comm)

    return ZMQIPCConfig()
