# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Communication configuration sub-package.

Includes the user-facing communication input models, the ZMQ output configs,
and the build_comm_config bridge between them.
"""

from aiperf.config.comm.base import (
    BaseZMQCommunicationConfig,
    BaseZMQProxyConfig,
)
from aiperf.config.comm.build import build_comm_config
from aiperf.config.comm.dual_bind import (
    ZMQDualBindConfig,
    ZMQDualBindProxyConfig,
)
from aiperf.config.comm.inputs import (
    CommunicationConfig,
    DualBindCommunicationConfig,
    IpcCommunicationConfig,
    TcpCommunicationConfig,
    TcpProxyConfig,
)
from aiperf.config.comm.ipc import ZMQIPCConfig, ZMQIPCProxyConfig
from aiperf.config.comm.tcp import ZMQTCPConfig, ZMQTCPProxyConfig

__all__ = [
    "BaseZMQCommunicationConfig",
    "BaseZMQProxyConfig",
    "CommunicationConfig",
    "DualBindCommunicationConfig",
    "IpcCommunicationConfig",
    "TcpCommunicationConfig",
    "TcpProxyConfig",
    "ZMQDualBindConfig",
    "ZMQDualBindProxyConfig",
    "ZMQIPCConfig",
    "ZMQIPCProxyConfig",
    "ZMQTCPConfig",
    "ZMQTCPProxyConfig",
    "build_comm_config",
]
