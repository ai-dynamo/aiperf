# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compat shim: every comm/proxy config type now lives under ``aiperf.config.comm``.
# Two duplicate definitions used to coexist (``aiperf.config._zmq_*`` and
# ``aiperf.config.comm.*``), and ``isinstance`` checks against one shape were
# silently False for instances built from the other. Re-export the canonical
# ``comm/*`` types so legacy import paths resolve to the same class objects.
from aiperf.config.comm.base import BaseZMQCommunicationConfig, BaseZMQProxyConfig
from aiperf.config.comm.dual_bind import ZMQDualBindConfig, ZMQDualBindProxyConfig
from aiperf.config.comm.ipc import ZMQIPCConfig, ZMQIPCProxyConfig
from aiperf.config.comm.tcp import ZMQTCPConfig, ZMQTCPProxyConfig

__all__ = [
    "BaseZMQCommunicationConfig",
    "BaseZMQProxyConfig",
    "ZMQDualBindConfig",
    "ZMQDualBindProxyConfig",
    "ZMQIPCConfig",
    "ZMQIPCProxyConfig",
    "ZMQTCPConfig",
    "ZMQTCPProxyConfig",
]
