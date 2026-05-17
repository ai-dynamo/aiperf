# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for split communication config modules."""

from aiperf.config.comm.ipc import ZMQIPCConfig, ZMQIPCProxyConfig

__all__ = ["ZMQIPCProxyConfig", "ZMQIPCConfig"]
