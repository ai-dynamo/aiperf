# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

This module re-exports all Pydantic models for the AIPerf YAML configuration
system. Implementations live in private submodules to keep any one file under
the ergonomics file-size cap:

* :mod:`aiperf.config._models_core`      — models, tokenizer, SLOs alias
* :mod:`aiperf.config._models_comm`      — IPC/TCP/DualBind communication configs
* :mod:`aiperf.config._models_runtime`   — runtime and logging configs
* :mod:`aiperf.config._models_benchmark` — multi-run and accuracy configs
"""

from __future__ import annotations

from aiperf.config._models_benchmark import AccuracyConfig, MultiRunConfig
from aiperf.config._models_comm import (
    CommunicationConfig,
    DualBindCommunicationConfig,
    IpcCommunicationConfig,
    TcpCommunicationConfig,
    TcpProxyConfig,
)
from aiperf.config._models_core import (
    ModelItem,
    ModelsAdvanced,
    SLOsConfig,
    TokenizerConfig,
    TokenizerOverride,
)
from aiperf.config._models_runtime import LoggingConfig, RuntimeConfig

__all__ = [
    # Accuracy benchmarking
    "AccuracyConfig",
    # Communication
    "CommunicationConfig",
    "DualBindCommunicationConfig",
    "IpcCommunicationConfig",
    # Logging
    "LoggingConfig",
    # Models
    "ModelItem",
    "ModelsAdvanced",
    # Multi-run
    "MultiRunConfig",
    # Runtime
    "RuntimeConfig",
    # SLOs
    "SLOsConfig",
    "TcpCommunicationConfig",
    "TcpProxyConfig",
    # Tokenizer
    "TokenizerConfig",
    "TokenizerOverride",
]
