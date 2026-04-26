# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config v1 - CLI-only input layer.

UserConfig and ServiceConfig are the cyclopts-facing input DTOs. They carry CLI
flag annotations and Pydantic field metadata, but NO validators - AIPerfConfig
is the single validation gate.

Hard rules (enforced by code review + the TID251 ban in pyproject.toml):

1. New CLI flags that fit an existing v1 nested class (EndpointConfig,
   InputConfig, LoadGeneratorConfig, OutputConfig, TokenizerConfig,
   AccuracyConfig) - add the field there.
2. New CLI flags that don't fit any existing nested class - add as a top-level
   field on UserConfig itself. NEVER add new nested classes to v1.
3. NO validators on v1 classes. Validation lives on AIPerfConfig.
4. The converter (aiperf.config.v1.converter) is the only module outside
   cli_commands/ that may read v1 attributes.

Anywhere downstream of cli_commands/, only AIPerfConfig / BenchmarkPlan /
BenchmarkRun flow.
"""

# Import nested classes so forward-ref strings on UserConfig/ServiceConfig can be
# resolved by model_rebuild() below. These names intentionally are NOT re-exported
# in __all__ - only UserConfig + ServiceConfig are public.
from aiperf.config.v1._accuracy import AccuracyConfig  # noqa: F401
from aiperf.config.v1._endpoint import EndpointConfig  # noqa: F401
from aiperf.config.v1._input import InputConfig  # noqa: F401
from aiperf.config.v1._loadgen import LoadGeneratorConfig  # noqa: F401
from aiperf.config.v1._output import OutputConfig  # noqa: F401
from aiperf.config.v1._tokenizer import TokenizerConfig  # noqa: F401
from aiperf.config.v1._workers import WorkersConfig  # noqa: F401
from aiperf.config.v1._zmq import (  # noqa: F401
    ZMQDualBindConfig,
    ZMQIPCConfig,
    ZMQTCPConfig,
)
from aiperf.config.v1.service_config import ServiceConfig
from aiperf.config.v1.user_config import UserConfig

# Resolve forward-ref string types on the top-level DTOs now that nested classes exist.
UserConfig.model_rebuild()
ServiceConfig.model_rebuild()

__all__ = ["ServiceConfig", "UserConfig"]
