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

from aiperf.config.v1.service_config import ServiceConfig
from aiperf.config.v1.user_config import UserConfig

__all__ = ["ServiceConfig", "UserConfig"]
