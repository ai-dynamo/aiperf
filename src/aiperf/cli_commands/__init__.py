# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for AIPerf."""

from aiperf.cli_commands.analyze_trace import (
    analyze_app,
    analyze_trace,
)

__all__ = ["analyze_app", "analyze_trace"]
