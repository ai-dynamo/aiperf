# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data models for the end-to-end testing framework.
"""

from dataclasses import dataclass


@dataclass
class Command:
    """Represents a command extracted from markdown"""

    tag_name: str
    command: str
    file_path: str
    start_line: int
    end_line: int


@dataclass
class Server:
    """Represents a server with its setup, health check, and aiperf commands"""

    name: str
    setup_command: Command | None
    health_check_command: Command | None
    aiperf_commands: list[Command]
