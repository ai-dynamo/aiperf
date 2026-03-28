# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data models for the end-to-end testing framework.
"""

from dataclasses import dataclass


@dataclass
class Command:
    """Represents a command extracted from markdown."""

    tag_name: str
    """HTML tag name that contained this command."""

    command: str
    """Shell command string to execute."""

    file_path: str
    """Path to the markdown file containing this command."""

    start_line: int
    """Line number where the command block starts."""

    end_line: int
    """Line number where the command block ends."""


@dataclass
class Server:
    """Represents a server with its setup, health check, and aiperf commands."""

    name: str
    """Server identifier name."""

    setup_command: Command | None
    """Command to start the server, if any."""

    health_check_command: Command | None
    """Command to verify server readiness, if any."""

    aiperf_commands: list[Command]
    """AIPerf benchmark commands to run against this server."""
