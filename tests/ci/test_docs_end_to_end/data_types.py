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
    # Estimated runtime in seconds; used by the matrix sharder to bin-pack
    # commands so one shard doesn't end up owning all the slow tests.
    # Tag-level annotation: ``<!-- aiperf-run-<server>-endpoint-server weight=300 -->``.
    # Default (80s) covers a typical synthetic-input tutorial command; the
    # observed mean across unweighted tests is ~95s with a heavy right tail,
    # so 80 is a conservative under-estimate that prefers to leave shard
    # headroom rather than over-allocate.
    weight: int = 80


@dataclass
class Server:
    """Represents a server with its setup, health check, and aiperf commands"""

    name: str
    setup_command: Command | None
    health_check_command: Command | None
    aiperf_commands: list[Command]
