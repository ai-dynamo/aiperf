# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data models for the end-to-end testing framework.
"""

from dataclasses import dataclass, field


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
    # Hard kill deadline in seconds. Defaults to AIPERF_COMMAND_TIMEOUT when
    # unset. Tag-level annotation: ``<!-- aiperf-run-<server>-endpoint-server
    # timeout=3600 -->``. Sweeps and multi-phase workflows legitimately run
    # far longer than a single-point benchmark, and capping them at the shared
    # default is what keeps those guides untestable.
    timeout: int | None = None


@dataclass
class FileFixture:
    """A file a guide needs on disk before its commands can run.

    Guides that drive AIPerf through ``--config foo.yaml`` (or a ``.jsonl``
    trace) already print the file contents in the page. Materializing that
    block is what makes such a guide testable at all; without it the command
    can only be tagged by rewriting the doc to point at a path the reader does
    not have.
    """

    path: str
    content: str
    file_path: str
    start_line: int


@dataclass
class Server:
    """Represents a server with its setup, health check, and aiperf commands"""

    name: str
    setup_command: Command | None
    health_check_command: Command | None
    aiperf_commands: list[Command]
    # Files written into the AIPerf container's working directory before any
    # of this server's commands run, in document order.
    files: list[FileFixture] = field(default_factory=list)
