# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic port detection and rewriting for parallel docs E2E test execution."""

import re
import socket

from .data_types import Command, Server
from .utils import extract_ports_from_command


def find_free_port() -> int:
    """Bind to :0 and return the OS-assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rewrite_command_ports(command: str, port_map: dict[int, int]) -> str:
    """Replace host-side ports in Docker -p mappings and address/URL references.

    For ``-p HOST:CONTAINER`` flags the HOST (left) side is rewritten while the
    container-internal port is preserved — the server process inside the
    container still listens on its default port.  All other occurrences
    (``localhost:PORT``, ``127.0.0.1:PORT``, ``://host:PORT``) are rewritten so
    client code and health checks target the new host port.
    """
    for orig, assigned in port_map.items():
        s = str(assigned)
        # Docker publish flag: replace only the host (left) side of -p HOST:CONTAINER
        command = re.sub(rf"(-p\s+){orig}(:\d+)", rf"\g<1>{s}\2", command)
        # Address references: localhost:PORT and 127.0.0.1:PORT
        command = re.sub(
            rf"((?:localhost|127\.0\.0\.1):){orig}\b", rf"\g<1>{s}", command
        )
        # Generic URL port: ://hostname:PORT
        command = re.sub(rf"(://[^/:\s]+:){orig}\b", rf"\g<1>{s}", command)
    return command


def _collect_ports(server: Server) -> set[int]:
    """Collect all localhost ports referenced across a server's full command set."""
    ports: set[int] = set()
    cmds: list[Command] = []
    if server.setup_command:
        cmds.append(server.setup_command)
    if server.health_check_command:
        cmds.append(server.health_check_command)
    cmds.extend(server.aiperf_commands)
    for cmd in cmds:
        ports.update(extract_ports_from_command(cmd.command))
    return ports


def assign_ports_to_server(server: Server) -> dict[int, int]:
    """Detect all localhost ports in server commands and rewrite them to free ports.

    Mutates server.setup_command.command, server.health_check_command.command,
    and each entry in server.aiperf_commands in place.
    Returns {original_port: assigned_port}; empty dict if no ports found.
    Each original port is guaranteed a unique assigned port.
    """
    original_ports = _collect_ports(server)
    if not original_ports:
        return {}

    reserved_ports: set[int] = set(original_ports)
    port_map: dict[int, int] = {}
    for orig in original_ports:
        while True:
            candidate = find_free_port()
            if candidate not in reserved_ports:
                reserved_ports.add(candidate)
                port_map[orig] = candidate
                break

    cmds: list[Command] = []
    if server.setup_command:
        cmds.append(server.setup_command)
    if server.health_check_command:
        cmds.append(server.health_check_command)
    cmds.extend(server.aiperf_commands)

    for cmd in cmds:
        cmd.command = rewrite_command_ports(cmd.command, port_map)

    return port_map
