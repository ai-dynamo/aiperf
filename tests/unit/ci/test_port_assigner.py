# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket

from tests.ci.test_docs_end_to_end.data_types import Command, Server
from tests.ci.test_docs_end_to_end.port_assigner import (
    assign_ports_to_server,
    find_free_port,
    rewrite_command_ports,
)


def _cmd(command: str) -> Command:
    return Command(
        tag_name="test", command=command, file_path="f.md", start_line=1, end_line=1
    )


def test_find_free_port_returns_bindable_port():
    port = find_free_port()
    assert 1024 < port < 65536
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def test_rewrite_command_ports_replaces_all_occurrences():
    cmd = "docker run -p 8000:8000 img && curl localhost:8000/health"
    result = rewrite_command_ports(cmd, {8000: 19100})
    assert "19100" in result
    assert "8000" not in result


def test_rewrite_command_ports_handles_multiple_ports():
    cmd = "curl localhost:8000/v1 && curl localhost:9000/metrics"
    result = rewrite_command_ports(cmd, {8000: 19100, 9000: 19101})
    assert "19100" in result
    assert "19101" in result
    assert "8000" not in result
    assert "9000" not in result


def test_assign_ports_to_server_rewrites_all_commands():
    setup = _cmd("docker run -p 8000:8000 my-server")
    health = _cmd("curl http://localhost:8000/health")
    run1 = _cmd("aiperf run --base-url http://localhost:8000/v1")
    run2 = _cmd("aiperf run --base-url http://localhost:8000/v1 --model gpt2")
    server = Server(
        name="test-server",
        setup_command=setup,
        health_check_command=health,
        aiperf_commands=[run1, run2],
    )
    port_map = assign_ports_to_server(server)
    assert 8000 in port_map
    new_port = str(port_map[8000])
    assert new_port in server.setup_command.command
    assert new_port in server.health_check_command.command
    assert new_port in server.aiperf_commands[0].command
    assert new_port in server.aiperf_commands[1].command
    assert "8000" not in server.setup_command.command


def test_assign_ports_to_server_no_ports_returns_empty():
    server = Server(
        name="no-ports",
        setup_command=_cmd("echo hello"),
        health_check_command=_cmd("echo ok"),
        aiperf_commands=[_cmd("aiperf run --base-url http://my-server/v1")],
    )
    port_map = assign_ports_to_server(server)
    assert port_map == {}
