# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parametrized docs E2E test — one test node per aiperf tutorial command."""

from __future__ import annotations

import pytest

from .data_types import Command, E2ETestConfig, Server
from .test_runner import run_aiperf_command


@pytest.mark.docs_e2e
def test_docs_aiperf_command(
    aiperf_command: tuple[str, Command],
    server_context: Server,  # noqa: ARG001 — ensures server is up before we run
    e2e_config: E2ETestConfig,
    aiperf_container_id: str | None,
) -> None:
    _server_name, cmd = aiperf_command
    success, output = run_aiperf_command(
        cmd, e2e_config, container_id=aiperf_container_id
    )
    assert success, (
        f"aiperf command failed\n"
        f"  file: {cmd.file_path}:{cmd.start_line}\n"
        f"  command: {cmd.command}\n"
        f"  output:\n{output}"
    )
