# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest infrastructure for docs end-to-end tests.

All settings are configurable via --docs-e2e-* CLI options or DOCS_E2E_*
environment variables. CLI takes precedence.

Usage (local dev, server already running):
    uv run pytest tests/ci/test_docs_end_to_end/ -m docs_e2e --docs-e2e-local-dev

Usage (CI shard 0/4 of vllm-default-openai):
    uv run pytest tests/ci/test_docs_end_to_end/ -m docs_e2e \\
        --docs-e2e-server vllm-default-openai \\
        --docs-e2e-shard-index 0 --docs-e2e-shard-total 4
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pytest
from data_types import E2ETestConfig, Server
from parser import MarkdownParser
from port_assigner import assign_ports_to_server
from test_runner import (
    build_aiperf_image,
    cleanup_all_containers,
    run_health_check,
    setup_server,
    verify_local_aiperf,
)
from utils import get_repo_root, setup_logging

logger = logging.getLogger(__name__)

_SETTINGS_KEY = pytest.StashKey["E2ETestSettings"]()


@dataclass(frozen=True)
class E2ETestSettings:
    """Resolved docs E2E configuration (CLI > env > default)."""

    use_local_aiperf: bool = False
    skip_server_setup: bool = False
    skip_health_check: bool = False
    local_dev: bool = False
    server: str | None = None
    shard_index: int = 0
    shard_total: int = 1

    @property
    def config(self) -> E2ETestConfig:
        return E2ETestConfig(
            use_local_aiperf=self.local_dev or self.use_local_aiperf,
            skip_server_setup=self.local_dev or self.skip_server_setup,
            skip_health_check=self.local_dev or self.skip_health_check,
            server_filter=self.server,
        )


_OPTIONS: list[tuple[str, str, str | None, str, str]] = [
    ("--docs-e2e-use-local-aiperf", "DOCS_E2E_USE_LOCAL_AIPERF", None, "bool",
     "Use pip-installed aiperf instead of a container"),
    ("--docs-e2e-skip-server-setup", "DOCS_E2E_SKIP_SERVER_SETUP", None, "bool",
     "Skip spinning up servers (use already-running)"),
    ("--docs-e2e-skip-health-check", "DOCS_E2E_SKIP_HEALTH_CHECK", None, "bool",
     "Skip health checks"),
    ("--docs-e2e-local-dev", "DOCS_E2E_LOCAL_DEV", None, "bool",
     "Local dev: implies --use-local-aiperf, --skip-server-setup, --skip-health-check"),
    ("--docs-e2e-server", "DOCS_E2E_SERVER", None, "str",
     "Only run tests for the named server"),
    ("--docs-e2e-shard-index", "DOCS_E2E_SHARD_INDEX", "0", "int",
     "0-based shard index"),
    ("--docs-e2e-shard-total", "DOCS_E2E_SHARD_TOTAL", "1", "int",
     "Total shards for the selected server"),
]  # fmt: skip


def _resolve_settings(config: pytest.Config) -> E2ETestSettings:
    def _resolve(cli_flag: str, env_var: str, default: str | None, typ: str) -> object:
        attr = cli_flag.lstrip("-").replace("-", "_")
        cli_val = getattr(config.option, attr, None)
        if cli_val is not None:
            raw = cli_val
        else:
            env_val = os.environ.get(env_var, "")
            raw = env_val if env_val else default
        if raw is None:
            return None
        if typ == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).lower() in ("1", "true", "yes")
        if typ == "int":
            return int(raw)
        return str(raw)

    resolved: dict[str, object] = {}
    for cli_flag, env_var, default, typ, _help in _OPTIONS:
        field_name = cli_flag.removeprefix("--docs-e2e-").replace("-", "_")
        resolved[field_name] = _resolve(cli_flag, env_var, default, typ)

    for key in (
        "use_local_aiperf",
        "skip_server_setup",
        "skip_health_check",
        "local_dev",
    ):
        if resolved.get(key) is None:
            resolved[key] = False
    if resolved.get("shard_index") is None:
        resolved["shard_index"] = 0
    if resolved.get("shard_total") is None:
        resolved["shard_total"] = 1

    return E2ETestSettings(**resolved)  # type: ignore[arg-type]


def _get_settings(config: pytest.Config) -> E2ETestSettings:
    return config.stash[_SETTINGS_KEY]


def _lpt_shard(server: Server, shard_index: int, shard_total: int) -> list:
    """LPT bin-pack server.aiperf_commands; return commands for shard_index in docs order."""
    from data_types import Command

    shard_bins: list[list[Command]] = [[] for _ in range(shard_total)]
    shard_load: list[int] = [0] * shard_total
    sorted_cmds = sorted(
        server.aiperf_commands,
        key=lambda c: (-c.weight, c.file_path, c.start_line),
    )
    for cmd in sorted_cmds:
        target = min(range(shard_total), key=lambda i: shard_load[i])
        shard_bins[target].append(cmd)
        shard_load[target] += cmd.weight
    my_bin = shard_bins[shard_index]
    my_bin.sort(key=lambda c: (c.file_path, c.start_line))
    logger.info(
        "Shard %d/%d of '%s': %d/%d commands, est. %ds",
        shard_index + 1,
        shard_total,
        server.name,
        len(my_bin),
        len(server.aiperf_commands),
        shard_load[shard_index],
    )
    return my_bin


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("docs-e2e", "Docs end-to-end test options")
    for cli_flag, _env_var, _default, typ, help_text in _OPTIONS:
        if typ == "bool":
            group.addoption(cli_flag, action="store_true", default=None, help=help_text)
        else:
            group.addoption(cli_flag, default=None, help=help_text)


def pytest_configure(config: pytest.Config) -> None:
    settings = _resolve_settings(config)
    config.stash[_SETTINGS_KEY] = settings
    config.addinivalue_line(
        "markers",
        "docs_e2e: docs end-to-end test — requires Docker and an LLM server",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    setup_logging()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize `aiperf_command` with one value per docs tutorial command."""
    if "aiperf_command" not in metafunc.fixturenames:
        return

    settings = _get_settings(metafunc.config)
    repo_root = get_repo_root()
    md_parser = MarkdownParser()
    servers = md_parser.parse_directory(str(repo_root))

    if settings.server:
        if settings.server not in servers:
            pytest.fail(
                f"--docs-e2e-server '{settings.server}' not found. "
                f"Known servers: {sorted(servers.keys())}"
            )
        servers = {settings.server: servers[settings.server]}

    params = []
    for server_name, server in servers.items():
        cmds = server.aiperf_commands
        if settings.shard_total > 1:
            cmds = _lpt_shard(server, settings.shard_index, settings.shard_total)
        for cmd in cmds:
            params.append(
                pytest.param(
                    (server_name, cmd),
                    id=f"{server_name}::{cmd.file_path}:{cmd.start_line}",
                    marks=[pytest.mark.docs_e2e],
                )
            )

    metafunc.parametrize("aiperf_command", params, indirect=False)


@pytest.fixture(scope="session")
def e2e_settings(request: pytest.FixtureRequest) -> E2ETestSettings:
    return _get_settings(request.config)


@pytest.fixture(scope="session")
def e2e_config(e2e_settings: E2ETestSettings) -> E2ETestConfig:
    return e2e_settings.config


@pytest.fixture(scope="session")
def aiperf_container_id(e2e_config: E2ETestConfig) -> str | None:
    """Build the AIPerf container once; yield its name; stop it after the session."""
    if e2e_config.use_local_aiperf:
        verify_local_aiperf(e2e_config)
        yield None
        return
    if not e2e_config.skip_server_setup:
        cleanup_all_containers()
    container_id = build_aiperf_image(e2e_config)
    try:
        yield container_id
    finally:
        from utils import docker_stop_and_remove

        docker_stop_and_remove(container_id)


@pytest.fixture(scope="session")
def _server_port_maps() -> dict[str, dict[int, int]]:
    return {}


@pytest.fixture(scope="session")
def parsed_servers() -> dict[str, Server]:
    repo_root = get_repo_root()
    md_parser = MarkdownParser()
    return md_parser.parse_directory(str(repo_root))


@pytest.fixture(scope="session")
def server_context(
    aiperf_command: tuple[str, object],
    e2e_config: E2ETestConfig,
    aiperf_container_id: str | None,
    _server_port_maps: dict[str, dict[int, int]],
    parsed_servers: dict[str, Server],
) -> Server:
    """Set up the server for this aiperf_command's server (once per session per server name)."""
    server_name, _ = aiperf_command
    server = parsed_servers[server_name]

    if server_name not in _server_port_maps:
        port_map = assign_ports_to_server(server)
        _server_port_maps[server_name] = port_map
        if port_map:
            logger.info("Port map for '%s': %s", server_name, port_map)
        if not e2e_config.skip_server_setup:
            setup_server(server, e2e_config)
        if not e2e_config.skip_health_check:
            run_health_check(server, e2e_config)

    return server
