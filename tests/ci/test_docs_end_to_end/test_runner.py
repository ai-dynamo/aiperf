# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone functions for executing server setup, health checks, and AIPerf tests.
"""

import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from .constants import (
    AIPERF_COMMAND_TIMEOUT,
    AIPERF_UI_TYPE,
    HEALTH_CHECK_TIMEOUT,
    SETUP_MONITOR_TIMEOUT,
)
from .data_types import Command, E2ETestConfig, Server
from .utils import (
    get_repo_root,
)

logger = logging.getLogger(__name__)

# Label applied to every Docker container we start so that cleanup stays
# scoped to this session and does not affect parallel shards on the same runner.
_SESSION_LABEL = f"aiperf-e2e-session={os.getpid()}"


class _ProcessGroupKillGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._finished = False

    def mark_finished(self) -> None:
        with self._lock:
            self._finished = True

    def mark_killing_if_running(self, proc: Any) -> bool:
        with self._lock:
            if self._finished or proc.poll() is not None:
                return False
            self._finished = True
            return True


def _make_process_group_timeout_killer(
    *,
    proc: Any,
    test_num: int,
    server_name: str,
    guard: _ProcessGroupKillGuard,
) -> Callable[[], None]:
    def _kill_on_timeout() -> None:
        if not guard.mark_killing_if_running(proc):
            return
        logger.error(
            f"AIPerf test {test_num} exceeded "
            f"{AIPERF_COMMAND_TIMEOUT}s timeout for {server_name}; "
            f"sending SIGKILL to process group"
        )
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)

    return _kill_on_timeout


def _drain_output(
    proc: subprocess.Popen,
    timeout: float | None,
    prefix: str,
) -> tuple[list[str], bool]:
    """Read stdout from proc in a daemon thread, print each line, return (lines, timed_out).

    The reader runs in a daemon thread so it is SIGALRM-safe. ``timeout`` is
    passed directly to ``thread.join``; ``None`` blocks until the reader exits
    naturally (i.e. until stdout is exhausted).  ``timed_out`` is True only
    when the thread is still alive after ``timeout`` seconds.
    """
    lines: list[str] = []
    lock = threading.Lock()

    def _reader() -> None:
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                with lock:
                    lines.append(line)
                print(f"{prefix}: {line}", flush=True)
        except Exception:
            pass

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    reader_thread.join(timeout=timeout)
    timed_out = reader_thread.is_alive()
    return lines, timed_out


def _inject_session_label(command: str) -> str:
    """Inject the session label into the first ``docker run`` in a command string."""
    return re.sub(
        r"docker run(\s)", f"docker run --label {_SESSION_LABEL}\\1", command, count=1
    )


def cleanup_all_containers() -> None:
    """Stop and prune containers started by this test session."""
    subprocess.run(
        f"docker ps -q --filter label={_SESSION_LABEL} | xargs -r docker stop 2>/dev/null || true",
        shell=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        f"docker container prune -f --filter label={_SESSION_LABEL}",
        shell=True,
        capture_output=True,
        timeout=10,
    )


def verify_local_aiperf(config: E2ETestConfig) -> None:
    """Verify that a locally installed aiperf is present and working.

    Raises RuntimeError on failure.
    """
    logger.info("Verifying local aiperf installation...")
    try:
        result = subprocess.run(
            "aiperf --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error("Local aiperf not found or not working")
            logger.error(f"Stderr: {result.stderr}")
            logger.error("Install with: uv run pip install -e . (from repo root)")
            raise RuntimeError("Local aiperf not found or not working")
        logger.info(f"Local aiperf version: {result.stdout.strip()}")
    except FileNotFoundError as exc:
        raise RuntimeError("aiperf command not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("aiperf --version timed out") from exc


def build_aiperf_image(config: E2ETestConfig) -> str:
    """Build the AIPerf Docker image and start a long-lived helper container.

    Returns the container name (container_id).  Raises RuntimeError on failure.
    """
    logger.info("Building AIPerf container...")
    repo_root = get_repo_root()
    build_command = f"cd {repo_root} && docker build --target test -t aiperf:test ."
    logger.info("Building AIPerf Docker image...")
    logger.info(f"Build command: {build_command}")
    logger.info("=" * 60)

    build_process = subprocess.Popen(
        build_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    while True:
        line = build_process.stdout.readline()
        if not line and build_process.poll() is not None:
            break
        if line:
            print(f"BUILD: {line.rstrip()}")

    build_process.wait()

    if build_process.returncode != 0:
        logger.error("Failed to build AIPerf container")
        logger.error(f"Return code: {build_process.returncode}")
        raise RuntimeError("Failed to build AIPerf container")

    logger.info("AIPerf Docker image built successfully")

    container_name = f"aiperf-test-{os.getpid()}"
    repo_root = get_repo_root()
    fixtures_mount = f"-v {repo_root}/tests/fixtures:/fixtures:ro"
    run_command = (
        f"docker run -d --name {container_name} --label {_SESSION_LABEL} "
        f"-e HF_TOKEN {fixtures_mount} "
        f"--network host --entrypoint bash aiperf:test -c 'tail -f /dev/null'"
    )

    result = subprocess.run(
        run_command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        logger.error("Failed to start AIPerf container")
        logger.error(f"Error: {result.stderr}")
        raise RuntimeError("Failed to start AIPerf container")

    logger.info(f"AIPerf container ready: {container_name}")

    verify_result = subprocess.run(
        f"docker exec {container_name} aiperf --version",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if verify_result.returncode != 0:
        logger.error("AIPerf verification failed")
        raise RuntimeError("AIPerf verification failed in container")

    logger.info(f"AIPerf version: {verify_result.stdout.strip()}")
    return container_name


def validate_servers(servers: dict[str, Server], config: E2ETestConfig) -> None:
    """Validate that all server configs are complete. Raises RuntimeError on failure."""
    logger.info(f"Validating {len(servers)} servers...")
    for server_name, server in servers.items():
        if server.setup_command is None and not config.skip_server_setup:
            raise RuntimeError(f"Server '{server_name}' missing setup command")
        if server.health_check_command is None and not config.skip_health_check:
            raise RuntimeError(f"Server '{server_name}' missing health-check command")
        if not server.aiperf_commands:
            raise RuntimeError(f"Server '{server_name}' missing aiperf-run commands")
        setup_status = "skipped" if config.skip_server_setup else "1 setup"
        health_status = "skipped" if config.skip_health_check else "1 health-check"
        logger.info(
            f"Server '{server_name}': {setup_status}, {health_status}, "
            f"{len(server.aiperf_commands)} aiperf commands"
        )
    logger.info("Server validation passed")


def setup_server(server: Server, config: E2ETestConfig) -> None:
    """Start a server process and wait for it to initialise.

    Monitors process output in a daemon thread so logs are visible in CI.
    Raises RuntimeError if the process exits with a non-zero code during the
    SETUP_MONITOR_TIMEOUT window.
    """
    if server.setup_command is None:
        raise RuntimeError(f"Server '{server.name}' has no setup command")
    logger.info(f"Setting up server: {server.name}")
    labeled_command = _inject_session_label(server.setup_command.command)
    setup_process = subprocess.Popen(
        labeled_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def _monitor_logs() -> None:
        try:
            while True:
                line = setup_process.stdout.readline()
                if not line:
                    if setup_process.poll() is not None:
                        break
                    continue
                print(f"SERVER[{server.name}]: {line.rstrip()}", flush=True)
                sys.stdout.flush()
        except Exception as exc:
            logger.debug(f"Log monitoring thread exception: {exc}")

    log_thread = threading.Thread(target=_monitor_logs, daemon=True)
    log_thread.start()

    start_time = time.time()
    while time.time() - start_time < SETUP_MONITOR_TIMEOUT:
        if setup_process.poll() is not None:
            log_thread.join(timeout=2)
            if setup_process.returncode != 0:
                raise RuntimeError(
                    f"Server {server.name} setup process exited with "
                    f"code {setup_process.returncode}"
                )
            break
        time.sleep(0.1)

    logger.info(f"Server {server.name} setup started successfully")


def run_health_check(server: Server, config: E2ETestConfig) -> None:
    """Run the server health-check command. Raises RuntimeError on failure."""
    if server.health_check_command is None:
        raise RuntimeError(f"Server '{server.name}' has no health-check command")
    logger.info(f"Starting health check for server: {server.name}")
    health_process = subprocess.Popen(
        server.health_check_command.command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    _, timed_out = _drain_output(
        health_process, timeout=HEALTH_CHECK_TIMEOUT, prefix="HEALTH"
    )
    if timed_out:
        health_process.kill()
        with suppress(subprocess.TimeoutExpired):
            health_process.wait(timeout=10)
        raise RuntimeError(
            f"Health check for server {server.name} exceeded {HEALTH_CHECK_TIMEOUT}s"
        )
    try:
        health_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        health_process.kill()
        health_process.wait()
    if health_process.returncode != 0:
        logger.error(f"Health check failed for server: {server.name}")
        raise RuntimeError(f"Health check failed for server: {server.name}")
    logger.info(f"Server {server.name} health check passed")


def run_aiperf_command(
    cmd: Command,
    config: E2ETestConfig,
    container_id: str | None = None,
) -> tuple[bool, str]:
    """Execute a single AIPerf command, either locally or inside a container.

    Returns (success, captured_output).  When ``container_id`` is None the
    command is run with the locally installed aiperf; otherwise it is run via
    ``docker exec``.  For container execution a watchdog timer sends SIGKILL to
    the process group after AIPERF_COMMAND_TIMEOUT seconds.
    """
    command = cmd.command.replace(
        "aiperf profile", f"aiperf profile --ui-type {AIPERF_UI_TYPE}"
    )
    prefix = f"AIPERF[{cmd.tag_name}]"

    if container_id is None:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        lines, _ = _drain_output(proc, timeout=None, prefix=prefix)
        proc.wait()
        return proc.returncode == 0, "\n".join(lines)

    exec_command = f"docker exec {container_id} bash -c {shlex.quote(command)}"
    proc = subprocess.Popen(
        exec_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,
    )
    kill_guard = _ProcessGroupKillGuard()
    watchdog = threading.Timer(
        AIPERF_COMMAND_TIMEOUT,
        _make_process_group_timeout_killer(
            proc=proc,
            test_num=cmd.start_line,
            server_name=cmd.tag_name,
            guard=kill_guard,
        ),
    )
    watchdog.daemon = True
    watchdog.start()
    try:
        lines, _ = _drain_output(proc, timeout=None, prefix=prefix)
        proc.wait()
    finally:
        kill_guard.mark_finished()
        watchdog.cancel()
    return proc.returncode == 0, "\n".join(lines)


def teardown_server(
    server: Server,
    config: E2ETestConfig,
    aiperf_container_id: str | None = None,
) -> None:
    """Stop server containers started by this session and prune stopped ones."""
    if config.skip_server_setup:
        return
    filter_flag = f"--filter label={_SESSION_LABEL}"
    if aiperf_container_id:
        inspect = subprocess.run(
            ["docker", "inspect", "--format={{.Id}}", aiperf_container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        helper_id = inspect.stdout.strip()[:12] if inspect.returncode == 0 else ""
        exclude = f"| grep -v '^{helper_id}' " if helper_id else ""
        stop_cmd = f"docker ps -q {filter_flag} {exclude}| xargs -r docker stop 2>/dev/null || true"
    else:
        stop_cmd = (
            f"docker ps -q {filter_flag} | xargs -r docker stop 2>/dev/null || true"
        )
    with suppress(subprocess.TimeoutExpired):
        subprocess.run(stop_cmd, shell=True, capture_output=True, timeout=30)
    with suppress(subprocess.TimeoutExpired):
        subprocess.run(
            f"docker container prune -f {filter_flag}",
            shell=True,
            capture_output=True,
            timeout=10,
        )
