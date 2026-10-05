# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test runner for executing server setup, health checks, and AIPerf tests.
"""

import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

from constants import (
    AIPERF_COMMAND_TIMEOUT,
    AIPERF_UI_TYPE,
    SETUP_MONITOR_TIMEOUT,
)
from data_types import Server
from utils import get_repo_root

logger = logging.getLogger(__name__)


# Matches an explicit --ui / --ui-type the doc already chose. The lookbehind
# keeps it from firing inside a longer token (``--no-ui-type-x``), and the
# trailing ``=|\s`` keeps ``--ui-types`` from counting as ``--ui-type``.
_UI_FLAG_RE = re.compile(r"(?<!\S)--ui(-type)?(=|\s)")


def inject_ui_type(command: str, ui_type: str = AIPERF_UI_TYPE) -> str:
    """Force the non-interactive UI unless the doc already selected one.

    cyclopts rejects a repeated parameter with "Parameter --ui-type specified
    multiple times" and exits 1, so injecting unconditionally would make every
    guide that teaches ``--ui-type``/``--ui`` impossible to tag for docs-e2e.
    """
    if _UI_FLAG_RE.search(command):
        return command
    return command.replace("aiperf profile", f"aiperf profile --ui-type {ui_type}")


def resolve_command_timeout(aiperf_cmd) -> int:
    """Pick the per-command ``timeout=`` if the guide set one, else the global.

    Shared with the unit tests so the per-command value is pinned where it is
    consumed: asserting only that the parser produced it leaves the runner free
    to ignore it, which is exactly the regression this guards.
    """
    return aiperf_cmd.timeout or AIPERF_COMMAND_TIMEOUT


def build_fixture_write_command(target: PurePosixPath) -> str:
    """Build the shell command that writes one fixture, with the path quoted.

    `path=` is an attribute of a markdown comment, so it is ordinary
    documentation text that reaches a shell. The absolute/`..` guard stops the
    file being written outside the working directory but does nothing about
    metacharacters, and an unquoted `cat > {target}` turns `path=x;touch PWNED`
    into a second command running inside the CI container. Shared with the unit
    tests so the quoting is pinned where it is actually used.
    """
    quoted = shlex.quote(str(target))
    parent = target.parent
    mkdir = f"mkdir -p {shlex.quote(str(parent))} && " if str(parent) != "." else ""
    return f"{mkdir}cat > {quoted}"


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
    timeout: int = AIPERF_COMMAND_TIMEOUT,
) -> Callable[[], None]:
    def _kill_on_timeout() -> None:
        if not guard.mark_killing_if_running(proc):
            return
        logger.error(
            f"AIPerf test {test_num} exceeded "
            f"{timeout}s timeout for {server_name}; "
            f"sending SIGKILL to process group"
        )
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)

    return _kill_on_timeout


def test_timeout_killer_skips_process_marked_finished(monkeypatch):
    killed = []
    guard = _ProcessGroupKillGuard()
    guard.mark_finished()
    proc = SimpleNamespace(pid=12345, poll=lambda: None)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    killer = _make_process_group_timeout_killer(
        proc=proc,
        test_num=1,
        server_name="test-server",
        guard=guard,
    )
    killer()

    assert killed == []


class EndToEndTestRunner:
    """Runs the end-to-end tests"""

    def __init__(self):
        self.aiperf_container_id = None
        self.setup_process = None
        self.log_monitoring_thread = None
        self.stop_log_monitoring = threading.Event()

    def _cleanup_all_containers(self):
        """Stop all containers and prune (nuclear cleanup)"""
        subprocess.run(
            "docker stop $(docker ps -q) 2>/dev/null || true",
            shell=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            "docker container prune -f",
            shell=True,
            capture_output=True,
            timeout=10,
        )

    def run_tests(self, servers: dict[str, Server]) -> bool:
        """Run complete test suite"""
        logger.info("Starting end-to-end test execution")
        start_time = time.time()

        try:
            # Step 0: Force cleanup any leftover containers from previous runs
            logger.info(
                "Cleaning up any leftover containers from previous test runs..."
            )
            self._cleanup_all_containers()
            logger.info("All leftover containers cleaned up")

            # Step 1: Build AIPerf container
            if not self._build_aiperf_container():
                logger.error("AIPerf container build failed - stopping all tests")
                return False

            # Step 2: Validate servers (no duplicates, complete definitions)
            if not self._validate_servers(servers):
                logger.error("Server validation failed - stopping all tests")
                return False

            # Step 3: Run tests for each server
            all_passed = True
            for server_name, server in servers.items():
                logger.info(f"Testing server: {server_name}")

                if not self._test_server(server):
                    logger.error(f"Server {server_name} failed")
                    all_passed = False
                else:
                    logger.info(f"Server {server_name} passed")

            return all_passed

        finally:
            self._cleanup()
            elapsed_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"Total test execution time: {elapsed_time:.2f} seconds")
            logger.info("=" * 60)

    def _build_aiperf_container(self) -> bool:
        """Build AIPerf container from Dockerfile"""
        logger.info("Building AIPerf container...")

        # Get repo root using centralized function
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

        # Show real-time build output
        build_output_lines = []
        while True:
            line = build_process.stdout.readline()
            if not line and build_process.poll() is not None:
                break
            if line:
                print(f"BUILD: {line.rstrip()}")
                build_output_lines.append(line)

        build_process.wait()

        if build_process.returncode != 0:
            logger.error("=" * 60)
            logger.error("Failed to build AIPerf container")
            logger.error(f"Return code: {build_process.returncode}")
            return False

        logger.info("=" * 60)
        logger.info("AIPerf Docker image built successfully")

        # Start the container with bash entrypoint override
        container_name = f"aiperf-test-{os.getpid()}"

        # Mount test fixtures directory for audio/image/video file tests
        repo_root = get_repo_root()
        fixtures_mount = f"-v {repo_root}/tests/fixtures:/fixtures:ro"

        run_command = f"docker run -d --name {container_name} -e HF_TOKEN {fixtures_mount} --network host --entrypoint bash aiperf:test -c 'tail -f /dev/null'"

        result = subprocess.run(
            run_command, shell=True, capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            logger.error("Failed to start AIPerf container")
            logger.error(f"Error: {result.stderr}")
            return False

        self.aiperf_container_id = container_name
        logger.info(f"AIPerf container ready: {container_name}")

        # Verify aiperf works
        verify_result = subprocess.run(
            f"docker exec {container_name} aiperf --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if verify_result.returncode != 0:
            logger.error("AIPerf verification failed")
            logger.error(f"Stdout: {verify_result.stdout}")
            logger.error(f"Stderr: {verify_result.stderr}")
            return False

        logger.info(f"AIPerf version: {verify_result.stdout.strip()}")
        return True

    def _validate_servers(self, servers: dict[str, Server]) -> bool:
        """Validate that all servers have required commands and no duplicates"""
        logger.info(f"Validating {len(servers)} servers...")

        for server_name, server in servers.items():
            # Check that server has setup command
            if server.setup_command is None:
                logger.error(f"Server '{server_name}' missing setup command")
                return False

            # Check that server has health check command
            if server.health_check_command is None:
                logger.error(f"Server '{server_name}' missing health-check command")
                return False

            # Check that server has at least one aiperf command
            if not server.aiperf_commands:
                logger.error(f"Server '{server_name}' missing aiperf-run commands")
                return False

            logger.info(
                f"Server '{server_name}': 1 setup, 1 health-check, {len(server.aiperf_commands)} aiperf commands"
            )

        logger.info("Server validation passed")
        return True

    def _monitor_server_logs(self, process: subprocess.Popen, server_name: str):
        """Continuously monitor and display server logs in background thread"""
        import sys

        try:
            while not self.stop_log_monitoring.is_set():
                line = process.stdout.readline()
                if not line:
                    # Check if process has ended
                    if process.poll() is not None:
                        break
                    continue
                # Display log line with server identification and flush immediately
                print(f"SERVER[{server_name}]: {line.rstrip()}", flush=True)
                sys.stdout.flush()
        except Exception as e:
            logger.debug(f"Log monitoring thread exception: {e}")

    def _test_server(self, server: Server) -> bool:
        """Test a single server: setup + health check + aiperf runs"""
        logger.info(f"Setting up server: {server.name}")

        # Execute setup command in background
        logger.info(f"Starting server setup for {server.name}:")
        logger.info(f"Command: {server.setup_command.command}")
        logger.info("=" * 60)

        setup_process = subprocess.Popen(
            server.setup_command.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Store the process for cleanup later
        self.setup_process = setup_process

        # Start background thread immediately to stream logs in real-time
        logger.info(f"Starting real-time log monitoring for {server.name}...")
        self.stop_log_monitoring.clear()
        self.log_monitoring_thread = threading.Thread(
            target=self._monitor_server_logs,
            args=(setup_process, server.name),
            daemon=True,
        )
        self.log_monitoring_thread.start()

        # Monitor for early failures without blocking log output
        start_time = time.time()
        while time.time() - start_time < SETUP_MONITOR_TIMEOUT:
            # Check if process failed early
            if setup_process.poll() is not None:
                if setup_process.returncode != 0:
                    logger.error("=" * 60)
                    logger.error(f"Server setup failed early: {server.name}")
                    logger.error(f"Return code: {setup_process.returncode}")
                    # Stop the log monitoring thread
                    self.stop_log_monitoring.set()
                    if self.log_monitoring_thread:
                        self.log_monitoring_thread.join(timeout=2)
                    return False
                else:
                    # Process completed successfully (some servers might do this)
                    break
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

        logger.info("=" * 60)
        logger.info(f"Server {server.name} setup started successfully")

        # Start health check immediately in parallel (it has built-in timeout)
        logger.info(f"Starting health check in parallel for server: {server.name}")
        logger.info(f"Health check command: {server.health_check_command.command}")
        logger.info("=" * 60)

        health_process = subprocess.Popen(
            server.health_check_command.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Wait for health check to complete (it has its own timeout logic)
        health_output_lines = []
        while True:
            line = health_process.stdout.readline()
            if not line and health_process.poll() is not None:
                break
            if line:
                print(f"HEALTH: {line.rstrip()}")
                health_output_lines.append(line)

        health_process.wait()

        if health_process.returncode != 0:
            logger.error("=" * 60)
            logger.error(f"Health check failed for server: {server.name}")
            logger.error(f"Return code: {health_process.returncode}")
            return False

        logger.info("=" * 60)
        logger.info(f"Server {server.name} health check passed - ready for testing")

        if not self._materialize_files(server):
            return False

        # Run all aiperf commands for this server
        all_aiperf_passed = True
        for i, aiperf_cmd in enumerate(server.aiperf_commands):
            logger.info(
                f"Running AIPerf test {i + 1}/{len(server.aiperf_commands)} for {server.name}"
            )

            # Execute aiperf command in the container with verbose output
            aiperf_command_with_ui = inject_ui_type(aiperf_cmd.command)
            # The command goes in over stdin, not interpolated into a
            # single-quoted `bash -c '...'`. Guides routinely pass JSON in
            # single quotes (--extra-inputs '{"temperature": 0}'), and wrapping
            # that in single quotes strips them: the payload arrives as
            # {temperature: 0} and fails JSON validation. Any guide using a
            # single quote is otherwise impossible to tag.
            exec_argv = [
                "docker",
                "exec",
                "-i",
                self.aiperf_container_id,
                "bash",
                "-s",
            ]

            logger.info(
                f"Executing AIPerf command {i + 1}/{len(server.aiperf_commands)} against {server.name}:"
            )
            logger.info(f"Server: {server.name}")
            logger.info(f"Command: {aiperf_cmd.command}")
            logger.info(f"With UI flag: {aiperf_command_with_ui}")
            logger.info("=" * 60)

            aiperf_process = subprocess.Popen(
                exec_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                start_new_session=True,
            )
            assert aiperf_process.stdin is not None
            aiperf_process.stdin.write(aiperf_command_with_ui + "\n")
            aiperf_process.stdin.close()

            kill_guard = _ProcessGroupKillGuard()
            command_timeout = resolve_command_timeout(aiperf_cmd)
            watchdog = threading.Timer(
                command_timeout,
                _make_process_group_timeout_killer(
                    proc=aiperf_process,
                    test_num=i + 1,
                    server_name=server.name,
                    guard=kill_guard,
                    timeout=command_timeout,
                ),
            )
            watchdog.daemon = True
            watchdog.start()

            try:
                # Show real-time output
                aiperf_output_lines = []
                while True:
                    line = aiperf_process.stdout.readline()
                    if not line and aiperf_process.poll() is not None:
                        break
                    if line:
                        print(f"AIPERF[{server.name}]: {line.rstrip()}")
                        aiperf_output_lines.append(line)

                aiperf_process.wait()
            finally:
                kill_guard.mark_finished()
                watchdog.cancel()

            if aiperf_process.returncode != 0:
                logger.error("=" * 60)
                logger.error(f"AIPerf test {i + 1} failed for {server.name}")
                logger.error(f"Return code: {aiperf_process.returncode}")
                all_aiperf_passed = False
            else:
                logger.info("=" * 60)
                logger.info(f"AIPerf test {i + 1} passed for {server.name}")

        # Cleanup: Stop all containers EXCEPT the aiperf test container
        logger.info(
            f"Test completed for {server.name}. Stopping all containers except aiperf test container..."
        )
        # Stop all containers except the aiperf test container by filtering out its name
        stop_cmd = f"docker ps --format '{{{{.Names}}}}' | grep -v '^{self.aiperf_container_id}$' | xargs -r docker stop 2>/dev/null || true"
        subprocess.run(
            stop_cmd,
            shell=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            "docker container prune -f", shell=True, capture_output=True, timeout=10
        )
        logger.info(
            "All server containers stopped, aiperf container preserved for next test"
        )

        return all_aiperf_passed

    def _materialize_files(self, server) -> bool:
        """Write a server's declared file fixtures into the AIPerf container.

        Content is piped in over stdin rather than interpolated into the shell
        command: YAML and JSON routinely contain quotes, ``$`` and newlines,
        which would otherwise be re-interpreted by the shell wrapping the
        docker exec.
        """
        for fixture in server.files:
            target = PurePosixPath(fixture.path)
            if target.is_absolute() or ".." in target.parts:
                logger.error(
                    f"Refusing to write fixture outside the working directory: "
                    f"{fixture.path} ({fixture.file_path}:{fixture.start_line})"
                )
                return False

            logger.info(f"Writing fixture {fixture.path} for {server.name}")
            write_command = build_fixture_write_command(target)
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    self.aiperf_container_id,
                    "bash",
                    "-c",
                    write_command,
                ],
                input=fixture.content,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(
                    f"Failed to write fixture {fixture.path}: {result.stderr.strip()}"
                )
                return False
        return True

    def _cleanup(self):
        """Cleanup all containers (nuclear approach)"""
        logger.info("Final cleanup - stopping all containers...")
        self._cleanup_all_containers()
        logger.info("Final cleanup completed")
