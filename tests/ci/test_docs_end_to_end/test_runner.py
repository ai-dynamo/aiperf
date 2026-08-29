# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test runner for executing server setup, health checks, and AIPerf tests.
"""

import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

from constants import (
    AIPERF_COMMAND_TIMEOUT,
    AIPERF_UI_TYPE,
    SETUP_MONITOR_TIMEOUT,
)
from data_types import Server, TestConfig
from utils import (
    docker_stop_and_remove,
    get_repo_root,
)

logger = logging.getLogger(__name__)


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

    def __init__(self, config: TestConfig | None = None):
        self.config = config or TestConfig()
        self.aiperf_container_id = None
        self.setup_process = None
        self.log_monitoring_thread = None
        self.stop_log_monitoring = threading.Event()
        self.server_containers: dict[str, set[str]] = {}  # Track containers per server

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
            # Skip in dev mode to preserve user's already-running containers
            if not self.config.skip_server_setup:
                logger.info(
                    "Cleaning up any leftover containers from previous test runs..."
                )
                self._cleanup_all_containers()
                logger.info("All leftover containers cleaned up")

            # Step 1: Build AIPerf container (skip if using local aiperf)
            if self.config.use_local_aiperf:
                logger.info("Using locally installed aiperf (skipping container build)")
                if not self._verify_local_aiperf():
                    logger.error("Local aiperf not found - stopping all tests")
                    return False
            else:
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

    def _verify_local_aiperf(self) -> bool:
        """Verify that aiperf is installed and accessible on the host."""
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
                return False
            logger.info(f"Local aiperf version: {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            logger.error("aiperf command not found in PATH")
            logger.error("Install with: uv run pip install -e . (from repo root)")
            return False
        except subprocess.TimeoutExpired:
            logger.error("aiperf --version timed out")
            return False

    def _validate_servers(self, servers: dict[str, Server]) -> bool:
        """Validate that all servers have required commands and no duplicates"""
        logger.info(f"Validating {len(servers)} servers...")

        for server_name, server in servers.items():
            # Check that server has setup command (skip if not managing servers)
            if server.setup_command is None and not self.config.skip_server_setup:
                logger.error(f"Server '{server_name}' missing setup command")
                return False

            # Check that server has health check command (skip if not checking health)
            if (
                server.health_check_command is None
                and not self.config.skip_health_check
            ):
                logger.error(f"Server '{server_name}' missing health-check command")
                return False

            # Check that server has at least one aiperf command
            if not server.aiperf_commands:
                logger.error(f"Server '{server_name}' missing aiperf-run commands")
                return False

            setup_status = "skipped" if self.config.skip_server_setup else "1 setup"
            health_status = (
                "skipped" if self.config.skip_health_check else "1 health-check"
            )
            logger.info(
                f"Server '{server_name}': {setup_status}, {health_status}, {len(server.aiperf_commands)} aiperf commands"
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
        logger.info(f"Testing server: {server.name}")

        # Skip server setup if configured (using already-running server)
        if self.config.skip_server_setup:
            logger.info(
                f"Skipping server setup for {server.name} (--skip-server-setup)"
            )
            self.server_containers[server.name] = set()
        else:
            if not self._setup_server(server):
                return False

        # Skip health check if configured
        if self.config.skip_health_check:
            logger.info(
                f"Skipping health check for {server.name} (--skip-health-check)"
            )
        else:
            if not self._run_health_check(server):
                return False

        logger.info(f"Server {server.name} ready for testing")

        # Run all aiperf commands for this server
        all_aiperf_passed = True
        for i, aiperf_cmd in enumerate(server.aiperf_commands):
            logger.info(
                f"Running AIPerf test {i + 1}/{len(server.aiperf_commands)} for {server.name}"
            )

            returncode = self._execute_aiperf(
                aiperf_cmd.command, server.name, i + 1, len(server.aiperf_commands)
            )

            if returncode != 0:
                logger.error("=" * 60)
                logger.error(f"AIPerf test {i + 1} failed for {server.name}")
                logger.error(f"Return code: {returncode}")
                all_aiperf_passed = False
            else:
                logger.info("=" * 60)
                logger.info(f"AIPerf test {i + 1} passed for {server.name}")

        # Cleanup server containers only if we managed them
        if not self.config.skip_server_setup:
            logger.info(
                f"All AIPerf tests completed for {server.name}. Stopping server containers..."
            )
            # Stop all containers except the aiperf test container by filtering out its name
            if self.aiperf_container_id:
                stop_cmd = f"docker ps --format '{{{{.Names}}}}' | grep -v '^{self.aiperf_container_id}$' | xargs -r docker stop 2>/dev/null || true"
                subprocess.run(stop_cmd, shell=True, capture_output=True, timeout=30)
            subprocess.run(
                "docker container prune -f", shell=True, capture_output=True, timeout=10
            )
            logger.info(
                "Server containers stopped, aiperf container preserved for next test"
            )
        else:
            logger.info(
                f"All AIPerf tests completed for {server.name} (server left running)"
            )

        return all_aiperf_passed

    def _execute_aiperf(
        self, command: str, server_name: str, test_num: int, total_tests: int
    ) -> int:
        """Execute an aiperf command, routing to local or container based on config."""
        aiperf_command_with_ui = command.replace(
            "aiperf profile", f"aiperf profile --ui-type {AIPERF_UI_TYPE}"
        )
        logger.info(
            f"Executing AIPerf command {test_num}/{total_tests} against {server_name}:"
        )
        logger.info(f"Command: {command}")
        logger.info(f"With UI flag: {aiperf_command_with_ui}")
        logger.info("=" * 60)

        if self.config.use_local_aiperf:
            return self._run_aiperf_local(aiperf_command_with_ui, server_name)
        else:
            return self._run_aiperf_container(
                aiperf_command_with_ui, server_name, test_num
            )

    def _run_aiperf_local(self, command: str, server_name: str) -> int:
        """Run aiperf command directly on the host."""
        aiperf_process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        while True:
            line = aiperf_process.stdout.readline()
            if not line and aiperf_process.poll() is not None:
                break
            if line:
                print(f"AIPERF[{server_name}]: {line.rstrip()}")
        aiperf_process.wait()
        return aiperf_process.returncode

    def _run_aiperf_container(
        self, command: str, server_name: str, test_num: int
    ) -> int:
        """Run aiperf command inside the Docker container with a watchdog timeout."""
        exec_command = f"docker exec {self.aiperf_container_id} bash -c '{command}'"

        aiperf_process = subprocess.Popen(
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
                proc=aiperf_process,
                test_num=test_num,
                server_name=server_name,
                guard=kill_guard,
            ),
        )
        watchdog.daemon = True
        watchdog.start()

        try:
            while True:
                line = aiperf_process.stdout.readline()
                if not line and aiperf_process.poll() is not None:
                    break
                if line:
                    print(f"AIPERF[{server_name}]: {line.rstrip()}")
            aiperf_process.wait()
        finally:
            kill_guard.mark_finished()
            watchdog.cancel()

        return aiperf_process.returncode

    def _setup_server(self, server: Server) -> bool:
        """Execute server setup command and monitor for early failure."""
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
        return True

    def _run_health_check(self, server: Server) -> bool:
        """Run health check for server."""
        logger.info(f"Starting health check for server: {server.name}")
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

        while True:
            line = health_process.stdout.readline()
            if not line and health_process.poll() is not None:
                break
            if line:
                print(f"HEALTH: {line.rstrip()}")

        health_process.wait()

        if health_process.returncode != 0:
            logger.error("=" * 60)
            logger.error(f"Health check failed for server: {server.name}")
            logger.error(f"Return code: {health_process.returncode}")
            return False

        logger.info("=" * 60)
        logger.info(f"Server {server.name} health check passed")
        return True

    def _cleanup(self):
        """Cleanup AIPerf container and server containers."""
        if self.aiperf_container_id:
            logger.info(f"Cleaning up AIPerf container: {self.aiperf_container_id}")
            docker_stop_and_remove(self.aiperf_container_id)

        if self.config.skip_server_setup:
            logger.info("Skipping container cleanup (--skip-server-setup mode)")
            return

        logger.info("Final cleanup - stopping all remaining containers...")
        self._cleanup_all_containers()
        logger.info("Final cleanup completed")
