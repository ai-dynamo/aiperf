# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test runner for executing server setup, health checks, and AIPerf tests.
"""

import logging
import os
import subprocess
import threading
import time

from constants import (
    AIPERF_UI_TYPE,
    SETUP_MONITOR_TIMEOUT,
)
from data_types import Server
from utils import (
    docker_stop_and_remove,
    force_cleanup_containers,
    get_all_container_ids,
    get_repo_root,
)

logger = logging.getLogger(__name__)


class EndToEndTestRunner:
    """Runs the end-to-end tests"""

    def __init__(self):
        self.aiperf_container_id = None
        self.setup_process = None
        self.log_monitoring_thread = None
        self.stop_log_monitoring = threading.Event()
        self.server_containers: dict[str, set[str]] = {}  # Track containers per server

    def run_tests(self, servers: dict[str, Server]) -> bool:
        """Run complete test suite"""
        logger.info("Starting end-to-end test execution")
        start_time = time.time()

        try:
            # Step 0: Force cleanup any leftover containers from previous runs
            logger.info(
                "Cleaning up any leftover containers from previous test runs..."
            )
            leftover_containers = get_all_container_ids()
            if leftover_containers:
                logger.info(f"Found {len(leftover_containers)} containers to clean up")
                force_cleanup_containers(leftover_containers)
            else:
                logger.info("No leftover containers found")

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

        # Build the container
        build_command = f"cd {repo_root} && docker build -t aiperf:test ."

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
        run_command = f"docker run -d --name {container_name} --network host --entrypoint bash aiperf:test -c 'tail -f /dev/null'"

        result = subprocess.run(
            run_command, shell=True, capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            logger.error("Failed to start AIPerf container")
            logger.error(f"Error: {result.stderr}")
            return False

        self.aiperf_container_id = container_name
        logger.info(f"AIPerf container ready: {container_name}")

        # Verify aiperf works by checking the virtual environment
        verify_result = subprocess.run(
            f"docker exec {container_name} bash -c 'source /opt/aiperf/venv/bin/activate && aiperf --version'",
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

        # Snapshot containers before server setup to track what gets created
        containers_before = get_all_container_ids()
        logger.info(
            f"Container snapshot before {server.name} setup: {len(containers_before)} containers"
        )

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

        # Add delay to allow Docker containers to be fully created
        # This is important when setup commands background docker run with &
        logger.info("Waiting for Docker containers to be created...")
        time.sleep(3)

        # Snapshot containers after server setup to identify new containers
        containers_after = get_all_container_ids()
        new_containers = containers_after - containers_before
        self.server_containers[server.name] = new_containers
        logger.info(
            f"Container snapshot after {server.name} setup: {len(containers_after)} total, {len(new_containers)} new containers created"
        )

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

        # Run all aiperf commands for this server
        all_aiperf_passed = True
        for i, aiperf_cmd in enumerate(server.aiperf_commands):
            logger.info(
                f"Running AIPerf test {i + 1}/{len(server.aiperf_commands)} for {server.name}"
            )

            # Execute aiperf command in the container with verbose output (use the virtual environment)
            # Add --ui-type simple to all aiperf commands
            aiperf_command_with_ui = aiperf_cmd.command.replace(
                "aiperf profile", f"aiperf profile --ui-type {AIPERF_UI_TYPE}"
            )
            exec_command = f"docker exec {self.aiperf_container_id} bash -c 'source /opt/aiperf/venv/bin/activate && {aiperf_command_with_ui}'"

            logger.info(
                f"Executing AIPerf command {i + 1}/{len(server.aiperf_commands)} against {server.name}:"
            )
            logger.info(f"Server: {server.name}")
            logger.info(f"Command: {aiperf_cmd.command}")
            logger.info(f"With UI flag: {aiperf_command_with_ui}")
            logger.info("=" * 60)

            aiperf_process = subprocess.Popen(
                exec_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

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

            if aiperf_process.returncode != 0:
                logger.error("=" * 60)
                logger.error(f"AIPerf test {i + 1} failed for {server.name}")
                logger.error(f"Return code: {aiperf_process.returncode}")
                all_aiperf_passed = False
            else:
                logger.info("=" * 60)
                logger.info(f"AIPerf test {i + 1} passed for {server.name}")

        # Gracefully shutdown the server after all tests complete
        logger.info(
            f"All AIPerf tests completed for {server.name}. Gracefully shutting down server..."
        )
        self._graceful_server_shutdown(server.name)

        # Cleanup server (basic cleanup - stop any containers)
        self._cleanup_server(server.name)

        # Force cleanup only the tracked containers for this server
        logger.info(f"Force cleaning up tracked containers after {server.name} test...")
        containers = self.server_containers.get(server.name, set())
        force_cleanup_containers(containers)

        return all_aiperf_passed

    def _graceful_server_shutdown(self, server_name: str):
        """Gracefully shutdown server using tracked containers"""
        logger.info(f"Gracefully shutting down server: {server_name}")

        # Get tracked containers for this server
        containers = self.server_containers.get(server_name, set())

        if not containers:
            logger.warning(
                f"No containers tracked for {server_name}, skipping shutdown"
            )
            return

        logger.info(f"Stopping {len(containers)} tracked containers for {server_name}")

        try:
            # Convert set to space-separated string
            container_list = " ".join(containers)

            # Stop containers gracefully
            shutdown_cmd = f"""
                timeout 30 bash -c '
                    echo "Stopping {len(containers)} containers for {server_name}..."
                    echo "{container_list}" | xargs -r docker stop -t 10 2>/dev/null || true
                    echo "Graceful shutdown completed"
                '
            """

            # Execute the shutdown command
            result = subprocess.run(
                shutdown_cmd, shell=True, capture_output=True, text=True, timeout=35
            )
            if result.returncode == 0:
                logger.info(f"Graceful shutdown completed for {server_name}")
                if result.stdout.strip():
                    logger.debug(f"Shutdown output: {result.stdout.strip()}")
            else:
                logger.warning(
                    f"Graceful shutdown had some issues for {server_name} (non-critical)"
                )

        except subprocess.TimeoutExpired:
            logger.warning(
                f"Warning: Graceful shutdown for {server_name} timed out after 30 seconds"
            )
        except Exception as e:
            logger.warning(f"Warning: Graceful shutdown for {server_name} failed: {e}")

    def _cleanup_server(self, server_name: str):
        """Basic cleanup for server"""
        logger.info(f"Cleaning up server: {server_name}")

        # Stop the log monitoring thread first
        if self.log_monitoring_thread and self.log_monitoring_thread.is_alive():
            logger.info("Stopping log monitoring thread...")
            self.stop_log_monitoring.set()
            self.log_monitoring_thread.join(timeout=5)
            if self.log_monitoring_thread.is_alive():
                logger.warning("Log monitoring thread did not stop gracefully")

        # Stop the setup process if it's still running
        if self.setup_process and self.setup_process.poll() is None:
            logger.info("Terminating server setup process...")
            self.setup_process.terminate()
            try:
                self.setup_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Setup process didn't terminate, killing...")
                self.setup_process.kill()

        # Note: Do NOT call cleanup_docker_resources() here as it could remove the aiperf container
        # The aiperf container must remain running until all tests are complete
        # Server-specific containers are cleaned up via force_cleanup_containers() with tracked IDs

    def _cleanup(self):
        """Cleanup AIPerf container and other resources"""
        if self.aiperf_container_id:
            logger.info(f"Cleaning up AIPerf container: {self.aiperf_container_id}")
            docker_stop_and_remove(self.aiperf_container_id)

        # Final force cleanup of all tracked containers
        logger.info("Final cleanup of all tracked containers...")
        all_tracked_containers = set()
        for server_name, containers in self.server_containers.items():
            logger.info(
                f"Adding {len(containers)} containers from {server_name} to cleanup list"
            )
            all_tracked_containers.update(containers)

        if all_tracked_containers:
            logger.info(
                f"Cleaning up {len(all_tracked_containers)} total tracked containers"
            )
            force_cleanup_containers(all_tracked_containers)
        else:
            logger.info("No tracked containers to clean up")
