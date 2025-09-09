# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test runner for executing server setup, health checks, and AIPerf tests.
"""

import logging
import os
import subprocess

from constants import (
    AIPERF_UI_TYPE,
    SETUP_MONITOR_TIMEOUT,
)
from data_types import Server
from utils import (
    cleanup_docker_resources,
    docker_stop_and_remove,
    get_repo_root,
)

logger = logging.getLogger(__name__)


class TestRunner:
    """Runs the end-to-end tests"""

    def __init__(self):
        self.aiperf_container_id = None
        self.setup_process = None

    def run_tests(self, servers: dict[str, Server]) -> bool:
        """Run complete test suite"""
        logger.info("Starting end-to-end test execution")

        try:
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

    def _test_server(self, server: Server) -> bool:
        """Test a single server: setup + health check + aiperf runs"""
        logger.info(f"Setting up server: {server.name}")

        # Execute setup command in background with initial output monitoring
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

        # Monitor initial output for a short time to catch immediate failures
        import time

        setup_output_lines = []
        start_time = time.time()

        while (
            time.time() - start_time < SETUP_MONITOR_TIMEOUT
        ):  # Monitor for initial failures
            line = setup_process.stdout.readline()
            if line:
                print(f"SETUP: {line.rstrip()}")
                setup_output_lines.append(line)

            # Check if process failed early
            if setup_process.poll() is not None:
                if setup_process.returncode != 0:
                    logger.error("=" * 60)
                    logger.error(f"Server setup failed early: {server.name}")
                    logger.error(f"Return code: {setup_process.returncode}")
                    return False
                else:
                    # Process completed successfully (some servers might do this)
                    break

        logger.info("=" * 60)
        logger.info(f"Server {server.name} setup started in background")

        # Store the process for cleanup later
        self.setup_process = setup_process

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
            aiperf_command_with_ui = f"{aiperf_cmd.command} --ui-type {AIPERF_UI_TYPE}"
            exec_command = f"docker exec {self.aiperf_container_id} bash -c 'source /opt/aiperf/venv/bin/activate && {aiperf_command_with_ui}'"

            logger.info(
                f"Executing AIPerf command {i + 1}/{len(server.aiperf_commands)}:"
            )
            logger.info(f"Original: {aiperf_cmd.command}")
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
                    print(f"AIPERF: {line.rstrip()}")
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

        return all_aiperf_passed

    def _graceful_server_shutdown(self, server_name: str):
        """Gracefully shutdown server based on server type"""
        logger.info(f"Gracefully shutting down server: {server_name}")

        try:
            if "dynamo" in server_name.lower():
                # Dynamo-specific graceful shutdown
                logger.info("Executing Dynamo graceful shutdown...")
                shutdown_cmd = """
                    timeout 30 bash -c '
                        echo "Stopping Docker Compose services..."
                        docker compose -f docker-compose.yml down 2>/dev/null || true

                        echo "Stopping Dynamo containers..."
                        # Stop containers by Dynamo image
                        docker ps --filter ancestor=*dynamo* --format "{{.ID}}" | xargs -r docker stop 2>/dev/null || true
                        docker ps --filter ancestor=*vllm-runtime* --format "{{.ID}}" | xargs -r docker stop 2>/dev/null || true

                        # Remove containers
                        docker ps -aq --filter ancestor=*dynamo* | xargs -r docker rm 2>/dev/null || true
                        docker ps -aq --filter ancestor=*vllm-runtime* | xargs -r docker rm 2>/dev/null || true

                        echo "Dynamo graceful shutdown completed"
                    '
                """

            elif "vllm" in server_name.lower():
                # vLLM-specific graceful shutdown
                logger.info("Executing vLLM graceful shutdown...")
                shutdown_cmd = """
                    timeout 30 bash -c '
                        echo "Stopping vLLM containers..."
                        # Stop containers by vLLM image
                        docker ps --filter ancestor=*vllm* --format "{{.ID}}" | xargs -r docker stop 2>/dev/null || true

                        # Remove containers
                        docker ps -aq --filter ancestor=*vllm* | xargs -r docker rm 2>/dev/null || true

                        echo "vLLM graceful shutdown completed"
                    '
                """

            else:
                # Generic server shutdown
                logger.info("Executing generic server shutdown...")
                shutdown_cmd = f"""
                    timeout 30 bash -c '
                        echo "Stopping containers for {server_name}..."
                        docker ps --filter name={server_name} --format "{{.ID}}" | xargs -r docker stop 2>/dev/null || true
                        docker ps -aq --filter name={server_name} | xargs -r docker rm 2>/dev/null || true
                        echo "Generic server shutdown completed"
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

        # Stop the setup process if it's still running
        if self.setup_process and self.setup_process.poll() is None:
            logger.info("Terminating server setup process...")
            self.setup_process.terminate()
            try:
                self.setup_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Setup process didn't terminate, killing...")
                self.setup_process.kill()

        # Stop any containers that might be related to this server using utility functions
        cleanup_docker_resources()

    def _cleanup(self):
        """Cleanup AIPerf container and other resources"""
        if self.aiperf_container_id:
            logger.info(f"Cleaning up AIPerf container: {self.aiperf_container_id}")
            docker_stop_and_remove(self.aiperf_container_id)
