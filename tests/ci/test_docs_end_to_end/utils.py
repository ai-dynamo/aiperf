# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for the end-to-end testing framework.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    """Get the repository root path from the current test location"""
    # We're in tests/ci/basic_end_to_end, so go up 4 levels to reach repo root
    return Path(__file__).parent.parent.parent.parent


def setup_logging(
    level: int = logging.INFO, log_file: str = "test_execution.log"
) -> None:
    """Setup consistent logging configuration across all modules"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def run_command_with_timeout(
    command: str, timeout: int = 30, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command with timeout and consistent error handling"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired as e:
        logger.warning(f"Command timed out after {timeout}s: {command}")
        raise e
    except Exception as e:
        logger.error(f"Command failed: {command} - {e}")
        raise e


def run_command_with_realtime_output(command: str, prefix: str = "CMD") -> int:
    """Run a command and show real-time output with consistent formatting"""
    logger.info(f"Executing: {command}")
    logger.info("=" * 60)

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Show real-time output
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"{prefix}: {line.rstrip()}")
            output_lines.append(line)

    process.wait()
    logger.info("=" * 60)

    return process.returncode


def extract_ports_from_command(command: str) -> list[int]:
    """Extract port numbers from a command string using regex"""
    import re

    ports = []

    # Look for localhost:PORT patterns
    port_matches = re.findall(r"localhost:(\d+)", command)
    for match in port_matches:
        try:
            port = int(match)
            if 1 <= port <= 65535:  # Valid port range
                ports.append(port)
        except ValueError:
            continue

    # Also look for :PORT patterns (without localhost)
    if not ports:
        port_matches = re.findall(r":(\d+)", command)
        for match in port_matches:
            try:
                port = int(match)
                if 1000 <= port <= 65535:  # Common service ports
                    ports.append(port)
            except ValueError:
                continue

    # Remove duplicates and return
    return list(set(ports))


def docker_container_exists(container_name: str) -> bool:
    """Check if a Docker container exists"""
    try:
        result = subprocess.run(
            f"docker ps -aq --filter name={container_name}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def docker_stop_and_remove(container_name: str, timeout: int = 10) -> bool:
    """Stop and remove a Docker container gracefully"""
    try:
        # Stop container
        stop_result = subprocess.run(
            f"docker stop {container_name}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Remove container
        rm_result = subprocess.run(
            f"docker rm {container_name}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

        return stop_result.returncode == 0 and rm_result.returncode == 0
    except Exception as e:
        logger.debug(f"Error stopping/removing container {container_name}: {e}")
        return False


def validate_file_exists(file_path: str) -> bool:
    """Validate that a file exists and is readable"""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def safe_kill_process(pid: int, signal: int = 9) -> bool:
    """Safely kill a process by PID with error handling"""
    try:
        subprocess.run(f"kill -{signal} {pid}", shell=True, timeout=5)
        return True
    except Exception as e:
        logger.debug(f"Could not kill process {pid}: {e}")
        return False


def get_container_id_by_filter(filter_criteria: str) -> str | None:
    """Get Docker container ID by filter criteria"""
    try:
        result = subprocess.run(
            f"docker ps --filter {filter_criteria} --format '{{{{.ID}}}}'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]  # Return first match
        return None
    except Exception:
        return None


def cleanup_docker_resources(
    force: bool = False, exclude_containers: list[str] = None
) -> None:
    """Clean up Docker resources (containers, networks, volumes)"""
    try:
        if exclude_containers is None:
            exclude_containers = []

        # Stop all running containers except excluded ones
        logger.info("Stopping all running containers (except excluded)...")
        result = subprocess.run(
            "docker ps -q",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split("\n")
            for container_id in container_ids:
                # Check if this container should be excluded
                should_exclude = False
                for exclude_pattern in exclude_containers:
                    name_check = subprocess.run(
                        f"docker ps --filter id={container_id} --format '{{{{.Names}}}}'",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if (
                        name_check.returncode == 0
                        and exclude_pattern in name_check.stdout
                    ):
                        should_exclude = True
                        logger.info(f"Skipping container {container_id} (excluded)")
                        break

                if not should_exclude:
                    logger.info(f"Stopping container {container_id}")
                    subprocess.run(
                        f"docker stop {container_id}",
                        shell=True,
                        capture_output=True,
                        timeout=30,
                    )

        commands = [
            "docker container prune -f",
            "docker network prune -f",
        ]

        if force:
            commands.extend(["docker volume prune -f", "docker system prune -f"])

        for cmd in commands:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=10)

    except Exception as e:
        logger.debug(f"Docker cleanup had some issues: {e}")
