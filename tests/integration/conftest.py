# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import os
import shlex
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from pathlib import Path

import aiohttp
import pytest

from tests.conftest import real_sleep
from tests.integration.models import AIPerfResults, AIPerfSubprocessResult, FakeAIServer

# Suppress faker debug messages
logging.getLogger("faker").setLevel(logging.WARNING)


def pytest_runtest_setup(item):
    """Print test name before running each test."""
    if item.config.getoption("verbose") > 0:
        print(f"\n{'=' * 80}")
        print(f"STARTING: {item.nodeid}")
        print(f"{'=' * 80}")


def pytest_runtest_teardown(item):
    """Print test result after running each test."""
    if item.config.getoption("verbose") > 0:
        print(f"\n{'=' * 80}")
        print(f"FINISHED: {item.nodeid}")
        print(f"{'=' * 80}\n")


def get_venv_python() -> str:
    """Get the Python executable from the virtual environment."""
    # Check if we're in a virtual environment
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        python_path = Path(venv_path) / "bin" / "python"
        if python_path.exists():
            return str(python_path)
    # Fall back to sys.executable if not in a venv
    return sys.executable


class AIPerfCLI:
    """Clean CLI wrapper for running AIPerf benchmarks."""

    def __init__(
        self,
        aiperf_runner: Callable[[list[str], float], AIPerfSubprocessResult],
    ) -> None:
        self._runner = aiperf_runner

    async def run(
        self, command: str, timeout: float = 60.0, assert_success: bool = True
    ) -> AIPerfResults:
        """Run aiperf command and return results.

        Args:
            command: The aiperf command to run (e.g., "aiperf profile ...")
            timeout: Command timeout in seconds
            assert_success: Whether to raise an error if the command fails

        Returns:
            AIPerfResults object containing all output artifacts

        Raises:
            AssertionError: If assert_success is True and the command fails
        """
        args = self._parse_command(command)
        result = await self._runner(args, timeout)
        perf_results = AIPerfResults(result)

        if assert_success and result.exit_code != 0:
            self._raise_failure_error(result, perf_results)

        return perf_results

    def _raise_failure_error(
        self, result: AIPerfSubprocessResult, perf_results: AIPerfResults
    ) -> None:
        """Raise detailed error for failed AIPerf run."""
        error_parts = [f"AIPerf process failed with exit code {result.exit_code}\n"]

        if perf_results.log:
            error_parts.append(
                f"\n{'=' * 80}\nAIPERF LOG (logs/aiperf.log):\n{'=' * 80}\n"
                f"{perf_results.log}\n"
            )

        raise AssertionError("".join(error_parts))

    @staticmethod
    def _parse_command(cmd: str) -> list[str]:
        """Parse command string into args.

        Args:
            cmd: Command string to parse

        Returns:
            List of command arguments
        """
        cmd = cmd.strip().replace("\\\n", " ")
        args = shlex.split(cmd)
        return args[1:] if args and args[0] == "aiperf" else args


@pytest.fixture
async def fakeai_server_port() -> int:
    """Get an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
async def fakeai_server(fakeai_server_port: int) -> AsyncGenerator[FakeAIServer, None]:
    """Start FakeAI server, wait for it to be ready, and yield the server."""

    host = "127.0.0.1"
    url = f"http://{host}:{fakeai_server_port}"

    python_exe = get_venv_python()
    process = await asyncio.create_subprocess_exec(
        python_exe,
        "-m",
        "fakeai",
        "server",
        "--host",
        host,
        "--port",
        str(fakeai_server_port),
        "--response-delay",
        "0.01",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                try:
                    async with session.get(
                        f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await real_sleep(0.1)
            else:
                # Loop completed without break - all health checks failed
                if process.returncode is None:
                    process.terminate()
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                raise RuntimeError(
                    f"FakeAI server failed to become healthy after 30 attempts "
                    f"(URL: {url}/health)"
                )

        yield FakeAIServer(host=host, port=fakeai_server_port, url=url, process=process)

    finally:
        if process.returncode is None:
            process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5.0)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for AIPerf output."""
    output_dir = tmp_path / "aiperf_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
async def aiperf_runner(
    temp_output_dir: Path,
) -> Callable[[list[str], float], AIPerfSubprocessResult]:
    """AIPerf subprocess runner."""

    async def runner(args: list[str], timeout: float = 60.0) -> AIPerfSubprocessResult:
        full_args = args + ["--artifact-dir", str(temp_output_dir)]
        python_exe = get_venv_python()
        cmd = [python_exe, "-m", "aiperf"] + full_args

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        }

        # Pass stdout/stderr directly through for live terminal UI
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=None,
            stderr=None,
            env=env,
        )

        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            process.kill()
            raise RuntimeError(f"AIPerf timed out after {timeout}s") from e

        return AIPerfSubprocessResult(
            exit_code=process.returncode or 0,
            output_dir=temp_output_dir,
        )

    return runner


@pytest.fixture
def cli(
    aiperf_runner: Callable[[list[str], float], AIPerfSubprocessResult],
    fakeai_server: FakeAIServer,
) -> AIPerfCLI:
    """AIPerf CLI wrapper."""
    return AIPerfCLI(aiperf_runner)
