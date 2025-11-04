# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import os
import platform
import shlex
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import pytest

from tests.conftest import real_sleep
from tests.integration.models import (
    AIPerfMockServer,
    AIPerfResults,
    AIPerfSubprocessResult,
)

logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.INFO)


@dataclass(frozen=True)
class IntegrationTestDefaults:
    """Default test parameters."""

    # Default model to use for integration tests.
    # Note that the openai/gpt-oss-120b model crashes on macOS for some reason.
    # Defining the default model differently so we can have more variety in the tests.
    if platform.system() == "Darwin":
        model = "Qwen/Qwen3-0.6B"
    else:
        model = "openai/gpt-oss-120b"
    workers_max: int = 1
    concurrency: int = 2
    request_count: int = 10
    timeout: float = 200.0
    ui: str = "simple"


class AIPerfCLI:
    """Clean CLI wrapper for running AIPerf benchmarks."""

    def __init__(
        self,
        aiperf_runner: Callable[[list[str], float], AIPerfSubprocessResult],
    ) -> None:
        self._runner = aiperf_runner

    async def run(
        self,
        command: str,
        timeout: float = IntegrationTestDefaults.timeout,
        assert_success: bool = True,
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
            # TODO: HACK: FIXME: This is a temporary workaround for a known issue with macOS where the
            # process can exit with -6 when the process is terminated by a signal, failing the test.
            # More research is needed to root cause this issue, so for now, we will ignore it.
            if result.exit_code == -6 and platform.system() == "Darwin":
                pytest.xfail(
                    "AIPerf exited with SIGABRT (-6) on macOS - known platform issue"
                )

            self._raise_failure_error(result, perf_results)

        return perf_results

    def _raise_failure_error(
        self, result: AIPerfSubprocessResult, perf_results: AIPerfResults
    ) -> None:
        """Raise detailed error for failed AIPerf run."""
        error_parts = [f"AIPerf process failed with exit code {result.exit_code}\n"]

        if hasattr(perf_results, "log") and perf_results.log:
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


@pytest.fixture
async def mock_server_port() -> int:
    """Get an available port for the mock server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
async def aiperf_mock_server(
    mock_server_port: int,
) -> AsyncGenerator[AIPerfMockServer, None]:
    """Start AIPerf Mock Server for testing."""

    host = "127.0.0.1"
    url = f"http://{host}:{mock_server_port}"

    python_exe = get_venv_python()

    # Start the aiperf-mock-server
    process = await asyncio.create_subprocess_exec(
        python_exe,
        "-m",
        "aiperf_mock_server",
        "--host",
        host,
        "--port",
        str(mock_server_port),
        "--ttft",
        "0",
        "--itl",
        "0",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    try:
        # Wait for server to be ready
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
                    f"AIPerf Mock Server failed to become healthy after 100 attempts "
                    f"(URL: {url}/health)"
                )

        yield AIPerfMockServer(
            host=host, port=mock_server_port, url=url, process=process
        )

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
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logging.warning(
                    "Process did not exit after kill(), forcing termination"
                )
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
    aiperf_mock_server: AIPerfMockServer,
) -> AIPerfCLI:
    """AIPerf CLI wrapper."""
    return AIPerfCLI(aiperf_runner)
