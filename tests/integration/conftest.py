# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import multiprocessing
import os
import platform
import signal
import socket
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiperf_mock_server import MockServerConfig
from aiperf_mock_server import serve as aiperf_mock_server_serve

from aiperf.common.logging import AIPerfLogger
from tests.harness.utils import (
    AIPerfCLI,
    AIPerfMockServer,
    AIPerfResults,
    AIPerfRunnerFn,
    AIPerfRunnerResult,
)

logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.INFO)

_logger = AIPerfLogger(__name__)


@dataclass(frozen=True)
class IntegrationTestDefaults:
    """Default test parameters."""

    # Default model to use for integration tests.
    # Note that the openai/gpt-oss-120b model crashes on macOS for some reason.
    # Defining the default model differently so we can have more variety in the tests.
    if platform.system() == "Darwin":
        model = "Qwen/Qwen3-0.6B"
        tokenizer = "Qwen/Qwen3-0.6B"
    else:
        model = "openai/gpt-oss-120b"
        tokenizer = "openai/gpt-oss-120b"
    workers_max: int = 1
    concurrency: int = 2
    request_count: int = 10
    timeout: float = 200.0
    ui: str = "simple"


@pytest.fixture(scope="session", autouse=True)
def setup_integration_tokenizer():
    """Set up tokenizer caching for integration tests.

    This fixture runs once per test session and:
    1. Pre-caches the default tokenizer to avoid 429 rate limits
    2. Enables offline mode to prevent network requests during tests

    This prevents 429 rate limiting errors from HuggingFace when running
    many integration tests that load tokenizers.
    """
    # Check if offline mode is explicitly disabled (for CI cache warming)
    if bool(os.environ.get("AIPERF_SKIP_HF_OFFLINE", False)):
        _logger.info("HuggingFace offline mode disabled via AIPERF_SKIP_HF_OFFLINE")
        yield
        return

    # Pre-cache the tokenizer before enabling offline mode
    # This ensures the tokenizer is available for offline use
    try:
        from aiperf.common.tokenizer import Tokenizer

        tokenizer_name = IntegrationTestDefaults.tokenizer
        _logger.info(f"Pre-caching tokenizer for integration tests: {tokenizer_name}")
        Tokenizer.from_pretrained(tokenizer_name)
        Tokenizer.from_pretrained("gpt2")  # used by a lot of tests
        _logger.info("Tokenizer cached successfully")
    except Exception as e:
        _logger.warning(f"Failed to pre-cache tokenizer: {e}")
        # Don't enable offline mode if caching failed
        yield
        return

    # Enable offline mode for all subsequent tokenizer loads
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _logger.info("HuggingFace offline mode enabled for integration tests")

    yield

    # Restore original environment (optional, session scope so not strictly needed)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


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


@asynccontextmanager
async def create_server(**kwargs: Any) -> AsyncIterator[AIPerfMockServer]:
    # Get a fresh port for each server

    mp_ctx = multiprocessing.get_context("spawn")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    os.environ["AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD"] = "0"

    process: SpawnProcess = mp_ctx.Process(
        target=aiperf_mock_server_serve,
        kwargs={
            "config": MockServerConfig(
                host=host, port=port, no_tokenizer=True, **kwargs
            )
        },
        daemon=False,
    )

    process.start()

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
                    if process.exitcode is not None:
                        raise RuntimeError(
                            f"AIPerf Mock Server failed to start (exit code: {process.poll()})"
                        )
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(0.1)
            else:
                # Loop completed without break - all health checks failed
                if process.exitcode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(process.join, timeout=5.0),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                raise RuntimeError(
                    f"AIPerf Mock Server failed to become healthy after 100 attempts "
                    f"(URL: {url}/health)"
                )

            # Wait for DCGM endpoints to be ready
            for _ in range(100):
                try:
                    async with session.get(
                        f"{url}/dcgm1/metrics",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(0.1)
            else:
                # log warning but continue so that we have visibility but not fail the test
                _logger.warning(
                    f"DCGM endpoints not ready after 100 attempts (URL: {url}/dcgm1/metrics). "
                    f"GPU telemetry tests may fail."
                )

        yield AIPerfMockServer(host=host, port=port, url=url, process=process)

    finally:
        if process.exitcode is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(process.join, timeout=5.0),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                process.kill()


@pytest.fixture
def mock_server_factory() -> Callable[..., AsyncIterator[AIPerfMockServer]]:
    """Factory fixture for creating mock servers with custom CLI args.

    Usage in tests:
        async def test_custom_latency(mock_server_factory):
            async with mock_server_factory(ttft=100, itl=50) as server:
                # server has custom latency settings
                ...

        async def test_with_error_injection(mock_server_factory):
            async with mock_server_factory(fast=True, error_rate=10) as server:
                # server has 10% error rate
                ...
    """
    return create_server


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def aiperf_mock_server() -> AsyncGenerator[AIPerfMockServer, None]:
    """Start AIPerf Mock Server for testing.

    This fixture starts a mock server with 8 workers and fast mode enabled.
    It will be shared across all tests that need a mock server, except for tests that need a custom mock server,
    which will use the mock_server_factory fixture.
    """
    async with create_server(fast=True, workers=8) as server:
        yield server


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for AIPerf output."""
    output_dir = tmp_path / "aiperf_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
async def aiperf_runner(
    temp_output_dir: Path,
) -> AIPerfRunnerFn:
    """AIPerf subprocess runner."""

    async def runner(
        args: list[str], timeout: float = IntegrationTestDefaults.timeout
    ) -> AIPerfRunnerResult:
        full_args = args
        # Only add --artifact-dir for profile command (not for plot)
        if args and args[0] == "profile":
            full_args += [
                "--artifact-dir",
                str(temp_output_dir),
            ]
            # Add default tokenizer if not specified to use pre-cached tokenizer
            # This avoids 429 rate limiting from HuggingFace during tests
            if "--tokenizer" not in args:
                full_args += [
                    "--tokenizer",
                    IntegrationTestDefaults.model,
                ]
        python_exe = get_venv_python()
        cmd = [python_exe, "-m", "aiperf"] + full_args

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        }

        # Capture stdout/stderr for validation error tests
        # Use PIPE to capture output while still allowing it to be displayed
        import subprocess

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        except asyncio.TimeoutError as e:
            _logger.warning(f"AIPerf timed out after {timeout}s, sending SIGINT")
            process.send_signal(signal.SIGINT)
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                _logger.warning(
                    "Process did not exit after SIGINT(), forcing termination"
                )
                process.terminate()
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=5
                    )
                    stdout = stdout_bytes.decode("utf-8", errors="replace")
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
                except asyncio.TimeoutError:
                    _logger.warning(
                        "Process did not exit after kill(), forcing termination"
                    )
                    process.kill()
                    stdout = ""
                    stderr = ""
            raise RuntimeError(f"AIPerf timed out after {timeout}s") from e

        return AIPerfRunnerResult(
            exit_code=process.returncode or 0,
            output_dir=temp_output_dir,
            stdout=stdout,
            stderr=stderr,
        )

    return runner


@pytest.fixture
def cli(
    aiperf_runner: AIPerfRunnerFn,
) -> AIPerfCLI:
    """AIPerf CLI wrapper."""
    return AIPerfCLI(aiperf_runner)


class AIPerfSignalCLI:
    """CLI wrapper with SIGINT signal support for testing Ctrl+C cancellation.

    Note: This class does not inherit from AIPerfCLI because it needs different
    subprocess handling (stdout capture, delayed SIGINT). It uses AIPerfCLI._parse_command
    as a static method for command parsing.
    """

    def __init__(
        self,
        temp_output_dir: Path,
    ) -> None:
        self._temp_output_dir = temp_output_dir

    async def run_with_sigint(
        self,
        command: str,
        sigint_delay: float | None = None,
        timeout: float = IntegrationTestDefaults.timeout,
        wait_for_profiling: bool = False,
    ) -> AIPerfResults:
        """Run aiperf command and send SIGINT after specified delay or when profiling starts.

        Args:
            command: The aiperf command to run
            sigint_delay: Seconds to wait before sending SIGINT (ignored if wait_for_profiling=True)
            timeout: Total command timeout
            wait_for_profiling: If True, wait for "AIPerf is PROFILING" log before sending SIGINT

        Returns:
            AIPerfResults object containing output artifacts
        """
        args = AIPerfCLI._parse_command(command)
        full_args = args + [
            "--artifact-dir",
            str(self._temp_output_dir),
        ]
        # Add default tokenizer if not specified to use pre-cached tokenizer
        if "--tokenizer" not in args:
            full_args += [
                "--tokenizer",
                IntegrationTestDefaults.model,
            ]
        python_exe = get_venv_python()
        cmd = [python_exe, "-m", "aiperf"] + full_args

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        }

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            if wait_for_profiling:
                # Read stdout line by line and wait for profiling to start
                _logger.info("Waiting for AIPerf to start profiling...")
                profiling_started = False
                while not profiling_started and process.returncode is None:
                    try:
                        # Read a line with timeout to avoid blocking forever
                        line = await asyncio.wait_for(
                            process.stdout.readline(), timeout=1.0
                        )
                        if not line:
                            break
                        line_str = line.decode().strip()
                        print(line_str)
                        if "AIPerf System is PROFILING" in line_str:
                            _logger.info(
                                "AIPerf profiling started, waiting for delay..."
                            )
                            profiling_started = True
                            break
                    except asyncio.TimeoutError:
                        # Timeout reading line, continue checking
                        continue

                if not profiling_started:
                    _logger.warning(
                        "AIPerf profiling message not found, sending SIGINT anyway"
                    )
                elif sigint_delay and sigint_delay > 0:
                    # Wait for the additional delay after profiling starts
                    await asyncio.sleep(sigint_delay)
                    _logger.info(
                        f"Sending SIGINT after {sigint_delay}s delay following profiling start"
                    )
                else:
                    _logger.info("Sending SIGINT immediately after profiling started")
            else:
                # Wait for the delay before sending SIGINT
                await asyncio.sleep(sigint_delay)

            if process.returncode is None:
                process.send_signal(signal.SIGINT)

                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(process.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    _logger.warning("Process did not exit after SIGINT, forcing kill")
                    process.kill()
                    await process.wait()

        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        return AIPerfResults(
            AIPerfRunnerResult(process.returncode or 0, self._temp_output_dir)
        )


@pytest.fixture
def signal_cli(temp_output_dir: Path) -> AIPerfSignalCLI:
    """AIPerf CLI wrapper with SIGINT signal support."""
    return AIPerfSignalCLI(temp_output_dir)
