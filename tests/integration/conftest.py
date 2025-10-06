# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from pathlib import Path

import aiohttp
import orjson
import pytest

from tests.integration.helpers import AIPerfCLI, AIPerfSubprocessResult, FakeAIServer


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

    process = await asyncio.create_subprocess_exec(
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
            for _ in range(30):
                try:
                    async with session.get(
                        f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(0.2)

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
        cmd = [sys.executable, "-m", "aiperf.cli"] + full_args

        process = await asyncio.create_subprocess_exec(*cmd, stdout=None, stderr=None)

        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            process.kill()
            raise RuntimeError(f"AIPerf timed out after {timeout}s") from e

        return AIPerfSubprocessResult(
            exit_code=process.returncode or 0,
            stdout="",
            stderr="",
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


@pytest.fixture
def create_rankings_dataset(tmp_path: Path) -> Callable[[int], Path]:
    """Rankings dataset creator."""

    def _create_dataset(num_entries: int = 5) -> Path:
        dataset_path = tmp_path / "rankings.jsonl"
        with open(dataset_path, "w") as f:
            for i in range(num_entries):
                entry = {
                    "texts": [
                        {"name": "query", "contents": [f"What is AI topic {i}?"]},
                        {"name": "passages", "contents": [f"AI passage {i}"]},
                    ]
                }
                f.write(orjson.dumps(entry).decode("utf-8") + "\n")
        return dataset_path

    return _create_dataset
