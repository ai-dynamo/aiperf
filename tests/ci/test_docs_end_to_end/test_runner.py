# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run documented benchmarks against native vLLM processes in a job container."""

import asyncio
import logging
import os
import re
import shlex
import signal
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from constants import AIPERF_COMMAND_TIMEOUT, AIPERF_UI_TYPE
from data_types import Server
from native_setup import NativeServerSetup, parse_native_setup
from utils import get_repo_root

from aiperf.common.constants import IS_WINDOWS

logger = logging.getLogger(__name__)


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop the owned process group, including children that outlive its leader."""
    with suppress(ProcessLookupError):
        if IS_WINDOWS:
            if process.returncode is None:
                process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        pass
    finally:
        with suppress(ProcessLookupError):
            if IS_WINDOWS:
                if process.returncode is None:
                    process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def stream_output(process: asyncio.subprocess.Process, prefix: str) -> None:
    """Drain output while the process runs so pipe buffers cannot stall it."""
    if process.stdout is not None:
        async for line in process.stdout:
            print(f"{prefix}: {line.decode(errors='replace').rstrip()}", flush=True)


async def run_script(
    command: str,
    *,
    cwd: Path,
    prefix: str,
    timeout: float,
) -> int:
    """Run Bash with bounded lifetime and preserve heredocs and quoting."""
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-eo",
        "pipefail",
        "-c",
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=not IS_WINDOWS,
        limit=1024 * 1024,
    )
    output = asyncio.create_task(stream_output(process, prefix))
    try:
        async with asyncio.timeout(timeout):
            await process.wait()
            await output
        return process.returncode
    finally:
        await stop_process(process)
        output.cancel()
        await asyncio.gather(output, return_exceptions=True)


class EndToEndTestRunner:
    """Share one native server and working directory across a server's commands."""

    def __init__(self) -> None:
        self.work_dir: Path | None = None

    async def run_tests(self, servers: dict[str, Server]) -> bool:
        """Run every selected server and report any setup or benchmark failure."""
        all_passed = True
        for server in servers.values():
            if (
                not server.setup_command
                or not server.health_check_command
                or not server.aiperf_commands
            ):
                logger.error(
                    f"{server.name}: setup, health check, and benchmark commands are required"
                )
                return False
            try:
                setup = parse_native_setup(server.setup_command.command)
                expected_image = os.environ.get("AIPERF_DOCS_SERVER_IMAGE")
                if expected_image and setup.image != expected_image:
                    raise ValueError(
                        f"Documented image {setup.image} differs from job image {expected_image}"
                    )
                self.work_dir = Path(
                    await asyncio.to_thread(tempfile.mkdtemp, prefix="aiperf-docs-")
                )
                await asyncio.to_thread(
                    (self.work_dir / "fixtures").symlink_to,
                    get_repo_root() / "tests/fixtures",
                    target_is_directory=True,
                )
                logger.info(f"{server.name}: working directory {self.work_dir}")
                if not await self._test_server(server, setup):
                    all_passed = False
            except (OSError, ValueError, TimeoutError):
                logger.exception(
                    f"{server.setup_command.file_path}:{server.setup_command.start_line}: {server.name} failed"
                )
                all_passed = False
        return all_passed

    async def _server_python(self, setup: NativeServerSetup) -> str:
        """Keep compiled server dependencies separate from the AIPerf environment."""
        python = os.environ.get("AIPERF_DOCS_VLLM_PYTHON", sys.executable)
        if not setup.requirements:
            return python
        venv = self.work_dir / "server-venv"
        freeze = (
            "from importlib.metadata import distributions; "
            "print('\\n'.join(f'{d.metadata[\"Name\"]}=={d.version}' for d in distributions()))"
        )
        commands = [
            shlex.join([python, "-c", freeze]) + " > server-constraints.txt",
            shlex.join(
                ["uv", "venv", "--system-site-packages", "--python", python, str(venv)]
            ),
            shlex.join(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv / "bin/python"),
                    "--constraint",
                    "server-constraints.txt",
                    *setup.requirements,
                ]
            ),
        ]
        if await run_script(
            "\n".join(commands), cwd=self.work_dir, prefix="INSTALL", timeout=900
        ):
            raise ValueError("Documented server dependencies failed to install")
        return str(venv / "bin/python")

    async def _test_server(self, server: Server, setup: NativeServerSetup) -> bool:
        python = await self._server_python(setup)
        server_argv = [python, "-m", "vllm.entrypoints.openai.api_server", *setup.args]
        logger.info(f"{server.name}: {shlex.join(server_argv)}")
        process = await asyncio.create_subprocess_exec(
            *server_argv,
            cwd=self.work_dir,
            env={**os.environ, **setup.env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=not IS_WINDOWS,
            limit=1024 * 1024,
        )
        output = asyncio.create_task(stream_output(process, f"SERVER[{server.name}]"))
        exited = asyncio.create_task(process.wait())
        health = asyncio.create_task(
            run_script(
                server.health_check_command.command,
                cwd=self.work_dir,
                prefix="HEALTH",
                timeout=960,
            )
        )
        try:
            done, _ = await asyncio.wait(
                {health, exited}, return_when=asyncio.FIRST_COMPLETED
            )
            if exited in done:
                logger.error(
                    f"{server.name}: server exited before readiness ({process.returncode})"
                )
                return False
            if await health:
                logger.error(f"{server.name}: health check failed")
                return False
            all_passed = True
            for index, benchmark in enumerate(server.aiperf_commands, start=1):
                if process.returncode is not None:
                    logger.error(f"{server.name}: server exited ({process.returncode})")
                    return False
                logger.info(
                    f"{benchmark.file_path}:{benchmark.start_line}: command {index}/{len(server.aiperf_commands)}"
                )
                script = benchmark.command.replace(
                    "aiperf profile", f"aiperf profile --ui-type {AIPERF_UI_TYPE}"
                )
                script = re.sub(r"(?<![\w./])/fixtures/", "./fixtures/", script)
                try:
                    returncode = await run_script(
                        script,
                        cwd=self.work_dir,
                        prefix=f"AIPERF[{server.name}]",
                        timeout=AIPERF_COMMAND_TIMEOUT,
                    )
                except TimeoutError:
                    logger.error(
                        f"{server.name}: command {index} exceeded {AIPERF_COMMAND_TIMEOUT}s"
                    )
                    all_passed = False
                else:
                    if returncode:
                        logger.error(
                            f"{server.name}: command {index} failed ({returncode})"
                        )
                        all_passed = False
            return all_passed
        finally:
            health.cancel()
            await asyncio.gather(health, return_exceptions=True)
            await stop_process(process)
            output.cancel()
            await asyncio.gather(output, exited, return_exceptions=True)
